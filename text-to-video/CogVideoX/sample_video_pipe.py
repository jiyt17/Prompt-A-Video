import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torch.nn as nn

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

class CV_PIPE(nn.Module):
    def __init__(self, args, model_cls, device):
        super().__init__()

        if isinstance(model_cls, type):
            self.model = get_model(args, model_cls)
        else:
            self.model = model_cls

        load_checkpoint(self.model, args)
        self.model.eval()
        self.args = args
        self.device = device

    def generate(self, input_prompts):
        res = []

        image_size = [480, 720]

        sample_func = self.model.sample
        T, H, W, C, F = self.args.sampling_num_frames, image_size[0], image_size[1], self.args.latent_channels, 8
        num_samples = [1]
        force_uc_zero_embeddings = ["txt"]
        device = self.model.device
        with torch.no_grad():
            for ind, text in tqdm(enumerate(input_prompts)):
                print(f"processing ind: {ind}, {text}")
                # reload model on GPU
                self.model.to(device)
                # print("rank:", rank, "start to process", text, cnt)
                # TODO: broadcast image2video
                value_dict = {
                    "prompt": text,
                    "negative_prompt": "",
                    "num_frames": torch.tensor(T).unsqueeze(0),
                }

                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(self.model.conditioner), value_dict, num_samples
                )
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, batch[key].shape)
                    elif isinstance(batch[key], list):
                        print(key, [len(l) for l in batch[key]])
                    else:
                        print(key, batch[key])
                c, uc = self.model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                
                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
                for index in range(self.args.batch_size):
                    # reload model on GPU
                    self.model.to(device)
                    samples_z = sample_func(
                        c,
                        uc=uc,
                        batch_size=1,
                        shape=(T, C, H // F, W // F),
                    )
                    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                    # Unload the model from GPU to save GPU memory
                    self.model.to("cpu")
                    torch.cuda.empty_cache() # ?
                    first_stage_model = self.model.first_stage_model
                    first_stage_model = first_stage_model.to(device)

                    latent = 1.0 / self.model.scale_factor * samples_z

                    # Decode latent serial to save GPU memory
                    recons = []
                    loop_num = (T - 1) // 2
                    for i in range(loop_num):
                        if i == 0:
                            start_frame, end_frame = 0, 3
                        else:
                            start_frame, end_frame = i * 2 + 1, i * 2 + 3
                        if i == loop_num - 1:
                            clear_fake_cp_cache = True
                        else:
                            clear_fake_cp_cache = False
                        with torch.no_grad():
                            recon = first_stage_model.decode(
                                latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                            )

                        recons.append(recon)

                    recon = torch.cat(recons, dim=2).to(torch.float32)
                    samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                    for i, vid in enumerate(samples):
                        gif_frames = []
                        for frame in vid:
                            frame = rearrange(frame, "c h w -> h w c")
                            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
                            gif_frames.append(frame)
                        res.append(gif_frames)
                    

        return res # list[list[numpy]]

