import os
import time
from pprint import pformat

import colossalai
import torch
import torch.nn as nn
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

class OS_PIPE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # == device and dtype ==
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        cfg_dtype = cfg.get("dtype", "fp32")
        assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
        dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # == init distributed env ==
        if is_distributed():
            colossalai.launch_from_torch({})
            coordinator = DistCoordinator()
            enable_sequence_parallelism = coordinator.world_size > 1
            if enable_sequence_parallelism:
                set_sequence_parallel_group(dist.group.WORLD)
        else:
            coordinator = None
            enable_sequence_parallelism = False
        set_random_seed(seed=cfg.get("seed", 1024))

        # ======================================================
        # build model & load weights
        # ======================================================
        # == build text-encoder and vae ==
        self.text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
        self.vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

        # == prepare video size ==
        image_size = cfg.get("image_size", None)
        if image_size is None:
            resolution = cfg.get("resolution", None)
            aspect_ratio = cfg.get("aspect_ratio", None)
            assert (
                resolution is not None and aspect_ratio is not None
            ), "resolution and aspect_ratio must be provided if image_size is not provided"
            image_size = get_image_size(resolution, aspect_ratio)
        num_frames = get_num_frames(cfg.num_frames)

        # == build diffusion model ==
        input_size = (num_frames, *image_size)
        self.latent_size = self.vae.get_latent_size(input_size)
        self.model = (
            build_module(
                cfg.model,
                MODELS,
                input_size=self.latent_size,
                in_channels=self.vae.out_channels,
                caption_channels=self.text_encoder.output_dim,
                model_max_length=self.text_encoder.model_max_length,
                enable_sequence_parallelism=enable_sequence_parallelism,
            )
            .to(device, dtype)
            .eval()
        )
        self.text_encoder.y_embedder = self.model.y_embedder  # HACK: for classifier-free guidance

        # == build scheduler ==
        self.scheduler = build_module(cfg.scheduler, SCHEDULERS)
        self.cfg = cfg
        self.image_size = image_size
        self.num_frames = num_frames

    def generate(self, prompts):

        # == prepare reference ==
        reference_path = self.cfg.get("reference_path", [""] * len(prompts))
        mask_strategy = self.cfg.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        # == prepare arguments ==
        fps = self.cfg.fps
        multi_resolution = self.cfg.get("multi_resolution", None)
        batch_size = self.cfg.get("batch_size", 1) # 1
        num_sample = self.cfg.get("num_sample", 1) # 1
        loop = self.cfg.get("loop", 1)
        condition_frame_length = self.cfg.get("condition_frame_length", 5)
        condition_frame_edit = self.cfg.get("condition_frame_edit", 0.0)
        align = self.cfg.get("align", None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg_dtype = self.cfg.get("dtype", "fp32")
        dtype = to_torch_dtype(self.cfg.get("dtype", "bf16"))

        verbose = self.cfg.get("verbose", 1)
        progress_wrap = tqdm if verbose == 1 else (lambda x: x)

        # == Iter over all samples ==
        res = []
        for i in progress_wrap(range(0, len(prompts))):
            # == prepare batch prompts ==
            batch_prompts = prompts[i : i + batch_size]
            print(batch_prompts)
            ms = mask_strategy[i : i + batch_size]
            refs = reference_path[i : i + batch_size]

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs = collect_references_batch(refs, self.vae, self.image_size)

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), self.image_size, self.num_frames, fps, device, dtype
            )

            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                # 1. refine prompt by openai
                # 2. append score
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = append_score_to_prompts(
                        prompt_segment_list,
                        aes=self.cfg.get("aes", None),
                        flow=self.cfg.get("flow", None),
                        camera_motion=self.cfg.get("camera_motion", None),
                    )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    # == get prompt for loop i ==
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        refs, ms = append_generated(
                            vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )

                    # == sampling ==
                    with torch.no_grad():
                        z = torch.randn(len(batch_prompts), self.vae.out_channels, *self.latent_size, device=device, dtype=dtype)
                        masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                        samples = self.scheduler.sample(
                            self.model,
                            self.text_encoder,
                            z=z,
                            prompts=batch_prompts_loop,
                            device=device,
                            additional_args=model_args,
                            progress=verbose >= 2,
                            mask=masks,
                        )
                        samples = self.vae.decode(samples.to(dtype), num_frames=self.num_frames)
                        video_clips.append(samples)

                # == save samples ==
                if is_main_process():
                    for idx, batch_prompt in enumerate(batch_prompts):
                        video = [video_clips[i][idx] for i in range(loop)]
                        video = torch.cat(video, dim=1)
                        low = -1
                        high = 1
                        video.clamp_(min=low, max=high)
                        video.sub_(low).div_(max(high - low, 1e-5))
                        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8).numpy()
                        video = [frame for frame in video]
                        res.append(video)
        
        return res
