'''
@File       :   utils.py
@Time       :   2023/04/05 19:18:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
* Based on CLIP code base
* https://github.com/openai/CLIP
* Checkpoint of CLIP/BLIP/Aesthetic are from:
* https://github.com/openai/CLIP
* https://github.com/salesforce/BLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''
import torch
import cv2
import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

from tqdm import tqdm
from huggingface_hub import hf_hub_download
from torch.cuda.amp import autocast


import numpy as np
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPImageProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Pad
from torch.utils.data import DataLoader

from dataclasses import dataclass
from transformers import CLIPModel as HFCLIPModel

import torch.nn.functional as F
from torch import nn, einsum

from MPS.trainer.models.base_model import BaseModelConfig
from MPS.trainer.models.cross_modeling import Cross_model
from MPS.trainer.models.clip_model import CLIPModel

from transformers import CLIPConfig
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from typing import Any, Optional, Tuple, Union

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class RM_MPS(nn.Module):
    def __init__(self, mdoel_ckpt_path=None, device='cpu', bf16=True, res=224):
        super().__init__()

        self.bf16 = bf16
        self.device = device
        self.dtype = torch.bfloat16 if bf16 else torch.float32
        
        pretrained_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

        image_processor = AutoProcessor.from_pretrained(pretrained_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path, trust_remote_code=True)

        if mdoel_ckpt_path is None:
            # model_ckpt_path = "/mnt/bn/zhangjc/code/Open-Sora/vidreward/MPS/MPS_overall_checkpoint.pth"
            mdoel_ckpt_path = "/cpfs01/shared/Gveval2/jiyatai/MPS/weight/MPS_overall_checkpoint.pth"
            # model = torch.load(model_ckpt_path).to(device, dtype=torch.bfloat16 if self.bf16 else torch.float32)     # NOTE: directly load as a model?
        
        # load the pretrain weights(e.g. original clip) then load the released MPS weights
        model = CLIPModel(ckpt=pretrained_name_or_path).to(device, dtype=self.dtype) 

        state_dict = torch.load(mdoel_ckpt_path, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=False).missing_keys
        print(f'### load MPS reward mdoel from {mdoel_ckpt_path} ... \n### missing keys: {msg}')

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor


    def infer_one_sample(self, prompt, image, condition=None):
        clip_model = self.model
        tokenizer = self.tokenizer
        device = self.device

        def _tokenize(caption):
            input_ids = tokenizer(
                caption,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            return input_ids

        image_input = image.unsqueeze(0).to(device)
        text_input = _tokenize(prompt).to(device)
        if condition is None:
            condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
        condition_batch = _tokenize(condition).repeat(text_input.shape[0],1).to(device)

        text_f, text_features = clip_model.model.get_text_features(text_input)

        image_f = clip_model.model.get_image_features(image_input)
        condition_f, _ = clip_model.model.get_text_features(condition_batch)

        sim_text_condition = einsum('b i d, b j d -> b j i', text_f, condition_f)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
        mask = mask.repeat(1,image_f.shape[1],1)
        # image_features = clip_model.cross_model(image_f, text_f, mask.half())[:,0,:]
        image_features = clip_model.cross_model(image_f, text_f, mask.to(self.dtype))[:,0,:]


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_score = clip_model.logit_scale.exp() * text_features @ image_features.T
        return image_score[0]
    

    def score(self, prompt, image, condition=None):
        ''' score batched image & prompt without gradient calculation 
            - `image_input`: (b, c, h, w)
            - `prompt`: list(str) if batch image are different, or a single str prompt if all image are with the same textual description
        '''
        def process_input_image(image):

            if isinstance(image, str) or isinstance(image, Image.Image):
                image_input = Image.open(image) if isinstance(image, str) else image
                reward_process = Compose([
                    Resize(224, interpolation=BICUBIC),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])

                image_input = reward_process(image_input).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor):
                image_input = image
            elif isinstance(image, list):
                image_input = torch.cat([process_input_image(image[i]) for i in range(len(image))])
            else:
                raise NotImplementedError
            return image_input

        image_input = process_input_image(image)
        with torch.no_grad(),autocast(enabled=self.bf16, dtype=torch.bfloat16):
            assert image_input.ndim == 4, 'score support the batch input, so organize your tensor in batch format.'
            clip_model = self.model
            tokenizer = self.tokenizer
            device = self.device

            def _tokenize(caption):
                input_ids = tokenizer(
                    caption,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                return input_ids

            text_input = _tokenize(prompt).to(device)
            if condition is None:
                condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
            condition_batch = _tokenize(condition).repeat(text_input.shape[0],1).to(device)

            text_f, text_features = clip_model.model.get_text_features(text_input)

            image_f = clip_model.model.get_image_features(image_input)
            condition_f, _ = clip_model.model.get_text_features(condition_batch)

            sim_text_condition = einsum('b i d, b j d -> b j i', text_f, condition_f)
            sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
            sim_text_condition = sim_text_condition / sim_text_condition.max()
            mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
            mask = mask.repeat(1,image_f.shape[1],1)
            image_features = clip_model.cross_model(image_f, text_f, mask.to(self.dtype))[:,0,:]


            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_score = clip_model.logit_scale.exp() * text_features @ image_features.T   # MPS take the cross product to generate the score
       
        image_score = image_score.float().detach().cpu().numpy()[0]

        return image_score
    

    def score_grad(self, prompt, image, max_length=77):
        reward = self.infer_one_sample(prompt, image)
        return reward


def load(
    mdoel_ckpt_path: str = "/cpfs01/shared/Gveval2/jiyatai/MPS/weight/MPS_overall_checkpoint.pth",
    bf16: bool = False, 
    res : int = 224, 
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ):

    model = RM_MPS(mdoel_ckpt_path=mdoel_ckpt_path, device=device, bf16=bf16, res=res).to(device)
    model.eval()
    return model


if __name__ == "__main__":
    reward_model = load()