import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT

import json
from PIL import Image

from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode

import sys
sys.path.append("/path/tp/Prompt-A-Video")
from text-to-video.OpenSora.opensora.utils.config_utils import parse_configs
from text-to-video.OpenSora.inference_pipe import OS_PIPE
from gpt4o_pipe import GPT_PIPE
from rewards.video_score_pipe import SCORE_PIPE
from rewards.aes_pipe import AES_PIPE
import rewards.MPS as mps_reward  

def _read_video_pyav(
        video_fs, 
        max_frames:int,
    ):
        frames = []
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(video_fs):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        # return np.stack([x.to_ndarray(format="rgb24") for x in frames])
        return frames

if __name__ == "__main__":
    cfg = parse_configs(training=False)

    # load models
    Opensora = OS_PIPE(cfg)

    gpt = GPT_PIPE(api_version="gpt-4-turbo-2024-04-09", api_key="xxxxxx")

    videoscore = SCORE_PIPE()

    aes_model = AES_PIPE()

    mps_reward_path = "/path/to/MPS/MPS_overall_checkpoint_new.pt"
    mps_reward_model = mps_reward.load(mps_reward_path, device='cuda', bf16=True)
    mps_reward_model.requires_grad_(False)

    # construct boost data
    with open('/path/to/prompts_webvid.txt', 'r') as f:
        res = f.readlines()
    res = [s.strip() for s in res]
    print(len(res)) 

    prompt_list = []
    for s in res:
        count = 0
        for c in s:
            if c.isdigit(): count += 1
        if count < 3:
            prompt_list.append(s)

    res = []
    # res = json.load(open(f'prompt_pairs_long/gpt4o_webvid_{cfg.data_id}.json', 'r'))
    for prompt in tqdm(prompt_list):
        prompt_score_iters = []
        input_text = [prompt]
        video = Opensora.generate(input_text)
        # print(prompt)
        v_score = videoscore.score_video(video, [prompt]) # ?
        # print(v_score)
        total_frames = len(video[0])
        indices = np.arange(0, total_frames, total_frames / 4).astype(int)
        frames = [Image.fromarray(x) for x in _read_video_pyav(video[0], indices)]
        mps_score = mps_reward_model.score(prompt=[prompt] * 4, image=frames)
        # print(mps_score)
        aes_score = aes_model.score(frames)
        # print(aes_score)
        init_score = v_score[0]
        init_score['AES'] = round(float(np.mean(aes_score) * 5 / 8), 2)
        init_score['MPS'] = round(float(np.mean(mps_score) / 4), 2)
        prompt_score_iters.append([{'prompt':prompt, 'scores':init_score}])

        iter_times = 0
        score_str = []
        score_overall_list = [init_score['VQ'] + init_score['FC'] + init_score['TC'] + 0.5 * init_score['DD'] + init_score['TVA'] + init_score['AES'] + init_score['MPS']]
        for s in v_score:
            vq = round(s['VQ'], 2)
            tc = round(s['TC'], 2)
            dd = round(s['DD'], 2)
            tva = round(s['TVA'], 2)
            fc = round(s['FC'], 2)
            score_str.append(f" (score, VQ:{vq}, TC:{tc}, DD:{dd}, TVA:{tva}, FC:{fc}, AES: {init_score['AES']}, MPS: {init_score['MPS']})\n")
        while True:
            st = len(input_text) - len(v_score)
            for i in range(len(v_score)):
                input_text[st+i] += score_str[i]
            if len(input_text) > 4:
                ind = 0
                prev_prompts = str(ind) + '. ' + input_text[0]
                top_indices = sorted(range(1, len(score_overall_list)), key=lambda i: score_overall_list[i], reverse=True)[:3]
                for i in top_indices:
                    ind += 1
                    prev_prompts += str(ind) + '. ' + input_text[i]
            else:
                prev_prompts = ""
                ind = 0
                for i in range(len(input_text)):
                    prev_prompts += str(ind) + '. ' + input_text[i]
                    ind += 1
            print(prev_prompts)
            new_prompts = gpt.generate_prompt(prev_prompts)
            if new_prompts == 'Error':
                print('produce prompts error')
                break
            input_text = input_text + new_prompts
            video = Opensora.generate(new_prompts)
            v_score = videoscore.score_video(video, [prompt] * 3)
            iter_times += 1
            score_str = []
            prompt_score_iter_cur = []
            for i, s in enumerate(v_score):
                frames = [Image.fromarray(x) for x in _read_video_pyav(video[i], indices)]
                mps_score = mps_reward_model.score(prompt=[prompt] * 4, image=frames)
                aes_score = aes_model.score(frames)
                s['AES'] = round(float(np.mean(aes_score) * 5 / 8), 2)
                s['MPS'] = round(float(np.mean(mps_score) / 4), 2)
                prompt_score_iter_cur.append({'prompt':new_prompts[i], 'scores':s})
                vq = round(s['VQ'], 2)
                tc = round(s['TC'], 2)
                dd = round(s['DD'], 2)
                tva = round(s['TVA'], 2)
                fc = round(s['FC'], 2)
                score_str.append(f" (score, VQ:{vq}, TC:{tc}, DD:{dd}, TVA:{tva}, FC:{fc}, AES: {s['AES']}, MPS: {s['MPS']})\n")
                score_overall_list.append(vq+tc+dd*0.5+fc+tva + s['AES'] + s['MPS'])
            prompt_score_iters.append(prompt_score_iter_cur)
            if iter_times >= 8:
                break
        
        res.append(prompt_score_iters)
        with open(f'prompt_pairs_long/gpt4o_webvid_{cfg.data_id}.json', 'w') as f:
            json.dump(res, f, indent=4)
