import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

from PIL import Image
import av

import sys
import MPS as mps_reward 

mps_reward_path = "/path/to/MPS_new.pt"
mps_reward_model = mps_reward.load(mps_reward_path, device='cuda', bf16=True)
mps_reward_model.requires_grad_(False)
def _read_video_pyav(
    frame_paths, 
    max_frames,
):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

video_folder="/path/to/candidates_videos"
video_txt="/path/to/prompts_candidates.json"
video_prompts = json.load(open(video_txt, 'r'))
res = []

for item in tqdm(video_prompts):
    mps_scores = []
    prompt = item['org_prompt']
    error_flag = 0
    for idx in range(5):
        video_path = os.path.join(video_folder, str(item['index']) + '-' + str(idx)+'.mp4')
        # video_path = os.path.join(video_folder, str(item['index']) + '-' + str(idx), '000000.mp4')
        try:
            container = av.open(video_path)
        except:
            error_flag = 1
            break
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 4).astype(int)
        frames = [Image.fromarray(x) for x in _read_video_pyav(container, indices)]
        with torch.no_grad():
            mps_score = mps_reward_model.score(prompt=[prompt] * 4, image=frames)
        
        mps_score = mps_score.mean().item()
        mps_scores.append(round(mps_score, 3))
    if error_flag == 0:
        item['mps_scores'] = mps_scores
        res.append(item)

        with open('/path/to/candidates_prompts_mps.json', 'w') as f:
            json.dump(res, f, indent=4)