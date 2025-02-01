import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

from PIL import Image
import open_clip
import av

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

amodel= get_aesthetic_model(clip_model="vit_l_14").cuda()
amodel.eval()

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model = model.cuda()
model.eval()

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
video_prompts = video_prompts[:1144]

res = []
for item in tqdm(video_prompts):
    aes_scores = []
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
        images = []
        for i in range(4):
            images.append(preprocess(frames[i]))
        images = torch.stack(images).cuda()
        
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = amodel(image_features)
            pred = prediction.mean().item()
        
        aes_scores.append(round(pred, 3))
    if error_flag == 0:
        item['aes_scores'] = aes_scores
        res.append(item)

        with open('/path/to/candidates_prompts_aes.json', 'w') as f:
            json.dump(res, f, indent=4)