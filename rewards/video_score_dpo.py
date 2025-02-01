import av
import numpy as np
from typing import List
from PIL import Image
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification

def _read_video_pyav(
    frame_paths:List[str], 
    max_frames:int,
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

MAX_NUM_FRAMES=16
ROUND_DIGIT=3
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, 
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""

model_name="/path/to/VideoScore/weights"
video_folder="/path/to/candidates_videos"

video_txt="/path/to/prompts_candidates.json"
video_prompts = json.load(open(video_txt, 'r'))

processor = AutoProcessor.from_pretrained(model_name,torch_dtype=torch.bfloat16)
model = Idefics2ForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# sample uniformly 8 frames from the video
print(len(video_prompts))
res = []
start_id = 0

error_list = []
for item in tqdm(video_prompts):
    refine_scores = []
    video_prompt = item['org_prompt']
    error_flag = 0
    if item['index'] < start_id:
        continue
    
    for idx in range(5):
    # for idx, video_prompt in enumerate(item['refined_prompts']):

        video_path = os.path.join(video_folder, str(item['index']) + '-' + str(idx)+'.mp4')
        # video_path = os.path.join(video_folder, str(item['index']) + '-' + str(idx), '000000.mp4')
        try:
            container = av.open(video_path)
        except:
            error_flag = 1
            error_list.append(item['index'])
            break
        total_frames = container.streams.video[0].frames
        if total_frames > MAX_NUM_FRAMES:
            indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
        else:
            indices = np.arange(total_frames)

        frames = [Image.fromarray(x) for x in _read_video_pyav(container, indices)]
        eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)
        num_image_token = eval_prompt.count("<image>")
        if num_image_token < len(frames):
            eval_prompt += "<image> " * (len(frames) - num_image_token)

        flatten_images = []
        for x in [frames]:
            if isinstance(x, list):
                flatten_images.extend(x)
            else:
                flatten_images.append(x)
        flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
        inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        num_aspects = logits.shape[-1]

        aspect_scores = []
        for i in range(num_aspects):
            aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
        # print(aspect_scores)
        refine_scores.append({'VQ': aspect_scores[0], 'TC': aspect_scores[1], 'DD': aspect_scores[2], 'TVA': aspect_scores[3], 'FC': aspect_scores[4]})

    if error_flag == 0:
        item['refined_scores'] = refine_scores
        res.append(item)

        with open('/path/to/candidates_prompts_videoscore.json', 'w') as f:
            json.dump(res, f, indent=4)
    
print(error_list)