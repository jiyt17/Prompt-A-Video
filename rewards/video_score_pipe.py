import av
import numpy as np
from typing import List
from PIL import Image
import os
import json
import torch
import torch.nn as nn
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification



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



class SCORE_PIPE(nn.Module):
    def __init__(self):
        super().__init__()
        model_name="/path/to/VideoScore/weight"
        self.processor = AutoProcessor.from_pretrained(model_name,torch_dtype=torch.bfloat16)
        self.model = Idefics2ForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        torch.manual_seed(43)

    def score_video(self, videos, video_prompt):

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


        res = []
        for index, video in enumerate(videos):

            total_frames = len(video)
            if total_frames > MAX_NUM_FRAMES:
                indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
            else:
                indices = np.arange(total_frames)

            frames = [Image.fromarray(x) for x in _read_video_pyav(video, indices)]
            
            eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt[index])
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
            inputs = self.processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            num_aspects = logits.shape[-1]

            aspect_scores = []
            for i in range(num_aspects):
                aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
            res.append({'index': index, 'VQ': aspect_scores[0], 'TC': aspect_scores[1], 'DD': aspect_scores[2], 'TVA': aspect_scores[3], 'FC': aspect_scores[4]})
        
        return res
