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



class AES_PIPE(nn.Module):
    def __init__(self):
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.model = self.model.cuda()
        self.model.eval()

        self.amodel = nn.Linear(768, 1)
        path_to_model = "/path/to/aesthetic_predictor/sa_0_4_vit_l_14_linear.pth"
        s = torch.load(path_to_model)
        self.amodel.load_state_dict(s)
        self.amodel = self.amodel.cuda()
        self.amodel.eval()

    def score(self, frames): # List[Image]

        images = []
        for i in range(4):
            images.append(self.preprocess(frames[i]))
        images = torch.stack(images).cuda()
        
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.amodel(image_features)
            # pred = prediction.mean().item()
            pred = prediction.float().squeeze().detach().cpu().numpy()

        return pred
        