import os
import sys
import time
# import gradio as gr
from tqdm import tqdm
import json
import requests
import random
import shutil

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model

from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader

# merge lora
base_model = AutoModelForCausalLM.from_pretrained("/path/to/base_model")
peft_model_id = "/path/to/lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
print(type(model))
merged_model = model.merge_and_unload()
print(type(merged_model))
new_path = "/path/to/merge_model"
merged_model.save_pretrained(new_path)
shutil.copy("/path/to/llama3-instruct/tokenizer_config.json", new_path)
shutil.copy("/path/to/llama3-instruct/tokenizer.json", new_path)
shutil.copy("/path/to/llama3-instruct/special_tokens_map.json", new_path)