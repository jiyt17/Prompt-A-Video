# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json
from tqdm import tqdm
import random
# import gradio as gr

import torch
from transformers import AutoTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model

from accelerate.utils import is_xpu_available

TASK_PROMPT = """You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task.
You will be prompted by people looking to create detailed, amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. You will only ever output a single video description per user request.
You should refactor the entire description to integrate the suggestions. Original prompt:\n"""

def main(
    model_name,
    peft_model: str=None,
    output_file: str=None,
    quantization: bool=False,
    max_new_tokens =250, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=False, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    with open('/path/to/prompts_webvid.txt', 'r') as f:
        res = f.readlines()
    res = [s.strip() for s in res]
    print(len(res))
    prompts_src = res[400000:] # different from fine-tuning
    prompt_list = []
    for s in prompts_src:
        count = 0
        for c in s:
            if c.isdigit(): count += 1
        if count < 3:
            prompt_list.append(s)
    prompts_src = prompt_list

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    res = []
    for ind, user_prompt in tqdm(enumerate(prompts_src)):
        refined_prompts = []

        print(f"User prompt:\n{user_prompt}")
        org_prompt = user_prompt

        user_prompt = ["<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + TASK_PROMPT + user_prompt + "\n" + "New prompt:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"] * 5
        
        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
            
        for i in range(5):
            output_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            st_index = output_text.find('New prompt:') + 22
            output_text = output_text[st_index:] #.split('\n')[0]
            refined_prompts.append(output_text)

        res.append({'index': ind, 'user_prompts': user_prompt[0], 'org_prompt': org_prompt, 'refined_prompts': refined_prompts})
    
        with open(output_file, 'w') as f:
            json.dump(res, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)
