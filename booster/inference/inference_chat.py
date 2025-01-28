# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import tqdm
import json

import torch
from transformers import AutoTokenizer

from llama_recipes.inference.chat_utils import read_dialogs_from_file
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker
from accelerate.utils import is_xpu_available

TASK_PROMPT = """You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task.
You will be prompted by people looking to create detailed, amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. You will only ever output a single video description per user request.
You should refactor the entire description to integrate the suggestions. Original prompt:\n"""

dialogs = [[
    {"role": "user", "content": ""}
]]

def main(
    model_name,
    peft_model: str=None,
    input_file: str=None,
    input_prompt: str=None,
    output_file: str=None,
    quantization: bool=False,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    enable_llamaguard_content_safety: bool = False,
    **kwargs
):
    res = []
    if input_file:
        with open(input_file, 'r') as f:
            prompts_src = f.readlines()
    elif input_prompt:
        prompts_src = [input_prompt]
    else:
        sys.exit('input is None.')
    prompts_src = [p.strip() for p in prompts_src]
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )

    for user_prompt in prompts_src:
        print(f"----------------\nUser prompt:\n{user_prompt}")

        dialogs[0][-1]['content'] = TASK_PROMPT + user_prompt + "\n" + "New prompt:\n"

        chats = tokenizer.apply_chat_template(dialogs)

        with torch.no_grad():
            for idx, chat in enumerate(chats):
                tokens= torch.tensor(chat).long()
                tokens= tokens.unsqueeze(0)
                if is_xpu_available():
                    tokens= tokens.to("xpu:0")
                else:
                    tokens= tokens.to("cuda:0")
                outputs = model.generate(
                    input_ids=tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    **kwargs
                )

                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = output_text.split('assistant')[-1].strip()

                print(response)
                res.append([user_prompt, response])
    
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
