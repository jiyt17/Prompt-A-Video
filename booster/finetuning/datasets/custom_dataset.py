# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools
import random
import json

from datasets import load_dataset, load_from_disk, Dataset


B_INST, E_INST = "[INST]", "[/INST]"
TASK_PROMPT = """You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task.
You will be prompted by people looking to create detailed, amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. You will only ever output a single video description per user request.
You should refactor the entire description to integrate the suggestions. Original prompt:\n"""

def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        # dialog_tokens = dialog_tokens[:-4] # Remove generation prompt <|start_header_id|>assistant<|end_header_id|>\n\n
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
        labels = copy.copy(dialog_tokens)
        last_idx = 0
        for n, idx in enumerate(eot_indices):
            if n % 2 == 1:
                last_idx = idx
            else:
                # labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
                labels[last_idx:idx+5] = [-100] * (idx-last_idx+5)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
        answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

        #Add labels, convert prompt token to -100 in order to ignore in loss function
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):

    dataset = []
    
    new_pairs = json.load(open('/nas/shared/Gveval2/jiyatai/cpfs/llama_code/SFT_data/OS_prompt_pairs_gpt4o_webvid.json', 'r'))
    random.shuffle(new_pairs)
    print('prompt pairs:', len(new_pairs))

    for example in new_pairs:
        dataset.append({
            "dialog": [
                {
                    "role": "user",
                    "content": TASK_PROMPT + example[0] + "\n" + "New prompt:\n"
                },
                {
                    "role": "assistant",
                    "content": example[1]
                }
            ]
        })
    
    # dataset = DatasetDict({"train": Dataset.from_list(dataset)})
    print(dataset[0])
    dataset = Dataset.from_list(dataset)
    print(dataset)
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

    return dataset
