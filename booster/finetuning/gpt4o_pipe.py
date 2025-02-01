import json
import os
import re
import openai
import time
import torch

OPENAI_CLIENT = None
REFINE_PROMPTS = None
REFINE_PROMPTS_PATH = "/path/to/Open-Sora/assets/texts/t2v_pllava.txt"
REFINE_PROMPTS_TEMPLATE = """
You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task. Here are some examples of refined prompts:
{}

The refined prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The refined prompt should be in English.
User will provide an original prompt and your revised prompts, with their generated videos' scores (Visual Quality, Temporal Consistency, Dynamic Degree, Text Video Alignment, Factual Consistency, Aesthetic score, Image quality, 7 dimensions termed as VQ, TC, DD, TVA, FC, AES, MPS), and you need to give an improved prompt according to previous prompts and their scores on different dimensions. 
Each prompt is tagged with an index, and the sentence labeled as 0 is the initial prompt. Each prompt is followed by (VQ, TC, DD, TVA, FC, AES, MPS) scores. You need build upon the most successful prompts and learn from the high-scoring prompts. You need to observe the scores of each prompt in different aspects, learn from the experiences of previous prompts, and combine their strengths to generate better prompts.
The new prompts should keep the same semantic meaning with original prompt, should not add extra scene changing or too many actions, which is hard for video generation. 
Generate 3 paraphrases of the initial prompt which keep the semantic meaning and that have higher scores than all the prompts above. Respond with each new prompt in between <PROMPT> and </PROMPT>, e.g., <PROMPT>paraphrase 1</PROMPT>.
"""

def load_prompts(prompt_path, start_idx=None, end_idx=None):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    prompts = prompts[start_idx:end_idx]
    return prompts

if REFINE_PROMPTS is None:
    examples = load_prompts(REFINE_PROMPTS_PATH)
    REFINE_PROMPTS = REFINE_PROMPTS_TEMPLATE.format("\n".join(examples))
pattern = re.compile(r"<PROMPT>(.*?)</PROMPT>")

class GPT_PIPE():
    def __init__(self, api_version, api_key): 
        self.model = api_version 
        self.client = openai.AzureOpenAI( 
            azure_endpoint="xxxxx", 
            api_version=api_version, 
            api_key=api_key) 
        self.sys_prompt = REFINE_PROMPTS

    def generate_prompt(self, content): 
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model, 
                    messages=[{
                        'role': 'system',
                        'content': self.sys_prompt,
                    }, {
                        'role': 'user',
                        'content': content,
                    }],
                ) 
                response = completion.choices[0].message.content
                # print(response)
                break
            except Exception as e:
                print(e)
                time.sleep(5)
                continue
        try:
            matches = pattern.findall(response)
            assert len(matches) == 3
        except:
            return "Error"
        
        return matches

