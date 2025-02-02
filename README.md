# Prompt-A-Video

<img src="assets/teaser.png">

The code base for [Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](https://arxiv.org/pdf/2412.15156)

## 1. Data and Weights

| Text-to-Video Model | Fine-tuning data | prompt booster |
| --- | --- | --- |
|Open-Sora 1.2 | [json link](https://huggingface.co/datasets/jiyatai/Prompt-A-Video-SFT-data/blob/main/OS_prompt_pairs_gpt4o_webvid.json) | [HF Link](https://huggingface.co/jiyatai/Prompt_A_Video_OS) |
|CogVideoX | [json link](https://huggingface.co/datasets/jiyatai/Prompt-A-Video-SFT-data/blob/main/CV_prompt_pairs_glm_webvid.json) | [HF Link](https://huggingface.co/jiyatai/Prompt_A_Video_CV) |

Test benchmark: 

> In-domain: benchmark/prompts_webvid_test.txt \
> Out-of-domain: benchmark/prompts_vbench_test.txt

## 2. Inference

Prompt-A-Video is based on LLama3-instruct. The conda environment can be found in LLama3 or [requirements.txt](requirements.txt).

### 2.1 Prompt refinement

Under booster/inference:

> bash inference.sh

### 2.2 Text-to-video generation

Build the environment for CogVideoX or Open-Sora1.2.

Under text-to-video/CogVideoX:

> bash inference.sh

## 3. Fine-tuning

### 3.1 Data construction

We adopt reward-based prompt evolution pipeline, which consists of three parts: LLM (GPT4o/GLM4), text-to-video models (Open-Sora1.2/CogVideoX), reward models (VideoScore, MPS, Aes-predictor).

*We only provide core code of text-to-video and reward models in folders 'text-to-video' and 'rewards', please refer to original projects in Acknowledgement for more details.*

The constructed fine-tuning data can be found [here](https://huggingface.co/datasets/jiyatai/Prompt-A-Video-SFT-data). To run the process, under booster/finetuning:

> bash gpt_boost.sh

### 3.2 Training

To fine-tune llama3-instruct with Lora, under booster/finetuning:

> bash finetune.sh

## 4. DPO

### 4.1 Data construction

With the fine-tuned LLM, we construct DPO triplets data: {'prompt', 'chosen', 'rejected'}.

First, under booster/dpo, we generate five refined candidate prompts for each original prompt.

> bash inference.sh

Second, we produce corresponding videos for candidate prompts. (CogVideoX/sample_video_dpo.py, OpenSora/inference_dpo.py)

Then, we use reward models to evaluate the quality of videos. (rewards/aes_predictor_dpo.py, mps_dpo.py, video_score_dpo.py)

Finally, we combine three types of scores and conduct filter to gain DPO triplets data. 

> python filter.py

DPO triplet example:

```json
{
    "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are an expert prompt optimizer for text-to-video models. You translate prompts written by humans into better prompts for the text-to-video models. Original prompt:\nFisherman in boat at sunset\nNew prompt:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "chosen": "In the twilight, a fisherman is shown at the helm of his boat on a peaceful lake. The sun has set, casting a warm orange glow across the water and the sky\u2019s soft colors. The water is still, reflecting the serene atmosphere of the evening. The boat is trimmed with the essentials for fishing, providing a sense of purpose and activity despite the tranquil setting. The focus is on the fisherman and the serene conditions around him.",
    "rejected": "An old man stands in a small, rustic boat on a serene lake in the calm of a warm sunset. The sky is ablaze with diverse hues of orange and pink, and the surrounding nature includes tall trees with green leaves. He is wearing a baseball cap, a sweater, and waders. In his hand, he holds a fishing pole, and the boat drifts gently across the lake's tranquil water. There is no other object inside the boat."
}
```

### 4.2 Training

We first merge the fine-tuned Lora with the base model with booster/merge.py. Then we train DPO for the merged LLM under booster/dpo:

> python dpo.py

To further improve metrics, we can continue to conduct DPO. Merge the lora after DPO training, then build DPO data and train again.

## Acknowledgement
* [LLama3](https://github.com/meta-llama/llama-cookbook)
* [CogVideoX](https://github.com/THUDM/CogVideo)
* [Open-Sora1.2](https://github.com/hpcaitech/Open-Sora)
* [VideoScore](https://github.com/TIGER-AI-Lab/VideoScore)
* [MPS](https://github.com/Kwai-Kolors/MPS)
* [Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor) 
