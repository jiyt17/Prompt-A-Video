# Prompt-A-Video

<img src="assets/teaser.png">

The code base for [Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](https://arxiv.org/pdf/2412.15156)

## Data and Weights

| Text-to-Video Model | Fine-tuning data | prompt booster |
| --- | --- | --- |
|Open-Sora 1.2 | [json link](https://huggingface.co/datasets/jiyatai/Prompt-A-Video-SFT-data/blob/main/OS_prompt_pairs_gpt4o_webvid.json) | [HF Link](https://huggingface.co/jiyatai/Prompt_A_Video_OS) |
|CogVideoX | [json link](https://huggingface.co/datasets/jiyatai/Prompt-A-Video-SFT-data/blob/main/CV_prompt_pairs_glm_webvid.json) | [HF Link](https://huggingface.co/jiyatai/Prompt_A_Video_CV) |

Test benchmark: 

> In-domain: benchmark/prompts_webvid_test.txt \
> Out-of-domain: benchmark/prompts_vbench_test.txt

## Inference

Prompt-A-Video is based on LLama3. The conda environment can be found in LLama3 or [requirements.txt](requirements.txt).

### Prompt refinement

Under booster/inference:

> bash inference.sh

### Text-to-video generation

Build the environment for CogVideoX or Open-Sora1.2.

Under text-to-video/CogVideoX:

> bash inference.sh

## Fine-tuning

## DPO

## Acknowledge
* [LLama3](https://github.com/meta-llama/llama-cookbook)
* [CogVideoX](https://github.com/THUDM/CogVideo)
* [Open-Sora1.2](https://github.com/hpcaitech/Open-Sora)
* [VideoScore](https://github.com/TIGER-AI-Lab/VideoScore)
* [MPS](https://github.com/Kwai-Kolors/MPS)
* [Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor)