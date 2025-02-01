cuda_id=0
CUDA_VISIBLE_DEVICES=$cuda_id python gpt_boost.py Open-Sora/configs/opensora-v1-2/inference/sample.py \
  --num-frames 2s --resolution 720p --aspect-ratio 9:16 --data-id $cuda_id \
  --llm-refine False