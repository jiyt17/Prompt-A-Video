CUDA_VISIBLE_DEVICES=0 python scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 2s --resolution 720p --aspect-ratio 9:16 \
  --prompt-path /cpfs01/shared/Gveval2/jiyatai/llama_code/recipes/inference/local_inference/llama3_boost_gpt4o_OS/prompts_webvid_test_chat_lora1_dpo2_dpo1.json --save-dir ./samples/webvid_test_chat_lora1_dpo2_dpo1 \
  --start-index 0
