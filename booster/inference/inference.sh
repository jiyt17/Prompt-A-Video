CUDA_VISIBLE_DEVICES=1 \
python inference_chat.py --model_name path_to_Prompt-A-Video_CV \
    --output_file ./CV_prompt_boost.json
# python inference_chat.py --model_name /cpfs01/shared/Gveval2/jiyatai/llama_code/recipes/dpo/outputs/llama_OS_lora1_dpo2_merge --peft_model /cpfs01/shared/Gveval2/jiyatai/llama_code/recipes/dpo/outputs/llama_OS_lora1_dpo2_dpo1/checkpoint-96 \
#     --output_file ./llama3_boost_gpt4o_OS/prompts_webvid_test_chat_lora1_dpo2_dpo1.json