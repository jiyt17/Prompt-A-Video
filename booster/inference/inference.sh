CUDA_VISIBLE_DEVICES=0 \
python inference_chat.py --model_name /nas/shared/Gveval2/jiyatai/cpfs/llama_code/recipes/dpo/final_models/Prompt_A_Video_CV \
    --input_prompt 'A girl is playing guitar.' --output_file ./example_res.json
# python inference_chat.py --model_name /cpfs01/shared/Gveval2/jiyatai/llama_code/recipes/dpo/outputs/llama_OS_lora1_dpo2_merge --peft_model /cpfs01/shared/Gveval2/jiyatai/llama_code/recipes/dpo/outputs/llama_OS_lora1_dpo2_dpo1/checkpoint-96 \
#     --output_file ./llama3_boost_gpt4o_OS/prompts_webvid_test_chat_lora1_dpo2_dpo1.json