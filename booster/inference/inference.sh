CUDA_VISIBLE_DEVICES=0 \
python inference_chat.py --model_name /path/to/Prompt_A_Video_CV \
    --input_prompt 'A girl is playing guitar.' --output_file ./example_res.json
# python inference_chat.py --model_name /path/to/base_model --peft_model /path/to/lora \
#     --input_file vbench_test.txt --output_file ./example_res.json
