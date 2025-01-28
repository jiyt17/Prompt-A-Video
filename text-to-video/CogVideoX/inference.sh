#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python sample_video.py --base configs/cogvideox_5b.yaml configs/inference.yaml --seed 1 --start-id 0 --output-dir "./samples/" \
    --input-file input_prompts.txt
