#!/bin/bash

num_gpus=8
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --model_path /mnt/minhquan-local/data/outputs/hy_phase1_shift17_bs_16_HD/checkpoint-192 \
    --prompt_path "/mnt/minhquan/hummingbird-video/VBench/prompts/all_dimension.txt" \
    --num_frames 125 \
    --height 720 \
    --width 1280 \
    --num_inference_steps 50 \
    --output_path /mnt/minhquan-local/data/outputs_video/hunyuan_hf/ \
    --seed 0 \
    # --lora_checkpoint_dir /mnt/minhquan-local/data/outputs/hy_phase1_shift17_bs_16_HD/checkpoint-192 \

