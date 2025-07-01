#!/bin/bash

num_gpus=8
export MODEL_BASE=/mnt/iftekhar/minhquan-local/robin/data/hunyuan
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 720 \
    --width 1280 \
    --num_frames 125 \
    --num_inference_steps 16 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompt /media/minhquan/hummingbird-video/VBench/prompts/all_dimension.txt \
    --seed 0 \
    --output_path /mnt/iftekhar/minhquan-local/robin/data/outputs_video/hunyuan/vae_sp_16steps_POT_192/ \
    --model_path $MODEL_BASE \
    --dit-weight data/outputs/hy_phase1_shift17_bs_16_HD_pot/checkpoint-192/diffusion_pytorch_model.safetensors \
    --vae-sp
