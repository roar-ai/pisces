#!/bin/bash

# num_gpus=8
# export MODEL_BASE=/mnt/iftekhar/minhquan-local/robin/data/hunyuan
# torchrun --nnodes=1 --nproc_per_node=$num_gpus \
#     fastvideo/sample/sample_t2v_hunyuan.py \
#     --height 720 \
#     --width 1280 \
#     --num_frames 125 \
#     --num_inference_steps 16 \
#     --guidance_scale 1 \
#     --embedded_cfg_scale 6 \
#     --flow_shift 17 \
#     --flow-reverse \
#     --prompt assets/prompt-moviegenbench.txt \
#     --seed 1024 \
#     --output_path data/outputs_video/hunyuan/vae_sp_16steps_POT_cosine_192/ \
#     --model_path $MODEL_BASE \
#     --dit-weight data/outputs/hy_pot_gradacc32_cosineOTmap/checkpoint-192/diffusion_pytorch_model.safetensors \
#     --vae-sp


# num_gpus=8
# export MODEL_BASE=/mnt/iftekhar/minhquan-local/robin/data/hunyuan
# torchrun --nnodes=1 --nproc_per_node=$num_gpus \
#     fastvideo/sample/sample_t2v_hunyuan.py \
#     --height 720 \
#     --width 1280 \
#     --num_frames 125 \
#     --num_inference_steps 16 \
#     --guidance_scale 1 \
#     --embedded_cfg_scale 6 \
#     --flow_shift 17 \
#     --flow-reverse \
#     --prompt assets/prompt-moviegenbench.txt \
#     --seed 1024 \
#     --output_path data/outputs_video/hunyuan/vae_sp_16steps_without_OT_192/ \
#     --model_path $MODEL_BASE \
#     --dit-weight data/outputs/hy_without_OT_gradacc32/checkpoint-192/diffusion_pytorch_model.safetensors \
#     --vae-sp


# num_gpus=8
# export MODEL_BASE=/mnt/iftekhar/minhquan-local/robin/data/hunyuan
# torchrun --nnodes=1 --nproc_per_node=$num_gpus \
#     fastvideo/sample/sample_t2v_hunyuan.py \
#     --height 720 \
#     --width 1280 \
#     --num_frames 125 \
#     --num_inference_steps 25 \
#     --guidance_scale 1 \
#     --embedded_cfg_scale 6 \
#     --flow_shift 17 \
#     --flow-reverse \
#     --prompt assets/test-long.txt \
#     --seed 1024 \
#     --output_path data/outputs_video/hunyuan/vae_sp_25steps_POT_128_test/ \
#     --model_path $MODEL_BASE \
#     --dit-weight data/outputs/hy_pot_gradacc32/checkpoint-128/diffusion_pytorch_model.safetensors \
#     --vae-sp

# num_gpus=8
# export MODEL_BASE=/mnt/iftekhar/minhquan-local/robin/data/hunyuan
# torchrun --nnodes=1 --nproc_per_node=$num_gpus \
#     fastvideo/sample/sample_t2v_hunyuan.py \
#     --height 720 \
#     --width 1280 \
#     --num_frames 125 \
#     --num_inference_steps 25 \
#     --guidance_scale 1 \
#     --embedded_cfg_scale 6 \
#     --flow_shift 17 \
#     --flow-reverse \
#     --prompt assets/test-long.txt \
#     --seed 1024 \
#     --output_path data/outputs_video/hunyuan/vae_sp_25steps_POT_192_test/ \
#     --model_path $MODEL_BASE \
#     --dit-weight data/outputs/hy_pot_gradacc32/checkpoint-192/diffusion_pytorch_model.safetensors \
#     --vae-sp


num_gpus=8
export MODEL_BASE=/mnt/iftekhar/minhquan-local/robin/data/hunyuan
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 720 \
    --width 1280 \
    --num_frames 125 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompt assets/prompt-moviegenbench.txt \
    --seed 1024 \
    --output_path data/outputs_video/hunyuan/vae_sp_50steps_POT_grpo_128/ \
    --model_path $MODEL_BASE \
    --dit-weight data/outputs/hy_pot_grpo8_gradacc4/checkpoint-128/diffusion_pytorch_model.safetensors \
    --vae-sp
