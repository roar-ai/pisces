#!/usr/bin/env bash
set -euo pipefail

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE="${WANDB_MODE:-online}"

DATA_DIR="${DATA_DIR:-data}"
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained}"
RUN_NAME="${RUN_NAME:-pisces_hunyuan_full}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-256}"

MODEL_DIR="${MODEL_DIR:-$DATA_DIR/hunyuan}"
DATA_JSON_PATH="${DATA_JSON_PATH:-$DATA_DIR/HD-Mixkit-Finetune-Hunyuan/videos2caption.json}"
VALIDATION_PROMPT_DIR="${VALIDATION_PROMPT_DIR:-$DATA_DIR/HD-Mixkit-Finetune-Hunyuan/validation}"
REWARD_MODEL_CKPT_DIR="${REWARD_MODEL_CKPT_DIR:-$PRETRAINED_DIR/InternVideo2-stage2_1b-224p-f4.pt}"
OT_MAP_CKPT_DIR="${OT_MAP_CKPT_DIR:-$PRETRAINED_DIR/OT_map_156000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/outputs/$RUN_NAME}"

# The differentiable 3D VAE decode is the main activation-memory bottleneck.
# Decoder checkpointing is enabled by default. Set VAE_TILING=1 to additionally
# decode overlapping spatial/temporal tiles when checkpointing alone is not enough.
VAE_MEMORY_ARGS=(--vae_decode_checkpointing)
if [[ "${VAE_TILING:-0}" == "1" ]]; then
    VAE_MEMORY_ARGS+=(--vae_tiling --vae_tile_sample_size "${VAE_TILE_SAMPLE_SIZE:-256}")
fi

torchrun --standalone --nnodes 1 --nproc_per_node "$NUM_GPUS" \
    fastvideo/distill.py \
    --seed 42 \
    --pretrained_model_name_or_path "$MODEL_DIR" \
    --dit_model_name_or_path "$MODEL_DIR/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
    --model_type hunyuan \
    --cache_dir "$DATA_DIR/.cache" \
    --data_json_path "$DATA_JSON_PATH" \
    --validation_prompt_dir "$VALIDATION_PROMPT_DIR" \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 8 \
    --sp_size 2 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 32 \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --learning_rate 1e-6 \
    --mixed_precision bf16 \
    --master_weight_type bf16 \
    --checkpointing_steps 64 \
    --validation_steps 64 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 4 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir "$OUTPUT_DIR" \
    --tracker_project_name PISCES \
    --num_height 720 \
    --num_width 1280 \
    --num_frames 125 \
    --shift 17 \
    --validation_guidance_scale 1.0 \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule 4000-1 \
    --not_apply_cfg_solver \
    --reward_model_ckpt_dir "$REWARD_MODEL_CKPT_DIR" \
    --ot_map_ckpt_dir "$OT_MAP_CKPT_DIR" \
    --reward_num_frames 26 \
    --use_ot_reward \
    --use_global_reward_loss \
    --use_finegrained_reward_loss \
    --use_consistency_loss \
    --global_reward_loss_weight 1.0 \
    --finegrained_reward_loss_weight 1.0 \
    --consistency_loss_weight 1.0 \
    --use_lora \
    --lora_rank 256 \
    --lora_alpha 32 \
    --lora_target_modules "img_attn_qkv,img_attn_proj,txt_attn_qkv,txt_attn_proj,linear1,linear2" \
    "${VAE_MEMORY_ARGS[@]}"
