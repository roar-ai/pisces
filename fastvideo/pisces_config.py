"""Configuration helpers shared by the PISCES training entry point and tests."""

import torch


HUNYUAN_VAE_UPSAMPLE_CHANNELS = 256
HUNYUAN_VAE_TEMPORAL_SCALE = 4
HUNYUAN_VAE_SPATIAL_SCALE = 8


def get_lora_target_modules(model_type, target_modules):
    if target_modules is not None:
        return [
            module.strip() for module in target_modules.split(",") if module.strip()
        ]
    if model_type == "hunyuan":
        return [
            "img_attn_qkv",
            "img_attn_proj",
            "txt_attn_qkv",
            "txt_attn_proj",
            "linear1",
            "linear2",
        ]
    return ["to_k", "to_q", "to_v", "to_out.0"]


def validate_training_args(args):
    """Validate objective and memory settings before distributed initialization."""

    reward_enabled = (
        args.use_global_reward_loss or args.use_finegrained_reward_loss
    )
    active_objective = (
        args.use_consistency_loss
        or reward_enabled
        or args.pred_decay_weight > 0
    )
    if not active_objective:
        raise ValueError(
            "At least one training objective must be enabled: consistency, "
            "global reward, fine-grained reward, or prediction decay."
        )
    if args.use_ot_reward and not reward_enabled:
        raise ValueError("--use_ot_reward requires at least one reward loss.")
    if (args.vae_decode_checkpointing or args.vae_tiling) and not reward_enabled:
        raise ValueError(
            "VAE memory options only apply when a video reward loss is enabled."
        )
    if (
        args.use_ot_reward
        and args.use_finegrained_reward_loss
        and args.train_batch_size != 1
    ):
        raise ValueError(
            "Fine-grained token-level POT currently requires "
            "--train_batch_size 1."
        )
    if args.use_consistency_loss and args.consistency_loss_weight == 0:
        raise ValueError("The enabled consistency loss must have a non-zero weight.")
    if args.use_global_reward_loss and args.global_reward_loss_weight == 0:
        raise ValueError("The enabled global reward must have a non-zero weight.")
    if (
        args.use_finegrained_reward_loss
        and args.finegrained_reward_loss_weight == 0
    ):
        raise ValueError(
            "The enabled fine-grained reward must have a non-zero weight."
        )
    if args.use_lora and (args.lora_rank <= 0 or args.lora_alpha <= 0):
        raise ValueError("LoRA rank and alpha must be positive.")
    if args.max_train_steps is None or args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be a positive integer.")
    if args.multi_phased_distill_schedule is None:
        raise ValueError("--multi_phased_distill_schedule is required.")
    if args.num_latent_t % args.sp_size:
        raise ValueError(
            "--num_latent_t must be divisible by --sp_size, got "
            f"{args.num_latent_t} and {args.sp_size}."
        )


def validate_hunyuan_vae_decode_shape(latents, *, tiling_enabled):
    """Reject full-frame decodes that exceed PyTorch's 32-bit upsample limit."""

    if tiling_enabled:
        return
    if latents.ndim != 5:
        raise ValueError(
            "Hunyuan VAE latents must have shape [B, C, T, H, W]."
        )

    batch, _, latent_frames, latent_height, latent_width = latents.shape
    temporal_upsample_frames = HUNYUAN_VAE_TEMPORAL_SCALE * max(
        latent_frames - 1, 0
    )
    if temporal_upsample_frames == 0:
        return

    output_elements = (
        batch
        * HUNYUAN_VAE_UPSAMPLE_CHANNELS
        * temporal_upsample_frames
        * latent_height
        * HUNYUAN_VAE_SPATIAL_SCALE
        * latent_width
        * HUNYUAN_VAE_SPATIAL_SCALE
    )
    int32_max = torch.iinfo(torch.int32).max
    if output_elements > int32_max:
        raise RuntimeError(
            "The full-frame Hunyuan VAE decode would create an intermediate "
            f"upsample tensor with {output_elements:,} elements, exceeding "
            f"PyTorch's INT_MAX limit ({int32_max:,}). Increase --sp_size so "
            "each rank decodes fewer latent frames, or enable --vae_tiling "
            "(VAE_TILING=1 in scripts/distill/distill_hunyuan.sh)."
        )
