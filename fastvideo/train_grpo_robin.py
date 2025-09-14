import argparse
from email.policy import strict
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper, broadcast
from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
    resume_lora_optimizer,
)
from fastvideo.utils.logging_ import main_print
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
import cv2
from diffusers.video_processor import VideoProcessor
from fastvideo.utils.load import load_vae

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
from torch.nn import functional as F
from typing import List
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.reward_fn import get_reward_fn


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = -(
            (
                prev_sample.detach().to(torch.float32)
                - prev_sample_mean.to(torch.float32)
            )
            ** 2
        ) / (2 * (std_dev_t**2))
        -math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def run_sample_step(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    encoder_attention_mask,
    grpo_sample,
):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full(
                [encoder_hidden_states.shape[0]],
                timestep_value,
                device=z.device,
                dtype=torch.long,
            )
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                model_pred = transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    guidance=torch.tensor(
                        [6018.0], device=z.device, dtype=torch.bfloat16
                    ),
                    encoder_attention_mask=encoder_attention_mask,  # B, L
                    return_dict=False,
                )[0]
            z, pred_original, log_prob = flux_step(
                model_pred,
                z.to(torch.float32),
                args.eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                grpo=True,
                sde_solver=True,
            )
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original.to(torch.float32) / 0.476986
        all_latents = torch.stack(
            all_latents, dim=1
        )  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs


def grpo_one_step(
    args,
    latents,
    pre_latents,
    encoder_hidden_states,
    encoder_attention_mask,
    transformer,
    timesteps,
    i,
    sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    with torch.autocast("cuda", torch.bfloat16):
        transformer.train()
        model_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            guidance=torch.tensor(
                [6018.0], device=latents.device, dtype=torch.bfloat16
            ),
            encoder_attention_mask=encoder_attention_mask,  # B, L
            return_dict=False,
        )[0]
    z, pred_original, log_prob = flux_step(
        model_pred,
        latents.to(torch.float32),
        args.eta,
        sigma_schedule,
        i,
        prev_sample=pre_latents.to(torch.float32),
        grpo=True,
        sde_solver=True,
    )
    return log_prob


def sample_reference_model(
    args,
    step,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    encoder_attention_mask,
    inferencer,
    caption,
):
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)

    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_global_rewards = []
    all_finegrained_rewards = []
    if args.use_same_noise:
        input_latents = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w),  # （c,t,h,w)
            device=device,
            dtype=torch.bfloat16,
        )
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_encoder_attention_mask = encoder_attention_mask[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        if not args.use_same_noise:
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
            input_latents = torch.randn(
                (
                    len(batch_idx),
                    IN_CHANNELS,
                    latent_t,
                    latent_h,
                    latent_w,
                ),  # （c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )
        grpo_sample = True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents.clone(),
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_encoder_attention_mask,
                grpo_sample,
            )

        # 累积所有批次的latents和log_probs
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        vae.enable_tiling()

        video_processor = VideoProcessor(vae_scale_factor=8)

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = vae.decode(latents, return_dict=False)[0]
                videos = video_processor.postprocess_video(video)
        rank = int(os.environ["RANK"])
        from diffusers.utils import export_to_video

        export_to_video(videos[0], f"./videos/hunyuan_{rank}_{index}.mp4", fps=args.fps)
        if args.use_videoalign:
            try:
                with torch.no_grad():
                    absolute_path = os.path.abspath(
                        f"./videos/hunyuan_{rank}_{index}.mp4"
                    )
                    # print("starting video align")
                    reward = inferencer.reward(
                        [absolute_path],
                        [batch_caption[0]],
                        use_norm=True,
                    )
                    vq_reward = torch.tensor(reward[0]["VQ"]).to(device)
                    all_vq_rewards.append(vq_reward.unsqueeze(0))
                    mq_reward = torch.tensor(reward[0]["MQ"]).to(device)
                    all_mq_rewards.append(mq_reward.unsqueeze(0))
            except Exception as e:
                vq_reward = torch.tensor(-1.0).to(device)
                all_vq_rewards.append(vq_reward.unsqueeze(0))
                mq_reward = torch.tensor(-1.0).to(device)
                all_mq_rewards.append(mq_reward.unsqueeze(0))

        if args.use_robin:
            # try:
            with torch.no_grad():
                absolute_path = os.path.abspath(f"./videos/hunyuan_{rank}_{index}.mp4")
                torch_video = (
                    torch.from_numpy(videos[0])
                    .unsqueeze(0)
                    .permute(0, 1, 4, 2, 3)
                    .to(device)
                )
                global_reward, finegrained_reward = inferencer(
                    torch_video, [batch_caption[0]]
                )
                all_global_rewards.append(global_reward.unsqueeze(0))
                all_finegrained_rewards.append(finegrained_reward.unsqueeze(0))
            # except Exception as e:
            #     global_reward = torch.tensor(-1.0).to(device)
            #     all_global_rewards.append(global_reward.unsqueeze(0))
            #     finegrained_reward = torch.tensor(-1.0).to(device)
            #     all_finegrained_rewards.append(finegrained_reward.unsqueeze(0))

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_global_rewards = torch.cat(all_global_rewards, dim=0)
    all_finegrained_rewards = torch.cat(all_finegrained_rewards, dim=0)

    return (
        videos,
        z,
        all_global_rewards,
        all_finegrained_rewards,
        all_latents,
        all_log_probs,
        sigma_schedule,
    )


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def train_one_step(
    args,
    device,
    transformer,
    vae,
    inferencer,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    step,
):
    total_loss = 0.0
    optimizer.zero_grad()
    (
        encoder_hidden_states,
        encoder_attention_mask,
        caption,
    ) = next(loader)
    if args.use_group:

        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        encoder_attention_mask = repeat_tensor(encoder_attention_mask)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    (
        videos,
        latents,
        global_reward,
        finegrained_reward,
        all_latents,
        all_log_probs,
        sigma_schedule,
    ) = sample_reference_model(
        args,
        step,
        device,
        transformer,
        vae,
        encoder_hidden_states,
        encoder_attention_mask,
        inferencer,
        caption,
    )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][
        : args.sampling_steps
    ]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps = torch.tensor(
        timestep_values, device=all_latents.device, dtype=torch.long
    )
    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1][
            :, :-1
        ],  # each entry is the latent before timestep t
        "next_latents": all_latents[:, 1:][
            :, :-1
        ],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "global_rewards": global_reward.to(torch.float32),
        "finegrained_rewards": finegrained_reward.to(torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
    gathered_global_reward = gather_tensor(samples["global_rewards"])
    gathered_finegrained_reward = gather_tensor(samples["finegrained_rewards"])
    if dist.get_rank() == 0:
        print("gathered_global_reward", gathered_global_reward)
        with open("./global_reward.txt", "a") as f:
            f.write(f"{gathered_global_reward.mean().item()}\n")
        print("gathered_finegrained_reward", gathered_finegrained_reward)
        with open("./finegrained_reward.txt", "a") as f:
            f.write(f"{gathered_finegrained_reward.mean().item()}\n")

    n = len(samples["global_rewards"]) // (args.num_generations)
    global_advantages = torch.zeros_like(samples["global_rewards"])
    finegrained_advantages = torch.zeros_like(samples["finegrained_rewards"])

    for i in range(n):
        start_idx = i * args.num_generations
        end_idx = (i + 1) * args.num_generations
        group_rewards = samples["global_rewards"][start_idx:end_idx]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-6
        global_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

    samples["global_advantages"] = global_advantages

    for i in range(n):
        start_idx = i * args.num_generations
        end_idx = (i + 1) * args.num_generations
        group_rewards = samples["finegrained_rewards"][start_idx:end_idx]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-6
        finegrained_advantages[start_idx:end_idx] = (
            group_rewards - group_mean
        ) / group_std

    samples["finegrained_advantages"] = finegrained_advantages

    # best-of-n
    total_scores = (
        args.global_coef * samples["global_advantages"]
        + args.finegrained_coef * samples["finegrained_advantages"]
    )

    sorted_indices = torch.argsort(total_scores)

    top_indices = sorted_indices[-args.bestofn // 2 :]
    bottom_indices = sorted_indices[: args.bestofn // 2]
    selected_indices = torch.cat([top_indices, bottom_indices])
    shuffled_order = torch.randperm(
        len(selected_indices), device=selected_indices.device
    )
    selected_indices = selected_indices[shuffled_order]

    if args.num_generations != args.bestofn:
        for key in samples:
            samples[key] = samples[key][selected_indices]
        batch_size = len(selected_indices)

    perms = torch.stack(
        [torch.randperm(len(samples["timesteps"][0])) for _ in range(batch_size)]
    ).to(device)
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device)[:, None],
            perms,
        ]
    samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]

    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
    for i, sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = 1e-4
            adv_clip_max = 5.0
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:, _],
                sample["next_latents"][:, _],
                sample["encoder_hidden_states"],
                sample["encoder_attention_mask"],
                transformer,
                sample["timesteps"][:, _],
                perms[i][_],
                sigma_schedule,
            )
            ratio = torch.exp(new_log_probs - sample["log_probs"][:, _])

            global_advantages = torch.clamp(
                sample["global_advantages"],
                -adv_clip_max,
                adv_clip_max,
            )
            global_unclipped_loss = -global_advantages * ratio
            global_clipped_loss = -global_advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            global_loss = torch.mean(
                torch.maximum(global_unclipped_loss, global_clipped_loss)
            ) / (args.gradient_accumulation_steps * train_timesteps)

            finegrained_advantages = torch.clamp(
                sample["finegrained_advantages"],
                -adv_clip_max,
                adv_clip_max,
            )
            finegrained_unclipped_loss = -finegrained_advantages * ratio
            finegrained_clipped_loss = -finegrained_advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            finegrained_loss = torch.mean(
                torch.maximum(finegrained_unclipped_loss, finegrained_clipped_loss)
            ) / (args.gradient_accumulation_steps * train_timesteps)

            final_loss = (
                args.global_coef * global_loss
                + args.finegrained_coef * finegrained_loss
            )

            final_loss.backward()
            avg_loss = final_loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank() % 4 == 0:
            print(
                "global loss/finegrained loss",
                global_loss.item(),
                finegrained_loss.item(),
            )
            print(
                "global/finegrained reward",
                sample["global_rewards"].item(),
                sample["finegrained_rewards"].item(),
            )
            print(" ratio", ratio)
            print(
                "global advantage/finegrained advantage",
                global_advantages.item(),
                finegrained_advantages.item(),
            )
            print("final loss", final_loss.item())
        dist.barrier()
    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inferencer = None
    if args.use_videoalign:
        from fastvideo.models.videoalign.inference import VideoVLMRewardInference

        load_from_pretrained = "./videoalign_ckpt"
        dtype = torch.bfloat16
        inferencer = VideoVLMRewardInference(
            load_from_pretrained, device=f"cuda:{device}", dtype=dtype
        )

    elif args.use_robin:
        inferencer = get_reward_fn(
            "vi_clip2_OT",
            precision="fp16",
            rm_ckpt_dir="/mnt/minhquan-local/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt",
            OT_map_ckpt_dir="/media/minhquan/hummingbird-video/OT_maps_v1/OT_map_156000.pt",
            n_frames=53,
        )

    # reward_model =
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32

    main_print(f"--> loading model from {args.model_type}")

    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
    )
    # reference_transformer = load_reference_model(args)
    main_print(f"--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    # Set model as trainable.
    transformer.train()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.t, args.cfg)
    sampler = DistributedSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        seed=args.sampler_seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps
        * args.sp_size
        / args.train_sp_batch_size
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_path)
    # vae.enable_tiling()

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    # todo future
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)  # Crucial for distributed shuffling per epoch

        for step in range(init_steps + 1, args.max_train_steps + 1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, args.output_dir, step, epoch)
                dist.barrier()
            loss, grad_norm = train_one_step(
                args,
                device,
                transformer,
                vae,
                inferencer,
                optimizer,
                lr_scheduler,
                loader,
                args.max_grad_norm,
                step,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="hunyuan_hf",
        help="The type of model to train.",
    )
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument(
        "--validation_sampling_steps",
        type=str,
        default="64",
        help="use ',' to split multi sampling steps",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=str,
        default="4.5",
        help="use ',' to split multi scale",
    )
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=None,
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="noise eta",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="fps of stored video",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,
        help="seed of sampler",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_same_noise",
        action="store_true",
        default=False,
        help="whether to use same noise",
    )
    parser.add_argument(
        "--use_videoalign",
        action="store_true",
        default=False,
        help="whether to videoalign reward model",
    )
    parser.add_argument(
        "--use_robin",
        action="store_true",
        default=False,
        help="whether to use robin reward model",
    )
    parser.add_argument(
        "--timestep_fraction",
        type=float,
        default=1.0,
        help="timestep_fraction",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=1.0,
        help="shift value",
    )
    parser.add_argument(
        "--bestofn",
        type=int,
        default=8,
        help="the chosen samples in best-of-n",
    )
    parser.add_argument(
        "--vq_coef",
        type=float,
        default=1.0,
        help="vq coef",
    )
    parser.add_argument(
        "--mq_coef",
        type=float,
        default=0.0,
        help="mq coef",
    )
    parser.add_argument(
        "--global_coef",
        type=float,
        default=1.0,
        help="global coef",
    )
    parser.add_argument(
        "--finegrained_coef",
        type=float,
        default=1.0,
        help="finegrained coef",
    )
    args = parser.parse_args()
    main(args)
