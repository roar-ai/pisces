"""DDP training for an Optimal Transport map between InternVideo2 text and video features.

Launch with:
    torchrun --nproc_per_node=8 train_OT_map.py [args]

Computes mutual k-NN (cross-modal alignment) and Spearman structure preservation
at every checkpoint, and tracks separate best checkpoints for each metric.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import InterpolationMode, Normalize, RandomCrop
from tqdm import tqdm
from webdataset import WebLoader

from fastvideo.intern_vid2.demo_config import Config, eval_dict_leaf
from fastvideo.intern_vid2.demo_utils import setup_internvideo2
from fastvideo.optimal_transport import (
    OptimalTransportMap,
    PotentialNetwork,
    extract_ot_map_state_dict,
    load_ot_checkpoint,
)

# We rebuild the video2dataset pipeline ourselves so we can (a) infer
# original_height/width from the first decoded frame when the json sidecar
# doesn't carry them (true for our webvid tars), and (b) wrap the resizer
# in wds.warn_and_continue so a single corrupt mp4 doesn't kill a rank.
import webdataset as wds
from video2dataset.dataloader.custom_wds import dict_collation_fn
from video2dataset.dataloader.dataloader import reassemble as _reassemble
from video2dataset.dataloader.filters import KeyFilter
from video2dataset.dataloader.transform import VideoResizer
from video2dataset.dataloader.video_decode import VideoDecorder

plt.switch_backend("Agg")


def build_video_loader(
    urls,
    batch_size,
    decoder_kwargs,
    resize_size,
    crop_size,
    shuffle=0,
    repeat=False,
    drop_last=True,
    video_key="mp4",
    min_video_bytes=10000,
):
    """Drop-in replacement for video2dataset.get_video_dataset with:
    - warn_and_continue handlers on every step that touches video bytes
    - pre-decode size filter that drops `AccessDenied`-style stubs (~4% of
      webvid shards are <10 KB XML error responses saved with .mp4 extension).
    """
    dset = wds.WebDataset(
        urls,
        nodesplitter=wds.split_by_node,
        shardshuffle=shuffle,
        handler=wds.warn_and_continue,
    )
    if repeat:
        dset = dset.repeat()
    if shuffle:
        dset = dset.shuffle(shuffle)
    dset = dset.select(KeyFilter(video_key=video_key))

    def _size_filter(sample):
        v = sample.get(video_key)
        return isinstance(v, (bytes, bytearray)) and len(v) >= min_video_bytes

    dset = dset.select(_size_filter)
    dset = dset.decode(
        VideoDecorder(**decoder_kwargs), handler=wds.warn_and_continue
    ).map(_reassemble, handler=wds.warn_and_continue)
    dset = dset.map(
        VideoResizer(
            size=resize_size,
            crop_size=crop_size,
            random_crop=False,
            key=video_key,
        ),
        handler=wds.warn_and_continue,
    )
    dset = dset.batched(
        batch_size, partial=not drop_last, collation_fn=dict_collation_fn
    )
    return dset


VICLIP_MEAN = [0.485, 0.456, 0.406]
VICLIP_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Video preprocessing
# --------------------------------------------------------------------------- #
class ResizeCropMinSize(nn.Module):
    def __init__(self, min_size: int, interpolation=InterpolationMode.BICUBIC):
        super().__init__()
        self.min_size = int(min_size)
        self.interpolation = interpolation
        self.random_crop = RandomCrop((self.min_size, self.min_size))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        height, width = img.shape[-2:]
        scale = self.min_size / float(min(height, width))
        if scale != 1.0:
            new_size = tuple(round(d * scale) for d in (height, width))
            img = torchvision.transforms.functional.resize(
                img, new_size, self.interpolation
            )
        return self.random_crop(img)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mutual_knn(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    """Mean fraction of overlap between top-k cosine NN of x and of y across modalities.

    A standard cross-modal alignment metric (Platonic Representation Hypothesis, Huh et al. 2024).
    """
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    n = x.size(0)
    k = min(k, n - 1)
    sim_x = x @ x.T
    sim_y = y @ y.T
    diag = torch.eye(n, dtype=torch.bool, device=x.device)
    sim_x.masked_fill_(diag, float("-inf"))
    sim_y.masked_fill_(diag, float("-inf"))
    knn_x = sim_x.topk(k, dim=-1).indices
    knn_y = sim_y.topk(k, dim=-1).indices
    ohx = torch.zeros(n, n, dtype=torch.bool, device=x.device).scatter_(1, knn_x, True)
    ohy = torch.zeros(n, n, dtype=torch.bool, device=x.device).scatter_(1, knn_y, True)
    return ((ohx & ohy).sum(-1).float() / k).mean().item()


@torch.no_grad()
def structure_preservation(x: torch.Tensor, tx: torch.Tensor) -> float:
    """Spearman correlation between pairwise Euclidean distances of x and T(x)."""
    dx = squareform(pdist(x.detach().float().cpu().numpy(), metric="euclidean"))
    dt = squareform(pdist(tx.detach().float().cpu().numpy(), metric="euclidean"))
    rho, _ = spearmanr(dx.flatten(), dt.flatten())
    return float(rho)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
@torch.no_grad()
def visualize(
    T_module: nn.Module,
    text_feats: torch.Tensor,
    video_feats: torch.Tensor,
    out_path: Path,
    num_samples: int = 1024,
    title_suffix: str = "",
) -> None:
    """t-SNE projection of (X, T(X), Y) + pairwise-distance density (X vs T(X))."""
    T_module.eval()
    num_samples = min(num_samples, text_feats.size(0))
    device = next(T_module.parameters()).device
    X = text_feats[:num_samples].to(device).float()
    Y = video_feats[:num_samples].to(device).float()
    TX = T_module(X)

    X_np = X.detach().float().cpu().numpy()
    Y_np = Y.detach().float().cpu().numpy()
    TX_np = TX.detach().float().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    emb = tsne.fit_transform(np.concatenate([X_np, TX_np, Y_np], axis=0))
    labels = np.array([0] * num_samples + [1] * num_samples + [2] * num_samples)

    dx = squareform(pdist(X_np, metric="euclidean"))
    dt = squareform(pdist(TX_np, metric="euclidean"))
    rho, _ = spearmanr(dx.flatten(), dt.flatten())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    legend = ["X (text)", "T(X) (text after OT)", "Y (video)"]
    for i in [0, 2, 1]:
        idx = labels == i
        axes[0].scatter(
            emb[idx, 0], emb[idx, 1], c=colors[i], label=legend[i], alpha=0.5, s=10
        )
    axes[0].set_title(f"t-SNE projection{title_suffix}", fontsize=16)
    axes[0].legend(fontsize=11)
    axes[0].set_xlabel("dim 1")
    axes[0].set_ylabel("dim 2")
    axes[0].grid(True, alpha=0.3)

    sns.kdeplot(
        dx.flatten(),
        label="X (text)",
        color=colors[0],
        fill=True,
        ax=axes[1],
        alpha=0.4,
    )
    sns.kdeplot(
        dt.flatten(), label="T(X)", color=colors[1], fill=True, ax=axes[1], alpha=0.4
    )
    axes[1].set_title(
        f"Pairwise distances — Spearman ρ(X, T(X)) = {rho:.4f}{title_suffix}",
        fontsize=14,
    )
    axes[1].legend(fontsize=11, loc="upper left")
    axes[1].set_xlabel("distance")
    axes[1].set_ylabel("density")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# DDP helpers
# --------------------------------------------------------------------------- #
def setup_ddp() -> tuple[int, int, int]:
    from datetime import timedelta

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        # Long timeout: rank 0 may take many minutes to fill the val-feature cache
        # while ranks 1..N-1 are sitting at the next collective.
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),
        )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_batch(
    batch,
    vi_clip,
    tokenizer,
    resize: ResizeCropMinSize,
    normalize_t: Normalize,
    device,
    weight_dtype,
    max_txt_l: int,
):
    text_inputs = batch["txt"]
    video = batch["mp4"]

    text = tokenizer(
        text_inputs,
        padding="max_length",
        truncation=True,
        max_length=max_txt_l,
        return_tensors="pt",
    ).to(device)
    _, text_features = vi_clip.encode_text(text)
    text_features = vi_clip.text_proj(text_features).float()

    video = (video / 255.0).clamp(0.0, 1.0)
    video = video.permute(0, 1, 4, 2, 3)  # B,T,H,W,3 -> B,T,3,H,W
    video = video.to(device, non_blocking=True)
    b, t = video.shape[:2]
    pixel_values = normalize_t(resize(video.view(b * t, *video.shape[2:])))
    pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
    video_features = vi_clip.get_vid_feat(pixel_values)
    video_features = video_features.float()

    return text_features.to(weight_dtype), video_features.to(weight_dtype)


def load_internvideo2(args, device) -> tuple[nn.Module, object]:
    config = Config.from_file(args.iv2_config)
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = args.num_frames
    config["inputs"]["video_input"]["num_frames_test"] = args.num_frames
    config["model"]["vision_encoder"]["num_frames"] = args.num_frames
    config["model"]["vision_encoder"]["pretrained"] = args.iv2_ckpt
    config["pretrained_path"] = args.iv2_ckpt
    vi_clip, tokenizer = setup_internvideo2(config)
    vi_clip.eval()
    vi_clip.requires_grad_(False)
    vi_clip.to(device)
    return vi_clip, tokenizer


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def validate_args(args) -> None:
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.dim <= 0 or args.ot_hidden <= 0 or args.f_hidden <= 0:
        raise ValueError("Network dimensions must be positive.")
    if args.T_iters <= 0:
        raise ValueError("--T-iters must be positive.")
    if args.max_steps <= 0 or args.eval_every <= 0:
        raise ValueError("Training and evaluation step counts must be positive.")
    if args.val_samples < 2:
        raise ValueError("--val-samples must be at least 2.")
    if args.knn_k <= 0:
        raise ValueError("--knn-k must be positive.")


def train(args):
    validate_args(args)
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    weight_dtype = torch.bfloat16 if args.bf16 else torch.float32
    out_dir = Path(args.output_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "visualization").mkdir(exist_ok=True)
        (out_dir / "checkpoints").mkdir(exist_ok=True)

    # ---- InternVideo2 (frozen feature extractor on each rank) ----------- #
    vi_clip, tokenizer = load_internvideo2(args, device)
    resize = ResizeCropMinSize(args.image_size).to(device)
    normalize_t = Normalize(mean=VICLIP_MEAN, std=VICLIP_STD)

    # ---- Datasets ------------------------------------------------------- #
    tmpdir = args.decode_tmpdir
    os.makedirs(tmpdir, exist_ok=True)
    decoder_kwargs = {
        "n_frames": args.num_frames,
        "fps": 16,
        "num_threads": args.num_decode_threads,
        "tmpdir": tmpdir,
    }
    resolution = (args.image_size, args.image_size)
    import braceexpand

    def expand_and_replicate(url_spec: str, min_count: int) -> list[str]:
        urls = list(braceexpand.braceexpand(url_spec))
        if not urls:
            raise ValueError(f"No URLs matched: {url_spec}")
        while len(urls) < min_count:
            urls = urls + urls
        return urls

    # get_video_dataset uses wds.split_by_node, which assigns shards[rank::world_size].
    # Ensure every rank gets >= num_workers shards by replicating the URL list.
    train_min = max(world_size * args.num_workers, world_size)
    train_url_list = expand_and_replicate(args.train_urls, train_min)
    val_url_list = expand_and_replicate(args.val_urls, world_size)
    if is_main(rank):
        print(
            f"[rank0] train urls: {len(train_url_list)} (unique={len(set(train_url_list))})  "
            f"val urls: {len(val_url_list)}",
            flush=True,
        )

    train_ds = build_video_loader(
        urls=train_url_list,
        batch_size=args.batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resolution,
        crop_size=resolution,
        shuffle=1000,
        repeat=True,
        drop_last=True,
    )
    val_ds = build_video_loader(
        urls=val_url_list,
        batch_size=args.val_batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resolution,
        crop_size=resolution,
        shuffle=0,  # val shards are already pre-split per rank, no buffer-fill needed
        repeat=False,
        drop_last=False,
    )
    # workers per rank, clamped to that rank's shard count
    per_rank_train_shards = max(1, len(train_url_list) // world_size)
    per_rank_val_shards = max(1, len(val_url_list) // world_size)
    train_workers = max(1, min(args.num_workers, per_rank_train_shards))
    val_workers = max(1, min(args.num_workers, per_rank_val_shards))
    train_loader = WebLoader(train_ds, batch_size=None, num_workers=train_workers)
    val_loader = WebLoader(val_ds, batch_size=None, num_workers=val_workers)
    if is_main(rank):
        print(
            f"[rank0] per-rank shards train={per_rank_train_shards} val={per_rank_val_shards}  "
            f"workers train={train_workers} val={val_workers}",
            flush=True,
        )

    # ---- Models, optimizers (BEFORE val cache so NCCL/DDP init is fast) ---- #
    T_net = OptimalTransportMap(
        input_dim=args.dim,
        hidden_dim=args.ot_hidden,
        output_dim=args.dim,
    ).to(device)
    f_net = PotentialNetwork(
        input_dim=args.dim,
        hidden_dim=args.f_hidden,
    ).to(device)

    if world_size > 1:
        T_net = DDP(T_net, device_ids=[local_rank])
        f_net = DDP(f_net, device_ids=[local_rank])
        dist.barrier()  # confirm collectives are healthy before slow rank-0 work

    # Unwrapped modules. In each alternating phase the *frozen* network is only
    # run forward (no grad sync), so calling it through DDP would arm a reducer
    # that the backward never finalizes -> "Expected to have finished reduction
    # in the prior iteration" on the next forward. Bypass DDP for those.
    T_core = T_net.module if world_size > 1 else T_net
    f_core = f_net.module if world_size > 1 else f_net

    # ---- Cache validation features (all ranks encode in parallel) -------- #
    # Each GPU encodes val_samples / world_size samples and we all_gather to
    # rank 0 for save. Per-rank shuffle seeds make the 8 shares mostly disjoint.
    val_text_feats, val_video_feats = None, None
    val_cache_path = (
        Path(args.val_cache_path)
        if args.val_cache_path
        else (out_dir / "val_features.pt")
    )
    cache_exists = val_cache_path.exists()
    if cache_exists:
        if is_main(rank):
            blob = load_ot_checkpoint(val_cache_path, map_location="cpu")
            if not isinstance(blob, dict) or not {"text", "video"} <= blob.keys():
                raise ValueError(
                    f"Invalid validation cache at {val_cache_path}: expected "
                    "a mapping containing 'text' and 'video' tensors."
                )
            val_text_feats = blob["text"][: args.val_samples]
            val_video_feats = blob["video"][: args.val_samples]
            print(
                f"[rank0] Loaded val features from {val_cache_path}: "
                f"text={tuple(val_text_feats.shape)} video={tuple(val_video_feats.shape)}",
                flush=True,
            )
    else:
        per_rank = math.ceil(args.val_samples / max(world_size, 1))
        if is_main(rank):
            print(
                f"[rank0] Encoding val features in parallel: {world_size} GPUs × "
                f"{per_rank} samples = {per_rank * world_size} (target {args.val_samples}). "
                f"Will save to {val_cache_path}",
                flush=True,
            )
        print(
            f"[rank{rank}] val cache: starting (target {per_rank} samples)", flush=True
        )
        local_txt, local_vid = [], []
        local_n = 0
        t_start = time.time()
        for batch in val_loader:
            tf, vf = encode_batch(
                batch,
                vi_clip,
                tokenizer,
                resize,
                normalize_t,
                device,
                weight_dtype,
                args.max_txt_l,
            )
            local_txt.append(tf)
            local_vid.append(vf)
            local_n += tf.size(0)
            elapsed = max(1e-3, time.time() - t_start)
            bar_w = 20
            filled = int(bar_w * min(local_n / max(per_rank, 1), 1.0))
            bar = "█" * filled + "·" * (bar_w - filled)
            print(
                f"[rank{rank}] val cache |{bar}| {local_n:>4}/{per_rank} "
                f"({elapsed:5.1f}s, {local_n/elapsed:5.1f} sps)",
                flush=True,
            )
            if local_n >= per_rank:
                break
        if not local_txt:
            raise RuntimeError(
                "The validation dataloader produced no usable samples. Check "
                "--val-urls and the expected .mp4/.txt/.json shard contents."
            )
        local_text = torch.cat(local_txt, dim=0)[:per_rank].contiguous().to(device)
        local_video = torch.cat(local_vid, dim=0)[:per_rank].contiguous().to(device)
        # Pad if a rank got short (shouldn't happen but safe).
        if local_text.size(0) < per_rank:
            pad = per_rank - local_text.size(0)
            local_text = torch.cat(
                [local_text, local_text.new_zeros(pad, local_text.size(1))], 0
            )
            local_video = torch.cat(
                [local_video, local_video.new_zeros(pad, local_video.size(1))], 0
            )

        if world_size > 1:
            gathered_text = [torch.empty_like(local_text) for _ in range(world_size)]
            gathered_video = [torch.empty_like(local_video) for _ in range(world_size)]
            dist.all_gather(gathered_text, local_text)
            dist.all_gather(gathered_video, local_video)
        else:
            gathered_text = [local_text]
            gathered_video = [local_video]

        if is_main(rank):
            val_text_feats = torch.cat(gathered_text, dim=0)[: args.val_samples].cpu()
            val_video_feats = torch.cat(gathered_video, dim=0)[: args.val_samples].cpu()
            val_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"text": val_text_feats, "video": val_video_feats}, val_cache_path
            )
            elapsed = time.time() - t_start
            print(
                f"[rank0] Cached val features → {val_cache_path}: "
                f"text={tuple(val_text_feats.shape)} video={tuple(val_video_feats.shape)}  "
                f"({elapsed:.1f}s wall, ≈{args.val_samples/elapsed:.1f} samples/s)",
                flush=True,
            )
    if world_size > 1:
        dist.barrier()

    T_params = list(T_net.parameters())
    f_params = list(f_net.parameters())
    if is_main(rank):
        print(
            f"T params: {sum(p.numel() for p in T_params)/1e6:.2f}M | "
            f"f params: {sum(p.numel() for p in f_params)/1e6:.2f}M",
            flush=True,
        )

    T_opt = torch.optim.Adam(
        T_params, lr=args.lr_T, weight_decay=args.weight_decay
    )
    f_opt = torch.optim.Adam(
        f_params, lr=args.lr_f, weight_decay=args.weight_decay
    )
    sched_T = torch.optim.lr_scheduler.CosineAnnealingLR(
        T_opt, T_max=args.max_steps, eta_min=args.lr_T * 0.05
    )
    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(
        f_opt, T_max=args.max_steps, eta_min=args.lr_f * 0.05
    )

    # ---- WandB ---------------------------------------------------------- #
    if is_main(rank) and args.wandb:
        import wandb  # noqa

        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # ---- Train loop ----------------------------------------------------- #
    best_mknn, best_struct = -math.inf, -math.inf
    step = 0
    if args.resume_from:
        checkpoint = load_ot_checkpoint(args.resume_from, map_location=device)
        if not isinstance(checkpoint, dict) or "T" not in checkpoint:
            raise ValueError(
                "--resume-from requires a structured training checkpoint "
                "created by train_OT_map.py, not a raw OT-map state dict."
            )
        T_core.load_state_dict(extract_ot_map_state_dict(checkpoint), strict=True)
        f_core.load_state_dict(checkpoint["f"], strict=True)
        T_opt.load_state_dict(checkpoint["T_opt"])
        f_opt.load_state_dict(checkpoint["f_opt"])
        if "sched_T" in checkpoint:
            sched_T.load_state_dict(checkpoint["sched_T"])
        if "sched_f" in checkpoint:
            sched_f.load_state_dict(checkpoint["sched_f"])
        step = int(checkpoint.get("step", 0))
        best_mknn = float(checkpoint.get("best_mknn", -math.inf))
        best_struct = float(checkpoint.get("best_struct", -math.inf))
        if is_main(rank):
            print(f"Resumed OT-map training from {args.resume_from} at step {step}")

    t0 = time.time()
    pbar = tqdm(
        total=args.max_steps,
        initial=step,
        desc="train",
        disable=not is_main(rank),
    )

    train_iter = iter(train_loader)
    while step < args.max_steps:
        step += 1

        # --- T updates (T_ITERS inner steps) ---
        T_net.train()
        f_net.eval()
        for p in f_params:
            p.requires_grad_(False)
        for p in T_params:
            p.requires_grad_(True)

        for _ in range(args.T_iters):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            tf, _ = encode_batch(
                batch,
                vi_clip,
                tokenizer,
                resize,
                normalize_t,
                device,
                weight_dtype,
                args.max_txt_l,
            )
            tf = tf.float()
            T_opt.zero_grad(set_to_none=True)
            TX = T_net(tf)
            mse = F.mse_loss(TX, tf)
            critic_score = f_core(TX).mean()
            T_loss = args.mse_weight * mse - critic_score
            T_loss.backward()
            torch.nn.utils.clip_grad_norm_(T_params, args.grad_clip)
            T_opt.step()

        # --- f update (single critic step) ---
        T_net.eval()
        f_net.train()
        for p in f_params:
            p.requires_grad_(True)
        for p in T_params:
            p.requires_grad_(False)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        tf, vf = encode_batch(
            batch,
            vi_clip,
            tokenizer,
            resize,
            normalize_t,
            device,
            weight_dtype,
            args.max_txt_l,
        )
        tf = tf.float()
        vf = vf.float()
        with torch.no_grad():
            TX_det = T_core(tf)
        f_opt.zero_grad(set_to_none=True)
        # Single DDP forward -> single backward (real + fake in one pass);
        # two separate f_net(...) calls would each arm the reducer and trip DDP.
        n_fake = TX_det.size(0)
        scores = f_net(torch.cat([TX_det, vf], dim=0))
        f_loss = scores[:n_fake].mean() - scores[n_fake:].mean()
        f_loss.backward()
        torch.nn.utils.clip_grad_norm_(f_params, args.grad_clip)
        f_opt.step()

        sched_T.step()
        sched_f.step()

        # restore grad flags for the next outer iter
        for p in T_params:
            p.requires_grad_(True)
        for p in f_params:
            p.requires_grad_(True)

        # --- log scalars ---
        if is_main(rank):
            pbar.update(1)
            pbar.set_postfix(
                {
                    "T": f"{T_loss.item():.3f}",
                    "mse": f"{mse.item():.4f}",
                    "fX": f"{critic_score.item():.3f}",
                    "fL": f"{f_loss.item():.3f}",
                }
            )
            if args.wandb:
                import wandb

                wandb.log(
                    {
                        "T_loss": T_loss.item(),
                        "mse_loss": mse.item(),
                        "critic_score_on_TX": critic_score.item(),
                        "f_loss": f_loss.item(),
                        "lr_T": sched_T.get_last_lr()[0],
                        "lr_f": sched_f.get_last_lr()[0],
                        "time_per_step": (time.time() - t0) / step,
                    },
                    step=step,
                )

        # --- periodic eval + checkpoint ---
        if step % args.eval_every == 0 or step == args.max_steps:
            if is_main(rank):
                T_net.eval()
                with torch.no_grad():
                    X = val_text_feats.to(device).float()
                    Y = val_video_feats.to(device).float()
                    TX = T_net(X)
                mknn_before = mutual_knn(X, Y, k=args.knn_k)
                mknn_after = mutual_knn(TX, Y, k=args.knn_k)
                struct_rho = structure_preservation(X, TX)

                improved_lines = []
                improved_mknn = mknn_after > best_mknn
                improved_struct = struct_rho > best_struct
                if improved_mknn:
                    best_mknn = mknn_after
                    improved_lines.append(f"  ↑ best mutual-kNN: {best_mknn:.4f}")
                if improved_struct:
                    best_struct = struct_rho
                    improved_lines.append(f"  ↑ best Spearman: {best_struct:.4f}")

                ot_state = T_core.state_dict()
                ckpt_payload = {
                    "format_version": 1,
                    "step": step,
                    "T": ot_state,
                    "f": f_core.state_dict(),
                    "T_opt": T_opt.state_dict(),
                    "f_opt": f_opt.state_dict(),
                    "sched_T": sched_T.state_dict(),
                    "sched_f": sched_f.state_dict(),
                    "args": vars(args),
                    "metrics": {
                        "mknn_before": mknn_before,
                        "mknn_after": mknn_after,
                        "struct_rho": struct_rho,
                    },
                    "best_mknn": best_mknn,
                    "best_struct": best_struct,
                }
                checkpoints_dir = out_dir / "checkpoints"
                torch.save(ckpt_payload, checkpoints_dir / "last.pt")
                torch.save(ot_state, checkpoints_dir / "ot_map_last.pt")
                if improved_mknn:
                    torch.save(ckpt_payload, checkpoints_dir / "best_mknn.pt")
                    torch.save(ot_state, checkpoints_dir / "ot_map_best_mknn.pt")
                if improved_struct:
                    torch.save(ckpt_payload, checkpoints_dir / "best_struct.pt")
                    torch.save(ot_state, checkpoints_dir / "ot_map_best_struct.pt")

                print(
                    f"\n[step {step}] mknn_before={mknn_before:.4f} mknn_after={mknn_after:.4f} "
                    f"struct_rho={struct_rho:.4f}",
                    flush=True,
                )
                for ln in improved_lines:
                    print(ln, flush=True)

                vis_path = out_dir / "visualization" / f"step_{step:06d}.pdf"
                visualize(
                    T_core,
                    val_text_feats,
                    val_video_feats,
                    vis_path,
                    num_samples=min(args.vis_samples, val_text_feats.size(0)),
                    title_suffix=f"  (step {step})",
                )

                if args.wandb:
                    import wandb

                    log_payload = {
                        "eval/mknn_before": mknn_before,
                        "eval/mknn_after_T": mknn_after,
                        "eval/struct_rho": struct_rho,
                        "eval/best_mknn": best_mknn,
                        "eval/best_struct": best_struct,
                    }
                    try:
                        log_payload["eval/visualization"] = wandb.Image(str(vis_path))
                    except Exception:
                        pass
                    wandb.log(log_payload, step=step)

            if world_size > 1:
                dist.barrier()

    pbar.close()
    cleanup_ddp()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-urls", type=str, default="data/webvid_10m_train/{00001..00036}.tar"
    )
    p.add_argument("--val-urls", type=str, default="data/webvid_10m_train/00000.tar")
    p.add_argument(
        "--iv2-config",
        type=str,
        default="fastvideo/intern_vid2/configs/internvideo2_stage2_config.py",
    )
    p.add_argument(
        "--iv2-ckpt", type=str, default="pretrained/InternVideo2-stage2_1b-224p-f4.pt"
    )
    p.add_argument("--output-dir", type=str, default="data/outputs/ot_map")
    p.add_argument("--run-name", type=str, default="ot_map_ddp")
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a structured checkpoint such as checkpoints/last.pt.",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="PISCES OT Map")

    p.add_argument("--batch-size", type=int, default=128, help="per-GPU batch size")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-txt-l", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-decode-threads", type=int, default=6)
    p.add_argument("--decode-tmpdir", type=str, default="tmp/v2d_scratch")

    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--ot-hidden", type=int, default=1024)
    p.add_argument("--f-hidden", type=int, default=1024)

    p.add_argument("--lr-T", type=float, default=1e-4)
    p.add_argument("--lr-f", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--T-iters", type=int, default=10)

    p.add_argument("--max-steps", type=int, default=160000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument(
        "--val-samples",
        type=int,
        default=1024,
        help="Total val samples; split evenly across world_size GPUs.",
    )
    p.add_argument(
        "--val-batch-size",
        type=int,
        default=128,
        help="Encode batch size used during val cache. With default per_rank=128, "
        "each GPU does one batch of 128.",
    )
    p.add_argument(
        "--val-cache-path",
        type=str,
        default="data/outputs/val_features_cache.pt",
        help="Disk cache for val features. Extracted on first run, reused after.",
    )
    p.add_argument("--vis-samples", type=int, default=1024)
    p.add_argument("--knn-k", type=int, default=10)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
