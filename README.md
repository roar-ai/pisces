<div align="center">
<img src="assets/logo.jpg" width="30%"/>

# PISCES

### Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards

[Paper](https://arxiv.org/abs/2602.01624) · ICML 2026
</div>

PISCES is an annotation-free post-training method for text-to-video diffusion
models. It aligns the text and video embedding spaces of InternVideo2 with two
complementary rewards:

- **Quality reward:** a distributional Neural Optimal Transport (OT) map aligns
  global text and real-video embeddings.
- **Semantic reward:** a token-level partial OT plan adds semantic,
  spatio-temporal structure to InternVideo2 cross-attention.

The reward module supports direct backpropagation and reinforcement learning.
This repository focuses on the direct-backpropagation recipe used for
HunyuanVideo. It is built on the
[FastVideo](https://github.com/hao-ai-lab/FastVideo) training framework.

## Implementation map

The two OT components are separate and should not be confused:

| Component | Purpose | Learned? | Main implementation |
|---|---|---:|---|
| Distributional Neural OT | Maps global text embeddings into the real-video embedding space for the quality reward | Yes | `fastvideo/optimal_transport.py`, trained by `train_OT_map.py` |
| Token-level Partial OT (POT) | Builds a per-head text-to-video-patch transport prior for the semantic reward | No | `fastvideo/partial_optimal_transport.py`, injected from InternVideo2 `xbert.py` |

The direct-backpropagation path is:

```text
fastvideo/distill.py
  └─ decodes predicted latents with the HunyuanVideo VAE
     └─ fastvideo/reward_fn.py
        └─ InternVideo2 reward_OT()
           ├─ global [CLS] reward through the learned OT map
           └─ VTM semantic reward through POT-refined cross-attention
```

See [docs/pot_implementation.md](docs/pot_implementation.md) for the complete
POT tensor flow, equations, defaults, source locations, and implementation
assumptions.

## Installation

The tested environment uses Python 3.10, PyTorch 2.5.0, CUDA 12.4, and NVIDIA
A100/H100 GPUs.

```bash
# Install uv if it is not already available.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv and install the CUDA/Python dependencies.
uv sync --extra lint
```

`bash env_setup.sh` runs the same environment synchronization and installs the
additional HPSv2 tokenizer asset used by legacy FastVideo reward functions.
PISCES itself uses InternVideo2.

### Download model weights

Download the original HunyuanVideo weights:

```bash
uv run python scripts/huggingface/download_hf.py \
  --repo_id FastVideo/hunyuan \
  --local_dir data/hunyuan \
  --repo_type model
```

InternVideo2 is gated on Hugging Face. Accept its license, authenticate with
Hugging Face, and download the checkpoint:

```bash
uv run huggingface-cli login
uv run python scripts/huggingface/download_hf.py \
  --repo_id OpenGVLab/InternVideo2-Stage2_1B-224p-f4 \
  --local_dir pretrained \
  --repo_type model
```

The paper OT map is included at
`pretrained/OT_map_156000.pt`. You can use it directly or train a new map as
described below.

## Data

PISCES uses different data for OT-map training and video-model post-training.

### WebVid10M for the OT map

We train the distributional OT map on WebVid10M text-video pairs. See the
[official WebVid repository](https://github.com/m-bain/webvid) for its terms and
availability. A community copy of the metadata is available from
[TempoFunk/webvid-10M](https://huggingface.co/datasets/TempoFunk/webvid-10M).
You are responsible for ensuring that your use and download of the source
videos complies with the dataset and source-site terms.

Download the metadata and use
[video2dataset](https://github.com/iejMac/video2dataset) to create WebDataset
shards:

```bash
uv run huggingface-cli download TempoFunk/webvid-10M \
  --repo-type dataset \
  --local-dir data/webvid_metadata

# Merge the partitioned CSV metadata while keeping one header.
awk 'FNR == 1 && NR != 1 { next } { print }' \
  data/webvid_metadata/data/train/partitions/*.csv \
  > data/webvid_metadata/train.csv

uv run video2dataset \
  --url_list="data/webvid_metadata/train.csv" \
  --input_format="csv" \
  --output-format="webdataset" \
  --output_folder="data/webvid_10m_train" \
  --url_col="contentUrl" \
  --caption_col="name" \
  --save_additional_columns='[videoid,page_dir,duration]' \
  --config=default
```

The OT dataloader expects numbered `.tar` shards. Each sample must contain
matching video, caption, and metadata entries:

```text
data/webvid_10m_train/
├── 00000.tar
├── 00001.tar
└── ...

# Inside each tar:
000010007.mp4
000010007.txt
000010007.json
```

Use a held-out shard for validation. For multi-GPU validation, either provide
enough validation shards for all ranks or split one held-out shard:

```bash
uv run python scripts/split_val_shard.py \
  data/webvid_10m_train/00000.tar \
  data/webvid_10m_train/val_split \
  8
```

### MixKit for post-training

We use MixKit for video-model post-training. The source videos are available in
the [Open-Sora-Plan MixKit collection](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/all_mixkit).
The precomputed HunyuanVideo latents and text embeddings can be downloaded with:

```bash
uv run python scripts/huggingface/download_hf.py \
  --repo_id FastVideo/HD-Mixkit-Finetune-Hunyuan \
  --local_dir data/HD-Mixkit-Finetune-Hunyuan \
  --repo_type dataset
```

For custom datasets, follow [docs/data_preprocess.md](docs/data_preprocess.md)
to produce `videos2caption.json`, VAE latents, prompt embeddings, and attention
masks.

In our experiments, post-training is short: roughly **192–256 optimizer
steps**. At this scale we observed little sensitivity to the exact training
dataset, and use MixKit for the released recipe.

## Train the OT map

`train_OT_map.py` extracts frozen 512-dimensional InternVideo2 text and video
features from 8-frame clips. Both the transport map and potential network are
paper-faithful three-layer MLPs with ReLU and LayerNorm:

```text
T: 512 → 1024 → 1024 → 512
f: 512 → 1024 → 1024 → 1
```

The default command reproduces the single-GPU paper setting: batch size 128 and
learning rate `1e-4`. Adjust the example shard range to match the shards
produced by your WebVid10M download.

```bash
torchrun --standalone --nproc_per_node=1 train_OT_map.py \
  --train-urls "data/webvid_10m_train/{00001..00036}.tar" \
  --val-urls "data/webvid_10m_train/00000.tar" \
  --iv2-ckpt pretrained/InternVideo2-stage2_1b-224p-f4.pt \
  --output-dir data/outputs/ot_map \
  --wandb
```

For multi-GPU training, launch one process per GPU. `--batch-size` is per GPU;
reduce it if you want to preserve the paper's global batch size:

```bash
torchrun --standalone --nproc_per_node=8 train_OT_map.py \
  --train-urls "data/webvid_10m_train/{00001..00036}.tar" \
  --val-urls "data/webvid_10m_train/val_split/val_part_{0..7}.tar" \
  --batch-size 16 \
  --val-batch-size 16 \
  --iv2-ckpt pretrained/InternVideo2-stage2_1b-224p-f4.pt \
  --output-dir data/outputs/ot_map
```

Evaluation reports:

- **Mutual KNN:** cross-modal neighborhood alignment.
- **Spearman correlation:** preservation of pairwise text-embedding structure.

The output directory contains resumable full checkpoints and raw reward-ready
OT maps:

```text
checkpoints/
├── last.pt                    # T, f, optimizers, schedulers, and metrics
├── best_mknn.pt               # structured checkpoint
├── best_struct.pt             # structured checkpoint
├── ot_map_best_mknn.pt        # raw T state dict
└── ot_map_best_struct.pt      # raw T state dict
```

Either a raw OT map or a structured checkpoint can be passed to
`--ot_map_ckpt_dir`. Resume interrupted OT training with:

```bash
torchrun --standalone --nproc_per_node=1 train_OT_map.py \
  --resume-from data/outputs/ot_map/checkpoints/last.pt \
  --train-urls "data/webvid_10m_train/{00001..00036}.tar" \
  --val-urls "data/webvid_10m_train/00000.tar"
```

## Token-level Partial OT implementation

POT is computed online inside InternVideo2 and has no trainable weights or
separate checkpoint. It is activated only when both `--use_ot_reward` and
`--use_finegrained_reward_loss` are present.

For each InternVideo2 cross-attention head:

1. `fastvideo/reward_fn.py` selects lexical prompt tokens, excluding padding
   and tokenizer special tokens.
2. `fastvideo/partial_optimal_transport.py` constructs the paper's semantic,
   temporal, and spatial cost:

   ```text
   C(i,j) = 1 - cos(y_i, x_j)
            + 0.2 |E_A[frame | y_i] - frame_j|
            + 0.2 ||E_A[position | y_i] - position_j||₂
   ```

3. A log-domain, entropically regularized unbalanced Sinkhorn solver
   approximates partial transport with `epsilon=0.05` and transported mass
   `m=0.9`.
4. The row-normalized transport plan is detached and fused with ordinary
   cross-attention in log space:

   ```text
   A_tilde = softmax(log(A + eps) + log(P* + eps))
   ```

5. InternVideo2's existing VTM classifier consumes the POT-refined features;
   its positive-class probability is the fine-grained semantic reward.

The transport plan is treated as a structural prior, so gradients pass through
the original attention and reward model path, but not through the Sinkhorn
iterations. The current InternVideo2 Stage-2 layout assumes one visual CLS token
followed by a 16×16 patch grid per temporal frame. Token-level POT currently
supports one video-caption pair per reward call.

## Post-train HunyuanVideo

The canonical full PISCES configuration enables:

- distributional and token-level OT alignment;
- global quality and fine-grained semantic rewards;
- consistency distillation;
- LoRA;
- differentiable VAE decoder checkpointing.

```bash
bash scripts/distill/distill_hunyuan.sh
```

Useful environment overrides:

```bash
# Run the shorter schedule.
MAX_TRAIN_STEPS=192 bash scripts/distill/distill_hunyuan.sh

# Add tiled decoding when decoder checkpointing alone does not fit.
VAE_TILING=1 bash scripts/distill/distill_hunyuan.sh

# Override paths or GPU count.
NUM_GPUS=8 \
SP_SIZE=8 \
DATA_DIR=/path/to/data \
PRETRAINED_DIR=/path/to/pretrained \
bash scripts/distill/distill_hunyuan.sh
```

### Training settings and ablations

Start from `scripts/distill/distill_hunyuan.sh`, then add or omit these flags:

| Setting | Enabled | Disabled |
|---|---|---|
| OT alignment | `--use_ot_reward` | omit it for vanilla InternVideo2 rewards |
| Global quality reward | `--use_global_reward_loss` | omit it |
| Fine-grained semantic reward | `--use_finegrained_reward_loss` | omit it |
| LoRA | `--use_lora` plus LoRA options | omit it for full-model tuning |
| Consistency distillation | `--use_consistency_loss` | omit it |

The corresponding loss weights are:

```text
--global_reward_loss_weight
--finegrained_reward_loss_weight
--consistency_loss_weight
```

Important behavior:

- At least one consistency, reward, or prediction-decay objective must be
  active.
- `--use_ot_reward` only changes enabled reward objectives. It has no effect in
  a consistency-only run.
- POT requires both `--use_ot_reward` and
  `--use_finegrained_reward_loss`. Global-only OT uses the learned map without
  constructing a token transport plan.
- The fine-grained POT path currently requires
  `--train_batch_size 1`.
- We recommend keeping consistency distillation enabled. It anchors the updated
  model to the teacher distribution and helps reduce reward hacking.
- Omitting `--use_lora` performs full-model fine-tuning and requires
  substantially more optimizer and gradient memory.

Example objective-only variants:

```bash
# Rewards without OT: omit --use_ot_reward.
# Quality only: omit --use_finegrained_reward_loss.
# Semantic only: omit --use_global_reward_loss.
# Rewards without consistency: omit --use_consistency_loss.
# Consistency only: omit both reward-loss flags and --use_ot_reward.
```

### VAE decoder memory bottleneck

Reward-based direct backpropagation decodes predicted latents to pixels and
backpropagates through the 3D VAE decoder into the denoiser. For long,
high-resolution videos, saved decoder activations are often the dominant memory
bottleneck.

- `--vae_decode_checkpointing` recomputes decoder blocks during backward. It
  usually preserves the standard decode path at the cost of additional compute.
- `--vae_tiling` decodes overlapping spatial and temporal tiles, reducing peak
  memory further. Adjust the spatial tile with `--vae_tile_sample_size`.
- The options are independent and can be combined when either one alone is not
  sufficient.

For the released 8-GPU, 8-latent-frame recipe, keep `SP_SIZE=8` (the script
default). This gives each rank one temporal latent before VAE decoding and
matches the paper's effective batch size. Reducing `SP_SIZE` makes each rank
decode more frames; at 720p this can exceed PyTorch's `INT_MAX` limit for
`upsample_nearest3d`. If a smaller sequence-parallel group is required, enable
`VAE_TILING=1`.

Transformer activation checkpointing is controlled separately by
`--gradient_checkpointing`.

## Inference and GRPO

PISCES reward models are used only during post-training and add no inference
cost. Generated checkpoints can be used with the existing FastVideo Hunyuan
inference utilities. Experimental GRPO entry points remain under
`scripts/distill/`, but the maintained reproduction instructions above focus on
direct backpropagation.

## Testing

```bash
uv run pytest tests/test_optimal_transport.py tests/test_distill_config.py
uv run pytest tests/test_partial_optimal_transport.py
uv run ruff check fastvideo/optimal_transport.py \
  fastvideo/partial_optimal_transport.py fastvideo/reward_fn.py \
  fastvideo/distill.py train_OT_map.py
bash -n scripts/distill/distill_hunyuan.sh
```

## Citation

```bibtex
@inproceedings{le2026pisces,
  title     = {PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards},
  author    = {Le, Minh-Quan and Mittal, Gaurav and Zhao, Cheng and Gu, David and Samaras, Dimitris and Chen, Mei},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgements

This implementation builds on
[FastVideo](https://github.com/hao-ai-lab/FastVideo),
[InternVideo2](https://github.com/OpenGVLab/InternVideo),
[Phased Consistency Models](https://github.com/G-U-N/Phased-Consistency-Model),
[diffusers](https://github.com/huggingface/diffusers), and
[video2dataset](https://github.com/iejMac/video2dataset).
