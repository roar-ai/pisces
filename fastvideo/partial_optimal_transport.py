"""Token-level Partial Optimal Transport used by the PISCES semantic reward.

The learned distributional OT map lives in :mod:`fastvideo.optimal_transport`.
This module implements the separate, parameter-free POT prior that is injected
into InternVideo2 cross-attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_PATCHES_PER_FRAME = 256
DEFAULT_TEMPORAL_WEIGHT = 0.2
DEFAULT_SPATIAL_WEIGHT = 0.2
DEFAULT_SINKHORN_EPSILON = 0.05
DEFAULT_TRANSPORTED_MASS = 0.9


def build_video_token_geometry(
    num_video_tokens: int,
    *,
    patches_per_frame: int = DEFAULT_PATCHES_PER_FRAME,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frame indices and normalized 2D coordinates for visual tokens.

    InternVideo2 represents a video as one CLS token followed by a square patch
    grid for every temporal frame. The CLS token receives sentinel geometry
    ``(-1, -1)`` and frame index ``-1``.
    """

    if num_video_tokens < 2:
        raise ValueError("POT requires one visual CLS token and patch tokens.")
    if patches_per_frame <= 0:
        raise ValueError("patches_per_frame must be positive.")

    num_patch_tokens = num_video_tokens - 1
    if num_patch_tokens % patches_per_frame:
        raise ValueError(
            "InternVideo2 POT expects one CLS token followed by complete frame "
            f"grids of {patches_per_frame} patches, got {num_video_tokens} "
            "visual tokens."
        )

    grid_size = math.isqrt(patches_per_frame)
    if grid_size * grid_size != patches_per_frame:
        raise ValueError(
            "patches_per_frame must form a square spatial grid, got "
            f"{patches_per_frame}."
        )

    num_frames = num_patch_tokens // patches_per_frame
    frame_indices = torch.cat(
        (
            torch.full((1,), -1, device=device, dtype=torch.long),
            torch.arange(num_frames, device=device).repeat_interleave(
                patches_per_frame
            ),
        )
    )

    axis = torch.linspace(0.0, 1.0, grid_size, device=device)
    patch_coordinates = torch.stack(
        torch.meshgrid(axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 2)
    spatial_coordinates = torch.cat(
        (
            torch.full((1, 2), -1.0, device=device),
            patch_coordinates.repeat(num_frames, 1),
        )
    )
    return frame_indices, spatial_coordinates


def compute_spatiotemporal_cost(
    text_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    attention: torch.Tensor,
    frame_indices: torch.Tensor,
    spatial_coordinates: torch.Tensor,
    *,
    temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
    spatial_weight: float = DEFAULT_SPATIAL_WEIGHT,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Construct the semantic, temporal, and spatial POT cost matrix.

    Args:
        text_embeddings: Projected lexical-token features shaped ``[N, D]``.
        video_embeddings: Projected visual-token features shaped ``[M, D]``.
        attention: Cross-attention probabilities shaped ``[N, M]``.
        frame_indices: Frame index for each visual token, shaped ``[M]``.
        spatial_coordinates: Normalized visual-token coordinates, ``[M, 2]``.

    Returns:
        A finite float32 cost matrix shaped ``[N, M]`` and normalized to
        ``[0, 1]`` for stable Sinkhorn iterations.
    """

    if text_embeddings.ndim != 2 or video_embeddings.ndim != 2:
        raise ValueError("POT text and video embeddings must be rank-2 tensors.")
    if text_embeddings.shape[1] != video_embeddings.shape[1]:
        raise ValueError(
            "POT text and video embedding dimensions must match, got "
            f"{text_embeddings.shape[1]} and {video_embeddings.shape[1]}."
        )

    num_text_tokens = text_embeddings.shape[0]
    num_video_tokens = video_embeddings.shape[0]
    if attention.shape != (num_text_tokens, num_video_tokens):
        raise ValueError(
            "POT attention shape must match text/video token counts, got "
            f"{tuple(attention.shape)}."
        )
    if frame_indices.shape != (num_video_tokens,):
        raise ValueError("frame_indices must contain one entry per video token.")
    if spatial_coordinates.shape != (num_video_tokens, 2):
        raise ValueError(
            "spatial_coordinates must have shape [num_video_tokens, 2]."
        )

    attention = attention.float()
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(eps)

    text_normalized = F.normalize(text_embeddings.float(), dim=-1)
    video_normalized = F.normalize(video_embeddings.float(), dim=-1)
    semantic_cost = (1.0 - text_normalized @ video_normalized.T).clamp(0.0, 2.0)

    frames = frame_indices.to(device=attention.device, dtype=torch.float32)
    expected_frames = attention @ frames
    temporal_cost = (expected_frames[:, None] - frames[None, :]).abs()
    temporal_cost /= (frames.max() - frames.min()).clamp_min(1.0)

    coordinates = spatial_coordinates.to(
        device=attention.device, dtype=torch.float32
    )
    expected_coordinates = attention @ coordinates
    spatial_cost = torch.cdist(
        expected_coordinates, coordinates, p=2
    ) / math.sqrt(2.0)

    cost = (
        semantic_cost
        + temporal_weight * temporal_cost
        + spatial_weight * spatial_cost
    )
    cost_min, cost_max = cost.aminmax()
    cost = (cost - cost_min) / (cost_max - cost_min).clamp_min(eps)
    return torch.nan_to_num(
        cost, nan=1.0, posinf=1.0, neginf=1.0
    ).contiguous()


def solve_partial_ot(
    cost: torch.Tensor,
    *,
    epsilon: float = DEFAULT_SINKHORN_EPSILON,
    transported_mass: float = DEFAULT_TRANSPORTED_MASS,
    max_iterations: int = 200,
    tolerance: float = 1e-3,
) -> torch.Tensor:
    """Solve the entropic POT problem with an unbalanced Sinkhorn approximation.

    ``transported_mass`` controls the marginal-relaxation strength. The output
    remains on the input device and is intentionally detached by the caller
    before it is used as a structural attention prior.
    """

    if cost.ndim != 2:
        raise ValueError("POT cost must be a rank-2 matrix.")
    if epsilon <= 0:
        raise ValueError("Sinkhorn epsilon must be positive.")
    if not 0 < transported_mass <= 1:
        raise ValueError("transported_mass must lie in (0, 1].")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    num_text_tokens, num_video_tokens = cost.shape
    dtype = (
        torch.float32
        if cost.dtype in (torch.float16, torch.bfloat16)
        else cost.dtype
    )
    cost = cost.to(dtype=dtype)
    if num_text_tokens == 0 or num_video_tokens == 0:
        return cost.new_zeros((num_text_tokens, num_video_tokens))

    text_marginal = cost.new_full(
        (num_text_tokens,), 1.0 / num_text_tokens
    )
    video_marginal = cost.new_full(
        (num_video_tokens,), 1.0 / num_video_tokens
    )

    if transported_mass >= 0.999:
        marginal_exponent = 1.0
    else:
        rho = epsilon * transported_mass / (1.0 - transported_mass)
        marginal_exponent = rho / (rho + epsilon)

    log_kernel = -cost / epsilon
    log_text_scaling = cost.new_zeros(num_text_tokens)
    log_video_scaling = cost.new_zeros(num_video_tokens)
    log_text_marginal = text_marginal.log()
    log_video_marginal = video_marginal.log()

    for _ in range(max_iterations):
        new_text_scaling = marginal_exponent * (
            log_text_marginal
            - torch.logsumexp(
                log_kernel + log_video_scaling.unsqueeze(0), dim=-1
            )
        )
        new_video_scaling = marginal_exponent * (
            log_video_marginal
            - torch.logsumexp(
                log_kernel.T + new_text_scaling.unsqueeze(0), dim=-1
            )
        )
        update = torch.maximum(
            (new_text_scaling - log_text_scaling).abs().max(),
            (new_video_scaling - log_video_scaling).abs().max(),
        )
        log_text_scaling = new_text_scaling
        log_video_scaling = new_video_scaling
        if update.item() < tolerance:
            break

    log_plan = (
        log_text_scaling[:, None]
        + log_kernel
        + log_video_scaling[None, :]
    )
    return log_plan.exp().to(torch.float32)


def normalize_transport_rows(
    transport_plan: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Normalize each text-token row of a transport plan."""

    return transport_plan / transport_plan.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)


@torch.no_grad()
def build_partial_ot_prior(
    query_embeddings: torch.Tensor,
    key_layer: torch.Tensor,
    attention_probs: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    patches_per_frame: int = DEFAULT_PATCHES_PER_FRAME,
) -> torch.Tensor:
    """Build one POT plan per InternVideo2 cross-attention head.

    Args:
        query_embeddings: Projected text queries shaped ``[1, Q, D]``.
        key_layer: Per-head visual keys shaped ``[1, H, M, D_h]``.
        attention_probs: Cross-attention shaped ``[1, H, Q, M]``.
        valid_tokens: One-dimensional lexical-token indices in the text query.

    Returns:
        A detached, row-normalized prior shaped ``[H, N, M]``.
    """

    if (
        query_embeddings.ndim != 3
        or key_layer.ndim != 4
        or attention_probs.ndim != 4
    ):
        raise ValueError(
            "POT expects queries [B,Q,D], keys [B,H,M,Dh], and attention "
            "[B,H,Q,M]."
        )
    if query_embeddings.shape[0] != 1 or key_layer.shape[0] != 1:
        raise ValueError(
            "Token-level POT currently supports one video-caption pair per call."
        )
    if attention_probs.shape[0] != 1:
        raise ValueError(
            "Token-level POT currently supports one attention batch item."
        )
    if (
        attention_probs.shape[1] != key_layer.shape[1]
        or attention_probs.shape[2] != query_embeddings.shape[1]
        or attention_probs.shape[3] != key_layer.shape[2]
    ):
        raise ValueError("POT query, key, and attention dimensions do not agree.")
    if query_embeddings.shape[2] != key_layer.shape[1] * key_layer.shape[3]:
        raise ValueError(
            "Projected POT query width must equal all flattened key-head widths."
        )
    if valid_tokens is None:
        raise ValueError("POT requires lexical-token indices.")

    valid_tokens = valid_tokens.to(
        device=query_embeddings.device, dtype=torch.long
    )
    if valid_tokens.ndim != 1 or valid_tokens.numel() == 0:
        raise ValueError("POT requires at least one lexical text token.")
    if valid_tokens.min() < 0 or valid_tokens.max() >= query_embeddings.shape[1]:
        raise ValueError("POT valid-token indices are outside the text sequence.")

    num_video_tokens = key_layer.shape[2]
    frame_indices, spatial_coordinates = build_video_token_geometry(
        num_video_tokens,
        patches_per_frame=patches_per_frame,
        device=query_embeddings.device,
    )

    video_embeddings = (
        key_layer[0].transpose(0, 1).reshape(num_video_tokens, -1).float()
    )
    text_embeddings = query_embeddings[0, valid_tokens].float()
    selected_attention = attention_probs[0, :, valid_tokens, :].float()

    plans = []
    for head_attention in selected_attention:
        cost = compute_spatiotemporal_cost(
            text_embeddings,
            video_embeddings,
            head_attention,
            frame_indices,
            spatial_coordinates,
        )
        plans.append(normalize_transport_rows(solve_partial_ot(cost)))
    return torch.stack(plans, dim=0)


def fuse_attention_with_partial_ot(
    attention_probs: torch.Tensor,
    transport_prior: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fuse a detached POT prior into cross-attention in log space.

    The transport plan is a structural prior; gradients flow through the
    original attention probabilities, matching Equation 3 in the paper.
    """

    if attention_probs.ndim != 4 or attention_probs.shape[0] != 1:
        raise ValueError("POT fusion expects attention shaped [1, H, Q, M].")
    if valid_tokens is None:
        raise ValueError("POT fusion requires lexical-token indices.")
    valid_tokens = valid_tokens.to(
        device=attention_probs.device, dtype=torch.long
    )
    if valid_tokens.ndim != 1 or valid_tokens.numel() == 0:
        raise ValueError("POT fusion requires at least one lexical text token.")
    if valid_tokens.min() < 0 or valid_tokens.max() >= attention_probs.shape[2]:
        raise ValueError("POT valid-token indices are outside the text sequence.")
    selected_attention = attention_probs[0, :, valid_tokens, :]
    if transport_prior.shape != selected_attention.shape:
        raise ValueError(
            "POT prior shape must match selected attention, got "
            f"{tuple(transport_prior.shape)} and "
            f"{tuple(selected_attention.shape)}."
        )

    prior = transport_prior.to(
        device=attention_probs.device, dtype=torch.float32
    ).detach()
    log_fused = (
        selected_attention.float().clamp_min(eps).log()
        + prior.clamp_min(eps).log()
    )
    fused_attention = torch.softmax(log_fused, dim=-1).to(
        dtype=attention_probs.dtype
    )

    updated_attention = attention_probs.clone()
    updated_attention[0, :, valid_tokens, :] = fused_attention
    return updated_attention


def apply_partial_ot_attention(
    attention_probs: torch.Tensor,
    query_embeddings: torch.Tensor,
    key_layer: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Build and inject the PISCES POT prior into InternVideo2 attention."""

    prior = build_partial_ot_prior(
        query_embeddings,
        key_layer,
        attention_probs,
        valid_tokens,
    )
    return fuse_attention_with_partial_ot(
        attention_probs,
        prior,
        valid_tokens,
    )
