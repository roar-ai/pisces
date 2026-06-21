"""Neural optimal-transport networks and checkpoint utilities for PISCES."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class OptimalTransportMap(nn.Module):
    """Three-layer MLP that maps InternVideo2 text features to video features."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class PotentialNetwork(nn.Module):
    """Three-layer potential network used by the Neural OT objective."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


def _is_tensor_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    )


def extract_ot_map_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract an OT-map state dict from legacy or structured checkpoints."""

    state_dict = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("T", "ot_map", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if _is_tensor_state_dict(candidate):
                state_dict = candidate
                break

    if not _is_tensor_state_dict(state_dict):
        raise ValueError(
            "The OT checkpoint must be a raw state dict or a structured "
            "checkpoint containing a tensor state dict under 'T'."
        )

    prefixes = ("module.", "_orig_mod.")
    cleaned_state = {}
    for key, value in state_dict.items():
        cleaned_key = key
        for prefix in prefixes:
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned_state[cleaned_key] = value
    return cleaned_state


def load_ot_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Safely load a tensor-only OT checkpoint."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"OT checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Compatibility with PyTorch releases before ``weights_only``.
        return torch.load(path, map_location=map_location)


def load_optimal_transport_map(
    checkpoint_path: str | Path,
    *,
    input_dim: int = 512,
    hidden_dim: int = 1024,
    output_dim: int = 512,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> OptimalTransportMap:
    """Load and freeze a paper-compatible OT map with strict validation."""

    model = OptimalTransportMap(input_dim, hidden_dim, output_dim)
    checkpoint = load_ot_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = extract_ot_map_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "The OT checkpoint is incompatible with the PISCES three-layer "
            f"MLP ({input_dim}->{hidden_dim}->{hidden_dim}->{output_dim}): {error}"
        ) from error

    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model
