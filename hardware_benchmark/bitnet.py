"""Utilities for exported BitNet integer weights."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _layer_number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def load_quantized_layers(path: Path) -> list[dict]:
    """Load the integer weights, scale, and bias exported by BitLinear layers."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    names = sorted(
        (key[: -len(".int_weight")] for key in state if key.endswith(".int_weight")),
        key=_layer_number,
    )
    layers = []
    for name in names:
        beta_shift = state.get(f"{name}.beta_shift")
        beta = (
            float(2.0 ** int(beta_shift.item()))
            if beta_shift is not None
            else float(state[f"{name}.beta_scale"].item())
        )
        layers.append(
            {
                "name": name,
                "weight": state[f"{name}.int_weight"].numpy().astype(np.int8),
                "beta": beta,
                "bias": state[f"{name}.bias"].numpy().astype(np.float32),
            }
        )
    if not layers:
        raise ValueError(f"No quantized BitNet layers found in {path}")
    return layers


def predict_folded(layers: list[dict], values: np.ndarray) -> np.ndarray:
    """Evaluate the cumulative-scale representation used by the patched HLS project."""
    output = np.asarray(values, dtype=np.float64)
    cumulative_scale = 1.0
    for index, layer in enumerate(layers):
        cumulative_scale *= float(layer["beta"])
        folded_bias = np.asarray(layer["bias"], dtype=np.float64) / cumulative_scale
        output = output @ np.asarray(layer["weight"], dtype=np.float64).T + folded_bias
        if index != len(layers) - 1:
            output = np.maximum(output, 0.0)
    logits = output * cumulative_scale
    logits = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))
