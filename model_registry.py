"""Model registry for the published benchmark families."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable
    description: str
    backend: str = "pytorch"
    available: bool = True
    unavailable_reason: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register(name: str, description: str, backend: str = "pytorch"):
    def decorator(builder: Callable) -> Callable:
        MODEL_REGISTRY[name] = ModelSpec(name, builder, description, backend)
        return builder

    return decorator


class DenseMLP(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], layer_factory: Callable):
        if nn is None:
            raise RuntimeError("PyTorch is required to construct neural models")
        super().__init__()
        dimensions = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(
            layer_factory(left, right)
            for left, right in zip(dimensions[:-1], dimensions[1:])
        )

    def forward(self, values):
        for layer in self.layers[:-1]:
            values = torch.relu(layer(values))
        return self.layers[-1](values)


@register("mlp_baseline", "Configurable dense MLP")
def build_dense(input_dim: int, output_dim: int, config: dict):
    return DenseMLP(input_dim, output_dim, config["model"]["hidden_dims"], nn.Linear)


@register("mlp_topo", "Dense MLP compatibility alias")
def build_dense_alias(input_dim: int, output_dim: int, config: dict):
    return build_dense(input_dim, output_dim, config)


def build_bitnet(input_dim: int, output_dim: int, config: dict, *, ternary: bool):
    from bitnet_layers import BitLinear, BitLinear158b

    options = config["model"].get("bitnet", {})
    layer_type = BitLinear158b if ternary else BitLinear

    def layer_factory(left: int, right: int):
        return layer_type(
            left,
            right,
            b=options.get("activation_bits", 8),
            frac_bits=options.get("frac_bits"),
            beta_quant=options.get("beta_quant"),
            beta_shift_min=options.get("beta_shift_min"),
            beta_shift_max=options.get("beta_shift_max"),
            quant_export=False,
        )

    return DenseMLP(input_dim, output_dim, config["model"]["hidden_dims"], layer_factory)


@register("bitnet_mlp", "Binary-weight BitNet MLP")
def build_bitnet_binary(input_dim: int, output_dim: int, config: dict):
    return build_bitnet(input_dim, output_dim, config, ternary=False)


@register("bit158_mlp", "Sparse ternary BitNet-1.58 MLP")
def build_bitnet_158(input_dim: int, output_dim: int, config: dict):
    return build_bitnet(input_dim, output_dim, config, ternary=True)


MODEL_REGISTRY["qkeras_mlp"] = ModelSpec(
    "qkeras_mlp",
    lambda *_args, **_kwargs: None,
    "QKeras quantized dense MLP",
    backend="qkeras",
)

MODEL_REGISTRY["hgq_mlp"] = ModelSpec(
    "hgq_mlp",
    lambda *_args, **_kwargs: None,
    "HGQ dense MLP",
    backend="hgq",
    available=importlib.util.find_spec("HGQ") is not None,
    unavailable_reason="Install the HGQ environment to use this backend.",
)

MODEL_REGISTRY["xgboost_bdt"] = ModelSpec(
    "xgboost_bdt",
    lambda *_args, **_kwargs: None,
    "XGBoost boosted decision tree",
    backend="xgboost",
    available=importlib.util.find_spec("xgboost") is not None,
    unavailable_reason="Install XGBoost to use this backend.",
)


def resolve_model_name(config: dict) -> str:
    return config["model"]["name"]


def build_registered_model(config: dict, input_dim: int, output_dim: int):
    name = resolve_model_name(config)
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(sorted(MODEL_REGISTRY))}")
    spec = MODEL_REGISTRY[name]
    if not spec.available:
        raise RuntimeError(f"Model {name!r} is unavailable: {spec.unavailable_reason}")
    if spec.backend != "pytorch":
        raise RuntimeError(f"Model {name!r} requires backend {spec.backend!r}")
    return spec.builder(input_dim, output_dim, config)


def registry_rows() -> list[dict]:
    return [
        {
            "name": spec.name,
            "backend": spec.backend,
            "available": spec.available,
            "description": spec.description,
            "unavailable_reason": spec.unavailable_reason,
        }
        for spec in sorted(MODEL_REGISTRY.values(), key=lambda item: item.name)
    ]
