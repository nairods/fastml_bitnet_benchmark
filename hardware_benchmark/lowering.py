import json
from pathlib import Path

import numpy as np


ARCHITECTURE_OPERATORS = {
    "deepsets_hlf": [
        "feature_tokenizer",
        "shared_dense_relu",
        "token_mean",
        "dense_relu",
        "dense_logits",
    ],
    "mlp_mixer_hlf": [
        "feature_tokenizer",
        "layer_norm",
        "transpose",
        "token_mlp_gelu",
        "residual_add",
        "channel_mlp_gelu",
        "token_mean",
        "dense_logits",
    ],
    "multihead_attention_hlf": [
        "feature_tokenizer",
        "prepend_cls",
        "position_add",
        "multihead_self_attention",
        "attention_softmax",
        "residual_layer_norm",
        "feedforward_gelu",
        "cls_select",
        "dense_logits",
    ],
    "linformer_hlf": [
        "feature_tokenizer",
        "prepend_cls",
        "token_projection",
        "position_add",
        "multihead_self_attention",
        "attention_softmax",
        "residual_layer_norm",
        "feedforward_gelu",
        "token_mean",
        "dense_logits",
    ],
    "jetformer_hlf": [
        "feature_tokenizer",
        "prepend_cls",
        "position_add",
        "multihead_self_attention",
        "attention_softmax",
        "residual_layer_norm",
        "feedforward_gelu",
        "cls_select",
        "dense_logits",
    ],
}


def export_pytorch_lowering_package(checkpoint_path: Path, output_dir: Path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = config["model"]
    name = model.get("name") or {"baseline": "mlp_baseline", "bitnet": "bitnet_mlp"}.get(
        model.get("type"), model.get("type")
    )
    if name not in ARCHITECTURE_OPERATORS:
        raise ValueError(f"{name} is not a custom HLF lowering architecture")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.detach().cpu().numpy() for key, value in checkpoint["state_dict"].items()}
    np.savez_compressed(output_dir / "parameters.npz", **arrays)
    graph = {
        "format": "opendata-hls-lowering-v1",
        "architecture": name,
        "operators": ARCHITECTURE_OPERATORS[name],
        "input_shape": [16],
        "output_shape": [5],
        "output_semantics": "logits",
        "config": config["model"],
        "checkpoint": str(checkpoint_path),
        "status": "operator_lowering_required",
        "acceptance": "Generated HLS output must be compared with the transferred reference predictions.",
    }
    (output_dir / "graph.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8"
    )
    return graph
