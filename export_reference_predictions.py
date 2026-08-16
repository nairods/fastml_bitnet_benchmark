import argparse
import json
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from benchmark import (
    ROOT,
    artifact_dir,
    artifact_path,
    load_checkpoint,
    load_config,
    load_dataset,
    predict_torch,
)
from model_registry import MODEL_REGISTRY, resolve_model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    arrays = load_dataset(config)
    spec = MODEL_REGISTRY[resolve_model_name(config)]

    if spec.backend == "pytorch":
        if torch is None:
            raise RuntimeError("Torch is required to export PyTorch reference predictions")
        model, _, _ = load_checkpoint(config, torch.device("cpu"))
        probabilities = predict_torch(
            model,
            arrays["x_test"],
            config["evaluation"]["batch_size"],
            torch.device("cpu"),
        )
    elif spec.backend == "qkeras":
        from qkeras_benchmark import build_qkeras_model, load_qkeras_weights, predict_qkeras

        output_dim = int(
            config.get("model", {}).get(
                "output_dim", len(arrays["metadata"]["class_names"])
            )
        )
        model = build_qkeras_model(config, arrays["x_train"].shape[1], output_dim)
        load_qkeras_weights(
            model, artifact_path(config, "models", f"{config['run_name']}.weights.h5")
        )
        probabilities = predict_qkeras(
            model, arrays["x_test"], config["evaluation"]["batch_size"]
        )
    elif spec.backend == "hgq":
        from hgq_benchmark import build_hgq_model, predict_hgq

        output_dim = int(
            config.get("model", {}).get(
                "output_dim", len(arrays["metadata"]["class_names"])
            )
        )
        model = build_hgq_model(config, arrays["x_train"].shape[1], output_dim)
        model.load_weights(artifact_path(config, "models", f"{config['run_name']}.weights.h5"))
        probabilities = predict_hgq(
            model, arrays["x_test"], config["evaluation"]["batch_size"]
        )
    else:
        raise ValueError(f"Unsupported backend: {spec.backend}")

    output_dir = artifact_dir(config, "data") / "synthesis" / "reference_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{config['run_name']}.npy"
    np.save(output, np.asarray(probabilities, dtype=np.float32))
    metadata = {
        "run_name": config["run_name"],
        "backend": spec.backend,
        "shape": list(probabilities.shape),
        "classes": arrays["metadata"]["class_names"],
        "input": "load_dataset(config).x_test",
    }
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
