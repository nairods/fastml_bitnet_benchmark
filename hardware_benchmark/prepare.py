import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .artifacts import readiness_rows
from .bitnet import export_hls_project, load_quantized_layers, predict_exact
from .lowering import export_pytorch_lowering_package
from .metrics import compare_predictions
from .native import export_pytorch_hls


def _is_bitnet(route: str) -> bool:
    lowered = route.lower()
    return any(token in lowered for token in ("bitnet", "binary", "ternary"))


def _run_keras_worker(
    root: Path,
    backend: str,
    run_name: str,
    output_dir: Path,
    validation_samples: int,
) -> dict:
    python = Path(os.environ.get("FASTML_KERAS_PYTHON", "")) if os.environ.get("FASTML_KERAS_PYTHON") else root.parents[1] / "miniconda3" / "envs" / "hlsenv310" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"Compatible Keras environment not found: {python}")
    weights = root / "models" / f"{run_name}.weights.h5"
    result = subprocess.run(
        [
            str(python),
            "-m",
            "hardware_benchmark.keras_worker",
            "--backend",
            backend,
            "--run-name",
            run_name,
            "--weights",
            str(weights),
            "--output",
            str(output_dir),
            "--root",
            str(root),
            "--samples",
            str(validation_samples),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip())
    return json.loads((output_dir / "conversion.json").read_text(encoding="utf-8"))


def _validate_pytorch_checkpoint(
    root: Path,
    run_name: str,
    values: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
) -> dict:
    import sys

    import torch

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model_registry import build_registered_model

    checkpoint = torch.load(
        root / "models" / f"{run_name}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = build_registered_model(checkpoint["config"], 16, 5)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            batch = np.array(values[start : start + 1024], dtype=np.float32, copy=True)
            logits = model(torch.from_numpy(batch))
            predictions.append(torch.softmax(logits, dim=1).numpy())
    return compare_predictions(
        np.concatenate(predictions), reference, labels
    )


def prepare_all(root: Path, output_root: Path, validation_samples: int) -> dict:
    inputs = np.load(root / "data" / "synthesis" / "x_test.npy", mmap_mode="r")
    labels = np.load(root / "data" / "synthesis" / "y_test.npy", mmap_mode="r")
    limit = min(validation_samples, len(inputs))
    records = []
    for row in readiness_rows(root):
        run_name = row["representative_run"]
        route = row["conversion_route"]
        output_dir = output_root / run_name / "native"
        record = {
            "run_name": run_name,
            "base_run_name": row["base_run_name"],
            "conversion_route": route,
            "output_dir": str(output_dir.relative_to(root))
            if output_dir.is_relative_to(root)
            else str(output_dir),
        }
        try:
            if route == "ONNX/PyTorch":
                record.update(
                    export_pytorch_hls(
                        root / "models" / f"{run_name}.pt",
                        output_dir,
                        "xcvu13p-flga2577-2-e",
                        5.0,
                    )
                )
                reference = np.load(
                    root / row["reference_predictions"], mmap_mode="r"
                )
                record["validation"] = _validate_pytorch_checkpoint(
                    root,
                    run_name,
                    inputs[:limit],
                    reference[:limit],
                    labels[:limit],
                )
                record["status"] = "project_generated"
            elif _is_bitnet(route):
                layers = load_quantized_layers(
                    root / "onnx" / "hardware" / f"{run_name}_quantized.pt"
                )
                record.update(
                    export_hls_project(
                        layers,
                        output_dir,
                        run_name.replace("-", "_"),
                        "xcvu13p-flga2577-2-e",
                        5.0,
                    )
                )
                reference = np.load(
                    root / row["reference_predictions"], mmap_mode="r"
                )
                logits = predict_exact(layers, inputs[:limit])
                record["validation"] = compare_predictions(
                    logits, reference[:limit], labels[:limit]
                )
                record["status"] = "project_generated"
            elif route == "custom converter":
                record["lowering"] = export_pytorch_lowering_package(
                    root / "models" / f"{run_name}.pt", output_dir
                )
                reference = np.load(
                    root / row["reference_predictions"], mmap_mode="r"
                )
                record["validation"] = _validate_pytorch_checkpoint(
                    root,
                    run_name,
                    inputs[:limit],
                    reference[:limit],
                    labels[:limit],
                )
                record["status"] = "lowering_package_generated"
            elif route == "native QKeras":
                record.update(
                    _run_keras_worker(
                        root, "qkeras", run_name, output_dir, limit
                    )
                )
            elif route == "native HGQ":
                record.update(
                    _run_keras_worker(root, "hgq", run_name, output_dir, limit)
                )
            else:
                record["status"] = "unsupported_route"
        except Exception as error:
            record["status"] = "preparation_failed"
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)

    summary = {
        "synthesis_run": False,
        "validation_samples": limit,
        "output_root": str(output_root),
        "counts": {
            status: sum(record["status"] == status for record in records)
            for status in sorted({record["status"] for record in records})
        },
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preparation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
