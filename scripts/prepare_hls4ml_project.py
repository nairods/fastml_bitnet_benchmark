#!/usr/bin/env python3
"""Prepare a stock hls4ml project from a trained dense, QKeras, or HGQ run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_benchmark.native import export_pytorch_hls
from benchmark import artifact_path


def backend(config: dict) -> str:
    name = config["model"]["name"]
    if name == "qkeras_mlp":
        return "qkeras"
    if name == "hgq_mlp":
        return "hgq"
    if name in {"bitnet_mlp", "bit158_mlp"}:
        return "bitnet"
    return "pytorch"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_name = config["run_name"]
    selected_backend = backend(config)
    if selected_backend == "bitnet":
        raise RuntimeError(
            "The published BitNet hardware rows use patched or custom HLS paths; "
            "use synthesize_bitnet_hls4ml_patched.py for binary BitNet. Stock "
            "hls4ml preparation is not an equivalent replacement."
        )

    protocol = json.loads((ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8"))
    hardware = protocol["hardware"]
    profile = config.get("benchmark_profile")
    output = args.output or ROOT / "hls_projects" / profile / run_name / "native"
    if selected_backend == "pytorch":
        result = export_pytorch_hls(
            artifact_path(config, "models", f"{run_name}.pt"),
            output,
            hardware["part"],
            hardware["clock_period_ns"],
        )
        print(json.dumps(result, indent=2))
        return 0

    command = [
        sys.executable,
        "-m",
        "hardware_benchmark.keras_worker",
        "--backend",
        selected_backend,
        "--run-name",
        run_name,
        "--config",
        str(config_path),
        "--weights",
        str(artifact_path(config, "models", f"{run_name}.weights.h5")),
        "--output",
        str(output),
        "--root",
        str(ROOT),
        "--samples",
        str(args.samples),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
