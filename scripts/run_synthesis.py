#!/usr/bin/env python3
"""Synthesize one trained benchmark run with its declared implementation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STOCK_IMPLEMENTATION = "hls4ml_latency_rf1"
BITNET_IMPLEMENTATION = "hls4ml_patched_bitnet_latency_rf1"


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def backend(config: dict) -> str:
    name = config["model"]["name"]
    if name in {"bitnet_mlp", "bit158_mlp"}:
        return "bitnet"
    if name in {"qkeras_mlp", "hgq_mlp", "xgboost_bdt"}:
        return name.removesuffix("_mlp").removesuffix("_bdt")
    return "pytorch"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-unverified-license", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = config["benchmark_profile"]
    run_name = config["run_name"]
    selected_backend = backend(config)
    implementation = (
        BITNET_IMPLEMENTATION
        if selected_backend == "bitnet"
        else "conifer_unrolled" if selected_backend == "xgboost" else STOCK_IMPLEMENTATION
    )
    result = ROOT / "results" / profile / "synthesis" / f"{run_name}_{implementation}" / "result.json"
    if result.exists() and not args.force:
        print(f"skip {run_name}: {result.relative_to(ROOT)} exists")
        return 0
    result_backend = "pytorch" if selected_backend in {"pytorch", "bitnet"} else selected_backend
    evaluation = ROOT / "results" / profile / "raw" / f"{run_name}_{result_backend}.json"
    if not evaluation.exists():
        raise FileNotFoundError(
            f"Evaluation must finish before synthesis: {evaluation.relative_to(ROOT)}"
        )

    license_flag = ["--allow-unverified-license"] if args.allow_unverified_license else []
    if selected_backend == "bitnet":
        checkpoint = ROOT / "models" / profile / f"{run_name}.pt"
        with tempfile.TemporaryDirectory(prefix=f"fastml-hls-{run_name}-", dir="/tmp") as directory:
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "synthesize_bitnet_hls4ml_patched.py"),
                    "--profile",
                    profile,
                    "--run-name",
                    run_name,
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    directory,
                    *license_flag,
                ]
            )
        return 0

    if selected_backend == "xgboost":
        with tempfile.TemporaryDirectory(prefix=f"fastml-hls-{run_name}-", dir="/tmp") as directory:
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "synthesize_xgboost_conifer.py"),
                    "--profile",
                    profile,
                    "--config",
                    str(config_path),
                    "--unroll",
                    "--output-dir",
                    directory,
                ]
            )
        return 0

    if selected_backend in {"qkeras", "hgq"}:
        run(
            [
                sys.executable,
                str(ROOT / "export_reference_predictions.py"),
                "--config",
                str(config_path),
            ]
        )
    with tempfile.TemporaryDirectory(prefix=f"fastml-hls-{run_name}-", dir="/tmp") as directory:
        project = Path(directory) / "project"
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_hls4ml_project.py"),
                "--config",
                str(config_path),
                "--output",
                str(project),
            ]
        )
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "synthesize_hls4ml_project.py"),
                "--profile",
                profile,
                "--run-name",
                run_name,
                "--project-dir",
                str(project),
                *license_flag,
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
