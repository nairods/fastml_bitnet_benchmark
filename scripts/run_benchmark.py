#!/usr/bin/env python3
"""Train and evaluate selected models from the public benchmark configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "benchmark.json"

BACKEND_SUFFIX = {
    "pytorch": "pytorch",
    "qkeras": "qkeras",
    "hgq": "hgq",
    "xgboost": "xgboost",
}

def read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def published_combinations(protocol: dict, task: str) -> list[tuple[str, str]]:
    combinations = []
    for model in protocol["models"]:
        architectures = (
            model["multiclass_architectures"]
            if task == "multiclass"
            else model["base_names"].keys()
        )
        combinations.extend((model["id"], architecture) for architecture in architectures)
    return combinations


def model_definition(protocol: dict, model_id: str) -> dict:
    return next(model for model in protocol["models"] if model["id"] == model_id)


def run_base_name(protocol: dict, task: str, model_id: str, architecture: str) -> str:
    prefix = {"qg_vs_wzt": "", "qg_vs_top": "topqg_", "multiclass": "multiclass_"}[task]
    return prefix + model_definition(protocol, model_id)["base_names"][architecture]


def build_run_config(
    protocol: dict,
    profile: str,
    task: str,
    model_id: str,
    architecture: str,
    seed: int,
) -> dict:
    model_spec = model_definition(protocol, model_id)
    task_spec = protocol["tasks"][task]
    binary = task_spec["output"] == "binary_sigmoid"
    base_name = run_base_name(protocol, task, model_id, architecture)
    hidden_dims = protocol["architectures"].get(architecture)
    model = {
        "name": model_spec["runner_model"],
        "output_dim": 1 if binary else 5,
        "output_mode": task_spec["output"],
    }
    if isinstance(hidden_dims, list):
        model["hidden_dims"] = hidden_dims
    if model_id.startswith("qkeras"):
        quantizer = model_spec.get("weight_quantizer", "quantized_bits")
        model["qkeras"] = {
            "quantizer": quantizer,
            "bits": model_spec.get("weight_bits", 7),
            "integer_bits": 0,
            "activation_bits": model_spec["activation_bits"],
            "alpha": 1,
        }
    elif model_id == "hgq":
        model["hgq"] = {"beta": 3e-6}
    elif model_id.startswith("bitnet"):
        model["bitnet"] = {
            "activation_bits": model_spec["activation_bits"],
            "frac_bits": model_spec["fractional_bits"],
        }
    elif model_id == "xgboost_bdt":
        model["xgboost"] = {
            **protocol["architectures"]["bdt"],
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": 1,
        }

    profile_spec = protocol["training_profiles"][profile]
    training = {
        **profile_spec["defaults"],
        **profile_spec.get("model_overrides", {}).get(model_id, {}),
        "selection_metric": profile_spec["selection_metric"],
        "selection_mode": profile_spec["selection_mode"],
        "early_stopping": profile_spec["early_stopping"],
    }
    if model_id == "xgboost_bdt":
        training = {"device": "cpu"}

    split = protocol["dataset"]["split"]
    return {
        "run_name": f"{base_name}__seed{seed}",
        "base_run_name": base_name,
        "benchmark_profile": profile,
        "seed": seed,
        "dataset": {
            "id": protocol["dataset"]["openml_id"],
            "cache": True,
            "max_samples": None,
            "sample_seed": split["seed"],
            "classification": {"mode": task_spec["class_mode"]},
        },
        "split": split,
        "training": training,
        "evaluation": {"batch_size": 4096, "latency": {"warmup": 20, "iterations": 200}},
        "model": model,
    }


def stage_script(backend: str, stage: str) -> str:
    scripts = {
        ("pytorch", "train"): "torchDNN.py",
        ("pytorch", "evaluate"): "testDNN.py",
        ("qkeras", "train"): "train_qkeras.py",
        ("qkeras", "evaluate"): "test_qkeras.py",
        ("hgq", "train"): "train_hgq.py",
        ("hgq", "evaluate"): "test_hgq.py",
        ("xgboost", "train"): "train_xgboost.py",
        ("xgboost", "evaluate"): "test_xgboost.py",
    }
    return scripts[(backend, stage)]


def artifact_path(config: dict, backend: str, stage: str) -> Path:
    run_name = config["run_name"]
    profile = config["benchmark_profile"]
    if stage == "train":
        return ROOT / "logs" / profile / f"{run_name}_history.json"
    return ROOT / "results" / profile / "raw" / f"{run_name}_{BACKEND_SUFFIX[backend]}.json"


def main() -> int:
    protocol = read_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(protocol["training_profiles"]),
        default=protocol["default_training_profile"],
    )
    parser.add_argument("--task", choices=tuple(protocol["tasks"]), required=True)
    model_ids = tuple(model["id"] for model in protocol["models"])
    parser.add_argument("--models", nargs="+", choices=model_ids, default=None)
    parser.add_argument("--architectures", nargs="+", choices=("standard", "wide", "bdt"), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=protocol["model_seeds"])
    parser.add_argument("--stages", nargs="+", choices=("train", "evaluate"), default=("train", "evaluate"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print selected run names without writing or executing them.")
    args = parser.parse_args()

    combinations = [
        pair
        for pair in published_combinations(protocol, args.task)
        if (args.models is None or pair[0] in args.models)
        and (args.architectures is None or pair[1] in args.architectures)
    ]
    if not combinations:
        parser.error("No published model/architecture combinations match the selection")
    if args.dry_run:
        for model_id, architecture in combinations:
            for seed in args.seeds:
                config = build_run_config(protocol, args.profile, args.task, model_id, architecture, seed)
                print(config["run_name"])
        return 0

    run_config_dir = ROOT / "logs" / args.profile / "run_configs"
    run_config_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for model_id, architecture in combinations:
        backend = model_definition(protocol, model_id)["backend"]
        for seed in args.seeds:
            config = build_run_config(protocol, args.profile, args.task, model_id, architecture, seed)
            path = run_config_dir / f"{config['run_name']}.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            for stage in args.stages:
                artifact = artifact_path(config, backend, stage)
                if artifact.exists() and not args.force:
                    print(f"skip {config['run_name']} {stage}: {artifact.relative_to(ROOT)} exists")
                    continue
                command = [sys.executable, str(ROOT / stage_script(backend, stage)), "--config", str(path)]
                print("run", config["run_name"], stage)
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode or not artifact.exists():
                    failures.append(f"{config['run_name']}:{stage}")
                    break
    if failures:
        raise RuntimeError("Failed stages: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
