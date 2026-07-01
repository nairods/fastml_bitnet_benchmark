import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from benchmark import ROOT, ensure_output_dirs
from model_registry import MODEL_REGISTRY
from summarize_results import rebuild_summary


STAGE_COMMANDS = {
    "train": "torchDNN.py",
    "evaluate": "testDNN.py",
    "export": "build_ONNX.py",
    "onnx": "test_ONNX.py",
}
QKERAS_STAGE_COMMANDS = {
    "train": "train_qkeras.py",
    "evaluate": "test_qkeras.py",
}
HGQ_STAGE_COMMANDS = {
    "train": "train_hgq.py",
    "evaluate": "test_hgq.py",
}
XGBOOST_STAGE_COMMANDS = {
    "train": "train_xgboost.py",
    "evaluate": "test_xgboost.py",
}


def deep_update(target, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def main():
    parser = argparse.ArgumentParser(description="Run a fair multi-model benchmark suite.")
    parser.add_argument("--config", required=True, help="Suite JSON file.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(STAGE_COMMANDS),
        default=["train", "evaluate"],
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Attempt models marked optional in the suite.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        suite = json.load(handle)
    ensure_output_dirs()
    generated_dir = ROOT / "logs" / "run_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for model_entry in suite["models"]:
        if not model_entry.get("enabled", True):
            continue
        if model_entry.get("optional", False) and not args.include_optional:
            print(f"Skipping optional model {model_entry['run_name']}")
            continue
        model_name = model_entry["model"]["name"]
        spec = MODEL_REGISTRY[model_name]
        if not spec.available:
            print(f"Skipping unavailable model {model_name}: {spec.unavailable_reason}")
            continue
        for seed in suite["seeds"]:
            config = deepcopy(suite["common"])
            deep_update(config, model_entry)
            base_name = model_entry["run_name"]
            config["base_run_name"] = base_name
            config["run_name"] = f"{base_name}__seed{seed}"
            config["seed"] = seed
            config_path = generated_dir / f"{config['run_name']}.json"
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
            for stage in args.stages:
                if spec.backend == "qkeras":
                    stage_commands = QKERAS_STAGE_COMMANDS
                elif spec.backend == "xgboost":
                    stage_commands = XGBOOST_STAGE_COMMANDS
                elif spec.backend == "hgq":
                    stage_commands = HGQ_STAGE_COMMANDS
                else:
                    stage_commands = STAGE_COMMANDS
                if stage not in stage_commands:
                    print(f"[{config['run_name']}] skipping unsupported stage {stage}")
                    continue
                print(f"[{config['run_name']}] {stage}")
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / stage_commands[stage]),
                        "--config",
                        str(config_path),
                    ],
                    cwd=ROOT,
                    check=True,
                )

    run_count, aggregate_count = rebuild_summary()
    print(f"Summary contains {run_count} runs and {aggregate_count} aggregates")


if __name__ == "__main__":
    main()
