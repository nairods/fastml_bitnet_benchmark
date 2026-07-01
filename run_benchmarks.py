import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from benchmark import ROOT, ensure_output_dirs, load_config
from model_registry import MODEL_REGISTRY, resolve_model_name
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


def seed_config(base_config, seed):
    config = deepcopy(base_config)
    config.pop("_config_path", None)
    base_name = config.get("base_run_name", config["run_name"])
    config["base_run_name"] = base_name
    config["run_name"] = f"{base_name}__seed{seed}"
    config["seed"] = int(seed)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run one config over repeated seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Training seeds. Defaults to config.seeds or config.seed.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(STAGE_COMMANDS),
        default=["train", "evaluate"],
    )
    args = parser.parse_args()

    ensure_output_dirs()
    base_config = load_config(args.config)
    spec = MODEL_REGISTRY[resolve_model_name(base_config)]
    if not spec.available:
        raise RuntimeError(spec.unavailable_reason)
    if spec.backend == "qkeras":
        stage_commands = QKERAS_STAGE_COMMANDS
    elif spec.backend == "hgq":
        stage_commands = HGQ_STAGE_COMMANDS
    else:
        stage_commands = STAGE_COMMANDS
    seeds = args.seeds or base_config.get("seeds") or [base_config["seed"]]
    generated_dir = ROOT / "logs" / "run_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        config = seed_config(base_config, seed)
        config_path = generated_dir / f"{config['run_name']}.json"
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        for stage in args.stages:
            if stage not in stage_commands:
                print(f"[{config['run_name']}] skipping unsupported stage {stage}")
                continue
            command = [
                sys.executable,
                str(ROOT / stage_commands[stage]),
                "--config",
                str(config_path),
            ]
            print(f"[{config['run_name']}] {stage}")
            subprocess.run(command, cwd=ROOT, check=True)

    run_count, aggregate_count = rebuild_summary()
    print(f"Summary contains {run_count} runs and {aggregate_count} aggregates")


if __name__ == "__main__":
    main()
