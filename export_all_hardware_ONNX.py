import json
import subprocess
import sys
from pathlib import Path

from benchmark import ROOT
from build_ONNX import BITNET_MODELS


def load_configs():
    configs = {}
    paths = list((ROOT / "configs").glob("*.json"))
    paths.extend((ROOT / "logs").glob("**/configs/*.json"))
    paths.extend((ROOT / "logs" / "run_configs").glob("*.json"))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        if "run_name" in config and "model" in config:
            configs[config["run_name"]] = path
    return configs


def main():
    configs = load_configs()
    checkpoints = sorted((ROOT / "models").glob("*.pt"))
    exported = 0
    missing = []
    for checkpoint in checkpoints:
        run_name = checkpoint.stem
        config_path = configs.get(run_name)
        if config_path is None:
            missing.append(run_name)
            continue
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if config["model"].get("name") not in BITNET_MODELS:
            continue
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "build_ONNX.py"),
                "--config",
                str(config_path),
            ],
            cwd=ROOT,
            check=True,
        )
        exported += 1
    print(f"Exported {exported} BitNet-family checkpoints")
    if missing:
        print(f"Skipped {len(missing)} checkpoints without run configs")


if __name__ == "__main__":
    main()
