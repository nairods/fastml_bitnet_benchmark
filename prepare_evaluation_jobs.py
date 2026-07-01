import argparse
import json
from pathlib import Path

from benchmark import ROOT


def load_configs():
    configs = {}
    paths = list((ROOT / "configs").glob("*.json"))
    paths.extend((ROOT / "logs").glob("**/configs/*.json"))
    paths.extend((ROOT / "logs" / "run_configs").glob("*.json"))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        if "training" not in config or "run_name" not in config:
            continue
        configs[config["run_name"]] = path.resolve()
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Create evaluation-only manifests for saved checkpoints."
    )
    parser.add_argument("--output", default="logs/slurm")
    args = parser.parse_args()
    configs = load_configs()
    manifests = {"evaluate": [], "evaluate_hgq": []}
    missing = []

    for path in sorted((ROOT / "results").glob("*.json")):
        if path.name in {
            "summary.json",
            "stage1_performance.json",
            "stage1_provenance.json",
        }:
            continue
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict) or result.get("backend") == "onnx":
            continue
        run_name = result["run_name"]
        config_path = configs.get(run_name)
        if config_path is None:
            missing.append(run_name)
            continue
        key = "evaluate_hgq" if result["backend"] == "hgq" else "evaluate"
        manifests[key].append(str(config_path))

    output = ROOT / args.output
    for name, entries in manifests.items():
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "manifest.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(entries) + "\n")
        print(f"Wrote {len(entries)} jobs to {directory / 'manifest.txt'}")
    if missing:
        print("Missing configs: " + ", ".join(missing))


if __name__ == "__main__":
    main()
