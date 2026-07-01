import argparse
import json
from copy import deepcopy
from pathlib import Path

from benchmark import ROOT
from model_registry import MODEL_REGISTRY
from run_suite import deep_update


def main():
    parser = argparse.ArgumentParser(description="Generate one config per suite job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="logs/slurm")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        suite = json.load(handle)
    output = ROOT / args.output
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for model_entry in suite["models"]:
        if not model_entry.get("enabled", True):
            continue
        spec = MODEL_REGISTRY[model_entry["model"]["name"]]
        if not spec.available:
            print(f"Skipping {spec.name}: {spec.unavailable_reason}")
            continue
        for seed in suite["seeds"]:
            config = deepcopy(suite["common"])
            deep_update(config, model_entry)
            base_name = model_entry["run_name"]
            config["base_run_name"] = base_name
            config["run_name"] = f"{base_name}__seed{seed}"
            config["seed"] = seed
            config_path = config_dir / f"{config['run_name']}.json"
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
            manifest.append(str(config_path))

    manifest_path = output / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(manifest) + "\n")
    print(f"Wrote {len(manifest)} jobs to {manifest_path}")


if __name__ == "__main__":
    main()
