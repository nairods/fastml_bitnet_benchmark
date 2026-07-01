import json
from pathlib import Path

from benchmark import ROOT


def configs_by_run():
    result = {}
    paths = list((ROOT / "configs").glob("*.json"))
    paths.extend((ROOT / "logs").glob("**/configs/*.json"))
    paths.extend((ROOT / "logs" / "run_configs").glob("*.json"))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        if "run_name" in config and "training" in config:
            result[config["run_name"]] = path.resolve()
    return result


def main():
    configs = configs_by_run()
    with open(ROOT / "results" / "hardware_readiness.json", encoding="utf-8") as handle:
        rows = json.load(handle)
    manifests = {"reference": [], "reference_hgq": []}
    for row in rows:
        run_name = row["representative_run"]
        path = configs.get(run_name)
        if path is None:
            raise FileNotFoundError(f"No config for {run_name}")
        key = "reference_hgq" if row["backend"] == "hgq" else "reference"
        manifests[key].append(str(path))
    for name, entries in manifests.items():
        directory = ROOT / "logs" / "slurm" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.txt").write_text(
            "\n".join(entries) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(entries)} entries to {directory / 'manifest.txt'}")


if __name__ == "__main__":
    main()
