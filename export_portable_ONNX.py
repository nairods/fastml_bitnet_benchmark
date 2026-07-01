import json
import subprocess
import sys
from pathlib import Path

from benchmark import ROOT


RUNS = [
    "mlp_baseline__seed42",
    "mlp_topo__seed42",
    "deepsets_hlf__seed42",
    "mlp_mixer_hlf__seed42",
]


def main():
    configs = {}
    for path in (ROOT / "logs").glob("**/configs/*.json"):
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        configs[config.get("run_name")] = path
    for run_name in RUNS:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "build_ONNX.py"),
                "--config",
                str(configs[run_name]),
                "--output-dir",
                "onnx/portable",
                "--trace",
            ],
            cwd=ROOT,
            check=True,
        )
    print(f"Exported {len(RUNS)} portable ONNX graphs")


if __name__ == "__main__":
    main()
