import argparse
import subprocess
import sys

from benchmark import ROOT, load_config
from model_registry import MODEL_REGISTRY, resolve_model_name


EVALUATION_SCRIPTS = {
    "pytorch": "testDNN.py",
    "qkeras": "test_qkeras.py",
    "xgboost": "test_xgboost.py",
    "hgq": "test_hgq.py",
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate one saved benchmark model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    spec = MODEL_REGISTRY[resolve_model_name(config)]
    if not spec.available:
        raise RuntimeError(spec.unavailable_reason)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / EVALUATION_SCRIPTS[spec.backend]),
            "--config",
            args.config,
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
