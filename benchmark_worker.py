import argparse
import subprocess
import sys

from benchmark import ROOT, load_config
from model_registry import MODEL_REGISTRY, resolve_model_name


BACKEND_COMMANDS = {
    "pytorch": ("torchDNN.py", "testDNN.py"),
    "qkeras": ("train_qkeras.py", "test_qkeras.py"),
    "xgboost": ("train_xgboost.py", "test_xgboost.py"),
    "hgq": ("train_hgq.py", "test_hgq.py"),
}


def main():
    parser = argparse.ArgumentParser(description="Run one generated benchmark job.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    spec = MODEL_REGISTRY[resolve_model_name(config)]
    if not spec.available:
        raise RuntimeError(spec.unavailable_reason)
    train_script, test_script = BACKEND_COMMANDS[spec.backend]
    for script in (train_script, test_script):
        subprocess.run(
            [sys.executable, str(ROOT / script), "--config", args.config],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
