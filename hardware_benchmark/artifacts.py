import json
from pathlib import Path


BITNET_NAMES = {
    "bitnet_mlp",
    "bitnet_topo",
    "bit158_mlp",
    "bit158_topo",
    "binary_large",
    "ternary_large",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def discover_configs(root: Path) -> dict[str, Path]:
    configs = {}
    candidates = list((root / "configs").glob("*.json"))
    candidates += list((root / "logs").glob("**/configs/*.json"))
    candidates += list((root / "logs" / "run_configs").glob("*.json"))
    for path in candidates:
        try:
            config = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        run_name = config.get("run_name")
        if run_name and "model" in config:
            configs[run_name] = path
    return configs


def checkpoint_path(root: Path, run_name: str, backend: str) -> Path:
    suffix = ".pt" if backend == "pytorch" else ".weights.h5"
    return root / "models" / f"{run_name}{suffix}"


def quantized_state_path(root: Path, run_name: str) -> Path:
    return root / "onnx" / "hardware" / f"{run_name}_quantized.pt"


def model_name(config: dict) -> str:
    model = config["model"]
    return model.get("name") or {"baseline": "mlp_baseline", "bitnet": "bitnet_mlp"}.get(
        model.get("type"), model.get("type")
    )


def readiness_rows(root: Path) -> list[dict]:
    return load_json(root / "results" / "hardware_readiness.json")


def readiness_by_run(root: Path) -> dict[str, dict]:
    return {row["representative_run"]: row for row in readiness_rows(root)}
