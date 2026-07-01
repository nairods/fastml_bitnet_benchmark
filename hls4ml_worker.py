import argparse
import json
from pathlib import Path

import hls4ml
import numpy as np
import onnx
from hls4ml.converters import convert_from_onnx_model
from hls4ml.utils.config import config_from_onnx_model


ROOT = Path(__file__).resolve().parent


def fixed_precision(width):
    integer = min(6, max(3, width // 3))
    return f"ap_fixed<{width},{integer},AP_RND,AP_SAT>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    with open(args.job, encoding="utf-8") as handle:
        job = json.load(handle)

    model = onnx.load(ROOT / job["onnx"])
    width = int(job["precision_bits"])
    precision = fixed_precision(width)
    config = config_from_onnx_model(
        model,
        granularity="name",
        backend=job["backend"],
        default_precision=precision,
    )
    config["Model"]["ReuseFactor"] = job["reuse_factor"]
    for name, layer in config.get("LayerName", {}).items():
        layer["ReuseFactor"] = job["reuse_factor"]
        if name.startswith("MatMul"):
            layer.setdefault("Precision", {})["weight"] = "ap_int<2>"
            layer["Precision"]["accum"] = "ap_fixed<24,10,AP_RND,AP_SAT>"

    output_dir = ROOT / job["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    hls_model = convert_from_onnx_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        project_name="jet_classifier",
        backend=job["backend"],
        part=job["part"],
        clock_period=job["clock_period"],
        io_type=job["io_type"],
    )
    hls_model.compile()
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 16)).astype(np.float32)
    predictions = hls_model.predict(x)
    np.save(output_dir / "smoke_predictions.npy", predictions)
    if args.build:
        report = hls_model.build(
            csim=False,
            synth=True,
            cosim=False,
            validation=False,
            export=False,
            vsynth=False,
        )
        with open(output_dir / "build_report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    with open(output_dir / "job.json", "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2)


if __name__ == "__main__":
    main()
