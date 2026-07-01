import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "synthesis"
RESULTS_DIR = ROOT / "results"
PROCESSED = (
    ROOT
    / "data"
    / "cache"
    / "openml_42468_nall_splitseed42_train0p64_val0p16_test0p2.npz"
)
METADATA = PROCESSED.with_suffix(".json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    arrays = np.load(PROCESSED)
    with open(METADATA, encoding="utf-8") as handle:
        metadata = json.load(handle)

    outputs = {
        "x_profile.npy": np.ascontiguousarray(
            arrays["x_train"][:16384], dtype=np.float32
        ),
        "y_profile.npy": arrays["y_train"][:16384].astype(np.int64),
        "x_test.npy": np.ascontiguousarray(
            arrays["x_test"], dtype=np.float32
        ),
        "y_test.npy": arrays["y_test"].astype(np.int64),
        "scaler_mean.npy": arrays["scaler_mean"].astype(np.float32),
        "scaler_scale.npy": arrays["scaler_scale"].astype(np.float32),
    }
    for name, values in outputs.items():
        np.save(DATA_DIR / name, values)

    x_train = arrays["x_train"]
    range_stats = {
        "source": str(PROCESSED.relative_to(ROOT)),
        "selection": "training data only",
        "feature_names": metadata["feature_names"],
        "global_abs_percentiles": {
            str(percentile): float(np.percentile(np.abs(x_train), percentile))
            for percentile in (99.0, 99.9, 99.99, 100.0)
        },
        "per_feature_min": x_train.min(axis=0).astype(float).tolist(),
        "per_feature_max": x_train.max(axis=0).astype(float).tolist(),
        "per_feature_abs_max": np.abs(x_train).max(axis=0).astype(float).tolist(),
    }
    with open(DATA_DIR / "input_range_statistics.json", "w", encoding="utf-8") as handle:
        json.dump(range_stats, handle, indent=2)

    policies = {
        "common_interface": {
            "input_samples": "identical standardized float32 arrays before fixed-point casting",
            "rounding": "AP_RND",
            "saturation": "AP_SAT",
            "integer_bits": 5,
            "reason": "covers the observed standardized training range up to abs(x)=8.96",
        },
        "native_comparison": {
            "purpose": "compare each method as designed",
            "float_mlp": "ap_fixed<16,5>",
            "bitnet_input": "ap_fixed<8,5>",
            "bitnet_activation": "8-bit native dynamic quantizer; reproduce explicitly in conversion",
            "qkeras": "model-configured weight and activation precision",
            "hgq": "learned precision from trained HGQ layers",
        },
        "controlled_comparison": {
            "purpose": "same input and HLS arithmetic width; model-specific weight representation remains",
            "formats": [
                f"ap_fixed<{width},5,AP_RND,AP_SAT>"
                for width in (16, 12, 10, 8, 6)
            ],
            "warning": "6 total bits with 5 integer bits has one fractional bit and is intentionally aggressive",
        },
        "accumulator_policy": {
            "initial": "ap_fixed<24,10,AP_RND,AP_SAT>",
            "requirement": "validate overflow and accuracy with x_profile before synthesis",
        },
    }
    with open(DATA_DIR / "precision_policies.json", "w", encoding="utf-8") as handle:
        json.dump(policies, handle, indent=2)

    files = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "dataset": metadata,
        "files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
            if path.is_file()
        },
    }
    with open(DATA_DIR / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Prepared synthesis data under {DATA_DIR}")


if __name__ == "__main__":
    main()
