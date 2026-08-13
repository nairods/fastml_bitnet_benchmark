import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

ROWS = [
    ("mlp_baseline", "pytorch", "mlp_baseline__seed42", "ONNX/PyTorch", "direct dense conversion"),
    ("mlp_topo", "pytorch", "mlp_topo__seed42", "ONNX/PyTorch", "direct dense conversion"),
    ("bitnet_mlp_f5_fixed", "pytorch", "bitnet_mlp_f5_fixed", "explicit BitNet ONNX", "ready"),
    ("bitnet_mlp_f7_fixed", "pytorch", "bitnet_mlp_f7_fixed", "explicit BitNet ONNX", "ready"),
    ("bitnet_mlp_f8_fixed", "pytorch", "bitnet_mlp_f8_fixed", "explicit BitNet ONNX", "ready"),
    ("bitnet_mlp_f10_fixed", "pytorch", "bitnet_mlp_f10_fixed", "explicit BitNet ONNX", "ready"),
    ("bitnet_mlp_f12_fixed", "pytorch", "bitnet_mlp_f12_fixed", "explicit BitNet ONNX", "ready"),
    ("bitnet_mlp_f7_power2", "pytorch", "bitnet_mlp_f7_power2", "explicit BitNet ONNX", "ready; beta_shift retained"),
    ("bitnet_topo_f7_fixed", "pytorch", "bitnet_topo_f7_fixed__seed42", "explicit BitNet ONNX", "ready"),
    ("bit158_mlp_f7_fixed", "pytorch", "bit158_mlp_f7_fixed__seed42", "explicit ternary ONNX", "ready"),
    ("bit158_topo_f7_fixed", "pytorch", "bit158_topo_f7_fixed__seed42", "explicit ternary ONNX", "ready"),
    ("binary_448_224_224", "pytorch", "binary_448_224_224__seed42", "explicit binary ONNX", "ready"),
    ("ternary_128_64_64_64", "pytorch", "ternary_128_64_64_64__seed42", "explicit ternary ONNX", "ready"),
    ("xgboost_bdt", "xgboost", "xgboost_bdt__seed42", "tree ensemble", "software benchmark only; hardware flow needs a tree-specific route such as Conifer/custom HLS"),
    ("qkeras_mlp", "qkeras", "qkeras_mlp__seed42", "native QKeras", "direct hls4ml Keras route"),
    ("qkeras_mlp_b5", "qkeras", "qkeras_mlp_b5__seed42", "native QKeras", "direct hls4ml Keras route"),
    ("qkeras_mlp_b7", "qkeras", "qkeras_mlp_b7__seed42", "native QKeras", "direct hls4ml Keras route"),
    ("qkeras_mlp_b8", "qkeras", "qkeras_mlp_b8__seed42", "native QKeras", "direct hls4ml Keras route"),
    ("qkeras_mlp_b10", "qkeras", "qkeras_mlp_b10__seed42", "native QKeras", "direct hls4ml Keras route"),
    ("qkeras_mlp_b12", "qkeras", "qkeras_mlp_b12__seed42", "native QKeras", "direct hls4ml Keras route"),
    ("hgq_mlp", "hgq", "hgq_mlp__seed42", "native HGQ", "requires HGQ-enabled hls4ml environment"),
    ("deepsets_hlf", "pytorch", "deepsets_hlf__seed42", "custom converter", "not directly supported"),
    ("mlp_mixer_hlf", "pytorch", "mlp_mixer_hlf__seed42", "custom converter", "not directly supported"),
]


def file_info(path):
    if not path.exists():
        return "", "", ""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path.relative_to(ROOT)), path.stat().st_size, digest


def artifact_for(backend, run_name, route):
    if "explicit" in route:
        return ROOT / "onnx" / "hardware" / f"{run_name}.onnx"
    if backend == "pytorch":
        return ROOT / "models" / f"{run_name}.pt"
    if backend == "xgboost":
        return ROOT / "models" / f"{run_name}.pkl"
    return ROOT / "models" / f"{run_name}.weights.h5"


def main():
    rows = []
    for base_name, backend, run_name, route, status in ROWS:
        artifact = artifact_for(backend, run_name, route)
        path, size, digest = file_info(artifact)
        result = ROOT / "results" / f"{run_name}_{backend}.json"
        result_path, _, _ = file_info(result)
        rows.append(
            {
                "base_run_name": base_name,
                "backend": backend,
                "representative_run": run_name,
                "conversion_route": route,
                "readiness": status,
                "primary_artifact": path,
                "artifact_bytes": size,
                "artifact_sha256": digest,
                "software_result": result_path,
                "reference_predictions": f"data/synthesis/reference_predictions/{run_name}.npy",
            }
        )
    with open(RESULTS / "hardware_readiness.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS / "hardware_readiness.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    direct = sum(
        ("not directly supported" not in row["readiness"])
        and ("software benchmark only" not in row["readiness"])
        for row in rows
    )
    tree_only = sum("software benchmark only" in row["readiness"] for row in rows)
    custom = sum("not directly supported" in row["readiness"] for row in rows)
    markdown = [
        "# Hardware Transfer Readiness",
        "",
        f"- Representative configurations: {len(rows)}",
        f"- Direct/native conversion routes: {direct}",
        f"- Custom-converter architectures: {custom}",
        f"- Tree-specific hardware routes: {tree_only}",
        "- Hardware synthesis uses one representative seed; software performance uses three seeds.",
        "- All hardware results must use the same FPGA part, clock, IO type, reuse factor, input samples, and reporting template.",
        "",
        "Deep Sets and MLP-Mixer still require custom neural-network conversion, and XGBoost BDT needs a tree-specific hardware flow rather than the dense-model hls4ml path.",
        "",
        "See `results/hardware_readiness.csv`, `data/synthesis/precision_policies.json`, and `configs/hardware_benchmark.json`.",
    ]
    (RESULTS / "hardware_readiness.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote readiness records for {len(rows)} configurations")


if __name__ == "__main__":
    main()
