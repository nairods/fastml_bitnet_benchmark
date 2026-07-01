import csv
import hashlib
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    with open(RESULTS / "hardware_readiness.json", encoding="utf-8") as handle:
        readiness = json.load(handle)

    for row in readiness:
        portable = ROOT / "onnx" / "portable" / f"{row['representative_run']}.onnx"
        row["portable_onnx"] = (
            str(portable.relative_to(ROOT)) if portable.exists() else ""
        )
        reference = ROOT / row["reference_predictions"]
        row["reference_ready"] = reference.exists()

    with open(RESULTS / "hardware_readiness.json", "w", encoding="utf-8") as handle:
        json.dump(readiness, handle, indent=2)
    with open(RESULTS / "hardware_readiness.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(readiness[0]))
        writer.writeheader()
        writer.writerows(readiness)

    direct = [
        row for row in readiness if "not directly supported" not in row["readiness"]
    ]
    policies = [
        "native",
        "controlled_16",
        "controlled_12",
        "controlled_10",
        "controlled_8",
        "controlled_6",
    ]
    plan = []
    for row in direct:
        for policy in policies:
            plan.append(
                {
                    "experiment_id": f"{row['base_run_name']}__{policy}",
                    "base_run_name": row["base_run_name"],
                    "representative_run": row["representative_run"],
                    "conversion_route": row["conversion_route"],
                    "precision_policy": policy,
                    "part": "xcvu13p-flga2577-2-e",
                    "clock_period_ns": 5.0,
                    "io_type": "io_parallel",
                    "reuse_factor": 1,
                    "input_data": "data/synthesis/x_test.npy",
                    "labels": "data/synthesis/y_test.npy",
                    "reference_predictions": row["reference_predictions"],
                }
            )
    with open(RESULTS / "synthesis_plan.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan[0]))
        writer.writeheader()
        writer.writerows(plan)
    with open(RESULTS / "synthesis_plan.json", "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    result_fields = [
        "experiment_id", "base_run_name", "precision_policy", "tool",
        "tool_version", "part", "clock_target_ns", "clock_achieved_ns",
        "latency_cycles_min", "latency_cycles_max", "latency_ns",
        "initiation_interval_cycles", "throughput_inferences_per_second",
        "lut", "ff", "dsp", "bram", "uram", "power_total_w",
        "power_dynamic_w", "post_hls_accuracy", "post_hls_macro_auc",
        "max_abs_error_vs_reference", "csim_pass", "cosim_pass", "notes",
    ]
    with open(RESULTS / "hardware_results_template.csv", "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=result_fields).writeheader()

    roots = [
        ROOT / "data" / "synthesis",
        ROOT / "models",
        ROOT / "onnx" / "hardware",
        ROOT / "onnx" / "portable",
        ROOT / "configs",
    ]
    files = [
        path for root in roots for path in root.rglob("*") if path.is_file()
    ]
    files.extend(
        ROOT / name
        for name in [
            "README.md", "benchmark.py", "model_registry.py",
            "qkeras_benchmark.py", "hgq_benchmark.py", "build_ONNX.py",
            "validate_hardware_ONNX.py", "requirements.txt",
            "requirements-qkeras.txt", "requirements-hgq.txt",
        ]
    )
    files.extend(
        RESULTS / name
        for name in [
            "stage1_performance.csv", "stage1_performance.json",
            "stage1_provenance.json", "hardware_readiness.csv",
            "synthesis_plan.csv", "hardware_results_template.csv",
        ]
    )
    unique = sorted({path.resolve() for path in files if path.exists()})
    manifest = {
        "created_for": "transfer to licensed Vivado/Vitis synthesis machine",
        "synthesis_run_on_this_cluster": False,
        "python_version_used_to_generate_manifest": platform.python_version(),
        "representative_configurations": len(readiness),
        "direct_or_native_routes": len(direct),
        "custom_converter_required": len(readiness) - len(direct),
        "planned_synthesis_experiments": len(plan),
        "files": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in unique
        },
    }
    with open(RESULTS / "hardware_transfer_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"Finalized {len(readiness)} configurations, {len(plan)} synthesis "
        f"experiments, and {len(unique)} checksummed files"
    )


if __name__ == "__main__":
    main()
