import argparse
import csv
import fcntl
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def flatten(result):
    legacy_names = {"baseline": "mlp_baseline", "bitnet": "bitnet_mlp"}
    model_name = result.get("model_name", result.get("model_type", ""))
    model_name = legacy_names.get(model_name, model_name)
    row = {
        "row_type": "run",
        "base_run_name": result.get("base_run_name", result["run_name"]),
        "run_name": result["run_name"],
        "backend": result["backend"],
        "model_name": model_name,
        "seed": result.get("seed", ""),
        "split_seed": result.get("split_seed", 42),
        "accuracy": result["accuracy"],
        "macro_auc": result["macro_auc"],
        "parameter_count": result["parameter_count"],
        "model_size_bytes": result["model_size_bytes"],
        "cpu_latency_ms": result.get("cpu_latency_ms", ""),
        "gpu_latency_ms": result.get("gpu_latency_ms", ""),
    }
    for class_name, value in result["per_class_auc"].items():
        row[f"auc_{class_name}"] = value
    return row


def load_run_rows():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "summary.json":
            continue
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        if isinstance(result, dict) and "accuracy" in result and "backend" in result:
            rows.append(flatten(result))
    return rows


def aggregate_rows(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[
            (row["base_run_name"], row["backend"], row["model_name"], row["split_seed"])
        ].append(row)

    metric_names = [
        "accuracy",
        "macro_auc",
        "parameter_count",
        "model_size_bytes",
        "cpu_latency_ms",
        "gpu_latency_ms",
        "auc_gluon",
        "auc_quark",
        "auc_W",
        "auc_Z",
        "auc_top",
    ]
    aggregates = []
    for (base_name, backend, model_name, split_seed), rows in sorted(grouped.items()):
        aggregate = {
            "row_type": "aggregate",
            "base_run_name": base_name,
            "run_name": f"{base_name}__mean",
            "backend": backend,
            "model_name": model_name,
            "seed": "",
            "split_seed": split_seed,
            "num_runs": len(rows),
        }
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in rows
                if row.get(metric, "") not in ("", None)
            ]
            aggregate[metric] = statistics.fmean(values) if values else ""
            aggregate[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ) if values else ""
        aggregates.append(aggregate)
    return aggregates


def _rebuild_summary_unlocked():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = load_run_rows()
    rows = run_rows + aggregate_rows(run_rows)
    preferred = [
        "row_type",
        "base_run_name",
        "run_name",
        "backend",
        "model_name",
        "seed",
        "split_seed",
        "num_runs",
    ]
    fieldnames = preferred + sorted(
        {key for row in rows for key in row if key not in preferred}
    )
    csv_path = RESULTS_DIR / "summary.csv"
    csv_temp = RESULTS_DIR / ".summary.csv.tmp"
    with open(csv_temp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temp, csv_path)
    json_path = RESULTS_DIR / "summary.json"
    json_temp = RESULTS_DIR / ".summary.json.tmp"
    with open(json_temp, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    os.replace(json_temp, json_path)
    return len(run_rows), len(rows) - len(run_rows)


def rebuild_summary():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / ".summary.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _rebuild_summary_unlocked()


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark result JSON files.")
    parser.parse_args()
    run_count, aggregate_count = rebuild_summary()
    print(
        f"Wrote results/summary.csv with {run_count} runs and "
        f"{aggregate_count} aggregate rows"
    )


if __name__ == "__main__":
    main()
