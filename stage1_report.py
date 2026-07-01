import csv
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CLASS_NAMES = ("gluon", "quark", "W", "Z", "top")
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def load_results():
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in {"summary.json", "stage1_performance.json"}:
            continue
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        if isinstance(result, dict) and "accuracy" in result:
            result["_path"] = path.name
            results.append(result)
    return results


def load_configs():
    configs = {}
    paths = list((ROOT / "configs").glob("*.json"))
    paths.extend((ROOT / "logs").glob("**/configs/*.json"))
    paths.extend((ROOT / "logs" / "run_configs").glob("*.json"))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        if "training" not in config:
            continue
        for key in (config.get("run_name"), path.stem):
            if key:
                configs[key] = config
    return configs


def classification_metrics(matrix):
    total = sum(sum(row) for row in matrix)
    metrics = {}
    precisions = []
    recalls = []
    f1_scores = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        support = sum(matrix[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"] = recall
        metrics[f"f1_{name}"] = f1
        metrics[f"support_{name}"] = support
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    metrics["macro_precision"] = statistics.fmean(precisions)
    metrics["macro_recall"] = statistics.fmean(recalls)
    metrics["balanced_accuracy"] = metrics["macro_recall"]
    metrics["macro_f1"] = statistics.fmean(f1_scores)
    metrics["test_samples"] = total
    return metrics


def normalize_result(result, configs):
    model_name = result.get("model_name", result.get("model_type", ""))
    model_name = {"baseline": "mlp_baseline", "bitnet": "bitnet_mlp"}.get(
        model_name, model_name
    )
    base_run_name = result.get("base_run_name", result["run_name"])
    config = configs.get(result["run_name"], configs.get(base_run_name, {}))
    training = config.get("training", {})
    row = {
        "base_run_name": base_run_name,
        "run_name": result["run_name"],
        "backend": result["backend"],
        "model_name": model_name,
        "seed": int(result.get("seed", 42)),
        "split_seed": int(result.get("split_seed", 42)),
        "dataset_id": int(result["dataset_id"]),
        "accuracy": float(result["accuracy"]),
        "macro_auc": float(result["macro_auc"]),
        "macro_average_precision": result.get("macro_average_precision", ""),
        "cross_entropy": result.get("cross_entropy", ""),
        "multiclass_brier_score": result.get("multiclass_brier_score", ""),
        "expected_calibration_error": result.get(
            "expected_calibration_error", ""
        ),
        "parameter_count": int(result["parameter_count"]),
        "model_size_bytes": int(result["model_size_bytes"]),
        "cpu_latency_ms": result.get("cpu_latency_ms", ""),
        "gpu_latency_ms": result.get("gpu_latency_ms", ""),
        "model_config": result.get("model_config", {}),
        "training_epochs": training.get("epochs", ""),
        "training_batch_size": training.get("batch_size", ""),
        "learning_rate": training.get("learning_rate", ""),
        "weight_decay": training.get("weight_decay", ""),
        "source_file": result["_path"],
    }
    for name in CLASS_NAMES:
        row[f"auc_{name}"] = float(result["per_class_auc"][name])
        row[f"average_precision_{name}"] = result.get(
            "per_class_average_precision", {}
        ).get(name, "")
    for threshold, values in result.get("confidence_coverage", {}).items():
        row[f"coverage_confidence_{threshold}"] = values["coverage"]
        row[f"accuracy_confidence_{threshold}"] = values["accuracy"]
    trigger_proxy = result.get("trigger_proxy", {})
    row["trigger_total_rate_khz"] = trigger_proxy.get(
        "total_min_bias_rate_khz", ""
    )
    for signal, values in trigger_proxy.get("signals", {}).items():
        row[f"trigger_auc_{signal}"] = values["auc"]
        for efficiency, point in values["signal_efficiency_points"].items():
            suffix = efficiency.replace(".", "p")
            row[f"trigger_rate_{signal}_eff{suffix}_khz"] = point[
                "proxy_trigger_rate_khz"
            ]
            row[f"background_efficiency_{signal}_eff{suffix}"] = point[
                "background_efficiency"
            ]
        for rate, point in values["rate_points_khz"].items():
            suffix = rate.replace(".", "p")
            row[f"signal_efficiency_{signal}_rate{suffix}_khz"] = point[
                "signal_efficiency"
            ]
    row.update(classification_metrics(result["confusion_matrix"]))
    return row


def confidence_interval(values):
    if len(values) < 2:
        return "", ""
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = T_CRITICAL_95.get(len(values), 1.96)
    half_width = critical * standard_error
    return mean - half_width, mean + half_width


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["base_run_name"],
                row["backend"],
                row["model_name"],
                row["split_seed"],
            )
        ].append(row)

    scalar_metrics = [
        "accuracy",
        "macro_auc",
        "macro_average_precision",
        "cross_entropy",
        "multiclass_brier_score",
        "expected_calibration_error",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "macro_f1",
        "parameter_count",
        "model_size_bytes",
        "cpu_latency_ms",
        "gpu_latency_ms",
    ]
    for name in CLASS_NAMES:
        scalar_metrics.extend(
            [
                f"auc_{name}",
                f"average_precision_{name}",
                f"precision_{name}",
                f"recall_{name}",
                f"f1_{name}",
            ]
        )
    scalar_metrics.extend(
        [
            "coverage_confidence_0.5",
            "coverage_confidence_0.7",
            "coverage_confidence_0.8",
            "coverage_confidence_0.9",
            "coverage_confidence_0.95",
            "accuracy_confidence_0.5",
            "accuracy_confidence_0.7",
            "accuracy_confidence_0.8",
            "accuracy_confidence_0.9",
            "accuracy_confidence_0.95",
            "trigger_total_rate_khz",
        ]
    )
    for signal in ("W", "Z", "top", "W_Z_top"):
        scalar_metrics.append(f"trigger_auc_{signal}")
        for efficiency in ("0p5", "0p7", "0p8", "0p9", "0p95"):
            scalar_metrics.extend(
                [
                    f"trigger_rate_{signal}_eff{efficiency}_khz",
                    f"background_efficiency_{signal}_eff{efficiency}",
                ]
            )
        for rate in ("1p0", "10p0", "100p0", "1000p0"):
            scalar_metrics.append(
                f"signal_efficiency_{signal}_rate{rate}_khz"
            )

    aggregates = []
    for (base_name, backend, model_name, split_seed), group in grouped.items():
        item = {
            "base_run_name": base_name,
            "backend": backend,
            "model_name": model_name,
            "split_seed": split_seed,
            "num_runs": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
            "evidence_level": "repeated" if len(group) >= 3 else "exploratory",
            "dataset_id": group[0]["dataset_id"],
            "test_samples": group[0]["test_samples"],
            "model_config": json.dumps(group[0]["model_config"], sort_keys=True),
            "training_epochs": group[0]["training_epochs"],
            "training_batch_size": group[0]["training_batch_size"],
            "learning_rate": group[0]["learning_rate"],
            "weight_decay": group[0]["weight_decay"],
        }
        for metric in scalar_metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric, "") not in ("", None)
            ]
            item[metric] = statistics.fmean(values) if values else ""
            item[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ) if values else ""
            if metric in {"accuracy", "macro_auc", "macro_f1"}:
                low, high = confidence_interval(values)
                item[f"{metric}_ci95_low"] = low
                item[f"{metric}_ci95_high"] = high
        aggregates.append(item)
    return aggregates


def add_baseline_deltas(aggregates, rows):
    baseline = {
        row["seed"]: row
        for row in rows
        if row["base_run_name"] == "mlp_baseline" and row["backend"] == "pytorch"
    }
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[(row["base_run_name"], row["backend"])].append(row)
    for aggregate_row in aggregates:
        matches = grouped_rows[
            (aggregate_row["base_run_name"], aggregate_row["backend"])
        ]
        paired = [(row, baseline[row["seed"]]) for row in matches if row["seed"] in baseline]
        aggregate_row["paired_baseline_runs"] = len(paired)
        for metric in ("accuracy", "macro_auc", "macro_f1"):
            deltas = [row[metric] - reference[metric] for row, reference in paired]
            aggregate_row[f"delta_{metric}_vs_mlp"] = (
                statistics.fmean(deltas) if deltas else ""
            )
            aggregate_row[f"delta_{metric}_vs_mlp_std"] = (
                statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            ) if deltas else ""


def write_csv(rows):
    preferred = [
        "base_run_name",
        "backend",
        "model_name",
        "evidence_level",
        "num_runs",
        "seeds",
        "accuracy",
        "accuracy_std",
        "accuracy_ci95_low",
        "accuracy_ci95_high",
        "macro_auc",
        "macro_auc_std",
        "macro_auc_ci95_low",
        "macro_auc_ci95_high",
        "macro_f1",
        "macro_f1_std",
        "macro_average_precision",
        "cross_entropy",
        "multiclass_brier_score",
        "expected_calibration_error",
        "macro_precision",
        "macro_recall",
        "delta_accuracy_vs_mlp",
        "delta_macro_auc_vs_mlp",
        "parameter_count",
        "model_size_bytes",
        "cpu_latency_ms",
        "training_epochs",
        "training_batch_size",
        "learning_rate",
        "weight_decay",
    ]
    fieldnames = preferred + sorted(
        {key for row in rows for key in row if key not in preferred}
    )
    with open(
        RESULTS_DIR / "stage1_performance.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, scale=1.0, digits=3):
    if value in ("", None):
        return "n/a"
    return f"{float(value) * scale:.{digits}f}"


def write_markdown(rows, onnx_results):
    repeated = sorted(
        [row for row in rows if row["evidence_level"] == "repeated"],
        key=lambda row: row["macro_auc"],
        reverse=True,
    )
    exploratory = sorted(
        [
            row
            for row in rows
            if row["evidence_level"] == "exploratory"
            and row["base_run_name"] != "baseline_mlp"
        ],
        key=lambda row: row["macro_auc"],
        reverse=True,
    )
    lines = [
        "# Stage 1: Predictive Performance",
        "",
        "## Scope",
        "",
        "- Dataset: OpenML 42468 (`hls4ml_lhc_jets_hlf`), 830,000 jets.",
        "- Inputs: 16 scalar high-level features.",
        "- Classes: gluon, quark, W, Z, top.",
        "- Fixed stratified split: 531,200 train, 132,800 validation, 166,000 test.",
        "- Split seed: 42. StandardScaler fitted on training indices only.",
        "- Primary evidence: three training seeds (42, 43, 44).",
        "- Accuracy, macro one-vs-rest AUC, macro F1, per-class AUC/precision/recall/F1, and confusion matrices are recorded.",
        "- Extended evaluation records average precision, cross-entropy, Brier score, calibration error, confidence coverage, and q/g-background proxy trigger rates.",
        "",
        "Confidence intervals use a two-sided 95% Student t interval across training seeds. With only three seeds they are descriptive, not strong significance claims.",
        "",
        "## Repeated-Seed Results",
        "",
        "| Model | Backend | Accuracy mean +/- SD (%) | Macro AUC mean +/- SD | Macro F1 | Delta accuracy vs MLP (pp) | Parameters |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in repeated:
        lines.append(
            "| {base_run_name} | {backend} | {acc} +/- {acc_std} | "
            "{auc} +/- {auc_std} | {f1} | {delta} | {params:.0f} |".format(
                base_run_name=row["base_run_name"],
                backend=row["backend"],
                acc=fmt(row["accuracy"], 100),
                acc_std=fmt(row["accuracy_std"], 100),
                auc=fmt(row["macro_auc"], digits=5),
                auc_std=fmt(row["macro_auc_std"], digits=5),
                f1=fmt(row["macro_f1"], digits=5),
                delta=fmt(row["delta_accuracy_vs_mlp"], 100),
                params=row["parameter_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Trigger And Calibration Summary",
            "",
            "The trigger columns use the combined `W+Z+top` score against q/g background.",
            "",
            "| Model | ECE | Macro AP | Proxy rate at 80% signal efficiency (kHz) | Signal efficiency at 100 kHz |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in repeated:
        lines.append(
            f"| {row['base_run_name']} | "
            f"{fmt(row['expected_calibration_error'], digits=5)} | "
            f"{fmt(row['macro_average_precision'], digits=5)} | "
            f"{fmt(row['trigger_rate_W_Z_top_eff0p8_khz'], digits=1)} | "
            f"{fmt(row['signal_efficiency_W_Z_top_rate100p0_khz'], digits=4)} |"
        )
    lines.extend(
        [
            "",
            "## Exploratory Single-Seed Results",
            "",
            "These runs are useful for selecting configurations but are not equivalent to the repeated-seed comparison.",
            "",
            "| Model | Accuracy (%) | Macro AUC | Macro F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in exploratory:
        lines.append(
            f"| {row['base_run_name']} | {fmt(row['accuracy'], 100)} | "
            f"{fmt(row['macro_auc'], digits=5)} | {fmt(row['macro_f1'], digits=5)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- The primary BitNet comparisons are topology-matched float MLP versus BitNet/BitNet-1.58, plus QKeras, HGQ, binary, and ternary dense models.",
            "- Transformer, Linformer, MLP-Mixer, and Deep Sets entries are HLF adaptations, not particle-constituent reproductions.",
            "- HGQ follows its 200-epoch reference schedule; the common suite uses 20 epochs.",
            "- Framework checkpoint bytes do not represent packed low-bit storage.",
            "- CPU latency is retained only as a software sanity measurement. Resource use, throughput, power, and synthesized latency belong to stage two.",
            "- Proxy trigger rate is `q/g background efficiency * 31,037.856 kHz` for 2760 colliding bunches. It is not a physical minimum-bias rate.",
            "- Physical trigger rates require an unbiased minimum-bias sample with the intended online preselection and pileup conditions.",
            "- Confidence coverage measures selective-classification coverage, not detector or kinematic phase-space coverage.",
            "- Physics robustness to pileup, detector response, calibration shifts, and changing run conditions cannot be established from this dataset alone.",
            "- A publication-grade paired bootstrap of AUC differences would require retaining per-event predictions; current uncertainty is across training seeds.",
            "",
            "## Artifact Coverage",
            "",
            f"- Canonical non-ONNX model runs: {sum(row['num_runs'] for row in rows)}.",
            f"- ONNX validation records: {len(onnx_results)}.",
            "- Every legacy run includes accuracy, macro and per-class AUC, confusion matrix, parameter count, model artifact size, and CPU latency.",
            "- Re-evaluated runs additionally include trigger operating points, average precision, proper scoring rules, calibration, and confidence coverage.",
            "- Per-class precision, recall, F1, macro F1, balanced accuracy, seed variation, confidence intervals, and paired MLP deltas are derived in `stage1_performance.csv`.",
            "",
            "## Stage-One Readiness",
            "",
            "- Complete: discrimination, class-wise behavior, calibration, confidence coverage, seed stability, topology-matched comparisons, and q/g proxy trigger curves.",
            "- Not available from OpenML 42468: physical minimum-bias trigger rate, pileup robustness, detector-systematic robustness, and kinematic coverage versus jet pT/eta.",
            "- These unavailable items require additional representative datasets, not further processing of the existing HLF table.",
            "",
            "Detailed machine-readable output: `results/stage1_performance.csv` and `results/stage1_performance.json`.",
        ]
    )
    with open(RESULTS_DIR / "stage1_performance.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    raw_results = load_results()
    configs = load_configs()
    onnx_results = [result for result in raw_results if result["backend"] == "onnx"]
    canonical = [
        normalize_result(result, configs)
        for result in raw_results
        if result["backend"] != "onnx"
    ]
    aggregates = aggregate(canonical)
    add_baseline_deltas(aggregates, canonical)
    aggregates.sort(key=lambda row: row["macro_auc"], reverse=True)
    write_csv(aggregates)
    with open(
        RESULTS_DIR / "stage1_performance.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregates, handle, indent=2)
    with open(
        ROOT / "data" / "cache" / "openml_42468_nall_splitseed42_train0p64_val0p16_test0p2.json",
        encoding="utf-8",
    ) as handle:
        dataset_metadata = json.load(handle)
    provenance = {
        "stage": "predictive_performance",
        "dataset": dataset_metadata,
        "input_representation": "16 scalar high-level features per jet",
        "classification_task": list(CLASS_NAMES),
        "loss": "sparse/categorical cross-entropy from logits",
        "primary_optimizer": "Adam",
        "primary_training_seeds": [42, 43, 44],
        "uncertainty": "sample standard deviation and two-sided 95% Student t interval across training seeds",
        "paired_reference": "mlp_baseline with matching training seed",
        "ranking_policy": "exclude ONNX duplicate validations",
        "onnx_validation_count": len(onnx_results),
        "canonical_run_count": len(canonical),
        "configuration_count": len(aggregates),
        "python_version": platform.python_version(),
        "artifacts": [
            "results/summary.csv",
            "results/summary.json",
            "results/stage1_performance.csv",
            "results/stage1_performance.json",
            "results/stage1_performance.md",
            "results/stage1_provenance.json",
        ],
        "stage_two_exclusions": [
            "synthesized resource utilization",
            "hardware latency",
            "throughput",
            "power",
            "packed low-bit model size",
        ],
        "performance_readiness": {
            "complete": [
                "accuracy",
                "macro and per-class ROC AUC",
                "macro and per-class average precision",
                "confusion matrix",
                "precision recall and F1",
                "cross-entropy and multiclass Brier score",
                "expected calibration error",
                "confidence coverage",
                "training-seed stability",
                "q/g-background proxy trigger operating points",
            ],
            "requires_additional_data": [
                "physical minimum-bias trigger rate",
                "pileup robustness",
                "detector systematic robustness",
                "run-condition robustness",
                "kinematic coverage versus jet pT and eta",
            ],
            "optional_publication_extension": [
                "paired per-event bootstrap confidence intervals",
                "out-of-distribution detection",
            ],
        },
    }
    with open(
        RESULTS_DIR / "stage1_provenance.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle, indent=2)
    write_markdown(aggregates, onnx_results)
    print(
        f"Wrote stage-one report for {len(canonical)} canonical runs, "
        f"{len(aggregates)} configurations, and {len(onnx_results)} ONNX checks"
    )


if __name__ == "__main__":
    main()
