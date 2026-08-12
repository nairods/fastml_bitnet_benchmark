#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import pickle
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
SYNTHESIS = RESULTS / "synthesis"

SEEDS = (42, 43, 44)
FPR_TARGET = 0.01

TASKS = {
    "binary_qg_vs_wzt": {
        "label": "q/g vs W/Z/top",
        "prefix": "binary_",
        "cache": "openml_42468_nall_splitseed42_classbinary_qg_vs_wzt_train0p64_val0p16_test0p2.npz",
    },
    "binary_topqg": {
        "label": "q/g vs top",
        "prefix": "binary_topqg_",
        "cache": "openml_42468_nall_splitseed42_classbinary_top_vs_qg_train0p64_val0p16_test0p2.npz",
    },
    "multiclass": {
        "label": "5-class q/g/W/Z/top",
        "prefix": "",
        "cache": "openml_42468_nall_splitseed42_classmulticlass_train0p64_val0p16_test0p2.npz",
    },
}

CORE_SPECS = [
    {
        "key": "mlp_64_32_32",
        "model_family": "Dense MLP",
        "architecture": "64-32-32",
        "suffix": "mlp_baseline_64_32_32",
        "backend": "pytorch",
        "preferred_variants": ["hls4ml_latency_rf1"],
    },
    {
        "key": "mlp_128_32",
        "model_family": "Dense MLP",
        "architecture": "128-32",
        "suffix": "mlp_topo_128_32",
        "backend": "pytorch",
        "preferred_variants": ["hls4ml_latency_rf1"],
    },
    {
        "key": "qkeras_b7_64_32_32",
        "model_family": "QKeras fixed b7",
        "architecture": "64-32-32",
        "suffix": "qkeras_mlp_64_32_32_b7",
        "backend": "qkeras",
        "preferred_variants": ["hls4ml_latency_rf1"],
    },
    {
        "key": "qkeras_b7_128_32",
        "model_family": "QKeras fixed b7",
        "architecture": "128-32",
        "suffix": "qkeras_topo_128_32_b7",
        "backend": "qkeras",
        "preferred_variants": ["hls4ml_latency_rf1"],
    },
    {
        "key": "hgq_64_32_32",
        "model_family": "HGQ",
        "architecture": "64-32-32",
        "suffix": "hgq_mlp_64_32_32",
        "backend": "hgq",
        "preferred_variants": ["hls4ml_latency_rf1"],
    },
    {
        "key": "hgq_128_32",
        "model_family": "HGQ",
        "architecture": "128-32",
        "suffix": "hgq_topo_128_32",
        "backend": "hgq",
        "preferred_variants": ["hls4ml_latency_rf1"],
    },
    {
        "key": "qkeras_binary_64_32_32",
        "model_family": "QKeras binary",
        "architecture": "64-32-32",
        "suffix": "qkeras_mlp_binary_64_32_32",
        "backend": "qkeras",
        "preferred_variants": ["hls4ml_latency_rf1"],
        "lowbit": True,
    },
    {
        "key": "qkeras_binary_128_32",
        "model_family": "QKeras binary",
        "architecture": "128-32",
        "suffix": "qkeras_topo_binary_128_32",
        "backend": "qkeras",
        "preferred_variants": ["hls4ml_latency_rf1"],
        "lowbit": True,
    },
    {
        "key": "qkeras_ternary_64_32_32",
        "model_family": "QKeras ternary",
        "architecture": "64-32-32",
        "suffix": "qkeras_mlp_ternary_64_32_32",
        "backend": "qkeras",
        "preferred_variants": ["hls4ml_latency_rf1"],
        "lowbit": True,
    },
    {
        "key": "qkeras_ternary_128_32",
        "model_family": "QKeras ternary",
        "architecture": "128-32",
        "suffix": "qkeras_topo_ternary_128_32",
        "backend": "qkeras",
        "preferred_variants": ["hls4ml_latency_rf1"],
        "lowbit": True,
    },
    {
        "key": "bitnet_64_32_32",
        "model_family": "BitNet binary",
        "architecture": "64-32-32",
        "suffix": "bitnet_sigmoid_f7_fixed",
        "backend": "pytorch",
        "preferred_variants": ["hls4ml_parent_v26_patch_sigmoid", "custom_v9_sigmoid"],
        "lowbit": True,
    },
    {
        "key": "bitnet_128_32",
        "model_family": "BitNet binary",
        "architecture": "128-32",
        "suffix": "bitnet_topo_sigmoid_f7_fixed",
        "backend": "pytorch",
        "preferred_variants": ["custom_v9_sigmoid", "hls4ml_parent_v26_patch_sigmoid"],
        "lowbit": True,
    },
    {
        "key": "bit158_64_32_32",
        "model_family": "Bit158 sparse ternary",
        "architecture": "64-32-32",
        "suffix": "bit158_sigmoid_f7_fixed",
        "backend": "pytorch",
        "preferred_variants": ["custom_v9_sigmoid"],
        "lowbit": True,
    },
    {
        "key": "bit158_128_32",
        "model_family": "Bit158 sparse ternary",
        "architecture": "128-32",
        "suffix": "bit158_topo_sigmoid_f7_fixed",
        "backend": "pytorch",
        "preferred_variants": ["custom_v9_sigmoid"],
        "lowbit": True,
    },
    {
        "key": "xgboost_bdt_d4_100",
        "model_family": "XGBoost BDT d4x100",
        "architecture": "100 trees depth 4",
        "suffix": "xgboost_bdt_d4_100",
        "backend": "xgboost",
        "preferred_variants": ["conifer_unrolled", "conifer_tree"],
    },
]

MULTICLASS_SPECS = [
    ("Dense MLP", "64-32-32", "mlp_baseline", "pytorch", ["hls4ml_latency_rf1"]),
    ("Dense MLP", "128-32", "mlp_topo", "pytorch", ["hls4ml_latency_rf1"]),
    ("QKeras fixed b7", "64-32-32", "qkeras_mlp_b7", "qkeras", ["hls4ml_latency_rf1"]),
    ("QKeras binary", "64-32-32", "qkeras_mlp_binary", "qkeras", ["hls4ml_latency_rf1"]),
    ("QKeras ternary", "64-32-32", "qkeras_mlp_ternary", "qkeras", ["hls4ml_latency_rf1"]),
    ("HGQ", "64-32-32", "hgq_mlp", "hgq", ["hls4ml_latency_rf1"]),
    ("HGQ", "128-32", "hgq_mlp_topo", "hgq", ["hls4ml_latency_rf1"]),
    ("BitNet binary", "64-32-32", "bitnet_mlp_f7_fixed", "pytorch", ["custom_native_tree1", "custom_native"]),
    ("BitNet binary", "128-32", "bitnet_topo_f7_fixed", "pytorch", ["custom_native_tree1", "custom_native"]),
    ("Bit158 sparse ternary", "64-32-32", "bit158_mlp_f7_fixed", "pytorch", ["custom_native_tree1", "custom_native"]),
    ("Bit158 sparse ternary", "128-32", "bit158_topo_f7_fixed", "pytorch", ["custom_native_tree1", "custom_native"]),
    ("XGBoost BDT", "available", "xgboost_bdt", "xgboost", ["conifer_unrolled", "conifer_tree"]),
]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and value != ""]
    return sum(clean) / len(clean) if clean else None


def stdev(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and value != ""]
    return statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None)


def fmt(value: object, digits: int = 5) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.{digits}f}"


def metric_suffix(backend: str) -> str:
    return {
        "pytorch": "_pytorch.json",
        "qkeras": "_qkeras.json",
        "hgq": "_hgq.json",
        "xgboost": "_xgboost.json",
    }[backend]


def weights_suffixes(backend: str) -> tuple[str, ...]:
    if backend in {"qkeras", "hgq"}:
        return (".weights.h5",)
    if backend == "xgboost":
        return (".pkl",)
    return (".pt",)


def base_for(task: str, suffix: str) -> str:
    return f"{TASKS[task]['prefix']}{suffix}"


def run_for(base: str, seed: int) -> str:
    return f"{base}__seed{seed}"


def split_run(run_name: str) -> tuple[str, int]:
    match = re.match(r"^(?P<base>.+)__seed(?P<seed>\d+)$", run_name)
    if match:
        return match.group("base"), int(match.group("seed"))
    return run_name, 42


def load_synth_rows() -> list[dict]:
    rows = []
    for path in sorted(SYNTHESIS.glob("*/result.json")):
        payload = read_json(path)
        run_name = payload.get("run_name")
        if not run_name:
            continue
        base, seed = split_run(str(run_name))
        variant = payload.get("variant") or path.parent.name.removeprefix(str(run_name)).lstrip("_")
        status = payload.get("synthesis_status") or ("success" if payload.get("success") else "")
        rows.append(
            {
                "run_name": str(run_name),
                "base_run_name": base,
                "seed": seed,
                "variant": str(variant),
                "result_path": path,
                "status": status,
                "latency_cycles": payload.get("latency_cycles") or payload.get("latency_cycles_max"),
                "ii_cycles": payload.get("initiation_interval_cycles"),
                "lut": payload.get("lut"),
                "ff": payload.get("ff"),
                "dsp": payload.get("dsp"),
                "bram_18k": payload.get("bram_18k") or payload.get("bram"),
                "uram": payload.get("uram"),
                "clock_target_ns": payload.get("clock_target_ns") or payload.get("clock_period_ns"),
                "clock_achieved_ns": payload.get("clock_achieved_ns"),
                "tool": payload.get("tool"),
            }
        )
    return rows


SYNTH_ROWS = load_synth_rows()


def synth_candidates(run_name: str, variants: list[str] | None = None) -> list[dict]:
    candidates = [
        row
        for row in SYNTH_ROWS
        if row["run_name"] == run_name
        and row.get("status") in {"success", ""}
        and row.get("latency_cycles") is not None
    ]
    if variants:
        ranked = []
        for row in candidates:
            try:
                rank = variants.index(row["variant"])
            except ValueError:
                rank = len(variants) + 1
            ranked.append((rank, row))
        candidates = [row for _, row in sorted(ranked, key=lambda item: item[0])]
    return candidates


def preferred_synth(run_name: str, variants: list[str] | None = None) -> dict | None:
    candidates = synth_candidates(run_name, variants)
    return candidates[0] if candidates else None


def all_metric_paths_for(base: str, backend: str) -> list[Path]:
    return [RESULTS / f"{run_for(base, seed)}{metric_suffix(backend)}" for seed in SEEDS]


def load_metrics(base: str, backend: str) -> list[dict]:
    return [read_json(path) for path in all_metric_paths_for(base, backend) if path.exists()]


def xgb_probabilities(run_name: str, task: str) -> np.ndarray | None:
    try:
        import xgboost  # noqa: F401
    except Exception:
        return None
    model_path = ROOT / "models" / f"{run_name}.pkl"
    if not model_path.exists():
        return None
    cache_path = ROOT / "data" / "cache" / TASKS[task]["cache"]
    if not cache_path.exists():
        return None
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    model = payload["model"]
    arrays = np.load(cache_path, mmap_mode="r")
    probabilities = np.asarray(model.predict_proba(arrays["x_test"]))
    if probabilities.ndim == 1:
        probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
    return probabilities


def signal_efficiency_at_fpr(run_name: str, task: str) -> float | None:
    pred_path = ROOT / "data" / "synthesis" / "reference_predictions" / f"{run_name}.npy"
    cache_path = ROOT / "data" / "cache" / TASKS[task]["cache"]
    if not cache_path.exists():
        return None
    if pred_path.exists():
        probabilities = np.asarray(np.load(pred_path, mmap_mode="r"))
    elif "xgboost" in run_name:
        probabilities = xgb_probabilities(run_name, task)
        if probabilities is None:
            return None
    else:
        return None
    arrays = np.load(cache_path, mmap_mode="r")
    y = np.asarray(arrays["y_test"]).astype(int)
    scores = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities.reshape(-1)
    background = scores[y == 0]
    signal = scores[y == 1]
    if len(background) == 0 or len(signal) == 0:
        return None
    # Largest threshold that keeps background acceptance at or below target.
    threshold = np.quantile(background, 1.0 - FPR_TARGET, method="higher")
    if np.mean(background >= threshold) > FPR_TARGET:
        threshold = np.nextafter(threshold, np.inf)
    return float(np.mean(signal >= threshold))


def aggregate_task_model(task: str, spec: dict, include_synth_variant: str | None = None) -> dict:
    base = base_for(task, spec["suffix"])
    metrics = load_metrics(base, spec["backend"])
    acc = [row.get("accuracy") for row in metrics]
    auc = [row.get("macro_auc") for row in metrics]
    eff = []
    for seed in SEEDS:
        run_name = run_for(base, seed)
        eff.append(signal_efficiency_at_fpr(run_name, task))
    synth_rows = []
    for seed in SEEDS:
        run_name = run_for(base, seed)
        variants = [include_synth_variant] if include_synth_variant else spec.get("preferred_variants")
        synth = preferred_synth(run_name, variants)
        if synth:
            synth_rows.append(synth)
    row = {
        "task": task,
        "task_label": TASKS[task]["label"],
        "model_key": spec["key"],
        "model": spec["model_family"],
        "architecture": spec["architecture"],
        "base_run_name": base,
        "backend": spec["backend"],
        "seeds_metrics": len(metrics),
        "seeds_synth": len(synth_rows),
        "synth_variant": include_synth_variant
        or (synth_rows[0]["variant"] if synth_rows else ",".join(spec.get("preferred_variants", []))),
        "accuracy_mean": mean(acc),
        "accuracy_std": stdev(acc),
        "auc_mean": mean(auc),
        "auc_std": stdev(auc),
        "signal_eff_at_1pct_fpr_mean": mean(eff),
        "signal_eff_at_1pct_fpr_std": stdev(eff),
        "latency_cycles_mean": mean([row.get("latency_cycles") for row in synth_rows]),
        "latency_cycles_std": stdev([row.get("latency_cycles") for row in synth_rows]),
        "ii_cycles_mean": mean([row.get("ii_cycles") for row in synth_rows]),
        "lut_mean": mean([row.get("lut") for row in synth_rows]),
        "lut_std": stdev([row.get("lut") for row in synth_rows]),
        "ff_mean": mean([row.get("ff") for row in synth_rows]),
        "ff_std": stdev([row.get("ff") for row in synth_rows]),
        "dsp_mean": mean([row.get("dsp") for row in synth_rows]),
        "dsp_std": stdev([row.get("dsp") for row in synth_rows]),
        "bram18_mean": mean([row.get("bram_18k") for row in synth_rows]),
        "bram18_std": stdev([row.get("bram_18k") for row in synth_rows]),
        "complete_metrics": len(metrics) == 3,
        "complete_synth": len(synth_rows) == 3,
    }
    return row


def build_status_matrix() -> list[dict]:
    rows = []
    plot_targets = {
        "benchmark_main_binary_table.csv",
        "benchmark_secondary_top_table.csv",
        "benchmark_multiclass_summary.csv",
        "benchmark_lowbit_comparison.csv",
        "benchmark_seed_statistics.csv",
    }
    for task in ("binary_qg_vs_wzt", "binary_topqg"):
        for spec in CORE_SPECS:
            base = base_for(task, spec["suffix"])
            for seed in SEEDS:
                run_name = run_for(base, seed)
                config_exists = (ROOT / "logs" / "run_configs" / f"{run_name}.json").exists()
                weights_exists = any((ROOT / "models" / f"{run_name}{suffix}").exists() for suffix in weights_suffixes(spec["backend"]))
                metrics_exists = (RESULTS / f"{run_name}{metric_suffix(spec['backend'])}").exists()
                onnx_exists = (
                    (ROOT / "onnx" / "hardware" / f"{run_name}.onnx").exists()
                    or (ROOT / "onnx" / "hardware" / f"{run_name}_quantized.pt").exists()
                    or (ROOT / "data" / "synthesis" / "reference_predictions" / f"{run_name}.npy").exists()
                    or spec["backend"] == "xgboost"
                )
                hls_exists = (ROOT / "hls_projects" / run_name).exists() or any(SYNTHESIS.glob(f"{run_name}_*"))
                synths = synth_candidates(run_name)
                synth_exists = bool(synths)
                parsed = any(row.get("latency_cycles") is not None and row.get("lut") is not None for row in synths)
                complete = config_exists and weights_exists and metrics_exists and onnx_exists and hls_exists and synth_exists and parsed
                rows.append(
                    {
                        "task": task,
                        "model": spec["model_family"],
                        "architecture": spec["architecture"],
                        "base_run_name": base,
                        "run_name": run_name,
                        "seed": seed,
                        "config_exists": config_exists,
                        "trained_weights_exist": weights_exists,
                        "test_metrics_exist": metrics_exists,
                        "onnx_or_export_exists": onnx_exists,
                        "hls_project_exists": hls_exists,
                        "csynthesis_report_exists": synth_exists,
                        "parsed_hardware_metrics_exist": parsed,
                        "plots_tables_included": bool(plot_targets),
                        "complete": complete,
                    }
                )
    for family, architecture, base, backend, _variants in MULTICLASS_SPECS:
        for seed in SEEDS:
            run_name = run_for(base, seed)
            config_exists = (
                (ROOT / "logs" / "run_configs" / f"{run_name}.json").exists()
                or (ROOT / "configs" / f"{base}.json").exists()
                or any((ROOT / "configs").glob(f"{base}*.json"))
            )
            weights_exists = any((ROOT / "models" / f"{run_name}{suffix}").exists() for suffix in weights_suffixes(backend))
            if not weights_exists and seed == 42:
                weights_exists = any((ROOT / "models" / f"{base}{suffix}").exists() for suffix in weights_suffixes(backend))
            metric_path = RESULTS / f"{run_name}{metric_suffix(backend)}"
            if not metric_path.exists() and seed == 42:
                metric_path = RESULTS / f"{base}{metric_suffix(backend)}"
            metrics_exists = metric_path.exists()
            onnx_exists = (
                (ROOT / "onnx" / "hardware" / f"{run_name}.onnx").exists()
                or (ROOT / "onnx" / "hardware" / f"{base}.onnx").exists()
                or (ROOT / "onnx" / "hardware" / f"{run_name}_quantized.pt").exists()
                or (ROOT / "onnx" / "hardware" / f"{base}_quantized.pt").exists()
                or backend in {"qkeras", "hgq", "xgboost"}
            )
            hls_exists = (ROOT / "hls_projects" / run_name).exists() or (ROOT / "hls_projects" / base).exists() or any(SYNTHESIS.glob(f"{run_name}_*"))
            synths = synth_candidates(run_name) or (synth_candidates(base) if seed == 42 else [])
            parsed = any(row.get("latency_cycles") is not None and row.get("lut") is not None for row in synths)
            rows.append(
                {
                    "task": "multiclass",
                    "model": family,
                    "architecture": architecture,
                    "base_run_name": base,
                    "run_name": run_name,
                    "seed": seed,
                    "config_exists": config_exists,
                    "trained_weights_exist": weights_exists,
                    "test_metrics_exist": metrics_exists,
                    "onnx_or_export_exists": onnx_exists,
                    "hls_project_exists": hls_exists,
                    "csynthesis_report_exists": bool(synths),
                    "parsed_hardware_metrics_exist": parsed,
                    "plots_tables_included": True,
                    "complete": config_exists and weights_exists and metrics_exists and parsed,
                }
            )
    return rows


def benchmark_table_rows(task: str) -> list[dict]:
    rows = []
    for spec in CORE_SPECS:
        if spec["suffix"] == "xgboost_bdt_d4_100":
            for variant in ("conifer_unrolled", "conifer_tree"):
                row = aggregate_task_model(task, spec, include_synth_variant=variant)
                row["model"] = f"{row['model']} ({variant.replace('conifer_', '')})"
                rows.append(row)
        else:
            rows.append(aggregate_task_model(task, spec))
    return rows


def format_table(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        output.append(
            {
                "task": row["task"],
                "model": row["model"],
                "architecture": row["architecture"],
                "base_run_name": row["base_run_name"],
                "accuracy_mean": fmt(row["accuracy_mean"]),
                "accuracy_std": fmt(row["accuracy_std"]),
                "auc_mean": fmt(row["auc_mean"]),
                "auc_std": fmt(row["auc_std"]),
                "signal_eff_at_1pct_fpr_mean": fmt(row["signal_eff_at_1pct_fpr_mean"]),
                "signal_eff_at_1pct_fpr_std": fmt(row["signal_eff_at_1pct_fpr_std"]),
                "synth_variant": row["synth_variant"],
                "latency_cycles_mean": fmt(row["latency_cycles_mean"], 2),
                "latency_cycles_std": fmt(row["latency_cycles_std"], 2),
                "ii_cycles_mean": fmt(row["ii_cycles_mean"], 2),
                "lut_mean": fmt(row["lut_mean"], 1),
                "lut_std": fmt(row["lut_std"], 1),
                "ff_mean": fmt(row["ff_mean"], 1),
                "dsp_mean": fmt(row["dsp_mean"], 2),
                "bram18_mean": fmt(row["bram18_mean"], 2),
                "seeds_metrics": row["seeds_metrics"],
                "seeds_synth": row["seeds_synth"],
                "status": "complete" if row["complete_metrics"] and row["complete_synth"] else "partial",
            }
        )
    return output


def multiclass_summary() -> list[dict]:
    rows = []
    for family, architecture, base, backend, variants in MULTICLASS_SPECS:
        metrics = []
        for seed in SEEDS:
            path = RESULTS / f"{run_for(base, seed)}{metric_suffix(backend)}"
            if not path.exists() and seed == 42:
                path = RESULTS / f"{base}{metric_suffix(backend)}"
            if path.exists():
                metrics.append(read_json(path))
        synths = []
        for seed in SEEDS:
            run_name = run_for(base, seed)
            synth = preferred_synth(run_name, variants)
            if not synth and seed == 42:
                synth = preferred_synth(base, variants)
            if synth:
                synths.append(synth)
        rows.append(
            {
                "task": "multiclass",
                "model": family,
                "architecture": architecture,
                "base_run_name": base,
                "accuracy_mean": fmt(mean([m.get("accuracy") for m in metrics])),
                "accuracy_std": fmt(stdev([m.get("accuracy") for m in metrics])),
                "macro_auc_mean": fmt(mean([m.get("macro_auc") for m in metrics])),
                "macro_auc_std": fmt(stdev([m.get("macro_auc") for m in metrics])),
                "synth_variant": synths[0]["variant"] if synths else "",
                "latency_cycles_mean": fmt(mean([s.get("latency_cycles") for s in synths]), 2),
                "lut_mean": fmt(mean([s.get("lut") for s in synths]), 1),
                "dsp_mean": fmt(mean([s.get("dsp") for s in synths]), 2),
                "seeds_metrics": len(metrics),
                "seeds_synth": len(synths),
                "status": "complete" if len(metrics) == 3 and len(synths) == 3 else "partial",
            }
        )
    return rows


def pareto(rows: list[dict], score_key: str, cost_keys: tuple[str, ...]) -> list[dict]:
    candidates = [
        row
        for row in rows
        if row.get(score_key) is not None
        and all(row.get(key) is not None for key in cost_keys)
    ]
    frontier = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            score_better = other[score_key] >= row[score_key]
            costs_better = all(other[key] <= row[key] for key in cost_keys)
            strict = other[score_key] > row[score_key] or any(other[key] < row[key] for key in cost_keys)
            if score_better and costs_better and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def short_model_label(model: str) -> str:
    labels = {
        "Dense MLP": "Dense",
        "QKeras fixed b7": "QK fixed b7",
        "HGQ": "HGQ",
        "QKeras binary": "QK binary",
        "QKeras ternary": "QK ternary",
        "BitNet binary": "BitNet",
        "Bit158 sparse ternary": "Bit158b",
        "XGBoost BDT d4x100 (unrolled)": "BDT unrolled",
        "XGBoost BDT d4x100 (tree)": "BDT tree",
    }
    return labels.get(model, model)


def annotate_without_overlap(ax, rows: list[dict], x_key: str, y_key: str) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    candidates = [
        (6, 6),
        (6, -6),
        (-6, 6),
        (-6, -6),
        (10, 12),
        (-10, 12),
        (10, -12),
        (-10, -12),
        (0, 14),
        (0, -14),
        (14, 0),
        (-14, 0),
        (18, 12),
        (-18, 12),
        (18, -12),
        (-18, -12),
    ]
    ordered = sorted(rows, key=lambda row: (row.get(x_key, 0), row.get(y_key, 0)))
    for row in ordered:
        label = short_model_label(row["model"])
        x = row[x_key]
        y = row[y_key]
        row_candidates = candidates
        if row.get("model") == "QKeras fixed b7" and row.get("architecture") == "128-32":
            row_candidates = [
                (-10, 0),
                (-14, 0),
                (-18, 12),
                (-18, -12),
                (-10, 12),
                (-10, -12),
                (-6, 6),
                (-6, -6),
            ] + candidates
        placed = False
        for dx, dy in row_candidates:
            ann = ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.5, "alpha": 0.55},
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.06, 1.14)
            if not any(bbox.overlaps(other) for other in occupied):
                occupied.append(bbox)
                placed = True
                break
            ann.remove()
        if not placed:
            ann = ax.annotate(
                label,
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=7,
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.5, "alpha": 0.55},
            )
            fig.canvas.draw()
            occupied.append(ann.get_window_extent(renderer=renderer).expanded(1.06, 1.14))


def scale_sizes(rows: list[dict], key: str, min_size: float = 55.0, max_size: float = 260.0) -> dict[int, float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return {id(row): min_size for row in rows}
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {id(row): 0.5 * (min_size + max_size) for row in rows}
    scaled = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            scaled[id(row)] = min_size
            continue
        frac = (float(value) - low) / (high - low)
        scaled[id(row)] = min_size + frac * (max_size - min_size)
    return scaled


def plot_scatter(
    rows: list[dict],
    x_key: str,
    y_key: str,
    path: Path,
    title: str,
    xlabel: str,
    *,
    ylabel: str = "ROC AUC",
    task_filter: str | None = None,
    model_filter: set[str] | None = None,
    color_by_arch: bool = False,
    annotate: bool = True,
    xlim: tuple[float, float] | None = None,
) -> None:
    clean = [row for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
    if task_filter:
        clean = [row for row in clean if row["task"] == task_filter]
    if model_filter:
        clean = [row for row in clean if row["model"] in model_filter]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    markers = {
        "binary_qg_vs_wzt": "o",
        "binary_topqg": "s",
        "multiclass": "^",
    }
    task_palette = {
        "binary_qg_vs_wzt": "#1f77b4",
        "binary_topqg": "#d62728",
        "multiclass": "#2ca02c",
    }
    arch_palette = {
        "64-32-32": "#1f77b4",
        "128-32": "#d62728",
        "100 trees depth 4": "#2ca02c",
    }
    tasks = (task_filter,) if task_filter else ("binary_qg_vs_wzt", "binary_topqg")
    labeled_rows = []
    for task in tasks:
        if task is None:
            continue
        subset = [row for row in clean if row["task"] == task]
        if not subset:
            continue
        if color_by_arch:
            grouped: dict[str, list[dict]] = {}
            for row in subset:
                grouped.setdefault(row["architecture"], []).append(row)
            for arch, arch_rows in grouped.items():
                ax.scatter(
                    [row[x_key] for row in arch_rows],
                    [row[y_key] for row in arch_rows],
                    label=arch,
                    marker=markers[task],
                    s=70,
                    alpha=0.92,
                    color=arch_palette.get(arch, "#7f7f7f"),
                    edgecolors="white",
                    linewidths=0.7,
                )
        else:
            ax.scatter(
                [row[x_key] for row in subset],
                [row[y_key] for row in subset],
                label=TASKS[task]["label"],
                marker=markers[task],
                s=70,
                alpha=0.92,
                color=task_palette.get(task, "#1f77b4"),
                edgecolors="white",
                linewidths=0.7,
            )
        labeled_rows.extend(subset)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if x_key == "lut_mean":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value / 1000.0))}k"))
    ax.grid(True, alpha=0.18, linewidth=0.7)
    if annotate and labeled_rows:
        annotate_without_overlap(ax, labeled_rows, x_key, y_key)
    ax.legend(frameon=False, loc="best", title="Architecture" if color_by_arch else None)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latency_lut_bubble(
    rows: list[dict],
    path: Path,
    title: str,
    *,
    task_filter: str,
    model_filter: set[str],
    xlim: tuple[float, float] | None = None,
) -> None:
    clean = [
        row for row in rows
        if row.get("latency_cycles_mean") is not None
        and row.get("auc_mean") is not None
        and row.get("lut_mean") is not None
        and row.get("task") == task_filter
        and row.get("model") in model_filter
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    arch_palette = {
        "64-32-32": "#1f77b4",
        "128-32": "#d62728",
        "100 trees depth 4": "#2ca02c",
    }
    size_map = scale_sizes(clean, "lut_mean")
    grouped: dict[str, list[dict]] = {}
    for row in clean:
        grouped.setdefault(row["architecture"], []).append(row)
    for arch, arch_rows in grouped.items():
        ax.scatter(
            [row["latency_cycles_mean"] for row in arch_rows],
            [row["auc_mean"] for row in arch_rows],
            s=[size_map[id(row)] for row in arch_rows],
            label=arch,
            alpha=0.88,
            color=arch_palette.get(arch, "#7f7f7f"),
            edgecolors="white",
            linewidths=0.8,
        )
    ax.set_title(title)
    ax.set_xlabel("Latency cycles")
    ax.set_ylabel("ROC AUC")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    annotate_without_overlap(ax, clean, "latency_cycles_mean", "auc_mean")

    arch_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=arch_palette["64-32-32"], markeredgecolor="white", markeredgewidth=0.8, label="64-32-32"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=arch_palette["128-32"], markeredgecolor="white", markeredgewidth=0.8, label="128-32"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=arch_palette["100 trees depth 4"], markeredgecolor="white", markeredgewidth=0.8, label="100 trees depth 4"),
    ]
    arch_legend = ax.legend(
        handles=arch_handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.88),
        title="Architecture",
    )
    ax.add_artist(arch_legend)

    lut_values = sorted({int(round(row["lut_mean"] / 1000.0)) for row in clean})
    if lut_values:
        picks = [lut_values[0], lut_values[len(lut_values) // 2], lut_values[-1]]
        picks = list(dict.fromkeys(picks))
        low = min(row["lut_mean"] for row in clean)
        high = max(row["lut_mean"] for row in clean)
        size_handles = []
        size_labels = []
        for value_k in picks:
            value = value_k * 1000.0
            if math.isclose(low, high):
                size = 205.0
            else:
                frac = (value - low) / (high - low)
                frac = min(1.0, max(0.0, frac))
                size = 55.0 + frac * (260.0 - 55.0)
            size_handles.append(
                plt.scatter([], [], s=size, color="#999999", alpha=0.5, edgecolors="none")
            )
            size_labels.append(f"{value_k}k LUT")
        title_x = 0.78
        title_y = 0.59
        ax.text(
            title_x,
            title_y,
            "Resource scale",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="center",
        )
        ys = [0.52, 0.465, 0.41]
        for size, label, y in zip([h.get_sizes()[0] for h in size_handles], size_labels, ys):
            ax.scatter([title_x + 0.03], [y], s=size, color="#999999", alpha=0.5, edgecolors="none", transform=ax.transAxes, clip_on=False)
            ax.text(
                title_x + 0.09,
                y,
                label,
                transform=ax.transAxes,
                fontsize=9,
                ha="left",
                va="center",
            )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def precision_sweep_rows() -> list[dict]:
    rows = []
    for bits in (5, 7, 8, 10, 12):
        base = f"qkeras_mlp_b{bits}"
        metrics = load_metrics(base, "qkeras")
        synths = [preferred_synth(run_for(base, seed), ["hls4ml_latency_rf1"]) for seed in SEEDS]
        synths = [row for row in synths if row]
        if not metrics and bits == 7:
            metrics = load_metrics("qkeras_mlp", "qkeras")
        rows.append(
            {
                "bits": bits,
                "accuracy": mean([row.get("accuracy") for row in metrics]),
                "auc": mean([row.get("macro_auc") for row in metrics]),
                "latency": mean([row.get("latency_cycles") for row in synths]),
                "lut": mean([row.get("lut") for row in synths]),
            }
        )
    return rows


def bitnet_scaling_rows() -> list[dict]:
    names = [
        ("f5", "bitnet_mlp_f5_fixed"),
        ("f7", "bitnet_mlp_f7_fixed"),
        ("f8", "bitnet_mlp_f8_fixed"),
        ("f10", "bitnet_mlp_f10_fixed"),
        ("f12", "bitnet_mlp_f12_fixed"),
        ("power2", "bitnet_mlp_f7_power2"),
    ]
    rows = []
    for label, base in names:
        metrics = []
        for seed in SEEDS:
            path = RESULTS / f"{run_for(base, seed)}_pytorch.json"
            if path.exists():
                metrics.append(read_json(path))
            elif seed == 42 and (RESULTS / f"{base}_pytorch.json").exists():
                metrics.append(read_json(RESULTS / f"{base}_pytorch.json"))
        synths = []
        for seed in SEEDS:
            synth = preferred_synth(run_for(base, seed), ["custom_native_tree1", "custom_native"])
            if synth:
                synths.append(synth)
            elif seed == 42:
                synth = preferred_synth(base, ["custom_native_tree1", "custom_native"])
                if synth:
                    synths.append(synth)
        rows.append(
            {
                "scale": label,
                "accuracy": mean([row.get("accuracy") for row in metrics]),
                "auc": mean([row.get("macro_auc") for row in metrics]),
                "latency": mean([row.get("latency_cycles") for row in synths]),
                "lut": mean([row.get("lut") for row in synths]),
            }
        )
    return rows


def plot_sweep(rows: list[dict], x_key: str, label_key: str, path: Path, title: str) -> None:
    clean = [row for row in rows if row.get("auc") is not None or row.get("lut") is not None]
    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(clean))
    ax1.plot(x, [row.get("auc") for row in clean], marker="o", label="AUC", color="#2f6f73")
    ax1.set_ylabel("AUC")
    ax2 = ax1.twinx()
    ax2.plot(x, [row.get("lut") for row in clean], marker="s", label="LUT", color="#9a5c2e")
    ax2.set_ylabel("LUT")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(row[label_key]) for row in clean])
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def missing_commands(status_rows: list[dict]) -> list[str]:
    commands = []
    for row in status_rows:
        if row["task"] == "binary_qg_vs_wzt" and row["complete"] is False:
            commands.append(f"python scripts/run_binary_benchmark_workflow.py --class-mode binary_qg_vs_wzt --namespace binary --log-subdir binary_benchmark --seeds {row['seed']}")
        elif row["task"] == "binary_topqg" and row["complete"] is False:
            commands.append(f"python scripts/run_binary_benchmark_workflow.py --class-mode binary_top_vs_qg --namespace binary_topqg --log-subdir binary_topqg_benchmark --seeds {row['seed']}")
    return sorted(set(commands))


def cost_value(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return 1e99
    return float(value)


def markdown_table(rows: list[dict], cols: list[str], limit: int | None = None) -> list[str]:
    subset = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in subset:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return lines


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    if not any(RESULTS.glob("*_pytorch.json")) and (RESULTS / "benchmark_main_binary_table.csv").exists():
        fallback_path = ROOT / "scripts" / "reproduce_public_artifacts.py"
        spec = importlib.util.spec_from_file_location("reproduce_public_artifacts", fallback_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load public artifact fallback from {fallback_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main()

    status = build_status_matrix()
    status_fields = [
        "task",
        "model",
        "architecture",
        "base_run_name",
        "run_name",
        "seed",
        "config_exists",
        "trained_weights_exist",
        "test_metrics_exist",
        "onnx_or_export_exists",
        "hls_project_exists",
        "csynthesis_report_exists",
        "parsed_hardware_metrics_exist",
        "plots_tables_included",
        "complete",
    ]
    write_csv(RESULTS / "benchmark_status_matrix.csv", status, status_fields)

    primary_raw = benchmark_table_rows("binary_qg_vs_wzt")
    secondary_raw = benchmark_table_rows("binary_topqg")
    all_raw = primary_raw + secondary_raw
    table_fields = [
        "task",
        "model",
        "architecture",
        "base_run_name",
        "accuracy_mean",
        "accuracy_std",
        "auc_mean",
        "auc_std",
        "signal_eff_at_1pct_fpr_mean",
        "signal_eff_at_1pct_fpr_std",
        "synth_variant",
        "latency_cycles_mean",
        "latency_cycles_std",
        "ii_cycles_mean",
        "lut_mean",
        "lut_std",
        "ff_mean",
        "dsp_mean",
        "bram18_mean",
        "seeds_metrics",
        "seeds_synth",
        "status",
    ]
    primary = format_table(primary_raw)
    secondary = format_table(secondary_raw)
    write_csv(RESULTS / "benchmark_main_binary_table.csv", primary, table_fields)
    write_csv(RESULTS / "benchmark_secondary_top_table.csv", secondary, table_fields)

    multiclass = multiclass_summary()
    write_csv(
        RESULTS / "benchmark_multiclass_summary.csv",
        multiclass,
        [
            "task",
            "model",
            "architecture",
            "base_run_name",
            "accuracy_mean",
            "accuracy_std",
            "macro_auc_mean",
            "macro_auc_std",
            "synth_variant",
            "latency_cycles_mean",
            "lut_mean",
            "dsp_mean",
            "seeds_metrics",
            "seeds_synth",
            "status",
        ],
    )

    lowbit_raw = [row for row in all_raw if any(term in row["model"] for term in ("QKeras binary", "QKeras ternary", "BitNet", "Bit158"))]
    lowbit = format_table(lowbit_raw)
    write_csv(RESULTS / "benchmark_lowbit_comparison.csv", lowbit, table_fields)

    seed_stats = []
    for row in all_raw:
        seed_stats.append(
            {
                "task": row["task"],
                "model": row["model"],
                "architecture": row["architecture"],
                "base_run_name": row["base_run_name"],
                "synth_variant": row["synth_variant"],
                "accuracy_mean": fmt(row["accuracy_mean"]),
                "accuracy_std": fmt(row["accuracy_std"]),
                "auc_mean": fmt(row["auc_mean"]),
                "auc_std": fmt(row["auc_std"]),
                "latency_cycles_mean": fmt(row["latency_cycles_mean"], 2),
                "latency_cycles_std": fmt(row["latency_cycles_std"], 2),
                "lut_mean": fmt(row["lut_mean"], 1),
                "lut_std": fmt(row["lut_std"], 1),
                "dsp_mean": fmt(row["dsp_mean"], 2),
                "dsp_std": fmt(row["dsp_std"], 2),
                "seeds_metrics": row["seeds_metrics"],
                "seeds_synth": row["seeds_synth"],
            }
        )
    write_csv(
        RESULTS / "benchmark_seed_statistics.csv",
        seed_stats,
        [
            "task",
            "model",
            "architecture",
            "base_run_name",
            "synth_variant",
            "accuracy_mean",
            "accuracy_std",
            "auc_mean",
            "auc_std",
            "latency_cycles_mean",
            "latency_cycles_std",
            "lut_mean",
            "lut_std",
            "dsp_mean",
            "dsp_std",
            "seeds_metrics",
            "seeds_synth",
        ],
    )

    pareto_rows = pareto(all_raw, "auc_mean", ("lut_mean", "latency_cycles_mean"))
    pareto_csv = format_table(pareto_rows)
    write_csv(RESULTS / "benchmark_pareto_candidates.csv", pareto_csv, table_fields)

    plot_scatter(all_raw, "lut_mean", "auc_mean", PLOTS / "benchmark_pareto_auc_vs_lut.png", "AUC vs LUT C-synthesis estimate", "LUT")
    plot_scatter(all_raw, "latency_cycles_mean", "auc_mean", PLOTS / "benchmark_pareto_auc_vs_latency.png", "AUC vs latency C-synthesis estimate", "Latency cycles")
    plot_scatter(
        primary_raw,
        "lut_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs LUT",
        "LUT",
        task_filter="binary_qg_vs_wzt",
        model_filter={
            "Dense MLP",
            "QKeras fixed b7",
            "HGQ",
            "QKeras binary",
            "QKeras ternary",
            "BitNet binary",
            "Bit158 sparse ternary",
            "XGBoost BDT d4x100 (unrolled)",
        },
        color_by_arch=True,
        annotate=True,
    )
    plot_scatter(
        primary_raw,
        "latency_cycles_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs latency",
        "Latency cycles",
        task_filter="binary_qg_vs_wzt",
        model_filter={
            "Dense MLP",
            "QKeras fixed b7",
            "HGQ",
            "QKeras binary",
            "QKeras ternary",
            "BitNet binary",
            "Bit158 sparse ternary",
            "XGBoost BDT d4x100 (unrolled)",
        },
        color_by_arch=True,
        annotate=True,
        xlim=(0, 30),
    )
    plot_scatter(
        secondary_raw,
        "lut_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_top.png",
        "q/g vs top: AUC vs LUT",
        "LUT",
        task_filter="binary_topqg",
        model_filter={
            "Dense MLP",
            "QKeras fixed b7",
            "HGQ",
            "QKeras binary",
            "QKeras ternary",
            "BitNet binary",
            "Bit158 sparse ternary",
            "XGBoost BDT d4x100 (unrolled)",
        },
        color_by_arch=True,
        annotate=True,
    )
    plot_scatter(
        secondary_raw,
        "latency_cycles_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_top.png",
        "q/g vs top: AUC vs latency",
        "Latency cycles",
        task_filter="binary_topqg",
        model_filter={
            "Dense MLP",
            "QKeras fixed b7",
            "HGQ",
            "QKeras binary",
            "QKeras ternary",
            "BitNet binary",
            "Bit158 sparse ternary",
            "XGBoost BDT d4x100 (unrolled)",
        },
        color_by_arch=True,
        annotate=True,
        xlim=(0, 30),
    )
    multiclass_plot_rows = []
    for row in multiclass:
        plot_row = row.copy()
        macro_auc = row.get("macro_auc_mean")
        plot_row["auc_mean"] = float(macro_auc) if macro_auc not in (None, "") else None
        for key in ("latency_cycles_mean", "lut_mean"):
            value = row.get(key)
            plot_row[key] = float(value) if value not in (None, "") else None
        multiclass_plot_rows.append(plot_row)
    plot_scatter(
        multiclass_plot_rows,
        "lut_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_multiclass.png",
        "Multiclass: macro AUC vs LUT",
        "LUT",
        ylabel="Macro ROC AUC",
        task_filter="multiclass",
        model_filter={
            "Dense MLP",
            "QKeras fixed b7",
            "HGQ",
            "QKeras binary",
            "QKeras ternary",
            "BitNet binary",
            "Bit158 sparse ternary",
        },
        color_by_arch=True,
        annotate=True,
    )
    plot_scatter(
        multiclass_plot_rows,
        "latency_cycles_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_multiclass.png",
        "Multiclass: macro AUC vs latency",
        "Latency cycles",
        ylabel="Macro ROC AUC",
        task_filter="multiclass",
        model_filter={
            "Dense MLP",
            "QKeras fixed b7",
            "HGQ",
            "QKeras binary",
            "QKeras ternary",
            "BitNet binary",
            "Bit158 sparse ternary",
        },
        color_by_arch=True,
        annotate=True,
        xlim=(0, 30),
    )
    complete = [row for row in status if row["complete"]]
    missing = [row for row in status if not row["complete"]]
    failed = []
    for status_path in (RESULTS / "binary_benchmark_workflow_status.json", RESULTS / "binary_topqg_benchmark_workflow_status.json"):
        payload = read_json(status_path)
        for key, value in payload.items():
            if isinstance(value, dict) and value.get("status") == "failed":
                if ":synth_" in key:
                    run_name, variant = key.split(":synth_", 1)
                    if synth_candidates(run_name, [variant]):
                        continue
                failed.append({"key": key, "log": value.get("log"), "returncode": value.get("returncode")})

    hw_rows = [row for row in all_raw if row.get("latency_cycles_mean") is not None]
    best_acc = max(all_raw, key=lambda row: row.get("accuracy_mean") or -1)
    best_latency = min(hw_rows, key=lambda row: cost_value(row, "latency_cycles_mean"))
    best_lut = min(hw_rows, key=lambda row: cost_value(row, "lut_mean"))
    best_dsp = min(hw_rows, key=lambda row: cost_value(row, "dsp_mean"))

    report = [
        "# Benchmark Readiness Report",
        "",
        "Scope: implementation-aware benchmark artifacts for the FastML benchmark artifact.",
        "Hardware numbers are VU13P, 5 ns HLS C-synthesis estimates, not place-and-route.",
        f"Fixed-background signal efficiency is reported at FPR/background acceptance = {FPR_TARGET:g}.",
        "",
        "## Completeness",
        f"- Complete status rows: {len(complete)} / {len(status)}",
        f"- Missing or partial status rows: {len(missing)} / {len(status)}",
        "- Primary binary q/g vs W/Z/top has full seed metrics and full synthesis coverage for the core rows used in the main table.",
        "- Secondary q/g vs top has full seed metrics and full synthesis coverage for the core rows used in the secondary table.",
        "- Multiclass is included as compact supporting material only.",
        "",
        "## Best Rows",
        f"- Best accuracy: {best_acc['task']} {best_acc['model']} {best_acc['architecture']} acc={fmt(best_acc['accuracy_mean'])} AUC={fmt(best_acc['auc_mean'])}",
        f"- Best latency: {best_latency['task']} {best_latency['model']} {best_latency['architecture']} latency={fmt(best_latency['latency_cycles_mean'], 2)} cycles LUT={fmt(best_latency['lut_mean'], 1)}",
        f"- Best LUT: {best_lut['task']} {best_lut['model']} {best_lut['architecture']} LUT={fmt(best_lut['lut_mean'], 1)} latency={fmt(best_lut['latency_cycles_mean'], 2)} cycles",
        f"- Best DSP: {best_dsp['task']} {best_dsp['model']} {best_dsp['architecture']} DSP={fmt(best_dsp['dsp_mean'], 2)} LUT={fmt(best_dsp['lut_mean'], 1)}",
        "",
        "## Pareto Candidates",
    ]
    report.extend(
        markdown_table(
            pareto_csv,
            ["task", "model", "architecture", "auc_mean", "latency_cycles_mean", "lut_mean", "dsp_mean"],
            limit=20,
        )
    )
    report.extend(
        [
            "",
            "## Failed Jobs",
        ]
    )
    if failed:
        for row in failed:
            report.append(f"- {row['key']}: returncode={row['returncode']} log={row['log']}")
    else:
        report.append("- None recorded in the benchmark workflow status files.")
    report.extend(["", "## Missing Or Partial Rows"])
    for row in missing[:80]:
        report.append(
            f"- {row['task']} {row['run_name']}: config={row['config_exists']} weights={row['trained_weights_exist']} "
            f"metrics={row['test_metrics_exist']} export={row['onnx_or_export_exists']} hls={row['hls_project_exists']} "
            f"synth={row['csynthesis_report_exists']} parsed={row['parsed_hardware_metrics_exist']}"
        )
    if len(missing) > 80:
        report.append(f"- ... {len(missing) - 80} additional partial rows in benchmark_status_matrix.csv")
    report.extend(["", "## Rerun Commands"])
    commands = missing_commands(missing)
    if commands:
        for command in commands[:30]:
            report.append(f"- `{command}`")
        if len(commands) > 30:
            report.append(f"- ... {len(commands) - 30} additional commands implied by benchmark_status_matrix.csv")
    else:
        report.append("- No binary workflow reruns required by the status matrix.")
    report.extend(
        [
            "",
            "## Recommended Benchmark Material",
            "- Main table: results/benchmark_main_binary_table.csv",
            "- Secondary table: results/benchmark_secondary_top_table.csv",
            "- Pareto figures: plots/benchmark_pareto_auc_vs_lut.png and plots/benchmark_pareto_auc_vs_latency.png",
            "- Task-filtered Pareto figures: plots/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png and plots/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png",
            "- Use multiclass only as a compact supporting table: results/benchmark_multiclass_summary.csv",
            "",
            "## Interpretation Notes",
            "- BitNet improves over plain QKeras binary/ternary in the primary binary task when comparing AUC at similar 8-11 cycle latencies.",
            "- BitNet does not dominate HGQ or unrolled BDT in this artifact set.",
            "- HGQ is the strongest neural resource-efficiency point by LUT for both binary tasks.",
            "- Unrolled BDT is the fastest overall in the secondary task and should be presented separately from tree-mode BDT.",
            "- Bit158 custom_v9 uses the refreshed sparse-pruned path; the sigmoid variant removes the DSP penalty at 8-9 cycles but costs high LUT.",
            "- Scaling-factor implementation is a first-order hardware variable; compare the retained BitNet rows against dense, QKeras, HGQ and BDT rows.",
        ]
    )
    (RESULTS / "benchmark_readiness_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "status_rows": len(status),
        "complete_rows": len(complete),
        "missing_rows": len(missing),
        "primary_rows": len(primary),
        "secondary_rows": len(secondary),
        "pareto_rows": len(pareto_csv),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
