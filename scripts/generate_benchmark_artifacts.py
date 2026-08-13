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
from matplotlib.ticker import FuncFormatter, MultipleLocator
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
OBSOLETE_RESULT_FILES = (
    "benchmark_seed_statistics.csv",
    "benchmark_lowbit_comparison.csv",
    "benchmark_pareto_candidates.csv",
    "benchmark_status_matrix.csv",
    "benchmark_readiness_report.md",
    "public_reproduction_check.json",
)

SEEDS = (42, 43, 44)
FPR_TARGET = 0.01

TASKS = {
    "binary_qg_vs_wzt": {
        "label": "q/g vs W/Z/top",
        "public_task": "qg_vs_wzt",
        "prefix": "binary_",
        "cache": "openml_42468_nall_splitseed42_classbinary_qg_vs_wzt_train0p64_val0p16_test0p2.npz",
    },
    "binary_topqg": {
        "label": "q/g vs top",
        "public_task": "qg_vs_top",
        "prefix": "binary_topqg_",
        "cache": "openml_42468_nall_splitseed42_classbinary_top_vs_qg_train0p64_val0p16_test0p2.npz",
    },
    "multiclass": {
        "label": "5-class q/g/W/Z/top",
        "prefix": "",
        "cache": "openml_42468_nall_splitseed42_classmulticlass_train0p64_val0p16_test0p2.npz",
    },
}

PUBLIC_TASK_LABELS = {
    "qg_vs_wzt": "q/g vs W/Z/top",
    "qg_vs_top": "q/g vs top",
    "multiclass": "5-class q/g/W/Z/top",
}

PUBLIC_BINARY_SUFFIXES = {
    "binary_qg_vs_wzt": {
        "mlp_baseline_64_32_32": "dense_baseline_64_32_32",
        "mlp_topo_128_32": "dense_128_32",
        "qkeras_mlp_64_32_32_b7": "qkeras_b7_64_32_32",
        "qkeras_topo_128_32_b7": "qkeras_b7_128_32",
        "hgq_mlp_64_32_32": "hgq_64_32_32",
        "hgq_topo_128_32": "hgq_128_32",
        "qkeras_mlp_binary_64_32_32": "qkeras_binary_64_32_32",
        "qkeras_topo_binary_128_32": "qkeras_binary_128_32",
        "qkeras_mlp_ternary_64_32_32": "qkeras_ternary_64_32_32",
        "qkeras_topo_ternary_128_32": "qkeras_ternary_128_32",
        "bitnet_sigmoid_f7_fixed": "bitnet_64_32_32",
        "bitnet_topo_sigmoid_f7_fixed": "bitnet_128_32",
        "bit158_sigmoid_f7_fixed": "bit158_64_32_32",
        "bit158_topo_sigmoid_f7_fixed": "bit158_128_32",
        "xgboost_bdt_d4_100": "xgboost_bdt_d4_100",
    },
    "binary_topqg": {
        "mlp_baseline_64_32_32": "topqg_dense_baseline_64_32_32",
        "mlp_topo_128_32": "topqg_dense_baseline_128_32",
        "qkeras_mlp_64_32_32_b7": "topqg_qkeras_b7_64_32_32",
        "qkeras_topo_128_32_b7": "topqg_qkeras_b7_128_32",
        "hgq_mlp_64_32_32": "topqg_hgq_64_32_32",
        "hgq_topo_128_32": "topqg_hgq_128_32",
        "qkeras_mlp_binary_64_32_32": "topqg_qkeras_binary_64_32_32",
        "qkeras_topo_binary_128_32": "topqg_qkeras_binary_128_32",
        "qkeras_mlp_ternary_64_32_32": "topqg_qkeras_ternary_64_32_32",
        "qkeras_topo_ternary_128_32": "topqg_qkeras_ternary_128_32",
        "bitnet_sigmoid_f7_fixed": "topqg_bitnet_64_32_32",
        "bitnet_topo_sigmoid_f7_fixed": "topqg_bitnet_128_32",
        "bit158_sigmoid_f7_fixed": "topqg_bit158_64_32_32",
        "bit158_topo_sigmoid_f7_fixed": "topqg_bit158_128_32",
        "xgboost_bdt_d4_100": "topqg_xgboost_bdt_d4_100",
    },
}

PUBLIC_MULTICLASS_BASES = {
    "mlp_baseline": "multiclass_dense_baseline_64_32_32",
    "mlp_topo": "multiclass_dense_baseline_128_32",
    "qkeras_mlp_b7": "multiclass_qkeras_b7_64_32_32",
    "qkeras_mlp_binary": "multiclass_qkeras_binary_64_32_32",
    "qkeras_mlp_ternary": "multiclass_qkeras_ternary_64_32_32",
    "hgq_mlp": "multiclass_hgq_64_32_32",
    "hgq_mlp_topo": "multiclass_hgq_128_32",
    "bitnet_mlp_f7_fixed": "multiclass_bitnet_64_32_32",
    "bitnet_topo_f7_fixed": "multiclass_bitnet_128_32",
    "bit158_mlp_f7_fixed": "multiclass_bit158_64_32_32",
    "bit158_topo_f7_fixed": "multiclass_bit158_128_32",
    "xgboost_bdt": "multiclass_xgboost_bdt_d4_100",
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
        "model_family": "XGBoost BDT",
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
    ("XGBoost BDT", "100 trees depth 4", "xgboost_bdt", "xgboost", []),
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


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


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


def public_task_for(task: str) -> str:
    return TASKS.get(task, {}).get("public_task", task)


def task_label_for(task: str) -> str:
    if task in TASKS:
        return TASKS[task]["label"]
    return PUBLIC_TASK_LABELS.get(task, task)


def public_binary_base_for(task: str, suffix: str) -> str:
    return PUBLIC_BINARY_SUFFIXES.get(task, {}).get(suffix, base_for(task, suffix))


def public_multiclass_base_for(base: str) -> str:
    return PUBLIC_MULTICLASS_BASES.get(base, base)


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
    public_base = public_binary_base_for(task, spec["suffix"])
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
        "task": public_task_for(task),
        "task_label": TASKS[task]["label"],
        "model_key": spec["key"],
        "model": spec["model_family"],
        "architecture": spec["architecture"],
        "base_run_name": public_base,
        "raw_base_run_name": base,
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
            row = aggregate_task_model(task, spec, include_synth_variant="conifer_unrolled")
            row["model"] = "XGBoost BDT (unrolled)"
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
        public_base = public_multiclass_base_for(base)
        metrics = []
        for seed in SEEDS:
            path = RESULTS / f"{run_for(base, seed)}{metric_suffix(backend)}"
            if not path.exists() and seed == 42:
                path = RESULTS / f"{base}{metric_suffix(backend)}"
            if path.exists():
                metrics.append(read_json(path))
        synths = []
        if variants:
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
                "base_run_name": public_base,
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
        "XGBoost BDT (unrolled)": "BDT unrolled",
        "XGBoost BDT d4x100 (unrolled)": "BDT unrolled",
        "XGBoost BDT d4x100 (tree)": "BDT tree",
    }
    return labels.get(model, model)


ABSTRACT_PARETO_MODELS = {
    "Dense MLP",
    "QKeras fixed b7",
    "HGQ",
    "QKeras binary",
    "QKeras ternary",
    "BitNet binary",
    "Bit158 sparse ternary",
    "XGBoost BDT (unrolled)",
}


def legend_model_label(model: str) -> str:
    labels = {
        "Dense MLP": "Dense",
        "QKeras fixed b7": "QKeras (7-bit)",
        "HGQ": "HGQ",
        "QKeras binary": "QKeras Binary",
        "QKeras ternary": "QKeras Ternary",
        "BitNet binary": "BitNet binary",
        "Bit158 sparse ternary": "BitNet-1.58",
        "XGBoost BDT (unrolled)": "BDT (unrolled)",
        "XGBoost BDT d4x100 (unrolled)": "BDT (unrolled)",
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
        "qg_vs_wzt": "o",
        "qg_vs_top": "s",
        "binary_qg_vs_wzt": "o",
        "binary_topqg": "s",
        "multiclass": "^",
    }
    task_palette = {
        "qg_vs_wzt": "#1f77b4",
        "qg_vs_top": "#d62728",
        "binary_qg_vs_wzt": "#1f77b4",
        "binary_topqg": "#d62728",
        "multiclass": "#2ca02c",
    }
    arch_palette = {
        "64-32-32": "#1f77b4",
        "128-32": "#d62728",
        "100 trees depth 4": "#2ca02c",
    }
    tasks = (task_filter,) if task_filter else ("qg_vs_wzt", "qg_vs_top")
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
                label=task_label_for(task),
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


def plot_abstract_style_pareto(
    rows: list[dict],
    x_key: str,
    y_key: str,
    path: Path,
    title: str,
    xlabel: str,
    *,
    ylabel: str = "ROC AUC",
    task_filter: str,
    model_filter: set[str] = ABSTRACT_PARETO_MODELS,
    xerr_key: str,
    yerr_key: str,
    x_formatter: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    model_legend_loc: str = "lower right",
) -> None:
    source_rows = rows
    if x_key == "latency_ns_mean":
        source_rows = [
            {
                **row,
                "latency_ns_mean": 5.0 * float(row["latency_cycles_mean"]),
                "latency_ns_std": 5.0 * float(row.get("latency_cycles_std") or 0.0),
            }
            for row in rows
            if row.get("latency_cycles_mean") not in {None, ""}
        ]
    clean = [
        row
        for row in source_rows
        if row.get("task") == task_filter
        and row.get("model") in model_filter
        and row.get(x_key) not in (None, "")
        and row.get(y_key) not in (None, "")
    ]
    model_palette = {
        "Dense MLP": "#0057B8",
        "QKeras fixed b7": "#E69F00",
        "HGQ": "#009E73",
        "QKeras binary": "#7A4EAB",
        "QKeras ternary": "#D55E00",
        "BitNet binary": "#56B4E9",
        "Bit158 sparse ternary": "#CC79A7",
        "XGBoost BDT (unrolled)": "#4D4D4D",
    }
    arch_markers = {
        "64-32-32": "o",
        "128-32": "s",
        "100 trees depth 4": "^",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for row in clean:
        x_value = finite_float(row[x_key])
        y_value = finite_float(row[y_key])
        if x_value is None or y_value is None:
            continue
        color = model_palette.get(row["model"], "#7f7f7f")
        marker = arch_markers.get(row["architecture"], "o")
        ax.errorbar(
            [x_value],
            [y_value],
            xerr=[finite_float(row.get(xerr_key)) or 0.0],
            yerr=[finite_float(row.get(yerr_key)) or 0.0],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            alpha=0.58,
            zorder=1,
        )
        ax.scatter(
            [x_value],
            [y_value],
            marker=marker,
            s=74,
            color=color,
            edgecolors="#333333",
            linewidths=0.45,
            alpha=0.92,
            zorder=2,
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if x_formatter == "k":
        if xlim is None:
            ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MultipleLocator(50000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value / 1000.0))}"))
    ax.grid(True, alpha=0.18, linewidth=0.7)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7.5,
            markerfacecolor=color,
            markeredgecolor="#333333",
            markeredgewidth=0.45,
            label=legend_model_label(model),
        )
        for model, color in model_palette.items()
        if any(row["model"] == model for row in clean)
    ]
    arch_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markersize=6,
            markerfacecolor="#555555",
            markeredgecolor="#333333",
            markeredgewidth=0.45,
            label="100 trees max depth 4" if arch == "100 trees depth 4" else arch,
        )
        for arch, marker in arch_markers.items()
        if any(row["architecture"] == arch for row in clean)
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Model family",
        loc=model_legend_loc,
        frameon=True,
        framealpha=0.78,
        fontsize=9.4,
        title_fontsize=10.2,
        borderpad=0.45,
        labelspacing=0.28,
        handletextpad=0.48,
    )
    ax.add_artist(model_legend)
    arch_legend = ax.legend(
        handles=arch_handles,
        title="Architecture",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=max(1, len(arch_handles)),
        frameon=True,
        framealpha=0.78,
        fontsize=7.6,
        title_fontsize=8.4,
        borderpad=0.30,
        columnspacing=0.75,
        handletextpad=0.30,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", bbox_extra_artists=(arch_legend,))
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


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    for name in OBSOLETE_RESULT_FILES:
        (RESULTS / name).unlink(missing_ok=True)

    if not any(RESULTS.glob("*_pytorch.json")) and (RESULTS / "benchmark_main_binary_table.csv").exists():
        fallback_path = ROOT / "scripts" / "reproduce_public_artifacts.py"
        spec = importlib.util.spec_from_file_location("reproduce_public_artifacts", fallback_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load public artifact fallback from {fallback_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main()

    status = build_status_matrix()
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

    pareto_rows = pareto(all_raw, "auc_mean", ("lut_mean", "latency_cycles_mean"))
    pareto_csv = format_table(pareto_rows)

    plot_abstract_style_pareto(
        primary_raw,
        "lut_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs LUT",
        r"LUT usage [$10^3$]",
        task_filter="qg_vs_wzt",
        xerr_key="lut_std",
        yerr_key="auc_std",
        x_formatter="k",
        ylim=(0.88, 0.935),
        model_legend_loc="lower right",
    )
    plot_abstract_style_pareto(
        primary_raw,
        "latency_ns_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs latency",
        "Latency [ns]",
        task_filter="qg_vs_wzt",
        xerr_key="latency_ns_std",
        yerr_key="auc_std",
        xlim=(0, 120),
        ylim=(0.88, 0.935),
        model_legend_loc="upper right",
    )
    plot_abstract_style_pareto(
        secondary_raw,
        "lut_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_top.png",
        "q/g vs top: AUC vs LUT",
        r"LUT usage [$10^3$]",
        task_filter="qg_vs_top",
        xerr_key="lut_std",
        yerr_key="auc_std",
        x_formatter="k",
        model_legend_loc="lower right",
    )
    plot_abstract_style_pareto(
        secondary_raw,
        "latency_ns_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_top.png",
        "q/g vs top: AUC vs latency",
        "Latency [ns]",
        task_filter="qg_vs_top",
        xerr_key="latency_ns_std",
        yerr_key="auc_std",
        xlim=(0, 120),
        model_legend_loc="lower left",
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
    plot_abstract_style_pareto(
        multiclass_plot_rows,
        "lut_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_multiclass.png",
        "Multiclass: macro AUC vs LUT",
        r"LUT usage [$10^3$]",
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
        xerr_key="lut_std",
        yerr_key="macro_auc_std",
        x_formatter="k",
        model_legend_loc="lower right",
    )
    plot_abstract_style_pareto(
        multiclass_plot_rows,
        "latency_ns_mean",
        "auc_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_multiclass.png",
        "Multiclass: macro AUC vs latency",
        "Latency [ns]",
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
        xerr_key="latency_ns_std",
        yerr_key="macro_auc_std",
        xlim=(0, 120),
        model_legend_loc="upper right",
    )
    complete = [row for row in status if row["complete"]]
    missing = [row for row in status if not row["complete"]]

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
