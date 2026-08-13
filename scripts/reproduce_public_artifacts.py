#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
OBSOLETE_RESULT_FILES = (
    "benchmark_seed_statistics.csv",
    "benchmark_lowbit_comparison.csv",
    "benchmark_pareto_candidates.csv",
    "benchmark_status_matrix.csv",
    "benchmark_readiness_report.md",
    "public_reproduction_check.json",
)

PRIMARY_MODELS = {
    "Dense MLP",
    "QKeras fixed b7",
    "HGQ",
    "QKeras binary",
    "QKeras ternary",
    "BitNet binary",
    "Bit158 sparse ternary",
    "XGBoost BDT (unrolled)",
}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def require_complete(rows: list[dict[str, str]], label: str) -> None:
    missing = []
    for row in rows:
        if row.get("model") not in PRIMARY_MODELS:
            continue
        for key in ("accuracy_mean", "auc_mean", "latency_cycles_mean", "lut_mean", "dsp_mean"):
            if row.get(key, "") == "":
                missing.append(f"{label}: {row.get('model')} {row.get('architecture')} missing {key}")
    if missing:
        raise RuntimeError("\n".join(missing[:20]))


def pareto(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = []
    clean = [
        row
        for row in rows
        if row.get("status") == "complete"
        and f(row, "auc_mean") is not None
        and f(row, "latency_cycles_mean") is not None
        and f(row, "lut_mean") is not None
    ]
    for row in clean:
        dominated = False
        auc = f(row, "auc_mean")
        lat = f(row, "latency_cycles_mean")
        lut = f(row, "lut_mean")
        assert auc is not None and lat is not None and lut is not None
        for other in clean:
            if other is row:
                continue
            other_auc = f(other, "auc_mean")
            other_lat = f(other, "latency_cycles_mean")
            other_lut = f(other, "lut_mean")
            assert other_auc is not None and other_lat is not None and other_lut is not None
            if (
                other_auc >= auc
                and other_lat <= lat
                and other_lut <= lut
                and (other_auc > auc or other_lat < lat or other_lut < lut)
            ):
                dominated = True
                break
        if not dominated:
            candidates.append(row)
    return sorted(candidates, key=lambda row: (-float(row["auc_mean"]), float(row["latency_cycles_mean"])))


def short_label(model: str) -> str:
    return {
        "Dense MLP": "Dense",
        "QKeras fixed b7": "QK fixed b7",
        "QKeras binary": "QK binary",
        "QKeras ternary": "QK ternary",
        "Bit158 sparse ternary": "Bit158b",
        "XGBoost BDT (unrolled)": "BDT unrolled",
    }.get(model, model.replace(" binary", ""))


def plot_pareto(
    rows: list[dict[str, str]],
    x_key: str,
    output: Path,
    title: str,
    xlabel: str,
    *,
    task: str,
    y_key: str = "auc_mean",
    ylabel: str = "ROC AUC",
) -> None:
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
    clean = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("model") in PRIMARY_MODELS
        and f(row, x_key) is not None
        and f(row, y_key) is not None
    ]
    palette = {"64-32-32": "#1f77b4", "128-32": "#d62728", "100 trees depth 4": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for arch, color in palette.items():
        subset = [row for row in clean if row["architecture"] == arch]
        if not subset:
            continue
        ax.scatter(
            [float(row[x_key]) for row in subset],
            [float(row[y_key]) for row in subset],
            s=70,
            alpha=0.92,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            label=arch,
        )

    for index, row in enumerate(clean):
        x = float(row[x_key])
        y = float(row[y_key])
        dx = 0.02 * (max(float(r[x_key]) for r in clean) - min(float(r[x_key]) for r in clean))
        dy = 0.0010 if index % 2 else 0.0003
        label_offsets = {
            ("latency_cycles_mean", "Dense MLP", "64-32-32"): (0.45, 0.0002),
            ("latency_cycles_mean", "Dense MLP", "128-32"): (0.45, -0.0020),
            ("latency_cycles_mean", "QKeras fixed b7", "64-32-32"): (0.35, 0.0005),
            ("latency_cycles_mean", "QKeras fixed b7", "128-32"): (-0.65, 0.0008),
            ("latency_cycles_mean", "HGQ", "64-32-32"): (0.35, -0.0019),
            ("latency_cycles_mean", "HGQ", "128-32"): (0.35, 0.0010),
            ("latency_cycles_mean", "Bit158 sparse ternary", "64-32-32"): (0.35, -0.0017),
            ("latency_cycles_mean", "Bit158 sparse ternary", "128-32"): (0.35, 0.0010),
            ("latency_cycles_mean", "BitNet binary", "128-32"): (0.35, -0.0020),
            ("lut_mean", "HGQ", "64-32-32"): (3500.0, -0.0020),
            ("lut_mean", "HGQ", "128-32"): (3500.0, 0.0010),
            ("lut_mean", "QKeras fixed b7", "64-32-32"): (4500.0, 0.0008),
            ("lut_mean", "QKeras fixed b7", "128-32"): (4500.0, -0.0030),
            ("lut_mean", "Dense MLP", "64-32-32"): (4500.0, 0.0008),
            ("lut_mean", "Dense MLP", "128-32"): (4500.0, 0.0008),
            ("lut_mean", "Bit158 sparse ternary", "64-32-32"): (4500.0, -0.0015),
            ("lut_mean", "Bit158 sparse ternary", "128-32"): (4500.0, 0.0010),
        }
        dx, dy = label_offsets.get((x_key, row["model"], row["architecture"]), (dx, dy))
        if x_key == "latency_cycles_mean" and row["model"] == "QKeras fixed b7" and row["architecture"] == "128-32":
            dx = -1.0
        if x_key == "lut_mean" and row["model"] == "QKeras fixed b7" and row["architecture"] == "128-32":
            dx = -8000
        ax.annotate(
            short_label(row["model"]),
            (x, y),
            xytext=(x + dx, y + dy),
            fontsize=8,
            ha="left" if dx >= 0 else "right",
            va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.35, color="0.5", shrinkA=0, shrinkB=0),
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_key == "latency_cycles_mean":
        ax.set_xlim(0, 30)
    if x_key == "lut_mean":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value / 1000.0))}k"))
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best", title="Architecture")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    for name in OBSOLETE_RESULT_FILES:
        (RESULTS / name).unlink(missing_ok=True)

    main_rows = read_csv(RESULTS / "benchmark_main_binary_table.csv")
    top_rows = read_csv(RESULTS / "benchmark_secondary_top_table.csv")
    multiclass_path = RESULTS / "benchmark_multiclass_summary.csv"
    multiclass_rows = read_csv(multiclass_path) if multiclass_path.exists() else []
    all_rows = main_rows + top_rows
    require_complete(main_rows, "primary")
    require_complete(top_rows, "secondary")

    plot_pareto(
        main_rows,
        "lut_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs LUT",
        "LUT",
        task="qg_vs_wzt",
    )
    plot_pareto(
        main_rows,
        "latency_cycles_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs latency",
        "Latency cycles",
        task="qg_vs_wzt",
    )
    plot_pareto(
        top_rows,
        "lut_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_top.png",
        "q/g vs top: AUC vs LUT",
        "LUT",
        task="qg_vs_top",
    )
    plot_pareto(
        top_rows,
        "latency_cycles_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_top.png",
        "q/g vs top: AUC vs latency",
        "Latency cycles",
        task="qg_vs_top",
    )
    if multiclass_rows:
        plot_pareto(
            multiclass_rows,
            "lut_mean",
            PLOTS / "benchmark_pareto_auc_vs_lut_multiclass.png",
            "Multiclass: macro AUC vs LUT",
            "LUT",
            task="multiclass",
            y_key="macro_auc_mean",
            ylabel="Macro ROC AUC",
        )
        plot_pareto(
            multiclass_rows,
            "latency_cycles_mean",
            PLOTS / "benchmark_pareto_auc_vs_latency_multiclass.png",
            "Multiclass: macro AUC vs latency",
            "Latency cycles",
            task="multiclass",
            y_key="macro_auc_mean",
            ylabel="Macro ROC AUC",
        )
    check = {
        "status": "ok",
        "primary_rows": len(main_rows),
        "secondary_rows": len(top_rows),
        "multiclass_rows": len(multiclass_rows),
        "pareto_rows": len(pareto(all_rows)),
        "note": "Fixed-FPR signal efficiency is read from shipped benchmark tables; raw prediction scores are not included.",
    }
    print(json.dumps(check, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
