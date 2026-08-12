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

PRIMARY_MODELS = {
    "Dense MLP",
    "QKeras fixed b7",
    "HGQ",
    "QKeras binary",
    "QKeras ternary",
    "BitNet binary",
    "Bit158 sparse ternary",
    "XGBoost BDT d4x100 (unrolled)",
}

LOWBIT_MODELS = {
    "QKeras binary",
    "QKeras ternary",
    "BitNet binary",
    "Bit158 sparse ternary",
}

TABLE_FIELDS = [
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
        "XGBoost BDT d4x100 (unrolled)": "BDT unrolled",
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


def plot_lowbit(rows: list[dict[str, str]]) -> None:
    clean = [
        row
        for row in rows
        if row.get("task") == "binary_qg_vs_wzt"
        and row.get("model") in LOWBIT_MODELS
        and f(row, "auc_mean") is not None
        and f(row, "lut_mean") is not None
    ]
    labels = [f"{short_label(row['model'])}\n{row['architecture']}" for row in clean]
    x = range(len(clean))
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.bar([i - 0.18 for i in x], [float(row["auc_mean"]) for row in clean], width=0.36, label="AUC", color="#2f6f73")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.88, 0.93)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], [float(row["lut_mean"]) for row in clean], width=0.36, label="LUT", color="#9a5c2e", alpha=0.75)
    ax2.set_ylabel("LUT")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_title("Low-bit primary binary comparison")
    ax1.grid(True, axis="y", alpha=0.2)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "benchmark_lowbit_comparison.png", dpi=300)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    main_rows = read_csv(RESULTS / "benchmark_main_binary_table.csv")
    top_rows = read_csv(RESULTS / "benchmark_secondary_top_table.csv")
    multiclass_path = RESULTS / "benchmark_multiclass_summary.csv"
    multiclass_rows = read_csv(multiclass_path) if multiclass_path.exists() else []
    all_rows = main_rows + top_rows
    require_complete(main_rows, "primary")
    require_complete(top_rows, "secondary")

    seed_stats_fields = [
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
    ]
    seed_stats = [{key: row.get(key, "") for key in seed_stats_fields} for row in all_rows]
    write_csv(RESULTS / "benchmark_seed_statistics.csv", seed_stats, seed_stats_fields)
    write_csv(RESULTS / "benchmark_lowbit_comparison.csv", [row for row in main_rows if row.get("model") in LOWBIT_MODELS], TABLE_FIELDS)
    write_csv(RESULTS / "benchmark_pareto_candidates.csv", pareto(all_rows), TABLE_FIELDS)

    status_fields = [
        "task",
        "model",
        "architecture",
        "base_run_name",
        "seeds_metrics",
        "seeds_synth",
        "status",
    ]
    status_rows = [
        {key: row.get(key, "") for key in status_fields}
        for row in main_rows + top_rows + multiclass_rows
    ]
    write_csv(RESULTS / "benchmark_status_matrix.csv", status_rows, status_fields)

    plot_pareto(
        main_rows,
        "lut_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs LUT",
        "LUT",
        task="binary_qg_vs_wzt",
    )
    plot_pareto(
        main_rows,
        "latency_cycles_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs latency",
        "Latency cycles",
        task="binary_qg_vs_wzt",
    )
    plot_pareto(
        top_rows,
        "lut_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_top.png",
        "q/g vs top: AUC vs LUT",
        "LUT",
        task="binary_topqg",
    )
    plot_pareto(
        top_rows,
        "latency_cycles_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_top.png",
        "q/g vs top: AUC vs latency",
        "Latency cycles",
        task="binary_topqg",
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
    plot_lowbit(main_rows)

    check = {
        "status": "ok",
        "primary_rows": len(main_rows),
        "secondary_rows": len(top_rows),
        "multiclass_rows": len(multiclass_rows),
        "pareto_rows": len(pareto(all_rows)),
        "note": "Fixed-FPR signal efficiency is read from shipped benchmark tables; raw prediction scores are not included.",
    }
    (RESULTS / "public_reproduction_check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Benchmark Readiness Report",
        "",
        "Scope: public benchmark artifact generated from committed summary tables.",
        "Hardware numbers are VU13P, 5 ns HLS C-synthesis estimates, not place-and-route.",
        "",
        "## Regenerated Files",
        "- `results/benchmark_seed_statistics.csv`",
        "- `results/benchmark_lowbit_comparison.csv`",
        "- `results/benchmark_pareto_candidates.csv`",
        "- `results/benchmark_status_matrix.csv`",
        "- `plots/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png`",
        "- `plots/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png`",
        "- `plots/benchmark_pareto_auc_vs_lut_qg_vs_top.png`",
        "- `plots/benchmark_pareto_auc_vs_latency_qg_vs_top.png`",
        "- `plots/benchmark_pareto_auc_vs_lut_multiclass.png`",
        "- `plots/benchmark_pareto_auc_vs_latency_multiclass.png`",
        "- `plots/benchmark_lowbit_comparison.png`",
        "",
        "## Coverage",
        f"- Primary q/g vs W/Z/top rows: {len(main_rows)}",
        f"- Secondary q/g vs top rows: {len(top_rows)}",
        f"- Multiclass rows: {len(multiclass_rows)}; rows remain marked partial when seed metrics or synthesis estimates are incomplete.",
        "",
        "## Notes",
        "- Fixed-FPR signal efficiency is preserved in committed tables because raw per-event prediction scores are not included.",
        "- The public artifact does not include trained checkpoints, ONNX exports, generated HLS projects, or raw C-synthesis reports.",
    ]
    (RESULTS / "benchmark_readiness_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(check, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
