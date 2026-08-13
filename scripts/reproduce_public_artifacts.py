#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


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
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


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


def legend_model_label(model: str) -> str:
    return {
        "Dense MLP": "Dense",
        "QKeras fixed b7": "QKeras (7-bit)",
        "HGQ": "HGQ",
        "QKeras binary": "QKeras Binary",
        "QKeras ternary": "QKeras Ternary",
        "BitNet binary": "BitNet binary",
        "Bit158 sparse ternary": "BitNet-1.58",
        "XGBoost BDT (unrolled)": "BDT (unrolled)",
    }.get(model, model)


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
    xerr_key: str = "",
    yerr_key: str = "",
    x_formatter: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    model_legend_loc: str = "lower right",
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
    source_rows = rows
    if x_key == "latency_ns_mean":
        source_rows = [
            {
                **row,
                "latency_ns_mean": 5.0 * float(row["latency_cycles_mean"]),
                "latency_ns_std": 5.0 * float(row.get("latency_cycles_std") or 0.0),
            }
            for row in rows
            if f(row, "latency_cycles_mean") is not None
        ]
    clean = [
        row
        for row in source_rows
        if row.get("task") == task
        and row.get("model") in PRIMARY_MODELS
        and f(row, x_key) is not None
        and f(row, y_key) is not None
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
        x_value = f(row, x_key)
        y_value = f(row, y_key)
        if x_value is None or y_value is None:
            continue
        color = model_palette.get(row["model"], "#7f7f7f")
        marker = arch_markers.get(row["architecture"], "o")
        ax.errorbar(
            [x_value],
            [y_value],
            xerr=[f(row, xerr_key) or 0.0] if xerr_key else None,
            yerr=[f(row, yerr_key) or 0.0] if yerr_key else None,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", bbox_extra_artists=(arch_legend,))
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
        r"LUT usage [$10^3$]",
        task="qg_vs_wzt",
        xerr_key="lut_std",
        yerr_key="auc_std",
        x_formatter="k",
        ylim=(0.88, 0.935),
        model_legend_loc="lower right",
    )
    plot_pareto(
        main_rows,
        "latency_ns_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_wzt.png",
        "q/g vs W/Z/top: AUC vs latency",
        "Latency [ns]",
        task="qg_vs_wzt",
        xerr_key="latency_ns_std",
        yerr_key="auc_std",
        xlim=(0, 120),
        ylim=(0.88, 0.935),
        model_legend_loc="upper right",
    )
    plot_pareto(
        top_rows,
        "lut_mean",
        PLOTS / "benchmark_pareto_auc_vs_lut_qg_vs_top.png",
        "q/g vs top: AUC vs LUT",
        r"LUT usage [$10^3$]",
        task="qg_vs_top",
        xerr_key="lut_std",
        yerr_key="auc_std",
        x_formatter="k",
        model_legend_loc="lower right",
    )
    plot_pareto(
        top_rows,
        "latency_ns_mean",
        PLOTS / "benchmark_pareto_auc_vs_latency_qg_vs_top.png",
        "q/g vs top: AUC vs latency",
        "Latency [ns]",
        task="qg_vs_top",
        xerr_key="latency_ns_std",
        yerr_key="auc_std",
        xlim=(0, 120),
        model_legend_loc="lower left",
    )
    if multiclass_rows:
        plot_pareto(
            multiclass_rows,
            "lut_mean",
            PLOTS / "benchmark_pareto_auc_vs_lut_multiclass.png",
            "Multiclass: macro AUC vs LUT",
            r"LUT usage [$10^3$]",
            task="multiclass",
            y_key="macro_auc_mean",
            ylabel="Macro ROC AUC",
            xerr_key="lut_std",
            yerr_key="macro_auc_std",
            x_formatter="k",
            model_legend_loc="lower right",
        )
        plot_pareto(
            multiclass_rows,
            "latency_ns_mean",
            PLOTS / "benchmark_pareto_auc_vs_latency_multiclass.png",
            "Multiclass: macro AUC vs latency",
            "Latency [ns]",
            task="multiclass",
            y_key="macro_auc_mean",
            ylabel="Macro ROC AUC",
            xerr_key="latency_ns_std",
            yerr_key="macro_auc_std",
            xlim=(0, 120),
            model_legend_loc="upper right",
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
