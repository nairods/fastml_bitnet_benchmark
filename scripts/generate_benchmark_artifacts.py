#!/usr/bin/env python3
"""Generate the public benchmark tables and plots from committed run records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "benchmark.json"
SEEDS = {42, 43, 44}

BINARY_FIELDS = [
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
    "selected_epoch_mean",
    "selected_epoch_std",
    "max_epochs",
    "synth_variant",
    "latency_cycles_mean",
    "latency_cycles_std",
    "ii_cycles_mean",
    "lut_mean",
    "lut_std",
    "ff_mean",
    "ff_std",
    "dsp_mean",
    "dsp_std",
    "bram18_mean",
    "bram18_std",
    "seeds_metrics",
    "seeds_synth",
    "status",
]

MULTICLASS_FIELDS = [
    "task",
    "model",
    "architecture",
    "base_run_name",
    "accuracy_mean",
    "accuracy_std",
    "macro_auc_mean",
    "macro_auc_std",
    "selected_epoch_mean",
    "selected_epoch_std",
    "max_epochs",
    "synth_variant",
    "latency_cycles_mean",
    "lut_mean",
    "ff_mean",
    "ff_std",
    "dsp_mean",
    "dsp_std",
    "bram18_mean",
    "bram18_std",
    "seeds_metrics",
    "seeds_synth",
    "status",
]

TABLE_FILES = (
    "benchmark_main_binary_table.csv",
    "benchmark_secondary_top_table.csv",
    "benchmark_multiclass_summary.csv",
)

PARETO_FILES = (
    "benchmark_pareto_auc_vs_lut_qg_vs_wzt.png",
    "benchmark_pareto_auc_vs_latency_qg_vs_wzt.png",
    "benchmark_pareto_auc_vs_lut_qg_vs_top.png",
    "benchmark_pareto_auc_vs_latency_qg_vs_top.png",
    "benchmark_pareto_auc_vs_lut_multiclass.png",
    "benchmark_pareto_auc_vs_latency_multiclass.png",
)

MODEL_PALETTE = {
    "Dense MLP": "#0057B8",
    "QKeras fixed b7": "#E69F00",
    "HGQ": "#009E73",
    "QKeras binary": "#7A4EAB",
    "QKeras ternary": "#D55E00",
    "BitNet binary": "#56B4E9",
    "BitNet-1.58": "#CC79A7",
    "XGBoost BDT (unrolled)": "#4D4D4D",
    "XGBoost BDT": "#4D4D4D",
}

MODEL_LABELS = {
    "QKeras fixed b7": "QKeras (7-bit)",
    "BitNet-1.58": "BitNet-1.58",
    "XGBoost BDT (unrolled)": "BDT (unrolled)",
    "XGBoost BDT": "BDT",
}

ARCH_MARKERS = {
    "64-32-32": "o",
    "128-32": "s",
    "100 trees depth 4": "^",
}


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def stdev(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return statistics.stdev(clean) if len(clean) > 1 else 0.0


def fmt(value: float | int | None, digits: int = 5) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def synthesis_runs(model: dict) -> list[dict]:
    return [run["synthesis"] for run in model["runs"] if run.get("synthesis")]


def training_values(runs: list[dict], key: str) -> list[float | int | None]:
    return [run.get("training_selection", {}).get(key) for run in runs]


def implementation_label(syntheses: list[dict]) -> str:
    names = list(dict.fromkeys(row["implementation"] for row in syntheses))
    return ",".join(names)


def aggregate_binary(task: str, model: dict) -> dict:
    runs = model["runs"]
    syntheses = synthesis_runs(model)
    return {
        "task": task,
        "model": model["model"],
        "architecture": model["architecture"],
        "base_run_name": model["base_run_name"],
        "accuracy_mean": fmt(mean([run["accuracy"] for run in runs])),
        "accuracy_std": fmt(stdev([run["accuracy"] for run in runs])),
        "auc_mean": fmt(mean([run["auc"] for run in runs])),
        "auc_std": fmt(stdev([run["auc"] for run in runs])),
        "signal_eff_at_1pct_fpr_mean": fmt(mean([run["signal_eff_at_1pct_fpr"] for run in runs])),
        "signal_eff_at_1pct_fpr_std": fmt(stdev([run["signal_eff_at_1pct_fpr"] for run in runs])),
        "selected_epoch_mean": fmt(mean(training_values(runs, "selected_epoch")), 2),
        "selected_epoch_std": fmt(stdev(training_values(runs, "selected_epoch")), 2),
        "max_epochs": fmt(mean(training_values(runs, "max_epochs")), 0),
        "synth_variant": implementation_label(syntheses),
        "latency_cycles_mean": fmt(mean([row["latency_cycles"] for row in syntheses]), 2),
        "latency_cycles_std": fmt(stdev([row["latency_cycles"] for row in syntheses]), 2),
        "ii_cycles_mean": fmt(mean([row["ii_cycles"] for row in syntheses]), 2),
        "lut_mean": fmt(mean([row["lut"] for row in syntheses]), 1),
        "lut_std": fmt(stdev([row["lut"] for row in syntheses]), 1),
        "ff_mean": fmt(mean([row["ff"] for row in syntheses]), 1),
        "ff_std": fmt(stdev([row["ff"] for row in syntheses]), 1),
        "dsp_mean": fmt(mean([row["dsp"] for row in syntheses]), 2),
        "dsp_std": fmt(stdev([row["dsp"] for row in syntheses]), 2),
        "bram18_mean": fmt(mean([row["bram18"] for row in syntheses]), 2),
        "bram18_std": fmt(stdev([row["bram18"] for row in syntheses]), 2),
        "seeds_metrics": len(runs),
        "seeds_synth": len(syntheses),
        "status": "complete" if len(runs) == 3 and len(syntheses) == 3 else "partial",
    }


def aggregate_multiclass(model: dict) -> dict:
    runs = model["runs"]
    syntheses = synthesis_runs(model)
    return {
        "task": "multiclass",
        "model": model["model"],
        "architecture": model["architecture"],
        "base_run_name": model["base_run_name"],
        "accuracy_mean": fmt(mean([run["accuracy"] for run in runs])),
        "accuracy_std": fmt(stdev([run["accuracy"] for run in runs])),
        "macro_auc_mean": fmt(mean([run["macro_auc"] for run in runs])),
        "macro_auc_std": fmt(stdev([run["macro_auc"] for run in runs])),
        "selected_epoch_mean": fmt(mean(training_values(runs, "selected_epoch")), 2),
        "selected_epoch_std": fmt(stdev(training_values(runs, "selected_epoch")), 2),
        "max_epochs": fmt(mean(training_values(runs, "max_epochs")), 0),
        "synth_variant": implementation_label(syntheses),
        "latency_cycles_mean": fmt(mean([row["latency_cycles"] for row in syntheses]), 2),
        "lut_mean": fmt(mean([row["lut"] for row in syntheses]), 1),
        "ff_mean": fmt(mean([row["ff"] for row in syntheses]), 1),
        "ff_std": fmt(stdev([row["ff"] for row in syntheses]), 1),
        "dsp_mean": fmt(mean([row["dsp"] for row in syntheses]), 2),
        "dsp_std": fmt(stdev([row["dsp"] for row in syntheses]), 2),
        "bram18_mean": fmt(mean([row["bram18"] for row in syntheses]), 2),
        "bram18_std": fmt(stdev([row["bram18"] for row in syntheses]), 2),
        "seeds_metrics": len(runs),
        "seeds_synth": len(syntheses),
        "status": "complete" if len(runs) == 3 and len(syntheses) == 3 else "partial",
    }


def expected_models(protocol: dict, task: str) -> set[tuple[str, str, str]]:
    prefix = {"qg_vs_wzt": "", "qg_vs_top": "topqg_", "multiclass": "multiclass_"}[task]
    expected = set()
    for model in protocol["models"]:
        architectures = model["multiclass_architectures"] if task == "multiclass" else model["base_names"].keys()
        for architecture in architectures:
            definition = protocol["architectures"][architecture]
            architecture_label = (
                "-".join(str(width) for width in definition)
                if isinstance(definition, list)
                else f"{definition['n_estimators']} trees depth {definition['max_depth']}"
            )
            expected.add((model["display_name"], architecture_label, prefix + model["base_names"][architecture]))
    return expected


def validate_source(records: dict, protocol: dict, profile: str) -> None:
    if records.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark record schema")
    if records.get("training_profile") != profile:
        raise ValueError(
            f"Record profile {records.get('training_profile')!r} does not match {profile!r}"
        )
    if records.get("training_protocol") != protocol["training_profiles"][profile]:
        raise ValueError("Record training protocol does not match configs/benchmark.json")
    tasks = records.get("tasks", {})
    if set(tasks) != {"qg_vs_wzt", "qg_vs_top", "multiclass"}:
        raise ValueError("Expected primary, secondary, and multiclass task records")
    for task in ("qg_vs_wzt", "qg_vs_top"):
        models = tasks[task]["models"]
        expected_count = len(expected_models(protocol, task))
        if len(models) != expected_count:
            raise ValueError(f"{task}: expected {expected_count} models, found {len(models)}")
        names = [model["base_run_name"] for model in models]
        if len(names) != len(set(names)):
            raise ValueError(f"{task}: duplicate base_run_name")
        for model in models:
            seeds = {run["seed"] for run in model["runs"]}
            if seeds != SEEDS:
                raise ValueError(f"{task}/{model['base_run_name']}: expected seeds 42, 43, 44")
            if not synthesis_runs(model):
                raise ValueError(f"{task}/{model['base_run_name']}: no synthesis record")
    for task, values in tasks.items():
        actual = {(model["model"], model["architecture"], model["base_run_name"]) for model in values["models"]}
        if actual != expected_models(protocol, task):
            raise ValueError(f"{task}: records do not match configs/benchmark.json")
    implementation_names = set(protocol["hardware"]["implementations"])
    for values in tasks.values():
        for model in values["models"]:
            for synthesis in synthesis_runs(model):
                if synthesis["implementation"] not in implementation_names:
                    raise ValueError(f"Unknown implementation {synthesis['implementation']!r}")
    if len(records.get("training_curves", [])) != 7:
        raise ValueError("Expected seven primary-task training curves")


def numeric(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def plot_pareto(
    rows: list[dict],
    x_key: str,
    output: Path,
    title: str,
    xlabel: str,
    *,
    score_key: str,
    ylabel: str,
    xerr_key: str = "",
    yerr_key: str = "",
    x_formatter: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    model_legend_loc: str = "lower right",
) -> None:
    plt.rcParams.update({
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
    })
    clean = []
    for source in rows:
        row = dict(source)
        if x_key == "latency_ns_mean" and numeric(row, "latency_cycles_mean") is not None:
            row[x_key] = 5.0 * float(row["latency_cycles_mean"])
            row["latency_ns_std"] = 5.0 * float(row.get("latency_cycles_std") or 0.0)
        if numeric(row, x_key) is not None and numeric(row, score_key) is not None:
            clean.append(row)

    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    for row in clean:
        color = MODEL_PALETTE[row["model"]]
        marker = ARCH_MARKERS[row["architecture"]]
        x_value = numeric(row, x_key)
        y_value = numeric(row, score_key)
        axis.errorbar(
            [x_value],
            [y_value],
            xerr=[numeric(row, xerr_key) or 0.0] if xerr_key else None,
            yerr=[numeric(row, yerr_key) or 0.0] if yerr_key else None,
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            alpha=0.58,
            zorder=1,
        )
        axis.scatter(
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

    axis.set_title(title, pad=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if xlim:
        axis.set_xlim(*xlim)
    if ylim:
        axis.set_ylim(*ylim)
    if x_formatter == "k":
        if not xlim:
            axis.set_xlim(left=0)
        axis.xaxis.set_major_locator(MultipleLocator(50000))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{int(round(value / 1000.0))}"))
    axis.grid(True, alpha=0.18, linewidth=0.7)

    models = list(dict.fromkeys(row["model"] for row in clean))
    model_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None", markersize=7.5,
            markerfacecolor=MODEL_PALETTE[model], markeredgecolor="#333333",
            markeredgewidth=0.45, label=MODEL_LABELS.get(model, model),
        )
        for model in models
    ]
    architectures = list(dict.fromkeys(row["architecture"] for row in clean))
    arch_handles = [
        Line2D(
            [0], [0], marker=ARCH_MARKERS[architecture], linestyle="None", markersize=6,
            markerfacecolor="#555555", markeredgecolor="#333333", markeredgewidth=0.45,
            label="100 trees max depth 4" if architecture == "100 trees depth 4" else architecture,
        )
        for architecture in architectures
    ]
    model_legend = axis.legend(
        handles=model_handles, title="Model family", loc=model_legend_loc,
        frameon=True, framealpha=0.78, fontsize=9.4, title_fontsize=10.2,
        borderpad=0.45, labelspacing=0.28, handletextpad=0.48,
    )
    axis.add_artist(model_legend)
    arch_legend = axis.legend(
        handles=arch_handles, title="Architecture", loc="upper center",
        bbox_to_anchor=(0.5, -0.17), ncol=max(1, len(arch_handles)),
        frameon=True, framealpha=0.78, fontsize=7.6, title_fontsize=8.4,
        borderpad=0.30, columnspacing=0.75, handletextpad=0.30,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", bbox_extra_artists=(arch_legend,))
    plt.close(figure)


def plot_training_curve(curve: dict, output: Path) -> None:
    history = curve["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["validation_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.18)
    axes[1].plot(epochs, history["validation_accuracy"], color="#0057B8")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].grid(True, alpha=0.18)
    tick_spacing = 2 if len(history["train_loss"]) <= 20 else 20
    for axis in axes:
        axis.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        axis.set_xlim(1, len(history["train_loss"]))
    figure.suptitle(f"{MODEL_LABELS.get(curve['model'], curve['model'])}, {curve['architecture']}, seed {curve['seed']}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def score_ylim(rows: list[dict], score_key: str, yerr_key: str) -> tuple[float, float]:
    """Return a padded range containing scores and their error bars."""
    extents = []
    for row in rows:
        score = numeric(row, score_key)
        if score is None:
            continue
        error = numeric(row, yerr_key) or 0.0
        extents.extend((score - error, score + error))
    if not extents:
        raise ValueError(f"No finite values available for {score_key}")
    low, high = min(extents), max(extents)
    padding = max(0.08 * max(high - low, 0.01), 0.0015)
    return low - padding, high + padding


def generate(records: dict, output_root: Path, profile: str) -> dict:
    results_dir = output_root / "results" / profile
    plots_dir = output_root / "plots" / profile
    for directory, pattern in ((results_dir, "*.csv"), (plots_dir, "*.png")):
        if directory.exists():
            for stale in directory.glob(pattern):
                stale.unlink()
    primary = [aggregate_binary("qg_vs_wzt", model) for model in records["tasks"]["qg_vs_wzt"]["models"]]
    secondary = [aggregate_binary("qg_vs_top", model) for model in records["tasks"]["qg_vs_top"]["models"]]
    multiclass = [aggregate_multiclass(model) for model in records["tasks"]["multiclass"]["models"]]
    primary_ylim = score_ylim(primary, "auc_mean", "auc_std") if profile == "200-epochs" else (0.88, 0.935)
    secondary_ylim = score_ylim(secondary, "auc_mean", "auc_std") if profile == "200-epochs" else None
    multiclass_ylim = score_ylim(multiclass, "macro_auc_mean", "macro_auc_std") if profile == "200-epochs" else None

    write_csv(results_dir / TABLE_FILES[0], primary, BINARY_FIELDS)
    write_csv(results_dir / TABLE_FILES[1], secondary, BINARY_FIELDS)
    write_csv(results_dir / TABLE_FILES[2], multiclass, MULTICLASS_FIELDS)

    plot_pareto(primary, "lut_mean", plots_dir / PARETO_FILES[0], "q/g vs W/Z/top: AUC vs LUT", r"LUT usage [$10^3$]", score_key="auc_mean", ylabel="ROC AUC", xerr_key="lut_std", yerr_key="auc_std", x_formatter="k", ylim=primary_ylim, model_legend_loc="lower right")
    plot_pareto(primary, "latency_ns_mean", plots_dir / PARETO_FILES[1], "q/g vs W/Z/top: AUC vs latency", "Latency [ns]", score_key="auc_mean", ylabel="ROC AUC", xerr_key="latency_ns_std", yerr_key="auc_std", xlim=(0, 120), ylim=primary_ylim, model_legend_loc="upper right")
    plot_pareto(secondary, "lut_mean", plots_dir / PARETO_FILES[2], "q/g vs top: AUC vs LUT", r"LUT usage [$10^3$]", score_key="auc_mean", ylabel="ROC AUC", xerr_key="lut_std", yerr_key="auc_std", x_formatter="k", ylim=secondary_ylim, model_legend_loc="lower right")
    plot_pareto(secondary, "latency_ns_mean", plots_dir / PARETO_FILES[3], "q/g vs top: AUC vs latency", "Latency [ns]", score_key="auc_mean", ylabel="ROC AUC", xerr_key="latency_ns_std", yerr_key="auc_std", xlim=(0, 120), ylim=secondary_ylim, model_legend_loc="lower left")
    plot_pareto(multiclass, "lut_mean", plots_dir / PARETO_FILES[4], "Multiclass: macro AUC vs LUT", r"LUT usage [$10^3$]", score_key="macro_auc_mean", ylabel="Macro ROC AUC", yerr_key="macro_auc_std", x_formatter="k", ylim=multiclass_ylim, model_legend_loc="lower right")
    plot_pareto(multiclass, "latency_ns_mean", plots_dir / PARETO_FILES[5], "Multiclass: macro AUC vs latency", "Latency [ns]", score_key="macro_auc_mean", ylabel="Macro ROC AUC", xerr_key="latency_ns_std", yerr_key="macro_auc_std", xlim=(0, 120), ylim=multiclass_ylim, model_legend_loc="upper right")

    training_files = []
    for curve in records["training_curves"]:
        name = f"{curve['base_run_name']}__seed{curve['seed']}_training.png"
        plot_training_curve(curve, plots_dir / name)
        training_files.append(name)
    return {
        "primary_rows": len(primary),
        "secondary_rows": len(secondary),
        "multiclass_rows": len(multiclass),
        "training_plots": len(training_files),
    }


def compare_generated(generated_root: Path, profile: str) -> None:
    expected = [
        *(Path("results") / profile / name for name in TABLE_FILES),
        *(Path("plots") / profile / name for name in PARETO_FILES),
    ]
    expected.extend(
        Path("plots") / profile / path.name
        for path in sorted((generated_root / "plots" / profile).glob("*_training.png"))
    )
    expected_set = set(expected)
    actual_set = {
        path.relative_to(ROOT)
        for directory, pattern in (
            (ROOT / "results" / profile, "*.csv"),
            (ROOT / "plots" / profile, "*.png"),
        )
        for path in directory.glob(pattern)
    }
    mismatches = [
        str(path)
        for path in expected
        if not (ROOT / path).exists()
        or (ROOT / path).read_bytes() != (generated_root / path).read_bytes()
    ]
    mismatches.extend(f"unexpected: {path}" for path in sorted(actual_set - expected_set))
    if mismatches:
        raise RuntimeError("Committed artifacts differ from generated output:\n" + "\n".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("20-epochs", "200-epochs"), default="20-epochs")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Verify committed outputs without modifying them.")
    args = parser.parse_args()

    source = args.source or ROOT / "data" / args.profile / "benchmark_records.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    validate_source(records, protocol, args.profile)
    if args.check:
        with tempfile.TemporaryDirectory(prefix="fastml-benchmark-") as directory:
            generated_root = Path(directory)
            summary = generate(records, generated_root, args.profile)
            compare_generated(generated_root, args.profile)
    else:
        summary = generate(records, args.output_root, args.profile)
    print(json.dumps({"status": "ok", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
