import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Plot aggregate benchmark results.")
    parser.add_argument("--suite", default="configs/fair_benchmark.json")
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args()

    with open(args.suite, encoding="utf-8") as handle:
        suite = json.load(handle)
    included = {
        entry["run_name"] for entry in suite["models"] if entry.get("enabled", True)
    }
    with open(args.summary, newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["row_type"] == "aggregate"
            and row["base_run_name"] in included
            and int(row.get("num_runs") or 0) > 1
        ]
    rows.sort(key=lambda row: float(row["macro_auc"]))
    labels = [row["base_run_name"] for row in rows]
    positions = np.arange(len(rows))

    figure, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    axes[0].barh(
        positions,
        [float(row["accuracy"]) for row in rows],
        xerr=[float(row["accuracy_std"]) for row in rows],
    )
    axes[0].set_xlabel("accuracy")
    axes[0].set_xlim(0.70, 0.78)
    axes[0].set_yticks(positions, labels)
    axes[0].grid(axis="x", alpha=0.3)
    axes[1].barh(
        positions,
        [float(row["macro_auc"]) for row in rows],
        xerr=[float(row["macro_auc_std"]) for row in rows],
    )
    axes[1].set_xlabel("macro AUC")
    axes[1].set_xlim(0.90, 0.95)
    axes[1].grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(ROOT / "plots" / "fair_benchmark_comparison.png", dpi=170)
    plt.close(figure)

    class_names = ["gluon", "quark", "W", "Z", "top"]
    matrix = np.asarray(
        [[float(row[f"auc_{name}"]) for name in class_names] for row in rows]
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, aspect="auto", vmin=0.85, vmax=0.98, cmap="viridis")
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(labels)), labels)
    for row_index in range(len(labels)):
        for column_index in range(len(class_names)):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] < 0.92 else "black",
            )
    figure.colorbar(image, ax=axis, label="one-vs-rest AUC")
    figure.tight_layout()
    figure.savefig(ROOT / "plots" / "fair_benchmark_per_class_auc.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
