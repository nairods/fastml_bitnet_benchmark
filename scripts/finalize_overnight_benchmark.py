#!/usr/bin/env python3
"""Regenerate the benchmark status matrix and final reporting artifacts.

The script is intentionally read-only with respect to model/synthesis inputs. It
only writes derived CSV/plot/report artifacts under results/ and plots/.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

METRIC_SUFFIXES = {
    "_pytorch.json": "pytorch",
    "_qkeras.json": "qkeras",
    "_hgq.json": "hgq",
    "_xgboost.json": "xgboost",
}

PRIORITY_MODELS = [
    ("tier1", "mlp_baseline"),
    ("tier1", "qkeras_mlp"),
    ("tier1", "hgq_mlp"),
    ("tier1", "bitnet_topo_f7_fixed"),
    ("tier1", "bit158_topo_f7_fixed"),
    ("tier1", "bit158_mlp_f7_fixed"),
    ("tier1", "xgboost_bdt"),
    ("tier2", "bitnet_mlp_f5_fixed"),
    ("tier2", "bitnet_mlp_f7_fixed"),
    ("tier2", "bitnet_mlp_f8_fixed"),
    ("tier2", "bitnet_mlp_f10_fixed"),
    ("tier2", "bitnet_mlp_f12_fixed"),
    ("tier2", "bitnet_mlp_f7_power2"),
    ("tier3", "deepsets_hlf"),
    ("tier3", "mlp_mixer_hlf"),
]
PRIORITY_MODEL_NAMES = {model for _tier, model in PRIORITY_MODELS}

IGNORED_IF_PARTIAL = [
    "multihead_attention_hlf",
    "linformer_hlf",
    "jetformer_hlf",
    "binary_448_224_224",
    "ternary_128_64_64_64",
    "bitnet_binary_sigmoid_f7_fixed",
]

CANONICAL_SEED42 = {
    "bitnet_mlp_f5_fixed",
    "bitnet_mlp_f7_fixed",
    "bitnet_mlp_f8_fixed",
    "bitnet_mlp_f10_fixed",
    "bitnet_mlp_f12_fixed",
    "bitnet_mlp_f7_power2",
    "bitnet_binary_sigmoid_f7_fixed",
    "bitnet_binary_f7_fixed",
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def split_run_name(run_name: str, payload: dict | None = None) -> tuple[str, int]:
    match = re.match(r"^(?P<base>.+)__seed(?P<seed>\d+)$", run_name)
    if match:
        return match.group("base"), int(match.group("seed"))
    if payload and isinstance(payload.get("seed"), int):
        return run_name, int(payload["seed"])
    return run_name, 42


def artifact_run_names() -> set[str]:
    runs: set[str] = set()
    for path in (ROOT / "models").glob("*"):
        name = path.name
        for suffix in (".weights.h5", ".pt", ".pkl"):
            if name.endswith(suffix):
                runs.add(name[: -len(suffix)])
    for directory in (ROOT / "onnx" / "hardware", ROOT / "onnx" / "portable"):
        if directory.exists():
            for path in directory.glob("*.onnx"):
                runs.add(path.stem)
    for path in RESULTS.glob("*.json"):
        for suffix in METRIC_SUFFIXES:
            if path.name.endswith(suffix):
                runs.add(path.name[: -len(suffix)])
    for path in (ROOT / "hls_projects").glob("*"):
        if path.is_dir() and path.name != "__pycache__":
            runs.add(path.name)
    for result in (RESULTS / "synthesis").glob("*/result.json"):
        payload = read_json(result)
        if payload.get("run_name"):
            runs.add(str(payload["run_name"]))
    return runs


def choose_run_name(base: str, seed: int, known_runs: set[str]) -> str:
    seeded = f"{base}__seed{seed}"
    if seeded in known_runs:
        return seeded
    if seed == 42 and base in known_runs:
        return base
    if seed == 42 and base in CANONICAL_SEED42:
        return base
    return seeded


def metric_path_for(run_name: str) -> tuple[Path | None, str | None]:
    for suffix, backend in METRIC_SUFFIXES.items():
        path = RESULTS / f"{run_name}{suffix}"
        if path.exists():
            return path, backend
    return None, None


def weights_exist(run_name: str) -> bool:
    return any(
        (ROOT / "models" / f"{run_name}{suffix}").exists()
        for suffix in (".pt", ".weights.h5", ".pkl")
    )


def onnx_exist(run_name: str) -> bool:
    candidates = [
        ROOT / "onnx" / "hardware" / f"{run_name}.onnx",
        ROOT / "onnx" / "portable" / f"{run_name}.onnx",
        ROOT / "onnx" / f"{run_name}.onnx",
    ]
    return any(path.exists() for path in candidates)


def hls_exists(run_name: str) -> bool:
    if (ROOT / "hls_projects" / run_name / "native").exists():
        return True
    if (ROOT / "hls_projects" / run_name / "conifer").exists():
        return True
    synth_root = RESULTS / "synthesis"
    return any(path.is_dir() for path in synth_root.glob(f"{run_name}*"))


def synthesis_results() -> list[dict]:
    rows = []
    for result_path in sorted((RESULTS / "synthesis").glob("*/result.json")):
        payload = read_json(result_path)
        run_name = str(payload.get("run_name") or "")
        if not run_name:
            continue
        base, seed = split_run_name(run_name, payload)
        variant = payload.get("variant") or result_path.parent.name.removeprefix(run_name).lstrip("_")
        status = payload.get("synthesis_status")
        if not status and payload.get("returncode") == 0:
            status = "success"
        rows.append(
            {
                "run_name": run_name,
                "model_name": base,
                "seed": seed,
                "result_dir": str(result_path.parent.relative_to(ROOT)),
                "variant": variant,
                "mode": payload.get("mode"),
                "synthesis_status": status,
                "place_and_route_status": payload.get("place_and_route_status", "unknown"),
                "tool": payload.get("tool"),
                "tool_version": payload.get("tool_version"),
                "clock_target_ns": payload.get("clock_target_ns"),
                "clock_achieved_ns": payload.get("clock_achieved_ns"),
                "latency_cycles": payload.get("latency_cycles")
                or payload.get("latency_cycles_max"),
                "initiation_interval_cycles": payload.get("initiation_interval_cycles"),
                "lut": payload.get("lut"),
                "ff": payload.get("ff"),
                "dsp": payload.get("dsp"),
                "bram_18k": payload.get("bram_18k") or payload.get("bram"),
                "uram": payload.get("uram"),
                "reference_class_agreement": (
                    payload.get("reference_validation") or {}
                ).get("class_agreement"),
                "reference_accuracy_delta": (
                    payload.get("reference_validation") or {}
                ).get("accuracy_delta"),
                "reference_macro_auc_delta": (
                    payload.get("reference_validation") or {}
                ).get("macro_auc_delta"),
                "reference_source": (
                    payload.get("reference_validation") or {}
                ).get("reference_source"),
            }
        )
    return rows


def select_synth(run_name: str, synth_rows: list[dict]) -> dict | None:
    candidates = [
        row
        for row in synth_rows
        if row["run_name"] == run_name and row.get("synthesis_status") == "success"
    ]
    if not candidates:
        return None

    def rank(row: dict) -> tuple[int, int, int]:
        result_dir = row["result_dir"]
        variant = str(row.get("variant") or "")
        is_custom = "custom_native" in result_dir
        is_tree = row.get("mode") == "tree"
        is_logits = "sigmoid" not in result_dir
        return (0 if is_custom else 1, 0 if is_tree else 1, 0 if is_logits else 1)

    return sorted(candidates, key=rank)[0]


def target_model_rows(known_runs: set[str]) -> list[dict]:
    rows = []
    seen: set[tuple[str, int]] = set()
    for tier, base in PRIORITY_MODELS:
        for seed in (42, 43, 44):
            rows.append({"tier": tier, "model_name": base, "seed": seed})
            seen.add((base, seed))

    discovered: set[tuple[str, int]] = set()
    for run_name in known_runs:
        base, seed = split_run_name(run_name)
        discovered.add((base, seed))

    for base in IGNORED_IF_PARTIAL:
        for seed in (42, 43, 44):
            if (base, seed) in discovered and (base, seed) not in seen:
                rows.append({"tier": "ignored_partial", "model_name": base, "seed": seed})
                seen.add((base, seed))

    for base, seed in sorted(discovered):
        if (base, seed) not in seen and not base.endswith("_quantized"):
            rows.append({"tier": "discovered", "model_name": base, "seed": seed})
            seen.add((base, seed))
    return rows


def build_status_matrix(known_runs: set[str], synth_rows: list[dict]) -> list[dict]:
    matrix = []
    for item in target_model_rows(known_runs):
        base = item["model_name"]
        seed = int(item["seed"])
        run_name = choose_run_name(base, seed, known_runs)
        metric_path, backend = metric_path_for(run_name)
        synth = select_synth(run_name, synth_rows)
        row = {
            "model_name": base,
            "seed": seed,
            "weights_exists": weights_exist(run_name),
            "metrics_exists": metric_path is not None,
            "onnx_exists": onnx_exist(run_name),
            "hls_exists": hls_exists(run_name),
            "synth_exists": synth is not None,
            "report_parsed": bool(synth and synth.get("latency_cycles") is not None and synth.get("lut") is not None),
            "complete": False,
            "tier": item["tier"],
            "run_name": run_name,
            "metrics_backend": backend or "",
        }
        row["complete"] = all(
            row[key]
            for key in (
                "weights_exists",
                "metrics_exists",
                "onnx_exists",
                "hls_exists",
                "synth_exists",
                "report_parsed",
            )
        )
        matrix.append(row)
    return matrix


def software_metric_rows() -> dict[str, dict]:
    rows = {}
    for path in sorted(RESULTS.glob("*.json")):
        backend = None
        run_name = None
        for suffix, candidate_backend in METRIC_SUFFIXES.items():
            if path.name.endswith(suffix):
                backend = candidate_backend
                run_name = path.name[: -len(suffix)]
                break
        if not backend or not run_name:
            continue
        payload = read_json(path)
        base, seed = split_run_name(run_name, payload)
        if run_name.startswith("xgboost_bdt__seed"):
            base = "xgboost_bdt"
        rows[run_name] = {
            "run_name": run_name,
            "model_name": base,
            "seed": seed,
            "backend": backend,
            "accuracy": payload.get("accuracy"),
            "macro_auc": payload.get("macro_auc"),
            "macro_average_precision": payload.get("macro_average_precision"),
            "cross_entropy": payload.get("cross_entropy"),
            "expected_calibration_error": payload.get("expected_calibration_error"),
            "cpu_latency_ms": payload.get("cpu_latency_ms"),
            "model_size_bytes": payload.get("model_size_bytes") or payload.get("model_size"),
            "tree_node_proxy": payload.get("tree_node_proxy"),
        }
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(matrix: list[dict], synth_rows: list[dict]) -> list[dict]:
    metrics = software_metric_rows()
    summary = []
    emitted: set[tuple[str, str]] = set()
    for status in matrix:
        run_name = status["run_name"]
        synth = select_synth(run_name, synth_rows)
        metric = metrics.get(run_name, {})
        row = {
            **status,
            **{key: metric.get(key) for key in (
                "backend",
                "accuracy",
                "macro_auc",
                "macro_average_precision",
                "cross_entropy",
                "expected_calibration_error",
                "cpu_latency_ms",
                "model_size_bytes",
                "tree_node_proxy",
            )},
        }
        if synth:
            row.update(
                {
                    "hardware_result_dir": synth["result_dir"],
                    "hardware_variant": synth["variant"],
                    "mode": synth.get("mode"),
                    "synthesis_status": synth.get("synthesis_status"),
                    "place_and_route_status": synth.get("place_and_route_status"),
                    "clock_target_ns": synth.get("clock_target_ns"),
                    "clock_achieved_ns": synth.get("clock_achieved_ns"),
                    "latency_cycles": synth.get("latency_cycles"),
                    "initiation_interval_cycles": synth.get("initiation_interval_cycles"),
                    "lut": synth.get("lut"),
                    "ff": synth.get("ff"),
                    "dsp": synth.get("dsp"),
                    "bram_18k": synth.get("bram_18k"),
                    "uram": synth.get("uram"),
                    "reference_class_agreement": synth.get("reference_class_agreement"),
                    "reference_accuracy_delta": synth.get("reference_accuracy_delta"),
                    "reference_macro_auc_delta": synth.get("reference_macro_auc_delta"),
                    "reference_source": synth.get("reference_source"),
                }
            )
        summary.append(row)
        emitted.add((run_name, str(row.get("hardware_result_dir") or "")))

    for synth in synth_rows:
        key = (synth["run_name"], synth["result_dir"])
        if key in emitted:
            continue
        metric = metrics.get(synth["run_name"], {})
        base, seed = split_run_name(synth["run_name"])
        row = {
            "tier": "synthesis_ablation",
            "model_name": base,
            "seed": seed,
            "run_name": synth["run_name"],
            "weights_exists": weights_exist(synth["run_name"]),
            "metrics_exists": synth["run_name"] in metrics,
            "onnx_exists": onnx_exist(synth["run_name"]),
            "hls_exists": True,
            "synth_exists": synth.get("synthesis_status") == "success",
            "report_parsed": synth.get("latency_cycles") is not None and synth.get("lut") is not None,
            "complete": False,
            **{key: metric.get(key) for key in (
                "backend",
                "accuracy",
                "macro_auc",
                "macro_average_precision",
                "cross_entropy",
                "expected_calibration_error",
                "cpu_latency_ms",
                "model_size_bytes",
                "tree_node_proxy",
            )},
            "hardware_result_dir": synth["result_dir"],
            "hardware_variant": synth["variant"],
            "mode": synth.get("mode"),
            "synthesis_status": synth.get("synthesis_status"),
            "place_and_route_status": synth.get("place_and_route_status"),
            "clock_target_ns": synth.get("clock_target_ns"),
            "clock_achieved_ns": synth.get("clock_achieved_ns"),
            "latency_cycles": synth.get("latency_cycles"),
            "initiation_interval_cycles": synth.get("initiation_interval_cycles"),
            "lut": synth.get("lut"),
            "ff": synth.get("ff"),
            "dsp": synth.get("dsp"),
            "bram_18k": synth.get("bram_18k"),
            "uram": synth.get("uram"),
            "reference_class_agreement": synth.get("reference_class_agreement"),
            "reference_accuracy_delta": synth.get("reference_accuracy_delta"),
            "reference_macro_auc_delta": synth.get("reference_macro_auc_delta"),
            "reference_source": synth.get("reference_source"),
        }
        summary.append(row)
    return summary


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_seed_statistics(summary: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in summary:
        if row.get("tier") == "synthesis_ablation":
            continue
        grouped[(row["model_name"], row.get("hardware_variant") or "software_only")].append(row)

    stats = []
    metrics = ["accuracy", "macro_auc", "latency_cycles", "lut", "ff", "dsp", "bram_18k"]
    for (model, variant), rows in sorted(grouped.items()):
        out = {"model_name": model, "hardware_variant": variant, "n_seeds": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows if finite(row.get(metric))]
            out[f"{metric}_mean"] = statistics.mean(values) if values else None
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            out[f"{metric}_min"] = min(values) if values else None
            out[f"{metric}_max"] = max(values) if values else None
        stats.append(out)
    return stats


def pareto_front(summary: list[dict]) -> list[dict]:
    candidates = [
        row for row in summary
        if row.get("model_name") in PRIORITY_MODEL_NAMES
        and finite(row.get("accuracy"))
        and finite(row.get("latency_cycles"))
        and finite(row.get("lut"))
    ]
    frontier = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            no_worse = (
                float(other["accuracy"]) >= float(row["accuracy"])
                and float(other["latency_cycles"]) <= float(row["latency_cycles"])
                and float(other["lut"]) <= float(row["lut"])
            )
            strictly_better = (
                float(other["accuracy"]) > float(row["accuracy"])
                or float(other["latency_cycles"]) < float(row["latency_cycles"])
                or float(other["lut"]) < float(row["lut"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda r: (-float(r["accuracy"]), float(r["latency_cycles"]), float(r["lut"])))


def plot_outputs(summary: list[dict]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on host env
        (PLOTS / "plot_generation_failed.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return

    hardware = [
        row for row in summary
        if finite(row.get("accuracy")) and finite(row.get("lut")) and finite(row.get("latency_cycles"))
    ]
    for x_key, output, xlabel in [
        ("lut", "accuracy_vs_luts.png", "LUTs (C-synthesis estimate)"),
        ("latency_cycles", "accuracy_vs_latency.png", "Latency cycles (C-synthesis estimate)"),
    ]:
        plt.figure(figsize=(9, 6))
        for row in hardware:
            plt.scatter(float(row[x_key]), float(row["accuracy"]), s=28)
            label = f"{row['model_name']} s{row['seed']}"
            if row.get("mode"):
                label += f" {row['mode']}"
            plt.annotate(label, (float(row[x_key]), float(row["accuracy"])), fontsize=6, alpha=0.75)
        plt.xlabel(xlabel)
        plt.ylabel("Test accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS / output, dpi=180)
        plt.close()

    sweep = []
    for row in hardware:
        match = re.match(r"bitnet_mlp_f(?P<bits>\d+)_(?P<kind>fixed|power2)$", row["model_name"])
        if match:
            sweep.append((int(match.group("bits")), match.group("kind"), row))
    if sweep:
        plt.figure(figsize=(9, 6))
        for kind in sorted({item[1] for item in sweep}):
            points = sorted(
                ((bits, row) for bits, row_kind, row in sweep if row_kind == kind),
                key=lambda item: (
                    item[0],
                    str(item[1].get("seed")),
                    str(item[1].get("hardware_variant")),
                    str(item[1].get("hardware_result_dir")),
                ),
            )
            plt.plot(
                [bits for bits, _ in points],
                [float(row["lut"]) for _, row in points],
                marker="o",
                label=f"LUT {kind}",
            )
            plt.plot(
                [bits for bits, _ in points],
                [float(row["latency_cycles"]) * 1000.0 for _, row in points],
                marker="x",
                linestyle="--",
                label=f"latency cycles x1000 {kind}",
            )
        plt.xlabel("BitNet fractional bits")
        plt.ylabel("LUTs; latency shown as cycles x1000")
        plt.title("BitNet fixed/power-of-two scaling sweep")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS / "bitnet_alpha_sweep.png", dpi=180)
        plt.close()


def write_report(matrix: list[dict], summary: list[dict], pareto: list[dict]) -> None:
    completed = [row for row in matrix if row["complete"]]
    incomplete = [row for row in matrix if not row["complete"] and row["tier"] != "discovered"]
    software = [
        row for row in summary
        if row.get("model_name") in PRIORITY_MODEL_NAMES and finite(row.get("accuracy"))
    ]
    hardware = [
        row for row in summary
        if row.get("model_name") in PRIORITY_MODEL_NAMES
        and finite(row.get("latency_cycles"))
        and finite(row.get("lut"))
    ]

    best_accuracy = max(software, key=lambda row: float(row["accuracy"])) if software else None
    best_latency = min(hardware, key=lambda row: float(row["latency_cycles"])) if hardware else None
    best_lut_eff = (
        max(hardware, key=lambda row: float(row.get("accuracy") or 0.0) / max(float(row["lut"]), 1.0))
        if hardware
        else None
    )

    def fmt_model(row: dict | None) -> str:
        if not row:
            return "n/a"
        return (
            f"{row['model_name']} seed {row['seed']} "
            f"({row.get('hardware_variant') or row.get('backend') or 'no variant'})"
        )

    lines = [
        "# Overnight Benchmark Report",
        "",
        "All FPGA numbers in this report are C-synthesis estimates unless a row explicitly states otherwise. Current BitNet custom-native results report `place_and_route_status=not_run`.",
        "",
        "## Completed Models",
    ]
    if completed:
        lines.extend(
            f"- {row['model_name']} seed {row['seed']} ({row['run_name']})"
            for row in completed
        )
    else:
        lines.append("- None fully complete across weights, metrics, ONNX, HLS project, synthesis report, and parsed report.")

    lines.extend(["", "## Failed Or Blocked Models"])
    bdt_dirs = sorted((ROOT / "hls_projects").glob("xgboost_bdt*/conifer"))
    if bdt_dirs:
        lines.append("- xgboost_bdt Conifer project generation exists, but no usable C-synthesis report was produced in the current artifacts.")
    lines.append("- deepsets_hlf and mlp_mixer_hlf have lowering packages only; custom operator lowering is still required before synthesis.")
    lines.append("- Transformer-style HLF models are ignored for this benchmark unless explicitly revived.")

    lines.extend(["", "## Missing Or Incomplete Rows"])
    lines.extend(
        f"- {row['model_name']} seed {row['seed']}: "
        f"weights={row['weights_exists']}, metrics={row['metrics_exists']}, onnx={row['onnx_exists']}, "
        f"hls={row['hls_exists']}, synth={row['synth_exists']}, parsed={row['report_parsed']}"
        for row in incomplete[:80]
    )
    if len(incomplete) > 80:
        lines.append(f"- ... {len(incomplete) - 80} more rows; see results/status_matrix.csv.")

    lines.extend(
        [
            "",
            "## Best Current Rows",
            "These best rows are restricted to the requested Tier 1-3 OpenML multi-class benchmark models.",
            f"- Best software accuracy: {fmt_model(best_accuracy)} accuracy={best_accuracy.get('accuracy') if best_accuracy else 'n/a'}",
            f"- Best C-synth latency: {fmt_model(best_latency)} latency_cycles={best_latency.get('latency_cycles') if best_latency else 'n/a'}",
            f"- Best LUT efficiency: {fmt_model(best_lut_eff)} accuracy_per_lut={(float(best_lut_eff.get('accuracy') or 0.0) / max(float(best_lut_eff['lut']), 1.0)) if best_lut_eff else 'n/a'}",
            "",
            "## Pareto Frontier Candidates",
        ]
    )
    if pareto:
        lines.extend(
            f"- {row['model_name']} seed {row['seed']} {row.get('hardware_variant') or ''}: "
            f"accuracy={row.get('accuracy')}, latency={row.get('latency_cycles')}, LUT={row.get('lut')}"
            for row in pareto[:30]
        )
    else:
        lines.append("- No rows have accuracy, latency, and LUT simultaneously.")

    lines.extend(
        [
            "",
            "## Preliminary Paper Observations",
            "- The current trustworthy FPGA numbers are C-synthesis estimates, not place-and-route timing/resource numbers.",
            "- BitNet custom-native logits-only kernels are the most complete synthesized family in the current repository.",
            "- QKeras and HGQ have trained weights, test metrics, and representative generated HLS projects, but no completed synthesis reports in the current artifacts.",
            "- The XGBoost BDT reaches MLP-like software accuracy for seed 42, but Conifer/Vitis synthesis is not yet producing a usable report for the large tuned ensemble.",
            "- Seed statistics should be interpreted by implementation mode; mixing tree and unrolled BitNet runs can hide implementation effects.",
        ]
    )
    (RESULTS / "overnight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    known_runs = artifact_run_names()
    synth_rows = synthesis_results()
    matrix = build_status_matrix(known_runs, synth_rows)
    summary = build_summary(matrix, synth_rows)
    seed_stats = write_seed_statistics(summary)
    pareto = pareto_front(summary)

    status_fields = [
        "model_name",
        "seed",
        "weights_exists",
        "metrics_exists",
        "onnx_exists",
        "hls_exists",
        "synth_exists",
        "report_parsed",
        "complete",
        "tier",
        "run_name",
        "metrics_backend",
    ]
    summary_fields = [
        "tier",
        "model_name",
        "seed",
        "run_name",
        "backend",
        "accuracy",
        "macro_auc",
        "macro_average_precision",
        "cross_entropy",
        "expected_calibration_error",
        "cpu_latency_ms",
        "model_size_bytes",
        "tree_node_proxy",
        "weights_exists",
        "metrics_exists",
        "onnx_exists",
        "hls_exists",
        "synth_exists",
        "report_parsed",
        "complete",
        "hardware_result_dir",
        "hardware_variant",
        "mode",
        "synthesis_status",
        "place_and_route_status",
        "clock_target_ns",
        "clock_achieved_ns",
        "latency_cycles",
        "initiation_interval_cycles",
        "lut",
        "ff",
        "dsp",
        "bram_18k",
        "uram",
        "reference_class_agreement",
        "reference_accuracy_delta",
        "reference_macro_auc_delta",
        "reference_source",
    ]
    stat_fields = ["model_name", "hardware_variant", "n_seeds"]
    for metric in ["accuracy", "macro_auc", "latency_cycles", "lut", "ff", "dsp", "bram_18k"]:
        stat_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])

    write_csv(RESULTS / "status_matrix.csv", matrix, status_fields)
    write_csv(RESULTS / "final_benchmark_summary.csv", summary, summary_fields)
    write_csv(RESULTS / "final_seed_statistics.csv", seed_stats, stat_fields)
    write_csv(RESULTS / "final_pareto_candidates.csv", pareto, summary_fields)
    plot_outputs(summary)
    write_report(matrix, summary, pareto)

    complete_count = sum(1 for row in matrix if row["complete"])
    print(
        json.dumps(
            {
                "status_rows": len(matrix),
                "complete_rows": complete_count,
                "summary_rows": len(summary),
                "synthesis_rows": len(synth_rows),
                "pareto_rows": len(pareto),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
