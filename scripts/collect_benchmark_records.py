#!/usr/bin/env python3
"""Collect profile-local raw metrics, histories, and synthesis reports."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_SUFFIX = {
    "pytorch": "pytorch",
    "qkeras": "qkeras",
    "hgq": "hgq",
    "xgboost": "xgboost",
}
CHECKPOINT_SUFFIX = {
    "pytorch": ".pt",
    "qkeras": ".weights.h5",
    "hgq": ".weights.h5",
    "xgboost": ".pkl",
}
TRAINING_CURVE_MODELS = {
    "Dense MLP",
    "QKeras fixed b7",
    "HGQ",
    "QKeras binary",
    "QKeras ternary",
    "BitNet binary",
    "BitNet-1.58",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def architecture_label(protocol: dict, architecture: str) -> str:
    definition = protocol["architectures"][architecture]
    if isinstance(definition, list):
        return "-".join(str(width) for width in definition)
    return f"{definition['n_estimators']} trees depth {definition['max_depth']}"


def synthesis_record(path: Path, implementation: str, source_variant: str) -> dict:
    raw = load_json(path)
    report = raw.get("parsed_report", raw)
    return {
        "implementation": implementation,
        "source_variant": source_variant,
        "tool": raw.get("tool", "Vitis HLS"),
        "tool_version": raw.get("tool_version"),
        "part": raw.get("part") or report.get("part"),
        "clock_target_ns": raw.get("clock_target_ns") or report.get("clock_target_ns"),
        "latency_cycles": raw.get("latency_cycles") or report.get("latency_cycles_max"),
        "ii_cycles": raw.get("initiation_interval_cycles") or report.get("initiation_interval_cycles"),
        "lut": raw.get("lut") or report.get("lut"),
        "ff": raw.get("ff") or report.get("ff"),
        "dsp": raw.get("dsp") or report.get("dsp"),
        "bram18": raw.get("bram_18k") or report.get("bram"),
        "place_and_route": "not_run",
    }


def verify_reused_xgboost_model(run_name: str) -> None:
    boosters = []
    for profile in ("20-epochs", "200-epochs"):
        path = ROOT / "models" / profile / f"{run_name}.pkl"
        with path.open("rb") as handle:
            boosters.append(pickle.load(handle)["model"].get_booster().save_raw(raw_format="json"))
    if boosters[0] != boosters[1]:
        raise ValueError(f"Cannot reuse synthesis: XGBoost models differ for {run_name}")


def find_synthesis(profile: str, run_name: str, model: dict, template_run: dict) -> dict | None:
    template = template_run.get("synthesis")
    if model["model"] in {"BitNet binary", "BitNet-1.58"}:
        implementation = "hls4ml_patched_bitnet_latency_rf1"
    elif template:
        implementation = template["implementation"]
    elif model["backend"] == "xgboost":
        implementation = "conifer_unrolled"
    else:
        implementation = "hls4ml_latency_rf1"
    if implementation is None:
        return None
    directory = ROOT / "results" / profile / "synthesis" / f"{run_name}_{implementation}"
    result = directory / "result.json"
    if not result.exists():
        return None
    source_variant = (
        template.get("source_variant", implementation)
        if template and template.get("implementation") == implementation
        else implementation
    )
    return synthesis_record(result, implementation, source_variant)


def collect_run(
    profile: str,
    task: str,
    model: dict,
    model_spec: dict,
    seed: int,
    template_run: dict,
    require_synthesis: bool,
    max_epochs: int,
) -> dict:
    seed = int(seed)
    run_name = f"{model['base_run_name']}__seed{seed}"
    backend = model_spec["backend"]
    checkpoint = ROOT / "models" / profile / f"{run_name}{CHECKPOINT_SUFFIX[backend]}"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    result_path = (
        ROOT / "results" / profile / "raw" / f"{run_name}_{BACKEND_SUFFIX[backend]}.json"
    )
    raw = load_json(result_path)
    if raw.get("run_name") != run_name or int(raw.get("seed", -1)) != seed:
        raise ValueError(f"Raw result identity mismatch in {result_path}")
    if raw.get("benchmark_profile") not in (None, profile):
        raise ValueError(f"Raw result profile mismatch in {result_path}")
    collected = {
        "seed": seed,
        "accuracy": raw["accuracy"],
    }
    if task != "multiclass":
        signal_efficiency = raw.get("signal_eff_at_1pct_fpr")
        if signal_efficiency is None:
            signal_efficiency = template_run.get("signal_eff_at_1pct_fpr")
        if signal_efficiency is None:
            raise ValueError(f"Missing binary operating-point metric for {profile}/{run_name}")
        collected.update(
            auc=raw["macro_auc"],
            signal_eff_at_1pct_fpr=signal_efficiency,
        )
    else:
        collected.update(
            macro_auc=raw["macro_auc"],
            per_class_auc=raw["per_class_auc"],
        )
    selection = raw.get("training_selection")
    history_path = ROOT / "logs" / profile / f"{run_name}_history.json"
    if history_path.exists():
        history = load_json(history_path)
        validation_loss = history.get("validation_loss")
        if validation_loss:
            for key in ("train_loss", "validation_loss", "validation_accuracy"):
                if len(history.get(key, [])) != max_epochs:
                    raise ValueError(f"{history_path}: expected {max_epochs} values for {key}")
            selected_index = min(range(len(validation_loss)), key=validation_loss.__getitem__)
            selection = {
                "max_epochs": len(validation_loss),
                "selected_epoch": selected_index + 1,
                "metric": "validation_loss",
                "validation_loss": validation_loss[selected_index],
                "early_stopping": False,
            }
    elif backend != "xgboost":
        raise FileNotFoundError(history_path)
    if selection:
        collected["training_selection"] = selection
    synthesis = find_synthesis(profile, run_name, model, template_run)
    if synthesis is None and profile == "20-epochs" and template_run.get("synthesis"):
        synthesis = copy.deepcopy(template_run["synthesis"])
    if synthesis is None and profile == "200-epochs" and model_spec["backend"] == "xgboost":
        if template_run.get("synthesis"):
            verify_reused_xgboost_model(run_name)
            synthesis = copy.deepcopy(template_run["synthesis"])
            synthesis["reused_from_profile"] = "20-epochs"
    if synthesis:
        collected["synthesis"] = synthesis
    elif require_synthesis and (template_run.get("synthesis") or model["backend"] == "xgboost"):
        raise FileNotFoundError(f"Missing synthesis result for {profile}/{run_name}")
    return collected


def normalized_history(path: Path, max_epochs: int) -> dict:
    history = load_json(path)
    required = ("train_loss", "validation_loss", "validation_accuracy")
    for key in required:
        if len(history.get(key, [])) != max_epochs:
            raise ValueError(f"{path}: expected {max_epochs} values for {key}")
    return {key: history[key] for key in required}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("20-epochs", "200-epochs"), required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "data" / "20-epochs" / "benchmark_records.json",
        help="Existing records used only for model layout, BDT rows, and synthesis provenance.",
    )
    parser.add_argument("--allow-missing-synthesis", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    protocol = load_json(ROOT / "configs" / "benchmark.json")
    template = load_json(args.template)
    profile_spec = protocol["training_profiles"][args.profile]
    max_epochs = int(profile_spec["defaults"]["epochs"])
    records = {
        "schema_version": 1,
        "training_profile": args.profile,
        "training_protocol": copy.deepcopy(profile_spec),
        "dataset": copy.deepcopy(template["dataset"]),
        "hardware": copy.deepcopy(template["hardware"]),
        "tasks": {},
        "training_curves": [],
    }

    for task in protocol["tasks"]:
        task_template = template["tasks"].get(task, {})
        template_models = {
            model["base_run_name"]: model for model in task_template.get("models", [])
        }
        prefix = {"qg_vs_wzt": "", "qg_vs_top": "topqg_", "multiclass": "multiclass_"}[task]
        models = []
        for spec in protocol["models"]:
            architectures = (
                spec["multiclass_architectures"]
                if task == "multiclass"
                else spec["base_names"].keys()
            )
            for architecture in architectures:
                base_run_name = prefix + spec["base_names"][architecture]
                model = {
                    "model": spec["display_name"],
                    "architecture": architecture_label(protocol, architecture),
                    "base_run_name": base_run_name,
                    "backend": spec["backend"],
                }
                model_template = template_models.get(base_run_name, {})
                template_runs = model_template.get("runs", [])
                templates_by_seed = {int(run["seed"]): run for run in template_runs}
                provenance_fallback = template_runs[0] if template_runs else {}
                model["runs"] = [
                    collect_run(
                        args.profile,
                        task,
                        model,
                        spec,
                        seed,
                        templates_by_seed.get(int(seed), provenance_fallback),
                        not args.allow_missing_synthesis
                        and (
                            task != "multiclass"
                            or int(seed)
                            in spec.get("multiclass_synthesis_seeds", protocol["model_seeds"])
                        ),
                        max_epochs,
                    )
                    for seed in protocol["model_seeds"]
                ]
                models.append(model)
        records["tasks"][task] = {
            key: copy.deepcopy(value)
            for key, value in task_template.items()
            if key != "models"
        }
        records["tasks"][task]["models"] = models

    primary = records["tasks"]["qg_vs_wzt"]["models"]
    for model in primary:
        if model["model"] not in TRAINING_CURVE_MODELS or model["architecture"] != "64-32-32":
            continue
        run_name = f"{model['base_run_name']}__seed42"
        history_path = ROOT / "logs" / args.profile / f"{run_name}_history.json"
        records["training_curves"].append(
            {
                "task": "qg_vs_wzt",
                "model": model["model"],
                "architecture": model["architecture"],
                "base_run_name": model["base_run_name"],
                "seed": 42,
                "history": normalized_history(history_path, max_epochs),
            }
        )

    output = args.output or ROOT / "data" / args.profile / "benchmark_records.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
