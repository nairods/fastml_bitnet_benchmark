import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HLSENV310 = Path(os.environ.get("FASTML_KERAS_PYTHON", "")) if os.environ.get("FASTML_KERAS_PYTHON") else ROOT.parents[1] / "miniconda3" / "envs" / "hlsenv310" / "bin" / "python"
HLSENV = Path(os.environ.get("FASTML_HLSENV_PYTHON", "")) if os.environ.get("FASTML_HLSENV_PYTHON") else ROOT.parents[1] / "miniconda3" / "envs" / "hlsenv" / "bin" / "python"
PYTHON = str(HLSENV if HLSENV.exists() else Path(sys.executable))
PART = "xcvu13p-flga2577-2-e"
CLOCK_NS = 5.0


def _json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def _task_label(class_mode: str) -> str:
    if class_mode == "binary_top_vs_qg":
        return "top signal versus quark/gluon background"
    return "quark/gluon background versus W/Z/top signal"


def _namespaced_run_name(namespace: str, run_name: str) -> str:
    if namespace == "binary":
        return run_name
    if run_name.startswith("binary_"):
        return f"{namespace}_{run_name[len('binary_'):]}"
    return f"{namespace}_{run_name}"


def _common(model_name: str, run_name: str, seed: int, hidden_dims=None, class_mode="binary_qg_vs_wzt", namespace="binary"):
    namespaced = _namespaced_run_name(namespace, run_name)
    model = {
        "name": model_name,
        "output_dim": 1,
        "output_mode": "binary_sigmoid",
    }
    if hidden_dims is not None:
        model["hidden_dims"] = hidden_dims
    return {
        "run_name": f"{namespaced}__seed{seed}",
        "base_run_name": namespaced,
        "seed": int(seed),
        "dataset": {
            "id": 42468,
            "cache": True,
            "max_samples": None,
            "sample_seed": 42,
            "classification": {"mode": class_mode},
        },
        "split": {"train": 0.64, "validation": 0.16, "test": 0.2, "seed": 42},
        "training": {
            "epochs": 20,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "device": "cpu",
        },
        "evaluation": {
            "batch_size": 4096,
            "latency": {"warmup": 20, "iterations": 200},
        },
        "onnx": {"opset": 17},
        "model": model,
    }


def model_configs(seeds, class_mode="binary_qg_vs_wzt", namespace="binary"):
    specs = []
    for seed in seeds:
        specs.append(_common("mlp_baseline", "binary_mlp_baseline_64_32_32", seed, [64, 32, 32], class_mode, namespace))
        specs.append(_common("mlp_topo", "binary_mlp_topo_128_32", seed, None, class_mode, namespace))

        qkeras = _common("qkeras_mlp", "binary_qkeras_mlp_64_32_32_b7", seed, [64, 32, 32], class_mode, namespace)
        qkeras["model"]["qkeras"] = {
            "quantizer": "quantized_bits",
            "bits": 7,
            "integer_bits": 0,
            "activation_bits": 7,
            "alpha": 1,
        }
        specs.append(qkeras)

        qkeras_topo = _common("qkeras_mlp", "binary_qkeras_topo_128_32_b7", seed, [128, 32], class_mode, namespace)
        qkeras_topo["model"]["qkeras"] = deepcopy(qkeras["model"]["qkeras"])
        specs.append(qkeras_topo)

        qkeras_binary = _common("qkeras_mlp", "binary_qkeras_mlp_binary_64_32_32", seed, [64, 32, 32], class_mode, namespace)
        qkeras_binary["model"]["qkeras"] = {
            "quantizer": "binary",
            "activation_bits": 7,
            "alpha": 1,
        }
        specs.append(qkeras_binary)

        qkeras_binary_topo = _common("qkeras_mlp", "binary_qkeras_topo_binary_128_32", seed, [128, 32], class_mode, namespace)
        qkeras_binary_topo["model"]["qkeras"] = deepcopy(qkeras_binary["model"]["qkeras"])
        specs.append(qkeras_binary_topo)

        qkeras_ternary = _common("qkeras_mlp", "binary_qkeras_mlp_ternary_64_32_32", seed, [64, 32, 32], class_mode, namespace)
        qkeras_ternary["model"]["qkeras"] = {
            "quantizer": "ternary",
            "activation_bits": 7,
            "alpha": 1,
        }
        specs.append(qkeras_ternary)

        qkeras_ternary_topo = _common("qkeras_mlp", "binary_qkeras_topo_ternary_128_32", seed, [128, 32], class_mode, namespace)
        qkeras_ternary_topo["model"]["qkeras"] = deepcopy(qkeras_ternary["model"]["qkeras"])
        specs.append(qkeras_ternary_topo)

        hgq = _common("hgq_mlp", "binary_hgq_mlp_64_32_32", seed, [64, 32, 32], class_mode, namespace)
        hgq["training"] = {
            "epochs": 200,
            "batch_size": 16384,
            "learning_rate": 0.02,
            "cosine_decay_steps": 200,
            "device": "cpu",
        }
        hgq["model"]["hgq"] = {"beta": 3e-6}
        specs.append(hgq)

        hgq_topo = deepcopy(hgq)
        hgq_topo["run_name"] = f"{_namespaced_run_name(namespace, 'binary_hgq_topo_128_32')}__seed{seed}"
        hgq_topo["base_run_name"] = _namespaced_run_name(namespace, "binary_hgq_topo_128_32")
        hgq_topo["model"]["hidden_dims"] = [128, 32]
        specs.append(hgq_topo)

        bitnet = _common("bitnet_mlp", "binary_bitnet_sigmoid_f7_fixed", seed, [64, 32, 32], class_mode, namespace)
        bitnet["model"]["bitnet"] = {"activation_bits": 8, "frac_bits": 7}
        specs.append(bitnet)

        bitnet_topo = _common("bitnet_mlp", "binary_bitnet_topo_sigmoid_f7_fixed", seed, [128, 32], class_mode, namespace)
        bitnet_topo["model"]["bitnet"] = {"activation_bits": 8, "frac_bits": 7}
        specs.append(bitnet_topo)

        bit158 = _common("bit158_mlp", "binary_bit158_sigmoid_f7_fixed", seed, [64, 32, 32], class_mode, namespace)
        bit158["model"]["bitnet"] = {"activation_bits": 8, "frac_bits": 7}
        specs.append(bit158)

        bit158_topo = _common("bit158_mlp", "binary_bit158_topo_sigmoid_f7_fixed", seed, [128, 32], class_mode, namespace)
        bit158_topo["model"]["bitnet"] = {"activation_bits": 8, "frac_bits": 7}
        specs.append(bit158_topo)

        xgb = _common("xgboost_bdt", "binary_xgboost_bdt_d4_100", seed, None, class_mode, namespace)
        xgb["training"] = {"device": "cpu"}
        xgb["model"]["xgboost"] = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": 1,
        }
        specs.append(xgb)
    return specs


def backend(config):
    name = config["model"]["name"]
    if name == "qkeras_mlp":
        return "qkeras"
    if name == "hgq_mlp":
        return "hgq"
    if name == "xgboost_bdt":
        return "xgboost"
    return "pytorch"


def python_for(config, stage):
    b = backend(config)
    if b in {"qkeras", "hgq"}:
        return str(HLSENV310 if HLSENV310.exists() else Path(sys.executable))
    if b == "xgboost":
        return str(HLSENV if HLSENV.exists() else Path(sys.executable))
    return PYTHON


def command_for(config, stage, config_path):
    b = backend(config)
    if stage == "train":
        script = {
            "qkeras": "train_qkeras.py",
            "hgq": "train_hgq.py",
            "xgboost": "train_xgboost.py",
            "pytorch": "torchDNN.py",
        }[b]
        return [python_for(config, stage), str(ROOT / script), "--config", str(config_path)]
    if stage == "evaluate":
        script = {
            "qkeras": "test_qkeras.py",
            "hgq": "test_hgq.py",
            "xgboost": "test_xgboost.py",
            "pytorch": "testDNN.py",
        }[b]
        return [python_for(config, stage), str(ROOT / script), "--config", str(config_path)]
    if stage == "reference":
        return [python_for(config, stage), str(ROOT / "export_reference_predictions.py"), "--config", str(config_path)]
    if stage == "onnx":
        return [PYTHON, str(ROOT / "build_ONNX.py"), "--config", str(config_path)]
    raise ValueError(stage)


def artifact_for(config, stage):
    run_name = config["run_name"]
    b = backend(config)
    if stage == "train":
        suffix = ".pkl" if b == "xgboost" else ".weights.h5" if b in {"qkeras", "hgq"} else ".pt"
        return ROOT / "models" / f"{run_name}{suffix}"
    if stage == "evaluate":
        result_backend = "xgboost" if b == "xgboost" else b if b in {"qkeras", "hgq"} else "pytorch"
        return ROOT / "results" / f"{run_name}_{result_backend}.json"
    if stage == "reference":
        return ROOT / "data" / "synthesis" / "reference_predictions" / f"{run_name}.npy"
    if stage == "onnx":
        return ROOT / "onnx" / "hardware" / f"{run_name}.onnx"
    return ROOT / "__never__"


def run_step(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n\n$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return {
        "returncode": result.returncode,
        "seconds": time.time() - started,
        "log": str(log_path.relative_to(ROOT)),
    }


def run_model_stage(config, stage, config_path, status, force=False, log_subdir="binary_benchmark"):
    run_name = config["run_name"]
    art = artifact_for(config, stage)
    key = f"{run_name}:{stage}"
    if art.exists() and not force:
        status[key] = {"status": "skipped_existing", "artifact": str(art.relative_to(ROOT))}
        return True
    cmd = command_for(config, stage, config_path)
    result = run_step(cmd, ROOT, ROOT / "logs" / log_subdir / f"{run_name}_{stage}.log")
    result["artifact"] = str(art.relative_to(ROOT))
    if result["returncode"] == 0 and art.exists():
        result["status"] = "ok"
        status[key] = result
        return True
    result["status"] = "failed"
    status[key] = result
    return False


def prepare_dense_hls(run_name, status, force=False, log_subdir="binary_benchmark"):
    project_dir = ROOT / "hls_projects" / run_name / "native"
    artifact = project_dir / "project.tcl"
    key = f"{run_name}:prepare_hls"
    if artifact.exists() and not force:
        status[key] = {"status": "skipped_existing", "artifact": str(project_dir.relative_to(ROOT))}
        return True
    cmd = [
        PYTHON,
        "-m",
        "hardware_benchmark.cli",
        "export-dense",
        "--run-name",
        run_name,
        "--output",
        str(project_dir),
        "--part",
        PART,
        "--clock",
        str(CLOCK_NS),
    ]
    result = run_step(cmd, ROOT, ROOT / "logs" / log_subdir / f"{run_name}_prepare_hls.log")
    ok = result["returncode"] == 0 and artifact.exists()
    result["status"] = "ok" if ok else "failed"
    result["artifact"] = str(project_dir.relative_to(ROOT))
    status[key] = result
    return ok


def prepare_keras_hls(config, status, force=False, log_subdir="binary_benchmark"):
    run_name = config["run_name"]
    project_dir = ROOT / "hls_projects" / run_name / "native"
    conversion = project_dir / "conversion.json"
    key = f"{run_name}:prepare_hls"
    if conversion.exists() and not force:
        status[key] = {"status": "skipped_existing", "artifact": str(project_dir.relative_to(ROOT))}
        return True
    cmd = [
        str(HLSENV310 if HLSENV310.exists() else Path(sys.executable)),
        "-m",
        "hardware_benchmark.keras_worker",
        "--backend",
        backend(config),
        "--run-name",
        run_name,
        "--weights",
        str(artifact_for(config, "train")),
        "--output",
        str(project_dir),
        "--root",
        str(ROOT),
        "--samples",
        "8192",
    ]
    result = run_step(cmd, ROOT, ROOT / "logs" / log_subdir / f"{run_name}_prepare_hls.log")
    ok = result["returncode"] == 0 and conversion.exists()
    result["status"] = "ok" if ok else "failed"
    result["artifact"] = str(project_dir.relative_to(ROOT))
    status[key] = result
    return ok


def synthesize_hls4ml(run_name, status, variant="hls4ml_latency_rf1", force=False, log_subdir="binary_benchmark"):
    result_json = ROOT / "results" / "synthesis" / f"{run_name}_{variant}" / "result.json"
    project_dir = ROOT / "hls_projects" / run_name / "native"
    key = f"{run_name}:synth_{variant}"
    if result_json.exists() and not force:
        status[key] = {"status": "skipped_existing", "artifact": str(result_json.relative_to(ROOT))}
        return True
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "synthesize_hls4ml_project.py"),
        "--run-name",
        run_name,
        "--project-dir",
        str(project_dir),
        "--variant",
        variant,
        "--allow-unverified-license",
    ]
    result = run_step(cmd, ROOT, ROOT / "logs" / log_subdir / f"{run_name}_synth_{variant}.log")
    ok = result["returncode"] == 0 and result_json.exists()
    result["status"] = "ok" if ok else "failed"
    result["artifact"] = str(result_json.relative_to(ROOT))
    status[key] = result
    if ok:
        shutil.rmtree(project_dir, ignore_errors=True)
    return ok


def synthesize_bitnet(run_name, status, force=False, log_subdir="binary_benchmark"):
    variants = []
    if "bit158" not in run_name and "topo" not in run_name:
        variants.append(
            (
                "bitnet_hls4ml_v26_sigmoid",
                [PYTHON, str(ROOT / "scripts" / "synthesize_bitnet_hls4ml_v26_sigmoid.py"), "--run-name", run_name, "--keep-project"],
                ROOT / "results" / "synthesis" / f"{run_name}_hls4ml_v26_sigmoid" / "result.json",
            )
        )
    variants.extend([
        (
            "bitnet_custom_v9_sigmoid",
            [PYTHON, str(ROOT / "scripts" / "synthesize_bitnet_custom_v9_sigmoid.py"), "--run-name", run_name, "--output-mode", "sigmoid"],
            ROOT / "results" / "synthesis" / f"{run_name}_custom_v9_sigmoid" / "result.json",
        ),
        (
            "bitnet_custom_v9_logits",
            [PYTHON, str(ROOT / "scripts" / "synthesize_bitnet_custom_v9_sigmoid.py"), "--run-name", run_name, "--output-mode", "logits"],
            ROOT / "results" / "synthesis" / f"{run_name}_custom_v9_logits" / "result.json",
        ),
    ])
    ok_any = False
    for variant, cmd, result_json in variants:
        key = f"{run_name}:synth_{variant}"
        if result_json.exists() and not force:
            status[key] = {"status": "skipped_existing", "artifact": str(result_json.relative_to(ROOT))}
            ok_any = True
            continue
        result = run_step(cmd, ROOT, ROOT / "logs" / log_subdir / f"{run_name}_synth_{variant}.log")
        ok = result["returncode"] == 0 and result_json.exists()
        result["status"] = "ok" if ok else "failed"
        result["artifact"] = str(result_json.relative_to(ROOT))
        status[key] = result
        ok_any = ok_any or ok
    return ok_any


def synthesize_xgboost(config, status, force=False, log_subdir="binary_benchmark"):
    run_name = config["run_name"]
    variants = [
        ("conifer_tree", "conifer", []),
        ("conifer_unrolled", "conifer_unrolled", ["--unroll"]),
    ]
    ok_any = False
    for variant, suffix, extra_args in variants:
        result_json = ROOT / "results" / "synthesis" / f"{run_name}_{suffix}" / "result.json"
        key = f"{run_name}:synth_{variant}"
        if result_json.exists() and not force:
            status[key] = {"status": "skipped_existing", "artifact": str(result_json.relative_to(ROOT))}
            ok_any = True
            continue
        cmd = [
            str(HLSENV if HLSENV.exists() else Path(sys.executable)),
            str(ROOT / "scripts" / "synthesize_xgboost_conifer.py"),
            "--config",
            str(config["_config_path"]),
            "--tool",
            "vitishls",
            "--clock-period",
            str(CLOCK_NS),
            "--part",
            PART,
            "--precision",
            "ap_fixed<18,8>",
            *extra_args,
        ]
        result = run_step(cmd, ROOT, ROOT / "logs" / log_subdir / f"{run_name}_synth_{variant}.log")
        ok = result["returncode"] == 0 and result_json.exists()
        result["status"] = "ok" if ok else "failed"
        result["artifact"] = str(result_json.relative_to(ROOT))
        status[key] = result
        ok_any = ok_any or ok
        if ok:
            output_dir = ROOT / "hls_projects" / run_name / suffix
            if suffix == "conifer":
                shutil.rmtree(output_dir, ignore_errors=True)
    return ok_any


def hardware_for_config(config, status, force=False, log_subdir="binary_benchmark"):
    run_name = config["run_name"]
    b = backend(config)
    if b in {"pytorch", "qkeras", "hgq"} and config["model"]["name"] not in {
        "bitnet_mlp",
        "bit158_mlp",
    }:
        result_json = (
            ROOT
            / "results"
            / "synthesis"
            / f"{run_name}_hls4ml_latency_rf1"
            / "result.json"
        )
        if result_json.exists() and not force:
            status[f"{run_name}:hardware"] = {
                "status": "skipped_existing",
                "artifact": str(result_json.relative_to(ROOT)),
            }
            return True
    if b == "xgboost":
        result_json = ROOT / "results" / "synthesis" / f"{run_name}_conifer_unrolled" / "result.json"
        if result_json.exists() and not force:
            status[f"{run_name}:hardware"] = {
                "status": "skipped_existing",
                "artifact": str(result_json.relative_to(ROOT)),
            }
            return True
    if b == "xgboost":
        return synthesize_xgboost(config, status, force=force, log_subdir=log_subdir)
    if config["model"]["name"] == "bitnet_mlp":
        return synthesize_bitnet(run_name, status, force=force, log_subdir=log_subdir)
    if config["model"]["name"] == "bit158_mlp":
        return synthesize_bitnet(run_name, status, force=force, log_subdir=log_subdir)
    if b in {"qkeras", "hgq"}:
        if prepare_keras_hls(config, status, force=force, log_subdir=log_subdir):
            return synthesize_hls4ml(run_name, status, force=force, log_subdir=log_subdir)
        return False
    if b == "pytorch":
        if prepare_dense_hls(run_name, status, force=force, log_subdir=log_subdir):
            return synthesize_hls4ml(run_name, status, force=force, log_subdir=log_subdir)
        return False
    return False


def collect_rows(configs):
    rows = []
    for config in configs:
        run_name = config["run_name"]
        metric_path = artifact_for(config, "evaluate")
        metrics = _json(metric_path, {})
        synths = []
        for path in sorted((ROOT / "results" / "synthesis").glob(f"{run_name}_*/result.json")):
            values = _json(path, {})
            if values:
                synths.append(values)
        if synths:
            for synth in synths:
                rows.append(row_from(config, metrics, synth))
        else:
            rows.append(row_from(config, metrics, {}))
    return rows


def row_from(config, metrics, synth):
    run_name = config["run_name"]
    return {
        "run_name": run_name,
        "base_run_name": config["base_run_name"],
        "seed": config["seed"],
        "model_name": config["model"]["name"],
        "backend": backend(config),
        "hidden_dims": "-".join(map(str, config["model"].get("hidden_dims", []))),
        "accuracy": metrics.get("accuracy"),
        "macro_auc": metrics.get("macro_auc"),
        "cross_entropy": metrics.get("cross_entropy"),
        "parameter_count": metrics.get("parameter_count"),
        "cpu_latency_ms": metrics.get("cpu_latency_ms"),
        "synthesis_variant": synth.get("variant") or synth.get("implementation") or "",
        "implementation_boundary": synth.get("implementation_boundary", ""),
        "synth_status": synth.get("synthesis_status") or ("success" if synth.get("success") else ""),
        "clock_target_ns": synth.get("clock_target_ns"),
        "clock_achieved_ns": synth.get("clock_achieved_ns"),
        "latency_cycles": synth.get("latency_cycles"),
        "latency_at_target_ns": synth.get("latency_at_target_ns"),
        "ii_cycles": synth.get("initiation_interval_cycles"),
        "lut": synth.get("lut"),
        "ff": synth.get("ff"),
        "dsp": synth.get("dsp"),
        "bram_18k": synth.get("bram_18k"),
        "uram": synth.get("uram"),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_stats(rows):
    groups = {}
    for row in rows:
        if row["accuracy"] is None:
            continue
        key = row["base_run_name"]
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        acc = [float(v["accuracy"]) for v in values if v["accuracy"] is not None]
        auc = [float(v["macro_auc"]) for v in values if v["macro_auc"] is not None]
        output.append(
            {
                "base_run_name": key,
                "n_rows": len(values),
                "accuracy_mean": sum(acc) / len(acc) if acc else None,
                "accuracy_min": min(acc) if acc else None,
                "accuracy_max": max(acc) if acc else None,
                "auc_mean": sum(auc) / len(auc) if auc else None,
                "auc_min": min(auc) if auc else None,
                "auc_max": max(auc) if auc else None,
            }
        )
    return output


def write_report(rows, status, report_path: Path, class_mode: str):
    completed = [row for row in rows if row["accuracy"] is not None]
    hw = [row for row in rows if row["latency_cycles"] is not None]
    best_acc = max(completed, key=lambda r: float(r["accuracy"])) if completed else None
    best_auc = max(completed, key=lambda r: float(r["macro_auc"])) if completed else None
    best_latency = min(hw, key=lambda r: float(r["latency_cycles"])) if hw else None
    best_lut = min(hw, key=lambda r: float(r["lut"])) if hw and any(r["lut"] is not None for r in hw) else None
    failed = {key: value for key, value in status.items() if value.get("status") == "failed"}
    lines = [
        "# Binary Benchmark Report",
        "",
        f"Task: {_task_label(class_mode)} on OpenML 42468 HLF.",
        "All generated configs use split seed 42; model seeds are 42, 43 and 44.",
        "Hardware target: VU13P, 5 ns clock, C-synthesis only unless noted.",
        "",
        "## Best Current Rows",
    ]
    for label, row in (
        ("accuracy", best_acc),
        ("AUC", best_auc),
        ("latency", best_latency),
        ("LUT", best_lut),
    ):
        if row:
            lines.append(
                f"- Best by {label}: {row['run_name']} {row['synthesis_variant']} "
                f"acc={row['accuracy']} auc={row['macro_auc']} "
                f"latency={row['latency_cycles']} cycles LUT={row['lut']}"
            )
    lines.extend(["", "## Hardware Rows"])
    for row in hw:
        lines.append(
            f"- {row['run_name']} {row['synthesis_variant']}: "
            f"acc={row['accuracy']} auc={row['macro_auc']} "
            f"latency={row['latency_cycles']} cycles II={row['ii_cycles']} "
            f"LUT={row['lut']} FF={row['ff']} DSP={row['dsp']} BRAM18={row['bram_18k']}"
        )
    lines.extend(["", "## Failures"])
    if failed:
        for key, value in sorted(failed.items()):
            lines.append(f"- {key}: log={value.get('log')} returncode={value.get('returncode')}")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Preliminary BitNet Interpretation",
            "The binary track is designed to test the strongest BitNet case: a one-logit sigmoid model where cumulative alpha scaling can be folded into the final sigmoid/table implementation. Compare `bitnet_hls4ml_v26_sigmoid` and `bitnet_custom_v9_sigmoid` rows against dense, QKeras, HGQ and Conifer rows at similar accuracy. If BitNet is not Pareto competitive here, the benchmark conclusion should be that alpha folding alone is not enough for this HLF task and toolchain; if it is competitive only in the sigmoid endpoint, the advantage is endpoint-specific rather than a general multiclass/logits advantage.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed42-hardware-only", action="store_true")
    parser.add_argument("--class-mode", default="binary_qg_vs_wzt")
    parser.add_argument("--namespace", default="binary")
    parser.add_argument("--log-subdir", default="binary_benchmark")
    args = parser.parse_args()

    for rel in ("logs/run_configs", f"logs/{args.log_subdir}", "results/synthesis", "hls_projects"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    configs = model_configs(args.seeds, class_mode=args.class_mode, namespace=args.namespace)
    status_path = ROOT / "results" / f"{args.namespace}_benchmark_workflow_status.json"
    summary_path = ROOT / "results" / f"{args.namespace}_benchmark_summary.csv"
    seed_stats_path = ROOT / "results" / f"{args.namespace}_seed_statistics.csv"
    report_path = ROOT / "results" / f"{args.namespace}_benchmark_report.md"
    status = _json(status_path, {})

    for config in configs:
        config_path = ROOT / "logs" / "run_configs" / f"{config['run_name']}.json"
        config["_config_path"] = str(config_path)
        serializable = {key: value for key, value in config.items() if key != "_config_path"}
        _write_json(config_path, serializable)
        for stage in ("train", "evaluate"):
            if not run_model_stage(
                config, stage, config_path, status, force=args.force, log_subdir=args.log_subdir
            ):
                _write_json(status_path, status)
                break
        else:
            if backend(config) != "xgboost":
                run_model_stage(
                    config, "reference", config_path, status, force=args.force, log_subdir=args.log_subdir
                )
            if config["model"]["name"] in {"bitnet_mlp", "bit158_mlp"}:
                run_model_stage(
                    config, "onnx", config_path, status, force=args.force, log_subdir=args.log_subdir
                )
        _write_json(status_path, status)
        rows = collect_rows(configs)
        write_csv(summary_path, rows)
        write_csv(seed_stats_path, seed_stats(rows))
        write_report(rows, status, report_path, args.class_mode)

    for config in configs:
        if args.seed42_hardware_only and int(config["seed"]) != 42:
            continue
        if not artifact_for(config, "evaluate").exists():
            continue
        hardware_for_config(config, status, force=args.force, log_subdir=args.log_subdir)
        _write_json(status_path, status)
        rows = collect_rows(configs)
        write_csv(summary_path, rows)
        write_csv(seed_stats_path, seed_stats(rows))
        write_report(rows, status, report_path, args.class_mode)

    rows = collect_rows(configs)
    write_csv(summary_path, rows)
    write_csv(seed_stats_path, seed_stats(rows))
    write_report(rows, status, report_path, args.class_mode)


if __name__ == "__main__":
    main()
