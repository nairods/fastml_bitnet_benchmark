#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_benchmark.preflight import run_preflight
from hardware_benchmark.reports import parse_csynth_xml


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _find_report(project_dir: Path, project_name: str) -> Path:
    expected = (
        project_dir
        / f"{project_name}_prj"
        / "solution1"
        / "syn"
        / "report"
        / f"{project_name}_csynth.xml"
    )
    if expected.exists():
        return expected
    reports = sorted(
        path
        for path in project_dir.glob("**/syn/report/*_csynth.xml")
        if path.name != "csynth_design_size.xml"
    )
    if not reports:
        raise FileNotFoundError(f"No csynth XML report under {project_dir}")
    top_reports = [path for path in reports if path.stem == f"{project_name}_csynth"]
    return top_reports[0] if top_reports else reports[0]


def synthesize(
    run_name: str,
    project_dir: Path,
    variant: str,
    profile: str | None,
    allow_unverified_license: bool,
) -> dict:
    preflight = run_preflight()
    vitis = preflight["vitis_hls"]
    if not vitis["available"]:
        raise RuntimeError("Vitis HLS is unavailable")
    if not preflight["license"]["available"] and not allow_unverified_license:
        raise RuntimeError("Xilinx license availability is unverified")

    project_tcl = project_dir / "project.tcl"
    if not project_tcl.exists():
        raise FileNotFoundError(f"Missing project.tcl: {project_tcl}")
    project_name = "jet_classifier"
    for line in project_tcl.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("set project_name"):
            project_name = line.split()[-1].strip('"')
            break

    source_cpp = project_dir / "firmware" / f"{project_name}.cpp"
    if not source_cpp.exists():
        raise FileNotFoundError(f"Missing firmware source: {source_cpp}")
    testbench = project_dir / f"{project_name}_test.cpp"
    clock_period = "5.0"
    clock_uncertainty = "12.5%"
    part = "xcvu13p-flga2577-2-e"
    for line in project_tcl.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("set clock_period"):
            clock_period = stripped.split()[-1]
        elif stripped.startswith("set clock_uncertainty"):
            clock_uncertainty = stripped.split()[-1]
        elif stripped.startswith("set part"):
            part = stripped.split()[-1].strip('"')

    compat_tcl = project_dir / f"csynth_{variant}.tcl"
    tb_lines = []
    if testbench.exists():
        tb_lines.append(f'add_files -tb {testbench.name} -cflags "-std=c++0x"')
    if (project_dir / "firmware" / "weights").exists():
        tb_lines.append("add_files -tb firmware/weights")
    if (project_dir / "tb_data").exists():
        tb_lines.append("add_files -tb tb_data")
    compat_tcl.write_text(
        "\n".join(
            [
                f"open_project -reset {project_name}_prj",
                f"set_top {project_name}",
                f'add_files firmware/{project_name}.cpp -cflags "-std=c++0x"',
                *tb_lines,
                'open_solution -reset "solution1"',
                "config_compile -name_max_length 80",
                "config_schedule -enable_dsp_full_reg=false",
                f"set_part {part}",
                f"create_clock -period {clock_period} -name default",
                f"set_clock_uncertainty {clock_uncertainty} default",
                "csynth_design",
                "exit",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = str(Path(vitis["path"]).parent) + os.pathsep + env.get("PATH", "")
    log_path = project_dir / f"synthesis_{variant}.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                vitis["path"],
                "-f",
                str(compat_tcl.name),
            ],
            cwd=project_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    status = {
        "run_name": run_name,
        "variant": variant,
        "project_dir": str(project_dir.relative_to(ROOT) if project_dir.is_relative_to(ROOT) else project_dir),
        "returncode": completed.returncode,
        "log": str(log_path.relative_to(ROOT) if log_path.is_relative_to(ROOT) else log_path),
    }
    if completed.returncode:
        result_dir = ROOT / "results" / profile / "synthesis" / f"{run_name}_{variant}"
        result = {
            **status,
            "synthesis_status": "failed",
            "place_and_route_status": "not_run",
            "tool": "Vitis HLS",
            "tool_version": preflight["vitis_hls"].get("version"),
        }
        _write_json(result_dir / "result.json", result)
        raise RuntimeError(f"Synthesis failed for {run_name}; see {log_path}")

    xml = _find_report(project_dir, project_name)
    report = parse_csynth_xml(xml)
    result_dir = ROOT / "results" / profile / "synthesis" / f"{run_name}_{variant}"
    result_dir.mkdir(parents=True, exist_ok=True)
    copied_xml = result_dir / f"{run_name}_{variant}_csynth.xml"
    shutil.copy2(xml, copied_xml)
    rpt = xml.with_suffix(".rpt")
    copied_rpt = None
    if rpt.exists():
        copied_rpt = result_dir / f"{run_name}_{variant}_csynth.rpt"
        shutil.copy2(rpt, copied_rpt)

    run_config_path = ROOT / "logs" / profile / "run_configs" / f"{run_name}.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.exists()
        else {}
    )
    model_config = run_config.get("model", {})
    binary_sigmoid = (
        int(model_config.get("output_dim", 5)) == 1
        and model_config.get("output_mode") == "binary_sigmoid"
    )
    result = {
        **status,
        "tool": "Vitis HLS",
        "tool_version": report.get("tool_version") or preflight["vitis_hls"].get("version"),
        "part": report.get("part"),
        "clock_target_ns": report.get("clock_target_ns"),
        "clock_achieved_ns": report.get("clock_achieved_ns"),
        "latency_cycles": report.get("latency_cycles_max"),
        "latency_cycles_min": report.get("latency_cycles_min"),
        "latency_at_target_ns": (
            report.get("latency_cycles_max") * report.get("clock_target_ns")
            if report.get("latency_cycles_max") is not None
            and report.get("clock_target_ns") is not None
            else None
        ),
        "initiation_interval_cycles": report.get("initiation_interval_cycles"),
        "lut": report.get("lut"),
        "ff": report.get("ff"),
        "dsp": report.get("dsp"),
        "bram_18k": report.get("bram"),
        "uram": report.get("uram"),
        "synthesis_status": "success",
        "place_and_route_status": "not_run",
        "implementation_boundary": (
            "binary sigmoid output; no softmax"
            if binary_sigmoid
            else "multiclass logits path; no softmax"
        ),
        "report_files": {
            "xml": str(copied_xml.relative_to(ROOT)),
            "rpt": str(copied_rpt.relative_to(ROOT)) if copied_rpt else None,
        },
        "notes": [
            "Stock hls4ml generated project.",
            "C-synthesis only; no cosimulation or place-and-route.",
            (
                "Binary sigmoid endpoint included in synthesized top."
                if binary_sigmoid
                else "Multiclass logits path; no softmax included in synthesized top."
            ),
        ],
    }
    _write_json(result_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--variant", default="hls4ml_latency_rf1")
    parser.add_argument("--profile", choices=("20-epochs", "200-epochs"), required=True)
    parser.add_argument("--allow-unverified-license", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            synthesize(
                args.run_name,
                args.project_dir,
                args.variant,
                args.profile,
                args.allow_unverified_license,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
