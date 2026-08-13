import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_benchmark.reports import parse_csynth_xml


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def hardware_target(config):
    target = config.get("target", config)
    return {
        "part": target["part"],
        "clock_period": target.get("clock_period", target.get("clock_period_ns")),
    }


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_csynth_xml(output_dir: Path, project_name: str) -> Path | None:
    report_dir = output_dir / project_name / "solution1" / "syn" / "report"
    candidates = []
    if report_dir.exists():
        top_candidates = [
            report_dir / "csynth.xml",
            report_dir / f"{project_name}_csynth.xml",
            report_dir / f"{project_name.replace('__', '_')}_csynth.xml",
        ]
        for path in top_candidates:
            if path.exists():
                return path
        candidates.extend(report_dir.glob("*_csynth.xml"))
        candidates.extend(report_dir.glob("csynth.xml"))
    if not candidates:
        candidates = list(output_dir.glob("**/*_csynth.xml")) + list(
            output_dir.glob("**/csynth.xml")
        )
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int]:
        name = path.stem
        if name == "csynth":
            return (4, len(name))
        if name == f"{project_name}_csynth" or name == f"{project_name.replace('__', '_')}_csynth":
            return (4, len(name))
        if "Pipeline" in name or "decision_function" in name:
            return (1, len(name))
        if "reduce" in name or "tree_scores" in name:
            return (0, len(name))
        return (2, len(name))

    return sorted(candidates, key=score, reverse=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Synthesize an XGBoost BDT with Conifer.")
    parser.add_argument("--config", required=True, help="Path to the trained XGBoost run config.")
    parser.add_argument("--hardware-config", default="configs/hardware_benchmark.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--precision", default="ap_fixed<18,8>")
    parser.add_argument("--clock-period", type=float, default=None)
    parser.add_argument("--part", default=None)
    parser.add_argument("--tool", choices=("vivadohls", "vitishls"), default="vivadohls")
    parser.add_argument("--unroll", action="store_true", help="Use unrolled-tree implementation.")
    parser.add_argument(
        "--model-run-name",
        default=None,
        help="Use a different run name for locating the trained model checkpoint.",
    )
    parser.add_argument("--vsynth", action="store_true", help="Run downstream Vivado synthesis after HLS csynth.")
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    hw = load_json(ROOT / args.hardware_config)
    target = hardware_target(hw)
    run_name = args.run_name or config["run_name"]
    model_run_name = args.model_run_name or config["run_name"]
    variant_suffix = "conifer_unrolled" if args.unroll else "conifer"
    project_name = args.project_name or f"{run_name}_{variant_suffix}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "hls_projects" / run_name / variant_suffix
    )
    model_path = ROOT / "models" / f"{model_run_name}.pkl"
    result_dir = ROOT / "results" / "synthesis" / f"{run_name}_{variant_suffix}"
    ensure_clean_dir(result_dir)
    ensure_clean_dir(output_dir)

    payload = load_pickle(model_path)
    xgb_model = payload["model"]

    import xgboost
    import conifer
    import conifer.converters.xgboost as conifer_xgb
    from conifer.model import make_model

    conversion_started = time.time()
    ensemble = conifer_xgb.convert(xgb_model)
    conversion_seconds = time.time() - conversion_started

    conifer_config = conifer.backends.xilinxhls.auto_config(granularity="full")
    conifer_config["ProjectName"] = project_name
    conifer_config["OutputDir"] = str(output_dir)
    conifer_config["XilinxPart"] = args.part or target["part"]
    conifer_config["ClockPeriod"] = args.clock_period or target["clock_period"]
    conifer_config["Unroll"] = bool(args.unroll)
    conifer_config["InputPrecision"] = args.precision
    conifer_config["ThresholdPrecision"] = args.precision
    conifer_config["ScorePrecision"] = args.precision

    model = make_model(ensemble, conifer_config)
    model.write()
    build_started = time.time()
    success = model.build(reset=True, synth=True, vsynth=args.vsynth)
    build_seconds = time.time() - build_started
    report = model.read_report()
    csynth_xml = find_csynth_xml(output_dir, project_name)
    parsed_report = parse_csynth_xml(csynth_xml) if csynth_xml else {}

    artifact_paths = []
    artifact_candidates = [
        Path("vivado_synth.rpt"),
        Path("build.log"),
        Path("vivado_build.log"),
        Path("conifer.json"),
    ]
    if csynth_xml:
        artifact_candidates.append(csynth_xml.relative_to(output_dir))
        rpt = csynth_xml.with_suffix(".rpt")
        if rpt.exists():
            artifact_candidates.append(rpt.relative_to(output_dir))
    for rel in artifact_candidates:
        path = output_dir / rel
        if path.exists():
            target = result_dir / path.name
            shutil.copy2(path, target)
            artifact_paths.append(str(target.relative_to(ROOT)))

    result = {
        "run_name": run_name,
        "variant": "conifer_unrolled" if args.unroll else "conifer_tree",
        "tool": args.tool,
        "success": bool(success),
        "project_name": project_name,
        "project_dir": str(output_dir.relative_to(ROOT)),
        "model_path": str(model_path.relative_to(ROOT)),
        "xgboost_version": xgboost.__version__,
        "conifer_version": str(conifer.__version__),
        "conversion_seconds": conversion_seconds,
        "build_seconds": build_seconds,
        "clock_period_ns": conifer_config["ClockPeriod"],
        "part": conifer_config["XilinxPart"],
        "precision": args.precision,
        "unroll": bool(args.unroll),
        "vsynth_requested": bool(args.vsynth),
        "report": report,
        "parsed_report": parsed_report,
        "clock_target_ns": parsed_report.get("clock_target_ns"),
        "clock_achieved_ns": parsed_report.get("clock_achieved_ns"),
        "latency_cycles": parsed_report.get("latency_cycles_max"),
        "latency_cycles_min": parsed_report.get("latency_cycles_min"),
        "latency_at_target_ns": (
            parsed_report.get("latency_cycles_max") * parsed_report.get("clock_target_ns")
            if parsed_report.get("latency_cycles_max") is not None
            and parsed_report.get("clock_target_ns") is not None
            else None
        ),
        "initiation_interval_cycles": parsed_report.get("initiation_interval_cycles"),
        "lut": parsed_report.get("lut"),
        "ff": parsed_report.get("ff"),
        "dsp": parsed_report.get("dsp"),
        "bram_18k": parsed_report.get("bram"),
        "uram": parsed_report.get("uram"),
        "artifacts": artifact_paths,
        "notes": [
            "Conifer warns that xgboost >= 2.0.0 is not fully supported.",
            "Latency/resource values come from Conifer HLS report extraction.",
        ],
    }
    with open(result_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
