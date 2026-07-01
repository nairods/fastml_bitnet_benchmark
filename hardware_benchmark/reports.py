import xml.etree.ElementTree as ET
from pathlib import Path


def _text(root, path, cast=None):
    node = root.find(path)
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return cast(value) if cast else value


def parse_csynth_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    timing = "./PerformanceEstimates/SummaryOfTimingAnalysis"
    latency = "./PerformanceEstimates/SummaryOfOverallLatency"
    resources = "./AreaEstimates/Resources"
    result = {
        "tool_version": _text(root, "./ReportVersion/Version"),
        "part": _text(root, "./UserAssignments/Part"),
        "top": _text(root, "./UserAssignments/TopModelName"),
        "clock_target_ns": _text(
            root, "./UserAssignments/TargetClockPeriod", float
        ),
        "clock_achieved_ns": _text(
            root, f"{timing}/EstimatedClockPeriod", float
        ),
        "latency_cycles_min": _text(
            root, f"{latency}/Best-caseLatency", int
        ),
        "latency_cycles_max": _text(
            root, f"{latency}/Worst-caseLatency", int
        ),
        "initiation_interval_cycles": _text(
            root, f"{latency}/Interval-min", int
        ),
        "bram": _text(root, f"{resources}/BRAM_18K", int),
        "ff": _text(root, f"{resources}/FF", int),
        "lut": _text(root, f"{resources}/LUT", int),
        "dsp": _text(root, f"{resources}/DSP", int),
        "uram": _text(root, f"{resources}/URAM", int),
        "report": str(path),
    }
    if result["clock_achieved_ns"] and result["initiation_interval_cycles"]:
        result["throughput_inferences_per_second"] = 1e9 / (
            result["clock_achieved_ns"] * result["initiation_interval_cycles"]
        )
    else:
        result["throughput_inferences_per_second"] = None
    return result
