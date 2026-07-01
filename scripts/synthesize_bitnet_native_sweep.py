#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_benchmark.bitnet import (
    fold_cumulative_alpha,
    load_quantized_layers,
    predict_exact,
    predict_v26_style,
)
from hardware_benchmark.metrics import compare_predictions
from hardware_benchmark.preflight import run_preflight
from hardware_benchmark.reports import parse_csynth_xml


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("/tmp/bitnet_native_sweep")


@dataclass(frozen=True)
class TypeSpec:
    input_t: str
    hidden_t: str
    accum_t: str
    output_t: str
    scale_t: str


FAMILY_SPECS = {
    "binary": TypeSpec(
        input_t="ap_fixed<14, 4, AP_RND, AP_SAT>",
        hidden_t="ap_ufixed<15, 5, AP_RND, AP_SAT>",
        accum_t="ap_fixed<17, 7, AP_RND, AP_SAT>",
        output_t="ap_fixed<16, 6, AP_RND, AP_SAT>",
        scale_t="ap_fixed<32, 4, AP_RND, AP_SAT>",
    ),
    "ternary": TypeSpec(
        input_t="ap_fixed<14, 4, AP_RND, AP_SAT>",
        hidden_t="ap_ufixed<14, 4, AP_RND, AP_SAT>",
        accum_t="ap_fixed<16, 6, AP_RND, AP_SAT>",
        output_t="ap_fixed<15, 5, AP_RND, AP_SAT>",
        scale_t="ap_fixed<32, 4, AP_RND, AP_SAT>",
    ),
}


MODEL_FAMILIES = {
    "bitnet_mlp_f5_fixed": "binary",
    "bitnet_mlp_f7_fixed": "binary",
    "bitnet_mlp_f8_fixed": "binary",
    "bitnet_mlp_f10_fixed": "binary",
    "bitnet_mlp_f12_fixed": "binary",
    "bitnet_mlp_f7_power2": "binary",
    "bitnet_binary_f7_fixed": "binary",
    "bitnet_topo_f7_fixed": "binary",
    "bitnet_topo_f7_fixed__seed42": "binary",
    "binary_448_224_224__seed42": "binary",
    "bit158_mlp_f7_fixed": "ternary",
    "bit158_mlp_f7_fixed__seed42": "ternary",
    "bit158_topo_f7_fixed": "ternary",
    "bit158_topo_f7_fixed__seed42": "ternary",
    "ternary_128_64_64_64__seed42": "ternary",
}


def _cpp_values(values, formatter=str, columns=16):
    flat = list(np.asarray(values).reshape(-1))
    lines = []
    for start in range(0, len(flat), columns):
        lines.append("    " + ", ".join(formatter(value) for value in flat[start : start + columns]))
    return ",\n".join(lines)


def _emit_custom_dense_simple(
    function_name: str,
    n_in: int,
    n_out: int,
    relu: bool,
    final_scale: bool,
    input_type: str,
    output_type: str,
    accum_type: str,
) -> str:
    lines = [
        f"static void {function_name}(",
        f"    const {input_type} input[{n_in}],",
        f"    {output_type} output[{n_out}],",
        f"    const weight_t weights[{n_in} * {n_out}],",
        f"    const {accum_type} biases[{n_out}]",
    ]
    if final_scale:
        lines.append("    , const scale_t final_scale")
    lines.extend(
        [
            ") {",
            "#pragma HLS INLINE off",
            "#pragma HLS PIPELINE II=1",
            f"    {accum_type} accumulators[{n_out}];",
            "#pragma HLS ARRAY_PARTITION variable=accumulators complete",
            "",
            "init_accumulators:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            "        accumulators[output_index] = biases[output_index];",
            "    }",
            "",
            "accumulate_inputs:",
            f"    for (int input_index = 0; input_index < {n_in}; input_index++) {{",
            "    accumulate_outputs:",
            f"        for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"            const int weight_index = input_index * {n_out} + output_index;",
            "            const weight_t weight = weights[weight_index];",
            "            if (weight > 0) {",
            "                accumulators[output_index] += input[input_index];",
            "            } else if (weight < 0) {",
            "                accumulators[output_index] -= input[input_index];",
            "            }",
            "        }",
            "    }",
            "",
            "write_outputs:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"        {accum_type} value = accumulators[output_index];",
        ]
    )
    if final_scale:
        lines.append("        value = value * final_scale;")
    if relu:
        lines.extend(
            [
                f"        output[output_index] = value > 0 ? static_cast<{output_type}>(value) : static_cast<{output_type}>(0);",
            ]
        )
    else:
        lines.extend([f"        output[output_index] = static_cast<{output_type}>(value);"])
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def _emit_custom_dense_tree(
    function_name: str,
    n_in: int,
    n_out: int,
    relu: bool,
    final_scale: bool,
    input_type: str,
    output_type: str,
    accum_type: str,
) -> str:
    stage_arrays = []
    current_terms = n_in
    stage_index = 0
    while current_terms > 4:
        next_terms = current_terms // 2
        stage_arrays.append((stage_index, current_terms, next_terms))
        current_terms = next_terms
        stage_index += 1

    lines = [
        f"static void {function_name}(",
        f"    const {input_type} input[{n_in}],",
        f"    {output_type} output[{n_out}],",
        f"    const weight_t weights[{n_in} * {n_out}],",
        f"    const {accum_type} biases[{n_out}]",
    ]
    if final_scale:
        lines.append("    , const scale_t final_scale")
    lines.extend(
        [
            ") {",
            "#pragma HLS INLINE off",
            "#pragma HLS PIPELINE II=1",
            f"    {accum_type} products[{n_in} * {n_out}];",
            f"    {accum_type} accumulators[{n_out}];",
            "#pragma HLS ARRAY_PARTITION variable=accumulators complete",
            "",
            "make_products:",
            f"    for (int input_index = 0; input_index < {n_in}; input_index++) {{",
            "    make_output_products:",
            f"        for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"            const int weight_index = input_index * {n_out} + output_index;",
            "            const weight_t weight = weights[weight_index];",
            "            if (weight > 0) {",
            "                products[weight_index] = input[input_index];",
            "            } else if (weight < 0) {",
            "                products[weight_index] = -input[input_index];",
            "            } else {",
            f"                products[weight_index] = {accum_type}(0);",
            "            }",
            "        }",
            "    }",
        ]
    )

    current_array = "products"
    for stage_index, _, next_terms in stage_arrays:
        stage_name = f"stage_{stage_index}"
        lines.extend(
            [
                f"    {accum_type} {stage_name}[{next_terms} * {n_out}];",
                "",
                f"pair_reduce_{stage_index}:",
                f"    for (int pair_index = 0; pair_index < {next_terms}; pair_index++) {{",
                f"    pair_reduce_{stage_index}_outputs:",
                f"        for (int output_index = 0; output_index < {n_out}; output_index++) {{",
                f"            const int first_index = (2 * pair_index) * {n_out} + output_index;",
                f"            const int second_index = first_index + {n_out};",
                f"            {stage_name}[pair_index * {n_out} + output_index] = {current_array}[first_index] + {current_array}[second_index];",
                "        }",
                "    }",
            ]
        )
        current_array = stage_name

    lines.extend(
        [
            "",
            "init_accumulators:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"        accumulators[output_index] = biases[output_index];",
            "    }",
            "",
            "accumulate_final:",
            f"    for (int partial_index = 0; partial_index < {current_terms}; partial_index++) {{",
            f"    accumulate_final_outputs:",
            f"        for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"            accumulators[output_index] += {current_array}[partial_index * {n_out} + output_index];",
            "        }",
            "    }",
            "",
            "write_outputs:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"        {accum_type} value = accumulators[output_index];",
        ]
    )
    if final_scale:
        lines.append("        value = value * final_scale;")
    if relu:
        lines.extend(
            [
                f"        output[output_index] = value > 0 ? static_cast<{output_type}>(value) : static_cast<{output_type}>(0);",
            ]
        )
    else:
        lines.extend([f"        output[output_index] = static_cast<{output_type}>(value);"])
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def _emit_custom_dense_unrolled(
    function_name: str,
    n_in: int,
    n_out: int,
    relu: bool,
    final_scale: bool,
    input_type: str,
    output_type: str,
    accum_type: str,
) -> str:
    lines = [
        f"static void {function_name}(",
        f"    const {input_type} input[{n_in}],",
        f"    {output_type} output[{n_out}],",
        f"    const weight_t weights[{n_in} * {n_out}],",
        f"    const {accum_type} biases[{n_out}]",
    ]
    if final_scale:
        lines.append("    , const scale_t final_scale")
    lines.extend(
        [
            ") {",
            "#pragma HLS INLINE off",
            "#pragma HLS PIPELINE II=1",
            f"    {accum_type} accumulators[{n_out}];",
            "#pragma HLS ARRAY_PARTITION variable=accumulators complete",
            "",
            "init_accumulators:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            "#pragma HLS UNROLL",
            "        accumulators[output_index] = biases[output_index];",
            "    }",
            "",
            "accumulate_inputs:",
            f"    for (int input_index = 0; input_index < {n_in}; input_index++) {{",
            "#pragma HLS UNROLL",
            "        const {input_type} value = input[input_index];".format(input_type=input_type),
            "accumulate_outputs:",
            f"        for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            "#pragma HLS UNROLL",
            f"            const int weight_index = input_index * {n_out} + output_index;",
            "            const weight_t weight = weights[weight_index];",
            "            if (weight > 0) {",
            "                accumulators[output_index] += value;",
            "            } else if (weight < 0) {",
            "                accumulators[output_index] -= value;",
            "            }",
            "        }",
            "    }",
            "",
            "write_outputs:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            "#pragma HLS UNROLL",
            f"        {accum_type} value = accumulators[output_index];",
        ]
    )
    if final_scale:
        lines.append("        value = value * final_scale;")
    if relu:
        lines.extend(
            [
                f"        output[output_index] = value > 0 ? static_cast<{output_type}>(value) : static_cast<{output_type}>(0);",
            ]
        )
    else:
        lines.extend([f"        output[output_index] = static_cast<{output_type}>(value);"])
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def _balanced_sum_lines(terms: list[str], accum_type: str, prefix: str) -> tuple[list[str], str]:
    if not terms:
        return [], f"{accum_type}(0)"

    lines: list[str] = []
    level = list(terms)
    temp_index = 0
    while len(level) > 1:
        next_level: list[str] = []
        for index in range(0, len(level), 2):
            if index + 1 < len(level):
                temp_name = f"{prefix}_{temp_index}"
                lines.append(f"        const {accum_type} {temp_name} = {level[index]} + {level[index + 1]};")
                next_level.append(temp_name)
                temp_index += 1
            else:
                next_level.append(level[index])
        level = next_level
    return lines, level[0]


def _emit_custom_sparse_layer(
    function_name: str,
    layer_weights: np.ndarray,
    relu: bool,
    final_scale: bool,
    input_type: str,
    output_type: str,
    accum_type: str,
) -> str:
    weights = np.asarray(layer_weights, dtype=np.int8)
    n_out, n_in = weights.shape
    lines = [
        f"static void {function_name}(",
        f"    const {input_type} input[{n_in}],",
        f"    {output_type} output[{n_out}],",
        f"    const {accum_type} biases[{n_out}]",
    ]
    if final_scale:
        lines.append("    , const scale_t final_scale")
    lines.extend(
        [
            ") {",
            "#pragma HLS INLINE off",
            "#pragma HLS PIPELINE II=1",
            f"    {accum_type} accumulators[{n_out}];",
            "#pragma HLS ARRAY_PARTITION variable=accumulators complete",
            "",
            "init_accumulators:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            "        accumulators[output_index] = biases[output_index];",
            "    }",
            "",
        ]
    )
    for output_index, row in enumerate(weights):
        terms = [
            f"input[{input_index}]"
            if int(weight) > 0
            else f"-input[{input_index}]"
            for input_index, weight in enumerate(row)
            if int(weight) != 0
        ]
        if not terms:
            continue
        sum_lines, sum_name = _balanced_sum_lines(terms, accum_type, f"{function_name}_{output_index}_sum")
        lines.extend(
            [
                f"    // output {output_index}",
                *sum_lines,
                f"        accumulators[{output_index}] += {sum_name};",
                "",
            ]
        )
    lines.extend(
        [
            "write_outputs:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"        {accum_type} value = accumulators[output_index];",
        ]
    )
    if final_scale:
        lines.append("        value = value * final_scale;")
    if relu:
        lines.extend(
            [
                f"        output[output_index] = value > 0 ? static_cast<{output_type}>(value) : static_cast<{output_type}>(0);",
            ]
        )
    else:
        lines.extend([f"        output[output_index] = static_cast<{output_type}>(value);"])
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def export_custom_native_project(
    layers: list[dict],
    output_dir: Path,
    project_name: str,
    part: str,
    clock_period: float,
    typespec: TypeSpec,
    mode: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    folded = fold_cumulative_alpha(layers)
    dimensions = [folded[0]["weight"].shape[1]]
    dimensions.extend(layer["weight"].shape[0] for layer in folded)
    final_scale = folded[-1]["cumulative_scale"]
    parameter_blocks = []
    ternary_sparse = typespec == FAMILY_SPECS["ternary"]
    for index, layer in enumerate(folded):
        if not ternary_sparse:
            parameter_blocks.append(
                f"static const weight_t W{index}[{layer['weight'].size}] = {{\n"
                f"{_cpp_values(layer['weight'], lambda value: str(int(value)))}\n}};"
            )
        parameter_blocks.append(
            f"static const {typespec.accum_t} B{index}[{len(layer['bias_folded'])}] = {{\n"
            f"{_cpp_values(layer['bias_folded'], lambda value: f'{float(value):.12g}')}\n}};"
        )

    layer_functions = []
    calls = []
    for index, layer in enumerate(folded):
        n_in = layer["weight"].shape[1]
        n_out = layer["weight"].shape[0]
        relu = index != len(folded) - 1
        function_name = f"bitnet_dense_custom_{index}"
        if ternary_sparse:
            emitter = _emit_custom_sparse_layer
        elif mode == "tree":
            emitter = _emit_custom_dense_tree
        elif mode == "unrolled":
            emitter = _emit_custom_dense_unrolled
        else:
            emitter = _emit_custom_dense_simple
        if ternary_sparse:
            layer_functions.append(
                emitter(
                    function_name,
                    layer["weight"],
                    relu=relu,
                    final_scale=not relu,
                    input_type=typespec.input_t if index == 0 else typespec.hidden_t,
                    output_type=typespec.hidden_t if relu else typespec.output_t,
                    accum_type=typespec.accum_t,
                )
            )
        else:
            layer_functions.append(
                emitter(
                    function_name,
                    n_in,
                    n_out,
                    relu=relu,
                    final_scale=not relu,
                    input_type=typespec.input_t if index == 0 else typespec.hidden_t,
                    output_type=typespec.hidden_t if relu else typespec.output_t,
                    accum_type=typespec.accum_t,
                )
            )
        source = "input" if index == 0 else f"layer{index}"
        target = "output" if index == len(folded) - 1 else f"layer{index + 1}"
        call = f"    {function_name}({source}, {target}, B{index}"
        if not ternary_sparse:
            call = f"    {function_name}({source}, {target}, W{index}, B{index}"
        if not relu:
            call += ", FINAL_SCALE"
        call += ");"
        calls.append(call)

    intermediates = "\n".join(
        f"    {typespec.hidden_t if index != len(dimensions) - 1 else typespec.output_t} layer{index}[{dimensions[index]}];\n"
        f"#pragma HLS ARRAY_PARTITION variable=layer{index} complete"
        for index in range(1, len(dimensions) - 1)
    )
    header = f"""#ifndef {project_name.upper()}_H_
#define {project_name.upper()}_H_
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_math.h"

typedef {typespec.input_t} data_t;
typedef {typespec.accum_t} accum_t;
typedef {typespec.scale_t} scale_t;
typedef ap_int<2> weight_t;
typedef {typespec.hidden_t} hidden_t;
typedef {typespec.output_t} output_t;

void {project_name}(data_t input[{dimensions[0]}], output_t output[{dimensions[-1]}]);
#endif
"""
    parameters = f"""#ifndef {project_name.upper()}_PARAMETERS_H_
#define {project_name.upper()}_PARAMETERS_H_
#include "{project_name}.h"
namespace parameters {{
{chr(10).join(parameter_blocks)}
static const scale_t FINAL_SCALE = {final_scale:.12g};
}}
#endif
"""
    source = f"""#include "{project_name}.h"
#include "{project_name}_parameters.h"
using namespace parameters;

{chr(10).join(layer_functions)}

void {project_name}(data_t input[{dimensions[0]}], output_t output[{dimensions[-1]}]) {{
#pragma HLS INTERFACE ap_vld port=input,output
#pragma HLS ARRAY_PARTITION variable=input complete
#pragma HLS ARRAY_PARTITION variable=output complete
#pragma HLS PIPELINE II=1
{intermediates}
{chr(10).join(calls)}
}}
"""
    tcl = f"""open_project -reset {project_name}_prj
set_top {project_name}
add_files firmware/{project_name}.cpp -cflags "-std=c++11"
open_solution -reset solution1
set_part {part}
create_clock -period {clock_period} -name default
set_clock_uncertainty 12.5% default
csynth_design
exit
"""
    firmware = output_dir / "firmware"
    firmware.mkdir(exist_ok=True)
    (firmware / f"{project_name}.h").write_text(header, encoding="utf-8")
    (firmware / f"{project_name}_parameters.h").write_text(parameters, encoding="utf-8")
    (firmware / f"{project_name}.cpp").write_text(source, encoding="utf-8")
    (output_dir / "run_hls.tcl").write_text(tcl, encoding="utf-8")
    metadata = {
        "project_name": project_name,
        "implementation": f"custom_native_{mode}",
        "dimensions": dimensions,
        "part": part,
        "clock_period_ns": clock_period,
        "final_scale": final_scale,
        "warning": "This exporter writes logits-only custom native HLS; no softmax is included.",
    }
    (output_dir / "project.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _synth_dir(root: Path, run_name: str) -> Path:
    return root / "results" / "synthesis" / f"{run_name}_custom_native"


def _project_dir(run_name: str) -> Path:
    return TMP_ROOT / f"{run_name}_custom_native"


def _select_report_pair(solution: Path, top: str):
    candidates = [
        f"{top}_csynth",
        f"{top.replace('__', '_')}_csynth",
        "csynth",
    ]
    for stem in candidates:
        rpt = solution / f"{stem}.rpt"
        xml = solution / f"{stem}.xml"
        if rpt.exists() and xml.exists():
            return rpt, xml

    for rpt in sorted(solution.glob("*_csynth.rpt")):
        if rpt.stem.startswith("bitnet_dense_custom_"):
            continue
        xml = solution / f"{rpt.stem}.xml"
        if xml.exists():
            return rpt, xml

    raise FileNotFoundError(f"No top-level csynth report pair found in {solution}")


def _copy_report_files(project_dir: Path, out_dir: Path, top: str):
    solution = project_dir / f"{project_dir.name}_prj" / "solution1" / "syn" / "report"
    rpt, xml = _select_report_pair(solution, top)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rpt, out_dir / rpt.name)
    shutil.copy2(xml, out_dir / xml.name)
    return out_dir / rpt.name, out_dir / xml.name


def _refresh_result(root: Path, run_name: str):
    project_dir = _project_dir(run_name)
    if not project_dir.exists():
        raise FileNotFoundError(f"Missing project directory for {run_name}: {project_dir}")

    metadata_path = project_dir / "project.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing project metadata for {run_name}: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    result_dir = _synth_dir(root, run_name)
    result_path = result_dir / "result.json"
    existing = {}
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))

    rpt, xml = _copy_report_files(project_dir, result_dir, metadata["project_name"])
    report = parse_csynth_xml(xml)

    clock_target_ns = existing.get("clock_target_ns", metadata.get("clock_period_ns"))
    result_json = {
        "run_name": run_name,
        "variant": existing.get("variant", "custom_native_tree1"),
        "mode": existing.get("mode", "tree"),
        "tool": "Vitis HLS",
        "tool_version": report.get("tool_version"),
        "part": report.get("part"),
        "clock_target_ns": clock_target_ns,
        "clock_achieved_ns": report.get("clock_achieved_ns"),
        "estimated_fmax_mhz": (
            1000.0 / report["clock_achieved_ns"] if report.get("clock_achieved_ns") else None
        ),
        "latency_cycles": report.get("latency_cycles_min"),
        "latency_at_target_ns": (
            report["latency_cycles_min"] * clock_target_ns
            if report.get("latency_cycles_min") is not None and clock_target_ns is not None
            else None
        ),
        "initiation_interval_cycles": report.get("initiation_interval_cycles"),
        "lut": report.get("lut"),
        "ff": report.get("ff"),
        "dsp": report.get("dsp"),
        "bram_18k": report.get("bram"),
        "uram": report.get("uram"),
        "reference_validation": existing.get("reference_validation"),
        "synthesis_status": "success",
        "place_and_route_status": existing.get("place_and_route_status", "not_run"),
        "approximate_model": existing.get("approximate_model", False),
        "notes": existing.get("notes", []),
        "report_files": {
            "rpt": str(rpt.relative_to(root)),
            "xml": str(xml.relative_to(root)),
        },
        "state": existing.get(
            "state", str((root / "onnx" / "hardware" / f"{run_name}_quantized.pt").relative_to(root))
        ),
        "folded_layers": existing.get("folded_layers"),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result_json, indent=2) + "\n", encoding="utf-8")
    return result_json


def _pick_types(run_name: str) -> TypeSpec:
    base_run_name = re.sub(r"__seed\d+$", "", run_name)
    family = MODEL_FAMILIES.get(run_name, MODEL_FAMILIES.get(base_run_name))
    if family is None:
        raise KeyError(run_name)
    return FAMILY_SPECS[family]


def _validate(run_name: str, reference: np.ndarray, labels: np.ndarray, logits: np.ndarray):
    return compare_predictions(logits, reference, labels)


def synthesize_run(
    root: Path, run_name: str, part: str, clock_period: float, samples: int, mode: str
):
    state_path = root / "onnx" / "hardware" / f"{run_name}_quantized.pt"
    layers = load_quantized_layers(state_path)
    inputs = np.load(root / "data" / "synthesis" / "x_test.npy", mmap_mode="r")
    labels = np.load(root / "data" / "synthesis" / "y_test.npy", mmap_mode="r")
    limit = min(samples, len(inputs))
    folded = fold_cumulative_alpha(layers)
    reference_path = root / "data" / "synthesis" / "reference_predictions" / f"{run_name}.npy"
    if reference_path.exists():
        reference = np.load(reference_path, mmap_mode="r")
        reference_source = str(reference_path.relative_to(root))
    else:
        reference = predict_exact(layers, inputs[:limit], batch_size=1024)
        reference_source = "exact_framework_prediction"
    logits = predict_v26_style(layers, inputs[:limit], batch_size=1024)
    validation = _validate(run_name, reference[:limit], labels[:limit], logits)

    project_dir = _project_dir(run_name)
    if project_dir.exists():
        shutil.rmtree(project_dir)
    typespec = _pick_types(run_name)
    metadata = export_custom_native_project(
        layers,
        project_dir,
        f"{run_name}_custom_native",
        part,
        clock_period,
        typespec,
        mode,
    )

    preflight = run_preflight(root)
    if not preflight["vitis_hls"]["available"]:
        raise RuntimeError("Vitis HLS is unavailable")
    if not preflight["license"]["available"]:
        print(f"WARNING: license availability is unverified for {run_name}")

    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(preflight["vitis_hls"]["path"]).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    log_path = project_dir / "synthesis.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [preflight["vitis_hls"]["path"], "-f", "run_hls.tcl"],
            cwd=project_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Synthesis failed for {run_name}; see {log_path}")

    result_dir = _synth_dir(root, run_name)
    rpt, xml = _copy_report_files(project_dir, result_dir, metadata["project_name"])
    report = parse_csynth_xml(xml)
    result_json = {
        "run_name": run_name,
        "variant": "custom_native_tree1",
        "mode": mode,
        "tool": "Vitis HLS",
        "tool_version": report.get("tool_version"),
        "part": report.get("part"),
        "clock_target_ns": clock_period,
        "clock_achieved_ns": report.get("clock_achieved_ns"),
        "estimated_fmax_mhz": (
            1000.0 / report["clock_achieved_ns"] if report.get("clock_achieved_ns") else None
        ),
        "latency_cycles": report.get("latency_cycles_min"),
        "latency_at_target_ns": (
            report["latency_cycles_min"] * clock_period
            if report.get("latency_cycles_min") is not None
            else None
        ),
        "initiation_interval_cycles": report.get("initiation_interval_cycles"),
        "lut": report.get("lut"),
        "ff": report.get("ff"),
        "dsp": report.get("dsp"),
        "bram_18k": report.get("bram"),
        "uram": report.get("uram"),
        "reference_validation": {
            **validation,
            "samples": int(limit),
            "reference_source": reference_source,
        },
        "synthesis_status": "success",
        "place_and_route_status": "not_run",
        "approximate_model": False,
        "notes": [
            "True handwritten HLS route, no hls4ml.",
            "Logits-only hardware path; no softmax in RTL.",
            f"Family precision profile: {typespec}.",
            "Final scaling stays outside the softmax stage.",
        ],
        "report_files": {
            "rpt": str(rpt.relative_to(root)),
            "xml": str(xml.relative_to(root)),
        },
        "state": str(state_path.relative_to(root)),
        "folded_layers": len(folded),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(result_json, indent=2) + "\n", encoding="utf-8"
    )
    return result_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument("--part", default="xcvu13p-flga2577-2-e")
    parser.add_argument("--mode", choices=("simple", "tree", "unrolled"), default="tree")
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--allow-unverified-license", action="store_true")
    args = parser.parse_args()
    if args.run_name:
        runs = args.run_name
    elif args.refresh_existing:
        runs = sorted(
            path.name.removesuffix("_custom_native")
            for path in (ROOT / "results" / "synthesis").glob("*_custom_native")
            if path.is_dir()
        )
    else:
        runs = list(MODEL_FAMILIES)
    summary = []
    for run_name in runs:
        if args.refresh_existing:
            summary.append(_refresh_result(ROOT, run_name))
        else:
            summary.append(
                synthesize_run(
                    ROOT, run_name, args.part, args.clock_period, args.samples, args.mode
                )
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
