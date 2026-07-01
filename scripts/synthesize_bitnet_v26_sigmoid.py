#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_benchmark.bitnet import fold_cumulative_alpha, load_quantized_layers
from hardware_benchmark.preflight import run_preflight
from hardware_benchmark.reports import parse_csynth_xml


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("/tmp/bitnet_v26_sigmoid")
ACCUM_TABLE_MIN = -32768
ACCUM_TABLE_STEP = 32
ACCUM_TABLE_SIZE = 2048


def _cpp_values(values, formatter=str, columns=16):
    flat = list(np.asarray(values).reshape(-1))
    lines = []
    for start in range(0, len(flat), columns):
        lines.append(
            "    "
            + ", ".join(formatter(value) for value in flat[start : start + columns])
        )
    return ",\n".join(lines)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def _make_sigmoid_table(final_scale: float, final_bias: float):
    accumulator_values = ACCUM_TABLE_MIN + ACCUM_TABLE_STEP * np.arange(
        ACCUM_TABLE_SIZE, dtype=np.float64
    )
    return _sigmoid(final_scale * (accumulator_values + final_bias))


def _emit_bitdense_bias_layer(function_name, n_in, n_out, input_type):
    return f"""static void {function_name}(
    const {input_type} input[{n_in}],
    accum_t output[{n_out}],
    const weight_t weights[{n_in} * {n_out}],
    const accum_t biases[{n_out}]
) {{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    {input_type} cache;
    accum_t mult[{n_in} * {n_out}];
    accum_t acc[{n_out}];
#pragma HLS function_instantiate variable=weights,biases
#pragma HLS ARRAY_PARTITION variable=mult complete
#pragma HLS ARRAY_PARTITION variable=acc complete

Product1BitBias:
    for (int ii = 0; ii < {n_in}; ii++) {{
        cache = input[ii];
    Product2BitBias:
        for (int jj = 0; jj < {n_out}; jj++) {{
            int index = ii * {n_out} + jj;
            weight_t weight = weights[index];
            if (weight == 0) {{
                mult[index] = 0;
            }} else if (weight < 0) {{
                mult[index] = -cache;
            }} else {{
                mult[index] = cache;
            }}
        }}
    }}

ResetAccumBitBias:
    for (int iacc = 0; iacc < {n_out}; iacc++) {{
        acc[iacc] = biases[iacc];
    }}

Accum1BitBias:
    for (int ii = 0; ii < {n_in}; ii++) {{
    Accum2BitBias:
        for (int jj = 0; jj < {n_out}; jj++) {{
            int index = ii * {n_out} + jj;
            acc[jj] += mult[index];
        }}
    }}

ResultBitBias:
    for (int ires = 0; ires < {n_out}; ires++) {{
        output[ires] = static_cast<accum_t>(acc[ires]);
    }}
}}"""


def _emit_relu_layer(function_name, n):
    return f"""static void {function_name}(
    const accum_t input[{n}],
    hidden_t output[{n}]
) {{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
ReluLoop:
    for (int index = 0; index < {n}; index++) {{
        accum_t value = input[index];
        output[index] = value > 0 ? static_cast<hidden_t>(value) : static_cast<hidden_t>(0);
    }}
}}"""


def _emit_hidden_layer(function_name, n_in, n_out, input_type, output_type):
    stage_arrays = []
    current_terms = n_in
    stage_index = 0
    while current_terms > 4:
        next_terms = current_terms // 2
        stage_arrays.append((stage_index, next_terms))
        current_terms = next_terms
        stage_index += 1

    lines = [
        f"static void {function_name}(",
        f"    const {input_type} input[{n_in}],",
        f"    {output_type} output[{n_out}],",
        f"    const weight_t weights[{n_in} * {n_out}],",
        f"    const accum_t biases[{n_out}]",
        ") {",
        "#pragma HLS INLINE off",
        "#pragma HLS PIPELINE II=1",
        f"    accum_t products[{n_in} * {n_out}];",
        f"    accum_t accumulators[{n_out}];",
        "#pragma HLS ARRAY_PARTITION variable=products complete",
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
        "                products[weight_index] = accum_t(0);",
        "            }",
        "        }",
        "    }",
    ]

    current_array = "products"
    for stage_index, next_terms in stage_arrays:
        stage_name = f"stage_{stage_index}"
        lines.extend(
            [
                f"    accum_t {stage_name}[{next_terms} * {n_out}];",
                f"#pragma HLS ARRAY_PARTITION variable={stage_name} complete",
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
            "        accumulators[output_index] = biases[output_index];",
            "    }",
            "",
            "accumulate_final:",
            f"    for (int partial_index = 0; partial_index < {current_terms}; partial_index++) {{",
            "    accumulate_final_outputs:",
            f"        for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            f"            accumulators[output_index] += {current_array}[partial_index * {n_out} + output_index];",
            "        }",
            "    }",
            "",
            "write_outputs:",
            f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
            "        const accum_t value = accumulators[output_index];",
            f"        output[output_index] = value > 0 ? static_cast<{output_type}>(value) : static_cast<{output_type}>(0);",
            "    }",
            "}",
        ]
    )
    return "\n".join(lines)


def _emit_final_sigmoid_layer(function_name, n_in):
    return f"""static void {function_name}(
    const hidden_t input[{n_in}],
    result_t output[1],
    const weight_t weights[{n_in} * 1]
) {{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    final_accum_t acc[1];
#pragma HLS function_instantiate variable=weights
#pragma HLS ARRAY_PARTITION variable=acc complete

ResetAccumBitDirectAccumSigmoid:
    for (int iacc = 0; iacc < 1; iacc++) {{
        acc[iacc] = 0;
    }}

Accum1BitDirectAccumSigmoid:
    for (int ii = 0; ii < {n_in}; ii++) {{
        hidden_t cache = input[ii];
    Accum2BitDirectAccumSigmoid:
        for (int jj = 0; jj < 1; jj++) {{
            int weight_index = ii * 1 + jj;
            weight_t weight = weights[weight_index];
            if (weight > 0) {{
                acc[jj] += cache;
            }} else if (weight < 0) {{
                acc[jj] -= cache;
            }}
        }}
    }}

ResultBitDirectAccumSigmoid:
    for (int ires = 0; ires < 1; ires++) {{
        int table_index = (static_cast<int>(acc[ires]) - ACCUM_TABLE_MIN) >> 5;
        if (table_index < 0)
            table_index = 0;
        if (table_index > ACCUM_TABLE_SIZE - 1)
            table_index = ACCUM_TABLE_SIZE - 1;
        output[ires] = static_cast<result_t>(SIGMOID_TABLE[table_index]);
    }}
}}"""


def _predict_v26_sigmoid_table(layers, inputs, batch_size=4096):
    import torch

    folded = fold_cumulative_alpha(layers)
    final = folded[-1]
    table = _make_sigmoid_table(
        final["cumulative_scale"], float(np.asarray(final["bias_folded"])[0])
    )
    torch_layers = [
        {
            "weight": torch.from_numpy(layer["weight"]).float(),
            "bias": torch.from_numpy(np.asarray(layer["bias_folded"], dtype=np.float32)),
        }
        for layer in folded[:-1]
    ]
    final_weight = torch.from_numpy(final["weight"]).float()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        output = torch.from_numpy(
            np.array(inputs[start : start + batch_size], dtype=np.float32, copy=True)
        )
        for layer in torch_layers:
            output = torch.relu(output @ layer["weight"].T + layer["bias"])
        raw_acc = (output @ final_weight.T).numpy()[:, 0]
        index = np.floor((raw_acc - ACCUM_TABLE_MIN) / ACCUM_TABLE_STEP).astype(np.int64)
        index = np.clip(index, 0, ACCUM_TABLE_SIZE - 1)
        signal = table[index]
        outputs.append(np.stack([1.0 - signal, signal], axis=1))
    return np.concatenate(outputs)


def _predict_reference_sigmoid(layers, inputs, batch_size=4096):
    import torch

    folded = fold_cumulative_alpha(layers)
    final_scale = folded[-1]["cumulative_scale"]
    torch_layers = [
        {
            "weight": torch.from_numpy(layer["weight"]).float(),
            "bias": torch.from_numpy(np.asarray(layer["bias_folded"], dtype=np.float32)),
        }
        for layer in folded
    ]
    outputs = []
    for start in range(0, len(inputs), batch_size):
        output = torch.from_numpy(
            np.array(inputs[start : start + batch_size], dtype=np.float32, copy=True)
        )
        for index, layer in enumerate(torch_layers):
            output = output @ layer["weight"].T + layer["bias"]
            if index != len(torch_layers) - 1:
                output = torch.relu(output)
            else:
                output = torch.sigmoid(output * final_scale)
        signal = output.numpy()
        outputs.append(np.concatenate([1.0 - signal, signal], axis=1))
    return np.concatenate(outputs)


def _validate(candidate, reference, labels):
    from sklearn.metrics import accuracy_score, roc_auc_score

    labels = np.asarray(labels).astype(np.int64)
    candidate_class = candidate.argmax(axis=1)
    reference_class = reference.argmax(axis=1)
    difference = np.abs(candidate - reference)
    return {
        "samples": int(len(labels)),
        "finite": bool(np.isfinite(candidate).all()),
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "class_agreement": float(np.mean(candidate_class == reference_class)),
        "accuracy": float(accuracy_score(labels, candidate_class)),
        "reference_accuracy": float(accuracy_score(labels, reference_class)),
        "auc": float(roc_auc_score(labels, candidate[:, 1])),
        "reference_auc": float(roc_auc_score(labels, reference[:, 1])),
    }


def export_project(layers, output_dir, project_name, part, clock_period):
    output_dir.mkdir(parents=True, exist_ok=True)
    folded = fold_cumulative_alpha(layers)
    if folded[-1]["weight"].shape[0] != 1:
        raise ValueError("v26 sigmoid export requires a one-output BitNet model")

    dimensions = [folded[0]["weight"].shape[1]]
    dimensions.extend(layer["weight"].shape[0] for layer in folded)
    final_layer = folded[-1]
    sigmoid_table = _make_sigmoid_table(
        final_layer["cumulative_scale"], float(np.asarray(final_layer["bias_folded"])[0])
    )

    parameter_blocks = []
    for index, layer in enumerate(folded):
        parameter_blocks.append(
            f"static const weight_t W{index}[{layer['weight'].size}] = {{\n"
            f"{_cpp_values(layer['weight'], lambda value: str(int(value)))}\n}};"
        )
        if index != len(folded) - 1:
            parameter_blocks.append(
                f"static const accum_t B{index}[{len(layer['bias_folded'])}] = {{\n"
                f"{_cpp_values(layer['bias_folded'], lambda value: f'{float(value):.12g}')}\n}};"
            )

    layer_functions = []
    calls = []
    for index, layer in enumerate(folded[:-1]):
        function_name = f"bitnet_dense_v26_{index}"
        layer_functions.append(
            _emit_hidden_layer(
                function_name,
                layer["weight"].shape[1],
                layer["weight"].shape[0],
                "data_t" if index == 0 else "hidden_t",
                "hidden_t",
            )
        )
        source = "input" if index == 0 else f"layer{index}"
        target = f"layer{index + 1}"
        calls.append(f"    {function_name}({source}, {target}, W{index}, B{index});")

    final_index = len(folded) - 1
    layer_functions.append(
        _emit_final_sigmoid_layer(
            f"bitnet_dense_v26_sigmoid_{final_index}",
            folded[-1]["weight"].shape[1],
        )
    )
    calls.append(
        f"    bitnet_dense_v26_sigmoid_{final_index}(layer{final_index}, output, W{final_index});"
    )

    intermediates = "\n".join(
        f"    hidden_t layer{index}[{dimensions[index]}];\n"
        f"#pragma HLS ARRAY_PARTITION variable=layer{index} complete"
        for index in range(1, len(dimensions) - 1)
    )
    header_guard = f"{project_name.upper()}_H_"
    header = f"""#ifndef {header_guard}
#define {header_guard}
#include "ap_fixed.h"
#include "ap_int.h"

typedef ap_fixed<16, 6, AP_RND, AP_SAT> data_t;
typedef ap_ufixed<15, 5, AP_RND, AP_SAT> hidden_t;
typedef ap_fixed<17, 7, AP_RND, AP_SAT> accum_t;
typedef ap_fixed<40, 22> final_accum_t;
typedef ap_ufixed<8, 0, AP_RND, AP_SAT> result_t;
typedef ap_int<2> weight_t;

void {project_name}(data_t input[{dimensions[0]}], result_t output[1]);
#endif
"""
    parameters_guard = f"{project_name.upper()}_PARAMETERS_H_"
    parameters = f"""#ifndef {parameters_guard}
#define {parameters_guard}
#include "{project_name}.h"
namespace parameters {{
static const int ACCUM_TABLE_MIN = {ACCUM_TABLE_MIN};
static const int ACCUM_TABLE_STEP = {ACCUM_TABLE_STEP};
static const int ACCUM_TABLE_SIZE = {ACCUM_TABLE_SIZE};
{chr(10).join(parameter_blocks)}
static const result_t SIGMOID_TABLE[{ACCUM_TABLE_SIZE}] = {{
{_cpp_values(sigmoid_table, lambda value: f'{float(value):.12g}')}
}};
}}
#endif
"""
    source = f"""#include "{project_name}.h"
#include "{project_name}_parameters.h"
using namespace parameters;

{chr(10).join(layer_functions)}

void {project_name}(data_t input[{dimensions[0]}], result_t output[1]) {{
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
        "implementation": "v26_direct_accumulator_sigmoid_table",
        "dimensions": dimensions,
        "part": part,
        "clock_period_ns": clock_period,
        "accum_table_min": ACCUM_TABLE_MIN,
        "accum_table_step": ACCUM_TABLE_STEP,
        "accum_table_size": ACCUM_TABLE_SIZE,
        "final_scale": final_layer["cumulative_scale"],
        "final_bias_folded": float(np.asarray(final_layer["bias_folded"])[0]),
    }
    (output_dir / "project.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _select_report_pair(solution, top):
    for stem in (f"{top}_csynth", f"{top.replace('__', '_')}_csynth"):
        rpt = solution / f"{stem}.rpt"
        xml = solution / f"{stem}.xml"
        if rpt.exists() and xml.exists():
            return rpt, xml
    for rpt in sorted(solution.glob("*_csynth.rpt")):
        xml = solution / f"{rpt.stem}.xml"
        if xml.exists():
            return rpt, xml
    raise FileNotFoundError(f"No top-level csynth report in {solution}")


def synthesize(run_name, part, clock_period, samples, class_mode="binary_qg_vs_wzt"):
    state_path = ROOT / "onnx" / "hardware" / f"{run_name}_quantized.pt"
    layers = load_quantized_layers(state_path)
    split_path = (
        ROOT
        / "data"
        / "cache"
        / f"openml_42468_nall_splitseed42_class{class_mode}_train0p64_val0p16_test0p2.npz"
    )
    if split_path.exists():
        split = np.load(split_path, mmap_mode="r")
        inputs = split["x_test"]
        labels = split["y_test"]
    else:
        inputs = np.load(ROOT / "data" / "synthesis" / "x_test.npy", mmap_mode="r")
        labels = np.load(ROOT / "data" / "synthesis" / "y_test.npy", mmap_mode="r")
    limit = min(samples, len(inputs))
    reference = _predict_reference_sigmoid(layers, inputs[:limit], batch_size=1024)
    candidate = _predict_v26_sigmoid_table(layers, inputs[:limit], batch_size=1024)
    validation = _validate(candidate, reference, labels[:limit])

    project_name = f"{run_name}_v26_sigmoid"
    project_dir = TMP_ROOT / project_name
    if project_dir.exists():
        shutil.rmtree(project_dir)
    metadata = export_project(layers, project_dir, project_name, part, clock_period)

    preflight = run_preflight(ROOT)
    if not preflight["vitis_hls"]["available"]:
        raise RuntimeError("Vitis HLS is unavailable")
    if not preflight["license"]["available"]:
        print(f"WARNING: license availability is unverified for {run_name}")

    env = os.environ.copy()
    env["PATH"] = (
        str(Path(preflight["vitis_hls"]["path"]).parent)
        + os.pathsep
        + env.get("PATH", "")
    )
    log_path = project_dir / "synthesis.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [preflight["vitis_hls"]["path"], "-f", "run_hls.tcl"],
            cwd=project_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Synthesis failed for {run_name}; see {log_path}")

    solution = project_dir / f"{project_name}_prj" / "solution1" / "syn" / "report"
    rpt, xml = _select_report_pair(solution, project_name)
    report = parse_csynth_xml(xml)
    result_dir = ROOT / "results" / "synthesis" / f"{run_name}_v26_sigmoid"
    result_dir.mkdir(parents=True, exist_ok=True)
    copied_rpt = result_dir / rpt.name
    copied_xml = result_dir / xml.name
    shutil.copy2(rpt, copied_rpt)
    shutil.copy2(xml, copied_xml)
    shutil.copy2(log_path, result_dir / "synthesis.log")

    result_json = {
        "run_name": run_name,
        "variant": "v26_direct_accumulator_sigmoid_table",
        "tool": "Vitis HLS",
        "tool_version": report.get("tool_version"),
        "part": report.get("part"),
        "clock_target_ns": clock_period,
        "clock_achieved_ns": report.get("clock_achieved_ns"),
        "estimated_fmax_mhz": (
            1000.0 / report["clock_achieved_ns"]
            if report.get("clock_achieved_ns")
            else None
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
        "reference_validation": validation,
        "synthesis_status": "success",
        "place_and_route_status": "not_run",
        "approximate_model": False,
        "notes": [
            "One-output binary sigmoid top, matching the parent v26 benchmark boundary.",
            "Final beta scale and bias are folded into an accumulator-domain sigmoid table.",
            "No runtime final softmax or sigmoid multiply is synthesized.",
        ],
        "report_files": {
            "rpt": str(copied_rpt.relative_to(ROOT)),
            "xml": str(copied_xml.relative_to(ROOT)),
        },
        "state": str(state_path.relative_to(ROOT)),
        "metadata": metadata,
    }
    (result_dir / "result.json").write_text(
        json.dumps(result_json, indent=2) + "\n", encoding="utf-8"
    )
    return result_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument("--part", default="xcvu13p-flga2577-2-e")
    parser.add_argument("--class-mode", default="binary_qg_vs_wzt")
    args = parser.parse_args()
    print(
        json.dumps(
            synthesize(
                args.run_name,
                args.part,
                args.clock_period,
                args.samples,
                args.class_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
