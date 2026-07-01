import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np


def _layer_number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def load_quantized_layers(path: Path) -> list[dict]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    names = sorted(
        (key[: -len(".int_weight")] for key in state if key.endswith(".int_weight")),
        key=_layer_number,
    )
    layers = []
    for name in names:
        beta_shift = state.get(f"{name}.beta_shift")
        beta = (
            float(2.0 ** int(beta_shift.item()))
            if beta_shift is not None
            else float(state[f"{name}.beta_scale"].item())
        )
        layers.append(
            {
                "name": name,
                "weight": state[f"{name}.int_weight"].numpy().astype(np.int8),
                "beta": beta,
                "bias": state[f"{name}.bias"].numpy().astype(np.float32),
            }
        )
    if not layers:
        raise ValueError(f"No quantized BitNet layers found in {path}")
    return layers


def dynamic_quantize(values: np.ndarray, bits: int = 8) -> np.ndarray:
    q_max = float(2 ** (bits - 1) - 1)
    maximum = np.maximum(np.max(np.abs(values), axis=-1, keepdims=True), 1e-8)
    gamma = q_max / maximum
    quantized = np.clip(np.rint(values * gamma), -(q_max + 1), q_max)
    return quantized / gamma


def predict_exact(layers: list[dict], values: np.ndarray, batch_size=4096):
    import torch

    outputs = []
    torch_layers = [
        {
            "weight": torch.from_numpy(layer["weight"]).float(),
            "beta": layer["beta"],
            "bias": torch.from_numpy(layer["bias"]).float(),
        }
        for layer in layers
    ]
    for start in range(0, len(values), batch_size):
        output = torch.from_numpy(
            np.array(values[start : start + batch_size], dtype=np.float32, copy=True)
        )
        for index, layer in enumerate(torch_layers):
            maximum = output.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            gamma = 127.0 / maximum
            quantized = (output * gamma).round().clamp(-128, 127) / gamma
            output = (
                quantized @ layer["weight"].T * layer["beta"]
                + layer["bias"]
            )
            if index != len(layers) - 1:
                output = torch.relu(output)
        outputs.append(output.numpy())
    return np.concatenate(outputs)


def predict_split_alpha(layers: list[dict], values: np.ndarray, batch_size=4096):
    import torch

    outputs = []
    torch_layers = [
        {
            "weight": torch.from_numpy(layer["weight"]).float(),
            "beta": layer["beta"],
            "bias": torch.from_numpy(layer["bias"]).float(),
        }
        for layer in layers
    ]
    for start in range(0, len(values), batch_size):
        output = torch.from_numpy(
            np.array(values[start : start + batch_size], dtype=np.float32, copy=True)
        )
        for index, layer in enumerate(torch_layers):
            output = output @ layer["weight"].T * layer["beta"] + layer["bias"]
            if index != len(layers) - 1:
                output = torch.relu(output)
        outputs.append(output.numpy())
    return np.concatenate(outputs)


def _beta_fraction(beta: float) -> tuple[int, int]:
    fraction = Fraction(beta).limit_denominator(4096)
    return fraction.numerator, fraction.denominator


def fold_cumulative_alpha(layers: list[dict]) -> list[dict]:
    cumulative_numerator = 1
    cumulative_denominator = 1
    folded = []
    for layer in layers:
        beta_numerator, beta_denominator = _beta_fraction(layer["beta"])
        cumulative_numerator *= beta_numerator
        cumulative_denominator *= beta_denominator
        cumulative_scale = cumulative_numerator / float(cumulative_denominator)
        folded.append(
            {
                **layer,
                "beta_numerator": beta_numerator,
                "beta_denominator": beta_denominator,
                "cumulative_numerator": cumulative_numerator,
                "cumulative_denominator": cumulative_denominator,
                "cumulative_scale": cumulative_scale,
                "bias_folded": np.asarray(layer["bias"], dtype=np.float64)
                / cumulative_scale,
            }
        )
    return folded


def predict_v26_style(layers: list[dict], values: np.ndarray, batch_size=4096):
    import torch

    folded = fold_cumulative_alpha(layers)
    final_scale = folded[-1]["cumulative_scale"]
    outputs = []
    torch_layers = [
        {
            "weight": torch.from_numpy(layer["weight"]).float(),
            "bias": torch.from_numpy(np.asarray(layer["bias_folded"], dtype=np.float32)),
        }
        for layer in folded
    ]
    for start in range(0, len(values), batch_size):
        output = torch.from_numpy(
            np.array(values[start : start + batch_size], dtype=np.float32, copy=True)
        )
        for index, layer in enumerate(torch_layers):
            output = output @ layer["weight"].T + layer["bias"]
            if index != len(torch_layers) - 1:
                output = torch.relu(output)
            else:
                output = output * final_scale
        outputs.append(output.numpy())
    return np.concatenate(outputs)


def _cpp_values(values, formatter=str, columns=16):
    flat = list(np.asarray(values).reshape(-1))
    lines = []
    for start in range(0, len(flat), columns):
        lines.append("    " + ", ".join(formatter(value) for value in flat[start : start + columns]))
    return ",\n".join(lines)


def export_hls_project(
    layers: list[dict],
    output_dir: Path,
    project_name: str,
    part: str,
    clock_period: float,
    dynamic_activation: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = [layers[0]["weight"].shape[1]]
    dimensions.extend(layer["weight"].shape[0] for layer in layers)
    maximum_width = max(dimensions)

    parameter_blocks = []
    for index, layer in enumerate(layers):
        parameter_blocks.append(
            f"static const weight_t W{index}[{layer['weight'].size}] = {{\n"
            f"{_cpp_values(layer['weight'], lambda value: str(int(value)))}\n}};\n"
            f"static const scale_t BETA{index} = {layer['beta']:.12g};\n"
            f"static const data_t B{index}[{len(layer['bias'])}] = {{\n"
            f"{_cpp_values(layer['bias'], lambda value: f'{float(value):.12g}')}\n}};"
        )

    calls = []
    for index, layer in enumerate(layers):
        n_in = layer["weight"].shape[1]
        n_out = layer["weight"].shape[0]
        source = "input" if index == 0 else f"layer{index}"
        target = "output" if index == len(layers) - 1 else f"layer{index + 1}"
        relu = "false" if index == len(layers) - 1 else "true"
        calls.append(
            f"    bitnet_dense<{n_in}, {n_out}, {relu}>"
            f"({source}, {target}, W{index}, BETA{index}, B{index});"
        )

    intermediates = "\n".join(
        f"    data_t layer{index}[{dimensions[index]}];\n"
        f"#pragma HLS ARRAY_PARTITION variable=layer{index} complete"
        for index in range(1, len(dimensions) - 1)
    )
    header = f"""#ifndef {project_name.upper()}_H_
#define {project_name.upper()}_H_
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_math.h"

typedef ap_fixed<18, 8, AP_RND, AP_SAT> data_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> accum_t;
typedef ap_fixed<18, 4, AP_RND, AP_SAT> scale_t;
typedef ap_int<2> weight_t;

void {project_name}(data_t input[{dimensions[0]}], data_t output[{dimensions[-1]}]);
#endif
"""
    parameters = f"""#ifndef {project_name.upper()}_PARAMETERS_H_
#define {project_name.upper()}_PARAMETERS_H_
#include "{project_name}.h"
namespace parameters {{
{chr(10).join(parameter_blocks)}
}}
#endif
"""
    quantization = (
        "    dynamic_quantize<N_IN>(input, quantized);"
        if dynamic_activation
        else """copy_input:
    for (int i = 0; i < N_IN; i++) {
#pragma HLS UNROLL
        quantized[i] = input[i];
    }"""
    )
    source = f"""#include "{project_name}.h"
#include "{project_name}_parameters.h"
using namespace parameters;

template<int N>
static void dynamic_quantize(const data_t input[N], data_t output[N]) {{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    data_t maximum = 0;
find_max:
    for (int i = 0; i < N; i++) {{
#pragma HLS UNROLL
        data_t absolute;
        if (input[i] < 0) absolute = -input[i];
        else absolute = input[i];
        if (absolute > maximum) maximum = absolute;
    }}
    data_t quantum = 0;
    if (maximum > 0) quantum = maximum / data_t(127);
quantize:
    for (int i = 0; i < N; i++) {{
#pragma HLS UNROLL
        accum_t scaled = 0;
        if (quantum > 0) scaled = input[i] / quantum;
        ap_int<10> rounded;
        if (scaled >= 0) rounded = ap_int<10>(scaled + accum_t(0.5));
        else rounded = ap_int<10>(scaled - accum_t(0.5));
        if (rounded > 127) rounded = 127;
        if (rounded < -128) rounded = -128;
        output[i] = data_t(rounded) * quantum;
    }}
}}

template<int N_IN, int N_OUT, bool RELU>
static void bitnet_dense(
    const data_t input[N_IN],
    data_t output[N_OUT],
    const weight_t weights[N_IN * N_OUT],
    const scale_t beta,
    const data_t bias[N_OUT]
) {{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    data_t quantized[N_IN];
#pragma HLS ARRAY_PARTITION variable=quantized complete
{quantization}
dense_outputs:
    for (int o = 0; o < N_OUT; o++) {{
#pragma HLS UNROLL
        accum_t accumulator = 0;
    dense_inputs:
        for (int i = 0; i < N_IN; i++) {{
#pragma HLS UNROLL
            weight_t weight = weights[o * N_IN + i];
            if (weight > 0) accumulator += quantized[i];
            else if (weight < 0) accumulator -= quantized[i];
        }}
        data_t value = accumulator * beta + bias[o];
        if (RELU && value < 0) output[o] = 0;
        else output[o] = value;
    }}
}}

void {project_name}(data_t input[{dimensions[0]}], data_t output[{dimensions[-1]}]) {{
#pragma HLS INTERFACE ap_vld port=input,output
#pragma HLS ARRAY_PARTITION variable=input complete
#pragma HLS ARRAY_PARTITION variable=output complete
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
        "implementation": (
            "faithful_dynamic_activation_quantization"
            if dynamic_activation
            else "simplified_without_dynamic_activation_quantization"
        ),
        "dimensions": dimensions,
        "maximum_width": maximum_width,
        "part": part,
        "clock_period_ns": clock_period,
        "expected_ii": None,
        "approximate_model": not dynamic_activation,
        "warning": "HLS timing and II must be confirmed; no synthesis was run by this exporter.",
    }
    (output_dir / "project.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _emit_reduction_layer(
    function_name: str,
    n_in: int,
    n_out: int,
    relu: bool,
    final_scale: float | None,
    weights_name: str,
    bias_name: str,
    input_type: str,
    output_type: str,
    scale_type: str = "scale_t",
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
        f"    const accum_t biases[{n_out}]",
    ]
    if final_scale is not None:
        lines.append(f"    , const {scale_type} final_scale")
    lines.extend(
        [
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
            "            if (weights[weight_index] > 0) {",
            "                products[weight_index] = input[input_index];",
            "            } else {",
            "                products[weight_index] = -input[input_index];",
            "            }",
            "        }",
            "    }",
        ]
    )
    current_array = "products"
    current_terms = n_in
    for stage_index, _, next_terms in stage_arrays:
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
        current_terms = next_terms
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
            f"        const accum_t value = accumulators[output_index];",
        ]
    )
    if final_scale is None:
        if relu:
            lines.extend(
                [
                    f"        output[output_index] = value > 0 ? static_cast<{output_type}>(value) : static_cast<{output_type}>(0);",
                ]
            )
        else:
            lines.extend([f"        output[output_index] = static_cast<{output_type}>(value);"])
    else:
        lines.extend(
            [
                f"        output[output_index] = static_cast<{output_type}>(value * final_scale);",
            ]
        )
    lines.extend(["    }", "}"])
    return "\n".join(lines)


def export_hls_project_v26_style(
    layers: list[dict],
    output_dir: Path,
    project_name: str,
    part: str,
    clock_period: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    folded = fold_cumulative_alpha(layers)
    dimensions = [folded[0]["weight"].shape[1]]
    dimensions.extend(layer["weight"].shape[0] for layer in folded)
    maximum_width = max(dimensions)
    final_scale = folded[-1]["cumulative_scale"]

    parameter_blocks = []
    for index, layer in enumerate(folded):
        parameter_blocks.append(
            f"static const weight_t W{index}[{layer['weight'].size}] = {{\n"
            f"{_cpp_values(layer['weight'], lambda value: str(int(value)))}\n}};\n"
            f"static const accum_t B{index}[{len(layer['bias_folded'])}] = {{\n"
            f"{_cpp_values(layer['bias_folded'], lambda value: f'{float(value):.12g}')}\n}};"
        )

    layer_functions = []
    calls = []
    for index, layer in enumerate(folded):
        n_in = layer["weight"].shape[1]
        n_out = layer["weight"].shape[0]
        relu = index != len(folded) - 1
        function_name = f"bitnet_dense_v26_{index}"
        layer_functions.append(
            _emit_reduction_layer(
                function_name,
                n_in,
                n_out,
                relu=relu,
                final_scale=final_scale if not relu else None,
                weights_name=f"W{index}",
                bias_name=f"B{index}",
                input_type="data_t" if index == 0 else "data_t",
                output_type="data_t",
            )
        )
        call = (
            f"    {function_name}({ 'input' if index == 0 else f'layer{index}'}"
            f", { 'output' if index == len(folded) - 1 else f'layer{index + 1}'}"
            f", W{index}, B{index}"
        )
        if not relu:
            call += ", FINAL_SCALE"
        call += ");"
        calls.append(call)

    intermediates = "\n".join(
        f"    data_t layer{index}[{dimensions[index]}];\n"
        f"#pragma HLS ARRAY_PARTITION variable=layer{index} complete"
        for index in range(1, len(dimensions) - 1)
    )
    header = f"""#ifndef {project_name.upper()}_H_
#define {project_name.upper()}_H_
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_math.h"

typedef ap_fixed<18, 8, AP_RND, AP_SAT> data_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> accum_t;
typedef ap_fixed<32, 4, AP_RND, AP_SAT> scale_t;
typedef ap_int<2> weight_t;

void {project_name}(data_t input[{dimensions[0]}], data_t output[{dimensions[-1]}]);
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

void {project_name}(data_t input[{dimensions[0]}], data_t output[{dimensions[-1]}]) {{
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
        "implementation": "cumulative_alpha_v26_style",
        "dimensions": dimensions,
        "maximum_width": maximum_width,
        "part": part,
        "clock_period_ns": clock_period,
        "expected_ii": 1,
        "approximate_model": False,
        "final_scale": final_scale,
        "warning": "HLS timing and II must be confirmed; no synthesis was run by this exporter.",
    }
    (output_dir / "project.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
