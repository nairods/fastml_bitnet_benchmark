#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
TMP_ROOT = Path("/tmp/bitnet_custom_v9_sigmoid")
ACCUM_TABLE_MIN = -32768
ACCUM_TABLE_STEP = 32
ACCUM_TABLE_SIZE = 2048
BINARY_CACHE = (
    ROOT
    / "data"
    / "cache"
    / "openml_42468_nall_splitseed42_classbinary_qg_vs_wzt_train0p64_val0p16_test0p2.npz"
)


def _cpp_values(values, formatter=str, columns=16):
    flat = list(np.asarray(values).reshape(-1))
    lines = []
    for start in range(0, len(flat), columns):
        lines.append("    " + ", ".join(formatter(value) for value in flat[start : start + columns]))
    return ",\n".join(lines)


def _cpp_matrix(values, formatter=str):
    matrix = np.asarray(values)
    rows = []
    for row in matrix:
        rows.append("    {" + ", ".join(formatter(value) for value in row) + "}")
    return ",\n".join(rows)


def _sparse_rows(weights: np.ndarray):
    weights = np.asarray(weights, dtype=np.int8)
    if weights.ndim != 2:
        weights = weights.reshape(weights.shape[0], -1)
    rows = []
    for row in weights:
        idx = np.flatnonzero(row)
        vals = row[idx].astype(np.int8)
        rows.append((idx, vals))
    return rows


def _make_sigmoid_table(final_scale: float, final_bias: float) -> np.ndarray:
    accumulator_values = ACCUM_TABLE_MIN + ACCUM_TABLE_STEP * np.arange(
        ACCUM_TABLE_SIZE, dtype=np.float64
    )
    return 1.0 / (1.0 + np.exp(-(final_scale * accumulator_values + final_bias)))


def _emit_dsp_add(function_name: str, typ: str) -> str:
    return f"""static {typ} {function_name}({typ} left, {typ} right) {{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    {typ} result = left + right;
#pragma HLS BIND_OP variable=result op=add impl=dsp latency=0
    return result;
}}"""


def _signed_int_bits(max_abs: float) -> int:
    maximum = max(float(max_abs), 1e-6)
    if maximum <= 1.0:
        return 2
    return int(math.ceil(math.log2(maximum))) + 1


def _unsigned_int_bits(max_value: float) -> int:
    maximum = max(float(max_value), 1e-6)
    if maximum <= 1.0:
        return 1
    return int(math.ceil(math.log2(maximum)))


def _ap_fixed(width: int, integer: int, unsigned: bool = False) -> str:
    kind = "ap_ufixed" if unsigned else "ap_fixed"
    return f"{kind}<{width}, {integer}, AP_RND, AP_SAT>"


def _choose_fractional_bits(integer_bits: int, target_width: int, minimum: int = 4, maximum: int = 10) -> int:
    return max(minimum, min(maximum, target_width - integer_bits))


def _binary_calibration_ranges(folded: list[dict]) -> dict[str, list[float]]:
    dataset = np.load(BINARY_CACHE)
    splits = [dataset["x_train"], dataset["x_validation"], dataset["x_test"]]
    pre_max = [0.0 for _ in folded]
    post_max = [0.0 for _ in folded[:-1]]
    for values in splits:
        output = values.astype(np.float64, copy=False)
        for index, layer in enumerate(folded):
            weights = np.asarray(layer["weight"], dtype=np.float64)
            bias = np.asarray(layer["bias_folded"], dtype=np.float64)
            output = output @ weights.T + bias
            pre_max[index] = max(pre_max[index], float(np.max(np.abs(output))))
            if index != len(folded) - 1:
                output = np.maximum(output, 0.0)
                post_max[index] = max(post_max[index], float(np.max(output)))
    return {"pre_max": pre_max, "post_max": post_max}


def _calibrated_type_defs(folded: list[dict]) -> dict[str, object]:
    ranges = _binary_calibration_ranges(folded)
    hidden_defs = []
    for index in range(len(folded) - 1):
        pre_bound = max(1.0, ranges["pre_max"][index] * 1.1)
        post_bound = max(1.0, ranges["post_max"][index] * 1.1)
        accum_integer = _signed_int_bits(pre_bound)
        output_integer = _unsigned_int_bits(post_bound)
        output_fractional = _choose_fractional_bits(output_integer, target_width=16)
        accum_fractional = max(4, min(8, output_fractional))
        output_width = output_integer + output_fractional
        accum_width = accum_integer + accum_fractional
        hidden_defs.append(
            {
                "accum_t": _ap_fixed(accum_width, accum_integer),
                "product_t": _ap_fixed(output_width, max(2, min(output_integer + 1, output_width - 1))),
                "dense_t": _ap_fixed(accum_width, accum_integer),
                "relu_t": _ap_fixed(output_width, output_integer, unsigned=True),
            }
        )
    final_bound = max(1.0, ranges["pre_max"][-1] * 1.1)
    final_integer = _signed_int_bits(final_bound)
    final_fractional = _choose_fractional_bits(final_integer, target_width=18, minimum=3, maximum=8)
    final_width = final_integer + final_fractional
    logits_integer = _signed_int_bits(final_bound * folded[-1]["cumulative_scale"] * 1.1)
    logits_fractional = _choose_fractional_bits(logits_integer, target_width=16, minimum=4, maximum=10)
    return {
        "hidden": hidden_defs,
        "final_accum_t": _ap_fixed(final_width, final_integer),
        "final_product_t": _ap_fixed(max(12, final_width), max(2, min(final_integer, max(12, final_width) - 1))),
        "result_t": "ap_ufixed<8, 0, AP_RND, AP_SAT>",
        "logits_t": _ap_fixed(logits_integer + logits_fractional, logits_integer),
        "ranges": ranges,
    }


def _tiled_sparse_tables(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    rows = _sparse_rows(weights)
    max_nnz = max((len(indexes) for indexes, _ in rows), default=1)
    index_table = np.zeros((len(rows), max_nnz), dtype=np.int32)
    sign_table = np.zeros((len(rows), max_nnz), dtype=np.int8)
    for row_index, (indexes, values) in enumerate(rows):
        if len(indexes):
            index_table[row_index, : len(indexes)] = indexes
            sign_table[row_index, : len(values)] = values
    return index_table, sign_table, max_nnz


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


def _emit_sparse_hidden_layer(
    function_name: str,
    weights: np.ndarray,
    input_type: str,
    output_type: str,
    accum_type: str,
    bias_name: str,
) -> str:
    weights = np.asarray(weights, dtype=np.int8)
    n_out, n_in = weights.shape
    lines = [
        f"static void {function_name}(const {input_type} input[{n_in}], {output_type} output[{n_out}]) {{",
        f"    {accum_type} accumulators[{n_out}];",
        "#pragma HLS INLINE off",
        "#pragma HLS PIPELINE II=1",
        "#pragma HLS ARRAY_PARTITION variable=output complete",
        "#pragma HLS ARRAY_PARTITION variable=accumulators complete",
        "",
        "init_accumulators:",
        f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
        f"        accumulators[output_index] = static_cast<{accum_type}>({bias_name}[output_index]);",
        "    }",
        "",
    ]
    for output_index, row in enumerate(weights):
        terms = [
            f"input[{input_index}]" if int(weight) > 0 else f"-input[{input_index}]"
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
            f"        {output_type} value = static_cast<{output_type}>(accumulators[output_index]);",
            "        if (value < 0) {",
                f"            output[output_index] = static_cast<{output_type}>(0);",
            "        } else {",
            "            output[output_index] = value;",
            "        }",
            "    }",
            "}",
        ]
    )
    return "\n".join(lines)


def _emit_tiled_sparse_hidden_layer(
    function_name: str,
    n_in: int,
    n_out: int,
    input_type: str,
    output_type: str,
    accum_type: str,
    bias_name: str,
    index_name: str,
    sign_name: str,
    max_nnz: int,
    tile_outputs: int,
) -> str:
    lines = [
        f"static void {function_name}(const {input_type} input[{n_in}], {output_type} output[{n_out}]) {{",
        "#pragma HLS INLINE off",
        "",
        "tile_outputs_loop:",
        f"    for (int tile_base = 0; tile_base < {n_out}; tile_base += {tile_outputs}) {{",
        "#pragma HLS PIPELINE II=1",
        "compute_tile:",
        f"        for (int lane = 0; lane < {tile_outputs}; lane++) {{",
        "#pragma HLS UNROLL",
        "            const int output_index = tile_base + lane;",
        f"            {accum_type} accumulator = static_cast<{accum_type}>({bias_name}[output_index]);",
        "accumulate_terms:",
        f"            for (int term_index = 0; term_index < {max_nnz}; term_index++) {{",
        "#pragma HLS UNROLL",
        f"                const ap_int<2> sign = {sign_name}[output_index][term_index];",
        "                if (sign > 0) {",
        f"                    accumulator += input[{index_name}[output_index][term_index]];",
        "                } else if (sign < 0) {",
        f"                    accumulator -= input[{index_name}[output_index][term_index]];",
        "                }",
        "            }",
        f"            {output_type} value = static_cast<{output_type}>(accumulator);",
        "            if (value < 0) {",
        f"                output[output_index] = static_cast<{output_type}>(0);",
        "            } else {",
        "                output[output_index] = value;",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(lines)


def _emit_sparse_final_layer(
    function_name: str,
    weights: np.ndarray,
    input_type: str,
    accum_type: str,
    output_mode: str,
) -> str:
    weights = np.asarray(weights, dtype=np.int8).reshape(-1)
    n_in = len(weights)
    terms = [
        f"input[{index}]" if int(weight) > 0 else f"-input[{index}]"
        for index, weight in enumerate(weights)
        if int(weight) != 0
    ]
    sum_lines, sum_name = _balanced_sum_lines(terms, accum_type, f"{function_name}_sum")
    lines = [
        f"static {'result_t' if output_mode == 'sigmoid' else 'logits_t'} {function_name}(const {input_type} input[{n_in}]) {{",
        f"    {accum_type} accumulator = 0;",
        "#pragma HLS INLINE off",
        "#pragma HLS PIPELINE II=1",
        "",
        *sum_lines,
        f"    accumulator += {sum_name};",
    ]
    if output_mode == "sigmoid":
        lines.extend(
            [
                "    int table_index = (accumulator - ACCUM_TABLE_MIN) / ACCUM_TABLE_STEP;",
                "    if (table_index < 0) {",
                "        table_index = 0;",
                "    } else if (table_index > ACCUM_TABLE_SIZE - 1) {",
                "        table_index = ACCUM_TABLE_SIZE - 1;",
                "    }",
                "    return FINAL_SIGMOID[table_index];",
            ]
        )
    else:
        lines.append("    return static_cast<logits_t>(accumulator * FINAL_SCALE);")
    lines.append("}")
    return "\n".join(line for line in lines if line != "")


def _emit_hidden_layer(
    function_name: str,
    add_function: str,
    n_in: int,
    n_out: int,
    input_type: str,
    product_type: str,
    accum_type: str,
    dense_type: str,
    output_type: str,
    weight_name: str,
    bias_name: str,
) -> str:
    body = [
        f"static void {function_name}(const {input_type} input[{n_in}], {output_type} output[{n_out}]) {{",
        f"    {accum_type} accumulators[{n_out}];",
        "#pragma HLS INLINE off",
        "#pragma HLS PIPELINE II=1",
        "#pragma HLS ARRAY_PARTITION variable=output complete",
        "#pragma HLS ARRAY_PARTITION variable=accumulators complete",
        "",
        "init_accumulators:",
        f"    for (int output_index = 0; output_index < {n_out}; output_index++) {{",
        f"        accumulators[output_index] = static_cast<{accum_type}>({bias_name}[output_index]);",
        "    }",
        "",
        "accumulate_inputs:",
        f"    for (int input_index = 0; input_index < {n_in}; input_index++) {{",
        f"        const {input_type} value = input[input_index];",
        "accumulate_outputs:",
        f"        for (int output_index = 0; output_index < {n_out}; output_index++) {{",
        f"            const int weight_index = input_index * {n_out} + output_index;",
        f"            const weight_t weight = {weight_name}[weight_index];",
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
        f"        {output_type} value = static_cast<{output_type}>(accumulators[output_index]);",
        "        if (value < 0) {",
        f"            output[output_index] = static_cast<{output_type}>(0);",
        "        } else {",
        "            output[output_index] = value;",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(body)


def _emit_final_sigmoid_layer(
    function_name: str,
    add_function: str,
    n_in: int,
    input_type: str,
    product_type: str,
    accum_type: str,
    weight_name: str,
) -> str:
    pair_count = n_in // 2
    quad_count = pair_count // 2
    pair_tail = pair_count % 2
    input_tail = n_in % 2
    pair_decl = (
        f"    {accum_type} pair_sums[{pair_count}];\n#pragma HLS ARRAY_PARTITION variable=pair_sums complete"
        if pair_count
        else ""
    )
    quad_decl = (
        f"    {accum_type} quad_sums[{quad_count}];\n#pragma HLS ARRAY_PARTITION variable=quad_sums complete"
        if quad_count
        else ""
    )
    body = [
        f"static result_t {function_name}(const {input_type} input[{n_in}]) {{",
        f"    {product_type} products[{n_in}];",
        pair_decl,
        quad_decl,
        f"    {accum_type} accumulator = 0;",
        "#pragma HLS PIPELINE II=1",
        "#pragma HLS ARRAY_PARTITION variable=products complete",
        "",
        f"make_{function_name}_products:",
        f"    for (int input_index = 0; input_index < {n_in}; input_index++) {{",
        f"        if ({weight_name}[input_index] > 0) {{",
        "            products[input_index] = input[input_index];",
        f"        }} else if ({weight_name}[input_index] < 0) {{",
        "            products[input_index] = -input[input_index];",
        "        } else {",
        f"            products[input_index] = {product_type}(0);",
        "        }",
        "    }",
        "",
    ]
    if pair_count:
        body.extend(
            [
                f"make_{function_name}_pairs:",
                f"    for (int pair_index = 0; pair_index < {pair_count}; pair_index++) {{",
                f"        pair_sums[pair_index] = {add_function}(",
                "            static_cast<final_accum_t>(products[2 * pair_index]),",
                "            static_cast<final_accum_t>(products[2 * pair_index + 1])",
                "        );",
                "    }",
                "",
            ]
        )
    if quad_count:
        body.extend(
            [
                f"make_{function_name}_quads:",
                f"    for (int quad_index = 0; quad_index < {quad_count}; quad_index++) {{",
                f"        quad_sums[quad_index] = {add_function}(",
                "            pair_sums[2 * quad_index],",
                "            pair_sums[2 * quad_index + 1]",
                "        );",
                "    }",
                "",
                f"accumulate_{function_name}_quads:",
                f"    for (int quad_index = 0; quad_index < {quad_count}; quad_index++) {{",
                "        accumulator += quad_sums[quad_index];",
                "    }",
                "",
            ]
        )
    elif pair_count:
        body.extend(
            [
                f"accumulate_{function_name}_pairs:",
                f"    for (int pair_index = 0; pair_index < {pair_count}; pair_index++) {{",
                "        accumulator += pair_sums[pair_index];",
                "    }",
                "",
            ]
        )
    if pair_tail:
        body.extend(
            [
                f"accumulate_{function_name}_tail_pair:",
                "    accumulator += pair_sums[pair_count - 1];",
                "",
            ]
        )
    if input_tail:
        body.extend(
            [
                f"accumulate_{function_name}_tail_input:",
                f"    accumulator += static_cast<final_accum_t>(products[{n_in - 1}]);",
                "",
            ]
        )
    body.extend(
        [
            "    int table_index = (accumulator - ACCUM_TABLE_MIN) / ACCUM_TABLE_STEP;",
            "    if (table_index < 0) {",
            "        table_index = 0;",
            "    } else if (table_index > ACCUM_TABLE_SIZE - 1) {",
            "        table_index = ACCUM_TABLE_SIZE - 1;",
            "    }",
            "    return FINAL_SIGMOID[table_index];",
            "}",
        ]
    )
    return "\n".join(line for line in body if line != "")


def _emit_final_logits_layer(
    function_name: str,
    add_function: str,
    n_in: int,
    input_type: str,
    product_type: str,
    accum_type: str,
    weight_name: str,
) -> str:
    pair_count = n_in // 2
    quad_count = pair_count // 2
    pair_tail = pair_count % 2
    input_tail = n_in % 2
    pair_decl = (
        f"    {accum_type} pair_sums[{pair_count}];\n#pragma HLS ARRAY_PARTITION variable=pair_sums complete"
        if pair_count
        else ""
    )
    quad_decl = (
        f"    {accum_type} quad_sums[{quad_count}];\n#pragma HLS ARRAY_PARTITION variable=quad_sums complete"
        if quad_count
        else ""
    )
    body = [
        f"static logits_t {function_name}(const {input_type} input[{n_in}]) {{",
        f"    {product_type} products[{n_in}];",
        pair_decl,
        quad_decl,
        f"    {accum_type} accumulator = 0;",
        "#pragma HLS PIPELINE II=1",
        "#pragma HLS ARRAY_PARTITION variable=products complete",
        "",
        f"make_{function_name}_products:",
        f"    for (int input_index = 0; input_index < {n_in}; input_index++) {{",
        f"        if ({weight_name}[input_index] > 0) {{",
        "            products[input_index] = input[input_index];",
        f"        }} else if ({weight_name}[input_index] < 0) {{",
        "            products[input_index] = -input[input_index];",
        "        } else {",
        f"            products[input_index] = {product_type}(0);",
        "        }",
        "    }",
        "",
    ]
    if pair_count:
        body.extend(
            [
                f"make_{function_name}_pairs:",
                f"    for (int pair_index = 0; pair_index < {pair_count}; pair_index++) {{",
                f"        pair_sums[pair_index] = {add_function}(",
                "            static_cast<final_accum_t>(products[2 * pair_index]),",
                "            static_cast<final_accum_t>(products[2 * pair_index + 1])",
                "        );",
                "    }",
                "",
            ]
        )
    if quad_count:
        body.extend(
            [
                f"make_{function_name}_quads:",
                f"    for (int quad_index = 0; quad_index < {quad_count}; quad_index++) {{",
                f"        quad_sums[quad_index] = {add_function}(",
                "            pair_sums[2 * quad_index],",
                "            pair_sums[2 * quad_index + 1]",
                "        );",
                "    }",
                "",
                f"accumulate_{function_name}_quads:",
                f"    for (int quad_index = 0; quad_index < {quad_count}; quad_index++) {{",
                "        accumulator += quad_sums[quad_index];",
                "    }",
                "",
            ]
        )
    elif pair_count:
        body.extend(
            [
                f"accumulate_{function_name}_pairs:",
                f"    for (int pair_index = 0; pair_index < {pair_count}; pair_index++) {{",
                "        accumulator += pair_sums[pair_index];",
                "    }",
                "",
            ]
        )
    if pair_tail:
        body.extend(
            [
                f"accumulate_{function_name}_tail_pair:",
                "    accumulator += pair_sums[pair_count - 1];",
                "",
            ]
        )
    if input_tail:
        body.extend(
            [
                f"accumulate_{function_name}_tail_input:",
                f"    accumulator += static_cast<final_accum_t>(products[{n_in - 1}]);",
                "",
            ]
        )
    body.extend(
        [
            "    return static_cast<logits_t>(accumulator * FINAL_SCALE);",
            "}",
        ]
    )
    return "\n".join(line for line in body if line != "")


def export_project(
    layers: list[dict],
    output_dir: Path,
    project_name: str,
    part: str,
    clock_period: float,
    output_mode: str,
    variant_tag: str = "custom_v9",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    folded = fold_cumulative_alpha(layers)
    if folded[-1]["weight"].shape[0] != 1:
        raise ValueError("custom_v9 sigmoid export requires a one-output BitNet model")

    dims = [folded[0]["weight"].shape[1]]
    dims.extend(layer["weight"].shape[0] for layer in folded)
    final_scale = folded[-1]["cumulative_scale"]
    final_bias = float(np.asarray(folded[-1]["bias_folded"], dtype=np.float64)[0])
    sigmoid_table = _make_sigmoid_table(final_scale, final_bias)
    calibrated_types = _calibrated_type_defs(folded) if variant_tag == "custom_v10_narrow" else None

    parameter_blocks = [
        f"static const int ACCUM_TABLE_MIN = {ACCUM_TABLE_MIN};",
        f"static const int ACCUM_TABLE_STEP = {ACCUM_TABLE_STEP};",
        f"static const int ACCUM_TABLE_SIZE = {ACCUM_TABLE_SIZE};",
    ]

    for index, layer in enumerate(folded[:-1]):
        weights = np.asarray(layer["weight"], dtype=np.int8)
        parameter_blocks.append(
            f"static const weight_t W{index + 1}[{weights.size}] = {{\n"
            f"{_cpp_values(weights, lambda value: str(int(value)))}\n}};"
        )
        parameter_blocks.append(
            f"static const bias_t B{index + 1}[{len(layer['bias_folded'])}] = {{\n"
            f"{_cpp_values(layer['bias_folded'], lambda value: f'{float(value):.12g}')}\n}};"
        )
        if variant_tag == "custom_v11_tiled":
            index_table, sign_table, max_nnz = _tiled_sparse_tables(weights)
            parameter_blocks.append(
                f"static const ap_uint<8> IDX{index + 1}[{weights.shape[0]}][{max_nnz}] = {{\n"
                f"{_cpp_matrix(index_table, lambda value: str(int(value)))}\n}};"
            )
            parameter_blocks.append(
                f"static const ap_int<2> S{index + 1}[{weights.shape[0]}][{max_nnz}] = {{\n"
                f"{_cpp_matrix(sign_table, lambda value: str(int(value)))}\n}};"
            )

    final_weights = np.asarray(folded[-1]["weight"], dtype=np.int8).reshape(-1)
    parameter_blocks.append(
        f"static const weight_t W{len(folded)}[{final_weights.size}] = {{\n"
        f"{_cpp_values(final_weights, lambda value: str(int(value)))}\n}};"
    )
    if output_mode == "sigmoid":
        parameter_blocks.append(
            f"static const result_t FINAL_SIGMOID[{ACCUM_TABLE_SIZE}] = {{\n"
            f"{_cpp_values(sigmoid_table, lambda value: f'{float(value):.12g}')}\n}};"
        )
    parameter_blocks.append(f"static const ap_fixed<32, 16> FINAL_SCALE = {final_scale:.12g};")

    if calibrated_types is None:
        type_block = """
typedef ap_fixed<16, 6> input_t;
typedef ap_fixed<32, 16> bias_t;
typedef ap_int<2> weight_t;

typedef ap_fixed<28, 10> accum1_t;
typedef ap_fixed<17, 7, AP_RND, AP_SAT> product1_t;
typedef ap_fixed<28, 10, AP_RND, AP_SAT> dense1_t;
typedef ap_ufixed<24, 8, AP_RND, AP_SAT> relu1_t;

typedef ap_fixed<32, 14> accum2_t;
typedef ap_fixed<25, 9, AP_RND, AP_SAT> product2_t;
typedef ap_fixed<32, 14, AP_RND, AP_SAT> dense2_t;
typedef ap_ufixed<26, 11, AP_RND, AP_SAT> relu2_t;

typedef ap_fixed<34, 16> accum3_t;
typedef ap_fixed<28, 12, AP_RND, AP_SAT> product3_t;
typedef ap_fixed<34, 16, AP_RND, AP_SAT> dense3_t;
typedef ap_ufixed<28, 14, AP_RND, AP_SAT> relu3_t;

typedef ap_fixed<40, 22> final_accum_t;
typedef ap_fixed<31, 15, AP_RND, AP_SAT> final_product_t;
typedef ap_ufixed<8, 0, AP_RND, AP_SAT> result_t;
typedef ap_fixed<20, 6, AP_RND, AP_SAT> logits_t;
"""
    else:
        hidden = calibrated_types["hidden"]
        type_block = f"""
typedef ap_fixed<16, 6> input_t;
typedef ap_fixed<24, 12> bias_t;
typedef ap_int<2> weight_t;

typedef {hidden[0]["accum_t"]} accum1_t;
typedef {hidden[0]["product_t"]} product1_t;
typedef {hidden[0]["dense_t"]} dense1_t;
typedef {hidden[0]["relu_t"]} relu1_t;

typedef {hidden[1]["accum_t"] if len(hidden) > 1 else hidden[0]["accum_t"]} accum2_t;
typedef {hidden[1]["product_t"] if len(hidden) > 1 else hidden[0]["product_t"]} product2_t;
typedef {hidden[1]["dense_t"] if len(hidden) > 1 else hidden[0]["dense_t"]} dense2_t;
typedef {hidden[1]["relu_t"] if len(hidden) > 1 else hidden[0]["relu_t"]} relu2_t;

typedef {hidden[2]["accum_t"] if len(hidden) > 2 else hidden[-1]["accum_t"]} accum3_t;
typedef {hidden[2]["product_t"] if len(hidden) > 2 else hidden[-1]["product_t"]} product3_t;
typedef {hidden[2]["dense_t"] if len(hidden) > 2 else hidden[-1]["dense_t"]} dense3_t;
typedef {hidden[2]["relu_t"] if len(hidden) > 2 else hidden[-1]["relu_t"]} relu3_t;

typedef {calibrated_types["final_accum_t"]} final_accum_t;
typedef {calibrated_types["final_product_t"]} final_product_t;
typedef {calibrated_types["result_t"]} result_t;
typedef {calibrated_types["logits_t"]} logits_t;
"""

    header = f"""#ifndef {project_name.upper()}_H_
#define {project_name.upper()}_H_

#include "ap_fixed.h"
#include "ap_int.h"
{type_block}

void {project_name}(input_t input[{dims[0]}], {"result_t" if output_mode == "sigmoid" else "logits_t"} output[1]);

#endif
"""

    parameters = f"""#ifndef {project_name.upper()}_PARAMETERS_H_
#define {project_name.upper()}_PARAMETERS_H_

#include "{project_name}.h"

namespace bitnet_parameters {{
{chr(10).join(parameter_blocks)}
}}

#endif
"""

    hidden_count = len(folded) - 1
    if hidden_count < 1 or hidden_count > 3:
        raise ValueError(
            f"custom_v9 supports one to three hidden layers, got {hidden_count}: {dims}"
        )
    hidden_specs = {
        1: ("input_t", "product1_t", "accum1_t", "dense1_t", "relu1_t", "dsp_add_layer1"),
        2: ("relu1_t", "product2_t", "accum2_t", "dense2_t", "relu2_t", "dsp_add_layer2"),
        3: ("relu2_t", "product3_t", "accum3_t", "dense3_t", "relu3_t", "dsp_add_layer3"),
    }
    hidden_blocks = []
    hidden_calls = []
    intermediate_decls = []
    sparse_layer_flags: list[bool] = []
    partition_pragmas = [
        "#pragma HLS ARRAY_RESHAPE variable=input complete dim=0",
        "#pragma HLS ARRAY_PARTITION variable=output complete dim=0",
    ]
    for layer_index in range(1, hidden_count + 1):
        input_type, product_type, accum_type, dense_type, output_type, add_function = hidden_specs[layer_index]
        layer_weights = np.asarray(folded[layer_index - 1]["weight"], dtype=np.int8)
        if variant_tag == "custom_v11_tiled":
            tile_outputs = 16 if dims[layer_index] >= 16 else dims[layer_index]
            index_table, sign_table, max_nnz = _tiled_sparse_tables(layer_weights)
            sparse_layer_flags.append(True)
            hidden_blocks.append(
                _emit_tiled_sparse_hidden_layer(
                    f"binary_dense_relu_layer{layer_index}_v9",
                    dims[layer_index - 1],
                    dims[layer_index],
                    input_type,
                    output_type,
                    accum_type,
                    f"B{layer_index}",
                    f"IDX{layer_index}",
                    f"S{layer_index}",
                    max_nnz,
                    tile_outputs,
                )
            )
        elif np.any(layer_weights == 0):
            sparse_layer_flags.append(True)
            hidden_blocks.append(
                _emit_sparse_hidden_layer(
                    f"binary_dense_relu_layer{layer_index}_v9",
                    layer_weights,
                    input_type,
                    output_type,
                    accum_type,
                    f"B{layer_index}",
                )
            )
        else:
            sparse_layer_flags.append(False)
            hidden_blocks.append(
                _emit_hidden_layer(
                    f"binary_dense_relu_layer{layer_index}_v9",
                    add_function,
                    dims[layer_index - 1],
                    dims[layer_index],
                    input_type,
                    product_type,
                    accum_type,
                    dense_type,
                    output_type,
                    f"W{layer_index}",
                    f"B{layer_index}",
                )
        )
        source_name = "input" if layer_index == 1 else f"layer{layer_index - 1}"
        target_name = f"layer{layer_index}"
        hidden_calls.append(f"    binary_dense_relu_layer{layer_index}_v9({source_name}, {target_name});")
        intermediate_decls.extend(
            [
                f"    {output_type} {target_name}[{dims[layer_index]}];",
                f"#pragma HLS ARRAY_PARTITION variable={target_name} complete",
            ]
        )
        if np.any(layer_weights == 0):
            partition_pragmas.append(f"#pragma HLS ARRAY_PARTITION variable=B{layer_index} complete dim=0")
            if variant_tag == "custom_v11_tiled":
                partition_pragmas.extend(
                    [
                        f"#pragma HLS ARRAY_PARTITION variable=IDX{layer_index} complete dim=2",
                        f"#pragma HLS ARRAY_PARTITION variable=S{layer_index} complete dim=2",
                    ]
                )
        else:
            partition_pragmas.extend(
                [
                    f"#pragma HLS ARRAY_PARTITION variable=W{layer_index} complete dim=0",
                    f"#pragma HLS ARRAY_PARTITION variable=B{layer_index} complete dim=0",
                ]
            )
    final_input_type = hidden_specs[hidden_count][4]
    final_input_name = f"layer{hidden_count}"
    final_weights = np.asarray(folded[-1]["weight"], dtype=np.int8).reshape(-1)
    final_sparse = np.any(final_weights == 0)
    if final_sparse:
        final_block = _emit_sparse_final_layer(
            "final_dense_sigmoid_v9" if output_mode == "sigmoid" else "final_dense_logits_v9",
            final_weights,
            final_input_type,
            "final_accum_t",
            output_mode,
        )
    else:
        final_weight_name = f"W{len(folded)}"
        partition_pragmas.append(f"#pragma HLS ARRAY_PARTITION variable={final_weight_name} complete dim=0")
        final_block = (
            _emit_final_sigmoid_layer(
                "final_dense_sigmoid_v9",
                "dsp_add",
                dims[-2],
                final_input_type,
                "final_product_t",
                "final_accum_t",
                final_weight_name,
            )
            if output_mode == "sigmoid"
            else _emit_final_logits_layer(
                "final_dense_logits_v9",
                "dsp_add",
                dims[-2],
                final_input_type,
                "final_product_t",
                "final_accum_t",
                final_weight_name,
            )
        )
    final_call = (
        f"final_dense_sigmoid_v9({final_input_name})"
        if output_mode == "sigmoid"
        else f"final_dense_logits_v9({final_input_name})"
    )

    source = f"""#include "{project_name}.h"
#include "{project_name}_parameters.h"

using namespace bitnet_parameters;

{_emit_dsp_add("dsp_add", "final_accum_t")}
{_emit_dsp_add("dsp_add_layer3", "accum3_t")}
{_emit_dsp_add("dsp_add_layer2", "accum2_t")}
{_emit_dsp_add("dsp_add_layer1", "accum1_t")}

{chr(10).join(hidden_blocks)}

{final_block}

void {project_name}(input_t input[{dims[0]}], {"result_t" if output_mode == "sigmoid" else "logits_t"} output[1]) {{
{chr(10).join(partition_pragmas)}
#pragma HLS INTERFACE ap_vld port=input,output
#pragma HLS PIPELINE II=1

{chr(10).join(intermediate_decls)}

{chr(10).join(hidden_calls)}
    output[0] = {final_call};
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
        "implementation": f"{variant_tag}_{output_mode}",
        "dimensions": dims,
        "part": part,
        "clock_period_ns": clock_period,
        "final_scale": final_scale,
        "final_bias_folded": final_bias,
        "accum_table_min": ACCUM_TABLE_MIN if output_mode == "sigmoid" else None,
        "accum_table_step": ACCUM_TABLE_STEP if output_mode == "sigmoid" else None,
        "accum_table_size": ACCUM_TABLE_SIZE if output_mode == "sigmoid" else None,
        "output_mode": output_mode,
        "calibrated_types": calibrated_types,
    }
    (output_dir / "project.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def synthesize(run_name: str, part: str, clock_period: float, output_mode: str, variant_tag: str) -> dict:
    state_path = ROOT / "onnx" / "hardware" / f"{run_name}_quantized.pt"
    layers = load_quantized_layers(state_path)
    project_name = f"{run_name}_{variant_tag}_{output_mode}".replace("__", "_")
    project_dir = TMP_ROOT / project_name
    if project_dir.exists():
        shutil.rmtree(project_dir)
    metadata = export_project(layers, project_dir, project_name, part, clock_period, output_mode, variant_tag)

    preflight = run_preflight(ROOT)
    if not preflight["vitis_hls"]["available"]:
        raise RuntimeError("Vitis HLS is unavailable")

    env = os.environ.copy()
    env["PATH"] = str(Path(preflight["vitis_hls"]["path"]).parent) + os.pathsep + env.get("PATH", "")
    log_path = project_dir / "synthesis.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [preflight["vitis_hls"]["path"], "-f", "run_hls.tcl"],
            cwd=project_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Synthesis failed for {run_name}; see {log_path}")

    report_dir = project_dir / f"{project_name}_prj" / "solution1" / "syn" / "report"
    rpt = report_dir / f"{project_name}_csynth.rpt"
    xml = report_dir / f"{project_name}_csynth.xml"
    report = parse_csynth_xml(xml)

    result_dir = ROOT / "results" / "synthesis" / f"{run_name}_{variant_tag}_{output_mode}"
    result_dir.mkdir(parents=True, exist_ok=True)
    copied_rpt = result_dir / rpt.name
    copied_xml = result_dir / xml.name
    shutil.copy2(rpt, copied_rpt)
    shutil.copy2(xml, copied_xml)
    shutil.copy2(log_path, result_dir / "synthesis.log")

    result = {
        "run_name": run_name,
        "variant": f"{variant_tag}_{output_mode}",
        "tool": "Vitis HLS",
        "tool_version": report.get("tool_version"),
        "part": report.get("part"),
        "clock_target_ns": clock_period,
        "clock_achieved_ns": report.get("clock_achieved_ns"),
        "latency_cycles": report.get("latency_cycles_min"),
        "initiation_interval_cycles": report.get("initiation_interval_cycles"),
        "lut": report.get("lut"),
        "ff": report.get("ff"),
        "dsp": report.get("dsp"),
        "bram_18k": report.get("bram"),
        "uram": report.get("uram"),
        "synthesis_status": "success",
        "place_and_route_status": "not_run",
        "implementation_boundary": "binary sigmoid output; no softmax" if output_mode == "sigmoid" else "binary logits scaling stage; no sigmoid or softmax",
        "metadata": metadata,
        "report_files": {
            "rpt": str(copied_rpt.relative_to(ROOT)),
            "xml": str(copied_xml.relative_to(ROOT)),
        },
        "state": str(state_path.relative_to(ROOT)),
    }
    (result_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="bitnet_binary_sigmoid_f7_fixed")
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument("--part", default="xcvu13p-flga2577-2-e")
    parser.add_argument("--output-mode", choices=["sigmoid", "logits"], default="sigmoid")
    parser.add_argument("--variant-tag", default="custom_v9")
    args = parser.parse_args()
    print(
        json.dumps(
            synthesize(args.run_name, args.part, args.clock_period, args.output_mode, args.variant_tag),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
