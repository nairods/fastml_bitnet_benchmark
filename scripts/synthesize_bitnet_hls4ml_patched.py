#!/usr/bin/env python3
"""Generate and synthesize patched hls4ml BitNet and BitNet-1.58 projects."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_benchmark.bitnet import load_quantized_layers, predict_folded, predict_folded_logits
from hardware_benchmark.preflight import run_preflight
from hardware_benchmark.reports import parse_csynth_xml


ACCUM_TABLE_SIZE = 2048
TARGET_SIGMOID_LOGIT_STEP = 0.1
IMPLEMENTATION = "hls4ml_patched_bitnet_latency_rf1"

ACCUM_PRECISIONS = (
    "ap_fixed<28,10>",
    "ap_fixed<32,14>",
    "ap_fixed<34,16>",
    "ap_fixed<40,22>",
)
ADD_PRECISIONS = (
    "ap_fixed<28,10,AP_RND,AP_SAT>",
    "ap_fixed<32,14,AP_RND,AP_SAT>",
    "ap_fixed<34,16,AP_RND,AP_SAT>",
    "ap_fixed<37,13,AP_RND,AP_SAT>",
)
RELU_PRECISIONS = (
    "ap_ufixed<24,8,AP_RND,AP_SAT>",
    "ap_ufixed<26,11,AP_RND,AP_SAT>",
    "ap_ufixed<28,14,AP_RND,AP_SAT>",
)
MULT_PRECISIONS = (
    "ap_fixed<17,7,AP_RND,AP_SAT>",
    "ap_fixed<25,9,AP_RND,AP_SAT>",
    "ap_fixed<28,12,AP_RND,AP_SAT>",
    "ap_fixed<31,15,AP_RND,AP_SAT>",
)

BITDENSE_HELPERS = r"""
namespace nnet {

// hls-bitdense insert accumulator sigmoid table

template <class data_T, class res_T, typename CONFIG_T>
void patched_bitdense_bias(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::alpha_bias_t biases[CONFIG_T::n_out]
) {
    typename CONFIG_T::mult_t products[CONFIG_T::n_in * CONFIG_T::n_out];
    typename CONFIG_T::accum_t accumulators[CONFIG_T::n_out];

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=products complete
    #pragma HLS ARRAY_PARTITION variable=accumulators complete

PatchedBitDenseProducts:
    for (int input_index = 0; input_index < CONFIG_T::n_in; input_index++) {
    PatchedBitDenseOutputs:
        for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
            int index = input_index * CONFIG_T::n_out + output_index;
            typename CONFIG_T::weight_t weight = weights[index];
            if (weight > 0) {
                products[index] =
                    static_cast<typename CONFIG_T::mult_t>(data[input_index]);
            } else if (weight < 0) {
                products[index] =
                    static_cast<typename CONFIG_T::mult_t>(-data[input_index]);
            } else {
                products[index] = 0;
            }
        }
    }

PatchedBitDenseReset:
    for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
        accumulators[output_index] =
            static_cast<typename CONFIG_T::accum_t>(biases[output_index]);
    }

PatchedBitDenseAccumulateInputs:
    for (int input_index = 0; input_index < CONFIG_T::n_in; input_index++) {
    PatchedBitDenseAccumulateOutputs:
        for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
            int index = input_index * CONFIG_T::n_out + output_index;
            accumulators[output_index] += products[index];
        }
    }

PatchedBitDenseResults:
    for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
        res[output_index] = static_cast<res_T>(accumulators[output_index]);
    }
}

template <class data_T, class res_T, typename CONFIG_T, typename SIGMOID_CONFIG_T>
void patched_bitdense_accum_sigmoid(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out]
) {
    typename CONFIG_T::accum_t accumulators[CONFIG_T::n_out];

    #pragma HLS function_instantiate variable=weights
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=accumulators complete

PatchedBitDenseSigmoidReset:
    for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
        accumulators[output_index] = 0;
    }

PatchedBitDenseSigmoidInputs:
    for (int input_index = 0; input_index < CONFIG_T::n_in; input_index++) {
        data_T value = data[input_index];
    PatchedBitDenseSigmoidOutputs:
        for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
            int index = input_index * CONFIG_T::n_out + output_index;
            typename CONFIG_T::weight_t weight = weights[index];
            if (weight > 0) {
                accumulators[output_index] += value;
            } else if (weight < 0) {
                accumulators[output_index] -= value;
            }
        }
    }

PatchedBitDenseSigmoidResults:
    for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
        int index =
            (accumulators[output_index] - CONFIG_T::accum_table_min)
            / CONFIG_T::accum_table_step;
        if (index < 0) {
            index = 0;
        } else if (index >= CONFIG_T::accum_table_size) {
            index = CONFIG_T::accum_table_size - 1;
        }
        res[output_index] =
            static_cast<res_T>(patched_accum_sigmoid_table[index]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void patched_bitdense_logits(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::alpha_bias_t biases[CONFIG_T::n_out]
) {
    typename CONFIG_T::accum_t accumulators[CONFIG_T::n_out];

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=accumulators complete

PatchedBitDenseLogitsReset:
    for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
        accumulators[output_index] =
            static_cast<typename CONFIG_T::accum_t>(biases[output_index]);
    }

PatchedBitDenseLogitsInputs:
    for (int input_index = 0; input_index < CONFIG_T::n_in; input_index++) {
        data_T value = data[input_index];
    PatchedBitDenseLogitsOutputs:
        for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
            int index = input_index * CONFIG_T::n_out + output_index;
            typename CONFIG_T::weight_t weight = weights[index];
            if (weight > 0) {
                accumulators[output_index] += value;
            } else if (weight < 0) {
                accumulators[output_index] -= value;
            }
        }
    }

PatchedBitDenseLogitsResults:
    for (int output_index = 0; output_index < CONFIG_T::n_out; output_index++) {
        res[output_index] = static_cast<res_T>(
            accumulators[output_index]
            * static_cast<typename CONFIG_T::alpha_bias_t>(CONFIG_T::final_scale)
        );
    }
}

} // namespace nnet
"""

DENSE_CALL = re.compile(
    r"(?P<indent>[ \t]*)nnet::dense<(?P<input_type>[^,]+), (?P<output_type>[^,]+), "
    r"(?P<config>[^>]+)>\((?P<input_var>[^,]+), (?P<output_var>[^,]+), "
    r"(?P<weight>[^,]+), (?P<bias>[^)]+)\); // Dense_MatMul_(?P<index>\d+)"
)
NORMALIZE_CALL = re.compile(
    r"(?P<indent>[ \t]*)nnet::normalize<(?P<input_type>[^,]+), (?P<output_type>[^,]+), "
    r"(?P<config>[^>]+)>\((?P<input_var>[^,]+), (?P<output_var>[^,]+), "
    r"(?P<scale>[^,]+), (?P<bias>[^)]+)\); // bn_Add_(?P<index>\d+)"
)
SIGMOID_CALL = re.compile(
    r"(?P<indent>[ \t]*)nnet::sigmoid<(?P<input_type>[^,]+), (?P<output_type>[^,]+), "
    r"(?P<config>[^>]+)>\((?P<input_var>[^,]+), (?P<output_var>[^)]+)\); // Sigmoid_0"
)


def _precision(values: tuple[str, ...], index: int) -> str:
    return values[min(index, len(values) - 1)]


def build_hls4ml_onnx(layers: list[dict], output_path: Path, binary_sigmoid: bool) -> None:
    import onnx
    from onnx import TensorProto, helper

    input_dim = int(layers[0]["weight"].shape[1])
    graph_inputs = [
        helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [None, input_dim])
    ]
    initializers = []
    value_infos = []
    nodes = []
    previous = "global_in"
    for index, layer in enumerate(layers):
        n_out = int(layer["weight"].shape[0])
        weight = np.asarray(layer["weight"], dtype=np.float32).T
        weight_name = f"w{index}"
        initializers.append(
            helper.make_tensor(
                weight_name, TensorProto.FLOAT, weight.shape, weight.reshape(-1)
            )
        )
        value_infos.append(
            helper.make_tensor_value_info(
                weight_name, TensorProto.FLOAT, weight.shape
            )
        )
        dot = f"matmul_{index}_out"
        value_infos.append(
            helper.make_tensor_value_info(dot, TensorProto.FLOAT, [None, n_out])
        )
        nodes.append(
            helper.make_node(
                "MatMul", [previous, weight_name], [dot], name=f"MatMul_{index}"
            )
        )

        beta_name = f"s{index}"
        initializers.append(
            helper.make_tensor(
                beta_name,
                TensorProto.FLOAT,
                [1],
                np.asarray([layer["beta"]], dtype=np.float32),
            )
        )
        value_infos.append(
            helper.make_tensor_value_info(beta_name, TensorProto.FLOAT, [1])
        )
        scaled = f"mul_{index}_out"
        value_infos.append(
            helper.make_tensor_value_info(scaled, TensorProto.FLOAT, [None, n_out])
        )
        nodes.append(
            helper.make_node("Mul", [dot, beta_name], [scaled], name=f"Mul_{index}")
        )

        bias = np.asarray(layer["bias"], dtype=np.float32)
        bias_name = f"b{index}"
        initializers.append(
            helper.make_tensor(bias_name, TensorProto.FLOAT, bias.shape, bias)
        )
        value_infos.append(
            helper.make_tensor_value_info(
                bias_name, TensorProto.FLOAT, bias.shape
            )
        )
        preactivation = f"add_{index}_out"
        value_infos.append(
            helper.make_tensor_value_info(
                preactivation, TensorProto.FLOAT, [None, n_out]
            )
        )
        nodes.append(
            helper.make_node(
                "Add", [scaled, bias_name], [preactivation], name=f"Add_{index}"
            )
        )
        if index == len(layers) - 1:
            previous = preactivation
        else:
            previous = f"relu_{index}_out"
            value_infos.append(
                helper.make_tensor_value_info(
                    previous, TensorProto.FLOAT, [None, n_out]
                )
            )
            nodes.append(
                helper.make_node("Relu", [preactivation], [previous], name=f"Relu_{index}")
            )

    if binary_sigmoid:
        nodes.append(
            helper.make_node("Sigmoid", [previous], ["probability"], name="Sigmoid_0")
        )
        output = helper.make_tensor_value_info(
            "probability", TensorProto.FLOAT, [None, 1]
        )
    else:
        output = helper.make_tensor_value_info(
            previous, TensorProto.FLOAT, [None, int(layers[-1]["weight"].shape[0])]
        )
    graph = helper.make_graph(
        nodes,
        "PatchedBinaryBitNet",
        graph_inputs,
        [output],
        initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="fastml_bitnet_benchmark",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def make_hls_config(model, layer_count: int, binary_sigmoid: bool) -> dict:
    from hls4ml.utils.config import config_from_onnx_model

    config = config_from_onnx_model(
        model,
        granularity="name",
        backend="Vivado",
        default_precision="fixed<16,6>",
    )
    config["Model"]["ReuseFactor"] = 1
    config["Model"]["Strategy"] = "Latency"
    for layer in config["LayerName"].values():
        layer["Trace"] = True
        layer["ReuseFactor"] = 1
        if "Strategy" in layer:
            layer["Strategy"] = "Latency"

    for index in range(layer_count):
        precision = config["LayerName"][f"MatMul_{index}"].setdefault("Precision", {})
        precision["weight"] = "ap_int<2>"
        precision["bias"] = "ap_int<1>"
        precision["accum"] = _precision(ACCUM_PRECISIONS, index)
        precision["result"] = "ap_fixed<20,6,AP_RND,AP_SAT>"
        config["LayerName"][f"Add_{index}"].setdefault("Precision", {})[
            "result"
        ] = _precision(ADD_PRECISIONS, index)
        if index < layer_count - 1:
            relu = config["LayerName"][f"Relu_{index}"]
            relu["TableSize"] = 1024
            relu.setdefault("Precision", {})["result"] = _precision(
                RELU_PRECISIONS, index
            )
            relu["Precision"]["table"] = "ap_fixed<18,8>"

    if binary_sigmoid:
        sigmoid = config["LayerName"]["Sigmoid_0"]
        sigmoid["TableSize"] = 1024
        sigmoid.setdefault("Precision", {})["result"] = (
            "ap_ufixed<8,0,AP_RND,AP_SAT>"
        )
        sigmoid["Precision"]["table"] = "ap_fixed<18,8>"
    return config


def _parse_generated_calls(
    source: str, layer_count: int, binary_sigmoid: bool
) -> tuple[list[dict], dict | None]:
    dense = {int(match["index"]): match.groupdict() for match in DENSE_CALL.finditer(source)}
    normalize = {
        int(match["index"]): match.groupdict()
        for match in NORMALIZE_CALL.finditer(source)
    }
    sigmoid_match = SIGMOID_CALL.search(source)
    expected = set(range(layer_count))
    if (
        set(dense) != expected
        or set(normalize) != expected
        or (binary_sigmoid and sigmoid_match is None)
        or (not binary_sigmoid and sigmoid_match is not None)
    ):
        raise RuntimeError(
            "Generated hls4ml call structure did not match the expected "
            "Dense/normalization/Sigmoid graph"
        )
    stages = [{"dense": dense[index], "normalize": normalize[index]} for index in range(layer_count)]
    return stages, sigmoid_match.groupdict() if sigmoid_match else None


def _parse_weight_values(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    return np.asarray(
        [
            float(item.strip())
            for item in text.replace("\n", " ").split(",")
            if item.strip()
        ],
        dtype=np.float64,
    )


def _write_weight_values(weights_dir: Path, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    payload = ", ".join(f"{value:.10f}" for value in values)
    (weights_dir / f"{name}.txt").write_text(payload, encoding="utf-8")
    header_path = weights_dir / f"{name}.h"
    text = header_path.read_text(encoding="utf-8")
    text = re.sub(r"//Min .+", f"//Min {values.min():.12f}", text, count=1)
    text = re.sub(r"//Max .+", f"//Max {values.max():.12f}", text, count=1)
    text, replacements = re.subn(
        rf"(#else\s*\nmodel_default_t {name}\[[0-9]+\] = )\{{[^}}]*\}};",
        rf"\1{{{payload}}};",
        text,
        count=1,
        flags=re.S,
    )
    if replacements != 1:
        raise RuntimeError(f"Could not update generated weight header {header_path}")
    header_path.write_text(text, encoding="utf-8")


def _format_hls_array(values: np.ndarray, values_per_line: int = 8) -> str:
    lines = []
    for start in range(0, len(values), values_per_line):
        chunk = values[start : start + values_per_line]
        lines.append("    " + ", ".join(f"{float(value):.10f}" for value in chunk))
    return ",\n".join(lines)


def _patch_parameters(
    project_dir: Path,
    stages: list[dict],
    final_scale: float,
    table: np.ndarray | None,
    table_minimum: int | None,
    table_step: int | None,
) -> None:
    parameters_path = project_dir / "firmware" / "parameters.h"
    text = parameters_path.read_text(encoding="utf-8")
    marker = "// hls-fpga-machine-learning insert layer-config"
    if marker not in text:
        raise RuntimeError(f"Missing layer-config marker in {parameters_path}")
    text = text.replace(marker, BITDENSE_HELPERS + "\n" + marker, 1)

    for index, stage in enumerate(stages):
        config_name = stage["dense"]["config"]
        start = text.index(f"struct {config_name} : nnet::dense_config {{")
        end = text.index("};", start)
        block = text[start:end]
        accum_match = re.search(r"\s+typedef dense_matmul_\d+_accum_t accum_t;", block)
        if accum_match is None:
            raise RuntimeError(f"Missing accumulator typedef in {config_name}")
        insert_at = start + accum_match.end()
        fields = (
            f"\n    typedef {_precision(MULT_PRECISIONS, index)} mult_t;"
            "\n    typedef model_default_t alpha_bias_t;"
            f"\n    static constexpr double final_scale = {final_scale:.17g};"
        )
        if table is not None:
            fields += (
                f"\n    static const int accum_table_min = {table_minimum};"
                f"\n    static const int accum_table_step = {table_step};"
                f"\n    static const int accum_table_size = {ACCUM_TABLE_SIZE};"
            )
        text = text[:insert_at] + fields + text[insert_at:]

    if table is not None:
        table_declaration = (
            f"static const result_t patched_accum_sigmoid_table[{ACCUM_TABLE_SIZE}] = {{\n"
            f"{_format_hls_array(table)}\n"
            "};"
        )
        table_marker = "// hls-bitdense insert accumulator sigmoid table"
        if table_marker not in text:
            raise RuntimeError(f"Missing sigmoid table marker in {parameters_path}")
        text = text.replace(table_marker, table_declaration + "\n\n" + table_marker, 1)
    else:
        table_marker = "// hls-bitdense insert accumulator sigmoid table"
        if table_marker not in text:
            raise RuntimeError(f"Missing sigmoid table marker in {parameters_path}")
        text = text.replace(
            table_marker,
            "static const result_t patched_accum_sigmoid_table[1] = {0};\n\n"
            + table_marker,
            1,
        )
    parameters_path.write_text(text, encoding="utf-8")


def _patch_cumulative_biases(
    project_dir: Path, layers: list[dict], stages: list[dict]
) -> float:
    defines_path = project_dir / "firmware" / "defines.h"
    text = defines_path.read_text(encoding="utf-8")
    text, replacements = re.subn(
        r"typedef ap_fixed<16,6> model_default_t;",
        "typedef ap_fixed<32,16> model_default_t;",
        text,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(f"Could not widen model_default_t in {defines_path}")
    defines_path.write_text(text, encoding="utf-8")

    weights_dir = project_dir / "firmware" / "weights"
    cumulative_scale = 1.0
    for layer, stage in zip(layers, stages):
        cumulative_scale *= float(layer["beta"])
        bias_name = stage["normalize"]["bias"]
        generated_bias = _parse_weight_values(weights_dir / f"{bias_name}.txt")
        _write_weight_values(
            weights_dir, bias_name, generated_bias / cumulative_scale
        )
    return cumulative_scale


def _accumulator_table_spec(final_scale: float) -> tuple[int, int]:
    maximum_step = max(1.0, TARGET_SIGMOID_LOGIT_STEP / final_scale)
    table_step = 2 ** int(np.floor(np.log2(maximum_step)))
    table_minimum = -(ACCUM_TABLE_SIZE // 2) * table_step
    return table_minimum, table_step


def _make_sigmoid_table(
    final_scale: float,
    final_bias: float,
    table_minimum: int,
    table_step: int,
) -> np.ndarray:
    centers = table_minimum + (
        np.arange(ACCUM_TABLE_SIZE, dtype=np.float64) + 0.5
    ) * table_step
    logits = np.clip(centers * final_scale + final_bias, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _guard_call(call: str) -> str:
    indent = call[: len(call) - len(call.lstrip())]
    return f"{indent}#ifndef __SYNTHESIS__\n{call}\n{indent}#endif"


def _patch_network(
    project_dir: Path, stages: list[dict], sigmoid: dict | None
) -> None:
    source_path = project_dir / "firmware" / "myproject.cpp"
    text = source_path.read_text(encoding="utf-8")
    for index, stage in enumerate(stages):
        dense = stage["dense"]
        normalize = stage["normalize"]
        dense_match = next(
            match
            for match in DENSE_CALL.finditer(text)
            if int(match["index"]) == index
        )
        text = text[: dense_match.start()] + _guard_call(dense_match.group(0)) + text[dense_match.end() :]

        normalize_match = next(
            match
            for match in NORMALIZE_CALL.finditer(text)
            if int(match["index"]) == index
        )
        if index < len(stages) - 1:
            replacement = (
                f'{normalize["indent"]}nnet::patched_bitdense_bias<'
                f'{dense["input_type"]}, {normalize["output_type"]}, {dense["config"]}>'
                f'({dense["input_var"]}, {normalize["output_var"]}, '
                f'{dense["weight"]}, {normalize["bias"]}); '
                f'// fused BitNet Dense_MatMul_{index} + bn_Add_{index}'
            )
        elif sigmoid is not None:
            replacement = _guard_call(normalize_match.group(0))
        else:
            replacement = (
                f'{normalize["indent"]}nnet::patched_bitdense_logits<'
                f'{dense["input_type"]}, {normalize["output_type"]}, {dense["config"]}>'
                f'({dense["input_var"]}, {normalize["output_var"]}, '
                f'{dense["weight"]}, {normalize["bias"]}); '
                f'// fused BitNet logits Dense_MatMul_{index} + bn_Add_{index}'
            )
        text = (
            text[: normalize_match.start()]
            + replacement
            + text[normalize_match.end() :]
        )

    if sigmoid is not None:
        sigmoid_match = SIGMOID_CALL.search(text)
        if sigmoid_match is None:
            raise RuntimeError("Could not find generated sigmoid call after dense patching")
        final_dense = stages[-1]["dense"]
        replacement = (
            f'{sigmoid["indent"]}nnet::patched_bitdense_accum_sigmoid<'
            f'{final_dense["input_type"]}, {sigmoid["output_type"]}, '
            f'{final_dense["config"]}, {sigmoid["config"]}>'
            f'({final_dense["input_var"]}, {sigmoid["output_var"]}, '
            f'{final_dense["weight"]}); '
            "// fused final BitNet Dense + accumulator-table Sigmoid"
        )
        text = text[: sigmoid_match.start()] + replacement + text[sigmoid_match.end() :]
    source_path.write_text(text, encoding="utf-8")


def patch_project(project_dir: Path, layers: list[dict]) -> dict:
    source = (project_dir / "firmware" / "myproject.cpp").read_text(encoding="utf-8")
    binary_sigmoid = int(layers[-1]["weight"].shape[0]) == 1
    stages, sigmoid = _parse_generated_calls(source, len(layers), binary_sigmoid)
    final_scale = _patch_cumulative_biases(project_dir, layers, stages)
    table_minimum = table_step = None
    table = None
    if binary_sigmoid:
        table_minimum, table_step = _accumulator_table_spec(final_scale)
        table = _make_sigmoid_table(
            final_scale,
            float(layers[-1]["bias"][0]),
            table_minimum,
            table_step,
        )
    _patch_parameters(
        project_dir,
        stages,
        final_scale,
        table,
        table_minimum,
        table_step,
    )
    _patch_network(project_dir, stages, sigmoid)
    metadata = {
        "implementation": IMPLEMENTATION,
        "layer_dimensions": [
            int(layers[0]["weight"].shape[1]),
            *(int(layer["weight"].shape[0]) for layer in layers),
        ],
        "dense_configs": [stage["dense"]["config"] for stage in stages],
        "normalization_configs": [
            stage["normalize"]["config"] for stage in stages
        ],
        "final_cumulative_scale": final_scale,
        "output_boundary": "binary_sigmoid" if binary_sigmoid else "multiclass_logits",
        "weight_values": sorted(
            {int(value) for layer in layers for value in np.unique(layer["weight"])}
        ),
    }
    if binary_sigmoid:
        metadata["final_bias"] = float(layers[-1]["bias"][0])
        metadata["accumulator_table"] = {
            "minimum": table_minimum,
            "step": table_step,
            "size": ACCUM_TABLE_SIZE,
            "target_logit_step": TARGET_SIGMOID_LOGIT_STEP,
        }
    (project_dir / "patched_bitnet_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _validate_hls_model(hls_model, layers: list[dict], samples: int) -> dict:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(samples, layers[0]["weight"].shape[1])).astype(
        np.float32
    )
    binary_sigmoid = int(layers[-1]["weight"].shape[0]) == 1
    expected = (
        predict_folded(layers, values)
        if binary_sigmoid
        else predict_folded_logits(layers, values)
    ).reshape(-1)
    # ModelGraph.compile() rewrites generated sources and would discard the patch.
    hls_model._compile()
    observed = np.asarray(hls_model.predict(values)).reshape(-1)
    difference = np.abs(expected - observed)
    result = {
        "samples": samples,
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "finite": bool(np.isfinite(observed).all()),
        "expected_range": [float(expected.min()), float(expected.max())],
        "observed_range": [float(observed.min()), float(observed.max())],
        "worst_expected": float(expected[difference.argmax()]),
        "worst_observed": float(observed[difference.argmax()]),
    }
    tolerance = 0.05 if binary_sigmoid else 0.1
    result["tolerance"] = tolerance
    result["output_boundary"] = "binary_sigmoid" if binary_sigmoid else "multiclass_logits"
    if not result["finite"] or result["maximum_absolute_error"] > tolerance:
        raise RuntimeError(f"Patched hls4ml validation failed: {result}")
    return result


def export_quantized_checkpoint(checkpoint_path: Path, output_path: Path) -> Path:
    import torch

    from bitnet_layers import BitLinear
    from model_registry import build_registered_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    input_dim = len(metadata["feature_names"])
    output_dim = int(
        metadata.get(
            "output_dim",
            checkpoint.get("config", {}).get("model", {}).get(
                "output_dim", len(metadata.get("class_names", [])) or 1
            ),
        )
    )
    model = build_registered_model(checkpoint["config"], input_dim, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, input_dim))
    bitnet_layers = [module for module in model.modules() if isinstance(module, BitLinear)]
    if not bitnet_layers:
        raise ValueError(f"Checkpoint does not contain BitNet layers: {checkpoint_path}")
    for layer in bitnet_layers:
        layer.quant_export = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


def write_project(
    run_name: str,
    quantized_state: Path,
    part: str,
    clock_period: float,
    output_dir: Path,
    validation_samples: int,
) -> dict:
    import onnx
    from hls4ml.converters import (
        convert_from_onnx_model,
        parse_onnx_model,
    )

    layers = load_quantized_layers(quantized_state)
    if any(not np.all(np.isin(layer["weight"], (-1, 0, 1))) for layer in layers):
        raise ValueError(
            "Patched BitNet weights must be binary {-1, +1} or ternary {-1, 0, +1}"
        )
    binary_sigmoid = int(layers[-1]["weight"].shape[0]) == 1
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{run_name}_hls4ml_input.onnx"
    build_hls4ml_onnx(layers, onnx_path, binary_sigmoid)
    model = onnx.load(onnx_path)
    parsed_layers, input_layers, output_layers = parse_onnx_model(model)
    config = make_hls_config(model, len(layers), binary_sigmoid)
    hls_model = convert_from_onnx_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        project_name="myproject",
        backend="Vivado",
        part=part,
        clock_period=clock_period,
        io_type="io_parallel",
    )
    hls_model.write()
    patch_metadata = patch_project(output_dir, layers)
    validation = (
        _validate_hls_model(hls_model, layers, validation_samples)
        if validation_samples
        else None
    )
    summary = {
        "run_name": run_name,
        "hls4ml_version": importlib.metadata.version("hls4ml"),
        "quantized_state": str(quantized_state),
        "output_dir": str(output_dir),
        "input_layers": input_layers,
        "output_layers": output_layers,
        "parsed_layers": [
            {"name": item.get("name"), "class_name": item.get("class_name")}
            for item in parsed_layers
        ],
        "patch": patch_metadata,
        "validation": validation,
    }
    (output_dir / "hls4ml_generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def synthesize(
    run_name: str,
    quantized_state: Path,
    part: str,
    clock_period: float,
    output_dir: Path,
    validation_samples: int,
    profile: str,
    allow_unverified_license: bool,
) -> dict:
    summary = write_project(
        run_name,
        quantized_state,
        part,
        clock_period,
        output_dir,
        validation_samples,
    )
    preflight = run_preflight()
    hls = preflight["vitis_hls"]
    if not hls["available"]:
        raise RuntimeError("Vitis HLS is unavailable")
    if not preflight["license"]["available"] and not allow_unverified_license:
        raise RuntimeError(
            "Xilinx license availability is unverified; pass "
            "--allow-unverified-license to attempt synthesis"
        )

    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(hls["path"]).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    synthesis_tcl = output_dir / "csynth_patched.tcl"
    synthesis_tcl.write_text(
        "\n".join(
            [
                "open_project -reset myproject_prj",
                "set_top myproject",
                'add_files firmware/myproject.cpp -cflags "-std=c++0x"',
                'open_solution -reset "solution1"',
                "config_compile -name_max_length 80",
                "config_schedule -enable_dsp_full_reg=false",
                f'set_part "{part}"',
                f"create_clock -period {clock_period} -name default",
                "set_clock_uncertainty 12.5% default",
                "csynth_design",
                "exit",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log_path = output_dir / "synthesis.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [hls["path"], "-f", synthesis_tcl.name],
            cwd=output_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Synthesis failed; see {log_path}")

    report_dir = (
        output_dir / "myproject_prj" / "solution1" / "syn" / "report"
    )
    xml = report_dir / "myproject_csynth.xml"
    report = parse_csynth_xml(xml)
    result_dir = ROOT / "results" / profile / "synthesis" / f"{run_name}_{IMPLEMENTATION}"
    result_dir.mkdir(parents=True, exist_ok=True)
    copied_xml = result_dir / f"{run_name}_{IMPLEMENTATION}_csynth.xml"
    shutil.copy2(xml, copied_xml)
    rpt = xml.with_suffix(".rpt")
    copied_rpt = None
    if rpt.exists():
        copied_rpt = result_dir / f"{run_name}_{IMPLEMENTATION}_csynth.rpt"
        shutil.copy2(rpt, copied_rpt)

    result = {
        "run_name": run_name,
        "implementation": IMPLEMENTATION,
        "tool": "Vitis HLS",
        "tool_version": report.get("tool_version") or hls.get("version"),
        "part": report.get("part"),
        "clock_target_ns": report.get("clock_target_ns"),
        "clock_achieved_ns": report.get("clock_achieved_ns"),
        "latency_cycles": report.get("latency_cycles_max"),
        "initiation_interval_cycles": report.get(
            "initiation_interval_cycles"
        ),
        "lut": report.get("lut"),
        "ff": report.get("ff"),
        "dsp": report.get("dsp"),
        "bram_18k": report.get("bram"),
        "uram": report.get("uram"),
        "synthesis_status": "success",
        "place_and_route_status": "not_run",
        "implementation_boundary": (
            "binary sigmoid output; no softmax"
            if summary["patch"]["output_boundary"] == "binary_sigmoid"
            else "multiclass logits output; no softmax"
        ),
        "generation": summary,
        "report_files": {
            "xml": str(copied_xml.relative_to(ROOT)),
            "rpt": str(copied_rpt.relative_to(ROOT)) if copied_rpt else None,
        },
    }
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--quantized-state", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--part", default="xcvu13p-flga2577-2-e")
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--profile", choices=("20-epochs", "200-epochs"), required=True)
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--allow-unverified-license", action="store_true")
    args = parser.parse_args()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "hls_projects" / args.profile / args.run_name / IMPLEMENTATION
    )
    quantized_state = (
        args.quantized_state.resolve()
        if args.quantized_state
        else export_quantized_checkpoint(
            args.checkpoint.resolve(),
            ROOT
            / "data"
            / args.profile
            / "synthesis"
            / "quantized"
            / f"{args.run_name}.pt",
        )
    )
    operation = write_project if args.write_only else synthesize
    keywords = {
        "run_name": args.run_name,
        "quantized_state": quantized_state,
        "part": args.part,
        "clock_period": args.clock_period,
        "output_dir": output_dir,
        "validation_samples": args.validation_samples,
    }
    if not args.write_only:
        keywords["profile"] = args.profile
        keywords["allow_unverified_license"] = args.allow_unverified_license
    print(json.dumps(operation(**keywords), indent=2))


if __name__ == "__main__":
    main()
