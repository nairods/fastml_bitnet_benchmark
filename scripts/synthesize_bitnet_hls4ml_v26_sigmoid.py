#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

import hls4ml
from hls4ml.converters import convert_from_onnx_model, parse_onnx_model
from hls4ml.utils.config import config_from_onnx_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_benchmark.bitnet import load_quantized_layers
from hardware_benchmark.preflight import run_preflight
from hardware_benchmark.reports import parse_csynth_xml


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path("/tmp/bitnet_hls4ml_v26_sigmoid")
PARENT_V26_SCRIPT = ROOT.parent / "test_hls_synthesis_onnx.py"
ACCUM_TABLE_MIN = -32768
ACCUM_TABLE_STEP = 32
ACCUM_TABLE_SIZE = 2048


def build_hls4ml_onnx(layers: list[dict], output_path: Path) -> None:
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
            helper.make_tensor_value_info(weight_name, TensorProto.FLOAT, weight.shape)
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

        beta = np.asarray([layer["beta"]], dtype=np.float32)
        beta_name = f"s{index}"
        initializers.append(helper.make_tensor(beta_name, TensorProto.FLOAT, [1], beta))
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
            helper.make_tensor(bias_name, TensorProto.FLOAT, bias.shape, bias.reshape(-1))
        )
        value_infos.append(
            helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, bias.shape)
        )
        preact = f"add_{index}_out"
        value_infos.append(
            helper.make_tensor_value_info(preact, TensorProto.FLOAT, [None, n_out])
        )
        nodes.append(
            helper.make_node("Add", [scaled, bias_name], [preact], name=f"Add_{index}")
        )
        if index == len(layers) - 1:
            previous = preact
        else:
            previous = f"relu_{index}_out"
            value_infos.append(
                helper.make_tensor_value_info(previous, TensorProto.FLOAT, [None, n_out])
            )
            nodes.append(
                helper.make_node("Relu", [preact], [previous], name=f"Relu_{index}")
            )

    nodes.append(helper.make_node("Sigmoid", [previous], ["probability"], name="Sigmoid_0"))
    value_infos.append(
        helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])
    )
    graph = helper.make_graph(
        nodes,
        "BitNet_hls4ml_v26_sigmoid",
        graph_inputs,
        [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])],
        initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="opendata_bitnet_hls4ml_v26",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def make_config(model):
    config = config_from_onnx_model(
        model,
        granularity="name",
        backend="Vivado",
        default_precision="fixed<16,6>",
    )
    config["Model"]["ReuseFactor"] = 1
    for name, layer in config["LayerName"].items():
        layer["Trace"] = True
        layer["ReuseFactor"] = 1

    matmuls = [name for name in config["LayerName"] if name.startswith("MatMul_")]
    relus = [name for name in config["LayerName"] if name.startswith("Relu_")]
    adds = [name for name in config["LayerName"] if name.startswith("Add_")]

    accum = ["ap_fixed<28,10>", "ap_fixed<32,14>", "ap_fixed<34,16>", "ap_fixed<40,22>"]
    for index, name in enumerate(sorted(matmuls, key=lambda item: int(item.rsplit("_", 1)[1]))):
        precision = config["LayerName"][name].setdefault("Precision", {})
        precision["weight"] = "ap_int<2>"
        precision["bias"] = "ap_int<1>"
        precision["accum"] = accum[index]
        precision["result"] = "ap_fixed<20,6,AP_RND,AP_SAT>"

    add_precision = [
        "ap_fixed<28,10,AP_RND,AP_SAT>",
        "ap_fixed<32,14,AP_RND,AP_SAT>",
        "ap_fixed<34,16,AP_RND,AP_SAT>",
        "ap_fixed<37,13,AP_RND,AP_SAT>",
    ]
    for index, name in enumerate(sorted(adds, key=lambda item: int(item.rsplit("_", 1)[1]))):
        config["LayerName"][name].setdefault("Precision", {})["result"] = add_precision[index]

    relu_precision = [
        "ap_ufixed<24,8,AP_RND,AP_SAT>",
        "ap_ufixed<26,11,AP_RND,AP_SAT>",
        "ap_ufixed<28,14,AP_RND,AP_SAT>",
    ]
    for index, name in enumerate(sorted(relus, key=lambda item: int(item.rsplit("_", 1)[1]))):
        config["LayerName"][name]["TableSize"] = 1024
        config["LayerName"][name].setdefault("Precision", {})["result"] = relu_precision[index]
        config["LayerName"][name]["Precision"]["table"] = "ap_fixed<18,8>"

    if "Sigmoid_0" in config["LayerName"]:
        config["LayerName"]["Sigmoid_0"]["TableSize"] = 1024
        config["LayerName"]["Sigmoid_0"].setdefault("Precision", {})["result"] = "ap_ufixed<8,0,AP_RND,AP_SAT>"
        config["LayerName"]["Sigmoid_0"]["Precision"]["table"] = "ap_fixed<18,8>"
    return config


def _extract_parent_bitdense_helpers() -> str:
    tree = ast.parse(PARENT_V26_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(target, "id", None) == "BITDENSE_ALPHA_FUNCTION" for target in node.targets):
            helpers = node.value.value
            if "bitdense_latency_direct_accum_sigmoid_table" not in helpers:
                raise RuntimeError("Parent helper block does not contain the v26 final kernel")
            return helpers
    raise RuntimeError(f"Could not find BITDENSE_ALPHA_FUNCTION in {PARENT_V26_SCRIPT}")


def _parse_weight_values(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.array([], dtype=np.float64)
    return np.array(
        [float(item.strip()) for item in text.replace("\n", " ").split(",") if item.strip()],
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
    text = re.sub(
        rf"(#else\s*\nmodel_default_t {name}\[[0-9]+\] = )\{{[^}}]*\}};",
        rf"\1{{{payload}}};",
        text,
        count=1,
        flags=re.S,
    )
    header_path.write_text(text, encoding="utf-8")


def _format_hls_array(values: np.ndarray, values_per_line: int = 8) -> str:
    lines = []
    for start in range(0, len(values), values_per_line):
        chunk = values[start : start + values_per_line]
        lines.append("    " + ", ".join(f"{float(value):.10f}" for value in chunk))
    return ",\n".join(lines)


def _patch_parameters(project_dir: Path) -> None:
    parameters_path = project_dir / "firmware" / "parameters.h"
    text = parameters_path.read_text(encoding="utf-8")
    if "void bitdense_alpha(" not in text:
        marker = "// hls-fpga-machine-learning insert layer-config"
        text = text.replace(marker, _extract_parent_bitdense_helpers() + "\n" + marker, 1)

    alpha_configs = [
        ("config38", 0),
        ("config39", 1),
        ("config40", 2),
        ("config41", 3),
    ]
    mult_t_by_layer = {
        0: "ap_fixed<17,7,AP_RND,AP_SAT>",
        1: "ap_fixed<25,9,AP_RND,AP_SAT>",
        2: "ap_fixed<28,12,AP_RND,AP_SAT>",
        3: "ap_fixed<31,15,AP_RND,AP_SAT>",
    }
    for config_name, dense_index in alpha_configs:
        start = text.index(f"struct {config_name} : nnet::dense_config {{")
        end = text.index("};", start)
        if "alpha_bias_t" in text[start:end]:
            continue
        typedef_line = f"    typedef dense_matmul_{dense_index}_accum_t accum_t;"
        insert_at = text.index(typedef_line, start) + len(typedef_line)
        fields = (
            "\n"
            f"    typedef {mult_t_by_layer[dense_index]} mult_t;\n"
            "    typedef model_default_t alpha_bias_t;\n"
            "    typedef ap_fixed<56,34,AP_RND,AP_SAT> alpha_prod_t;\n"
            "    typedef ap_fixed<20,6,AP_RND,AP_SAT> sigmoid_input_t;\n"
            "    static const int alpha_num = 1;\n"
            "    static const int alpha_shift = 0;\n"
            f"    static const int accum_table_min = {ACCUM_TABLE_MIN};\n"
            f"    static const int accum_table_step = {ACCUM_TABLE_STEP};\n"
            f"    static const int accum_table_size = {ACCUM_TABLE_SIZE};"
        )
        text = text[:insert_at] + fields + text[insert_at:]
    parameters_path.write_text(text, encoding="utf-8")


def _patch_cumulative_alpha(project_dir: Path, layers: list[dict]) -> float:
    defines_path = project_dir / "firmware" / "defines.h"
    text = defines_path.read_text(encoding="utf-8")
    text = text.replace(
        "typedef ap_fixed<16,6> model_default_t;",
        "typedef ap_fixed<32,16> model_default_t;",
    )
    defines_path.write_text(text, encoding="utf-8")

    weights_dir = project_dir / "firmware" / "weights"
    cumulative = []
    scale = 1.0
    for layer in layers:
        scale *= float(layer["beta"])
        cumulative.append(scale)
    for name, divisor in [("b31", cumulative[0]), ("b33", cumulative[1]), ("b35", cumulative[2])]:
        values = _parse_weight_values(weights_dir / f"{name}.txt")
        _write_weight_values(weights_dir, name, values / divisor)
    return cumulative[-1]


def _patch_sigmoid_table(project_dir: Path, final_scale: float) -> None:
    weights_dir = project_dir / "firmware" / "weights"
    final_bias = float(_parse_weight_values(weights_dir / "b37.txt")[0])
    indices = np.arange(ACCUM_TABLE_SIZE, dtype=np.float64)
    accum_centers = ACCUM_TABLE_MIN + (indices + 0.5) * ACCUM_TABLE_STEP
    table_values = 1.0 / (1.0 + np.exp(-(accum_centers * final_scale + final_bias)))
    table_text = (
        f"static const result_t accum_sigmoid_table_config41[{ACCUM_TABLE_SIZE}] = {{\n"
        f"{_format_hls_array(table_values)}\n"
        "};\n"
    )
    parameters_path = project_dir / "firmware" / "parameters.h"
    text = parameters_path.read_text(encoding="utf-8")
    marker = "// hls-bitdense insert accumulator sigmoid tables"
    if marker not in text:
        raise RuntimeError(f"Could not find sigmoid table marker in {parameters_path}")
    text = text.replace(marker, table_text + "\n" + marker, 1)
    parameters_path.write_text(text, encoding="utf-8")


def _patch_myproject(project_dir: Path) -> None:
    myproject_path = project_dir / "firmware" / "myproject.cpp"
    text = myproject_path.read_text(encoding="utf-8")
    for layer_var, size in [("layer38_out", 64), ("layer39_out", 32), ("layer40_out", 32), ("layer41_out", 1)]:
        declaration = (
            f"    {layer_var.replace('_out', '_t')} {layer_var}[{size}];\n"
            f"    #pragma HLS ARRAY_PARTITION variable={layer_var} complete dim=0\n"
        )
        guarded = (
            "#ifndef __SYNTHESIS__\n"
            f"    {layer_var.replace('_out', '_t')} {layer_var}[{size}];\n"
            f"    #pragma HLS ARRAY_PARTITION variable={layer_var} complete dim=0\n"
            "#endif\n"
        )
        if declaration in text:
            text = text.replace(declaration, guarded, 1)

    replacements = [
        (
            "    nnet::dense<input_t, layer38_t, config38>(global_in, layer38_out, w38, b38); // Dense_MatMul_0",
            "#ifndef __SYNTHESIS__\n"
            "    nnet::dense<input_t, layer38_t, config38>(global_in, layer38_out, w38, b38); // Dense_MatMul_0\n"
            "#endif",
        ),
        (
            "    nnet::normalize<layer38_t, layer31_t, config31>(layer38_out, layer31_out, s31, b31); // bn_Add_0",
            "    nnet::bitdense_bias<input_t, layer31_t, config38>(global_in, layer31_out, w38, b31); // fused Dense_MatMul_0 + bn_Add_0",
        ),
        (
            "    nnet::dense<layer17_t, layer39_t, config39>(layer17_out, layer39_out, w39, b39); // Dense_MatMul_1",
            "#ifndef __SYNTHESIS__\n"
            "    nnet::dense<layer17_t, layer39_t, config39>(layer17_out, layer39_out, w39, b39); // Dense_MatMul_1\n"
            "#endif",
        ),
        (
            "    nnet::normalize<layer39_t, layer33_t, config33>(layer39_out, layer33_out, s33, b33); // bn_Add_1",
            "    nnet::bitdense_bias<layer17_t, layer33_t, config39>(layer17_out, layer33_out, w39, b33); // fused Dense_MatMul_1 + bn_Add_1",
        ),
        (
            "    nnet::dense<layer21_t, layer40_t, config40>(layer21_out, layer40_out, w40, b40); // Dense_MatMul_2",
            "#ifndef __SYNTHESIS__\n"
            "    nnet::dense<layer21_t, layer40_t, config40>(layer21_out, layer40_out, w40, b40); // Dense_MatMul_2\n"
            "#endif",
        ),
        (
            "    nnet::normalize<layer40_t, layer35_t, config35>(layer40_out, layer35_out, s35, b35); // bn_Add_2",
            "    nnet::bitdense_bias<layer21_t, layer35_t, config40>(layer21_out, layer35_out, w40, b35); // fused Dense_MatMul_2 + bn_Add_2",
        ),
        (
            "    nnet::dense<layer25_t, layer41_t, config41>(layer25_out, layer41_out, w41, b41); // Dense_MatMul_3",
            "#ifndef __SYNTHESIS__\n"
            "    nnet::dense<layer25_t, layer41_t, config41>(layer25_out, layer41_out, w41, b41); // Dense_MatMul_3\n"
            "#endif",
        ),
        (
            "    nnet::normalize<layer41_t, layer37_t, config37>(layer41_out, layer37_out, s37, b37); // bn_Add_3",
            "#ifndef __SYNTHESIS__\n"
            "    nnet::bitdense_latency_alpha<layer25_t, layer37_t, config41>(layer25_out, layer37_out, w41, b37); // fused Dense_MatMul_3 + bn_Add_3\n"
            "#endif",
        ),
        (
            "    nnet::sigmoid<layer37_t, result_t, sigmoid_config29>(layer37_out, layer29_out); // Sigmoid_0",
            "    nnet::bitdense_latency_direct_accum_sigmoid_table<layer25_t, result_t, config41, sigmoid_config29>(layer25_out, layer29_out, w41, b37); // fused Dense_MatMul_3 + direct accumulator-table Sigmoid_0",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"Could not patch generated myproject.cpp; missing: {old}")
        text = text.replace(old, new, 1)
    myproject_path.write_text(text, encoding="utf-8")


def patch_project(project_dir: Path, layers: list[dict]) -> dict:
    _patch_parameters(project_dir)
    final_scale = _patch_cumulative_alpha(project_dir, layers)
    _patch_sigmoid_table(project_dir, final_scale)
    _patch_myproject(project_dir)
    metadata = {
        "patch": "parent_hls4ml_v26_bitdense_direct_accumulator_sigmoid_table",
        "parent_script": str(PARENT_V26_SCRIPT),
        "final_cumulative_scale": final_scale,
        "accum_table_min": ACCUM_TABLE_MIN,
        "accum_table_step": ACCUM_TABLE_STEP,
        "accum_table_size": ACCUM_TABLE_SIZE,
    }
    (project_dir / "v26_patch_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def write_project(run_name: str, part: str, clock_period: float, output_dir: Path) -> dict:
    layers = load_quantized_layers(ROOT / "onnx" / "hardware" / f"{run_name}_quantized.pt")
    onnx_path = output_dir / f"{run_name}_hls4ml_v26_input.onnx"
    build_hls4ml_onnx(layers, onnx_path)
    model = onnx.load(onnx_path)
    parsed_layers, input_layers, output_layers = parse_onnx_model(model)
    config = make_config(model)
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
    summary = {
        "run_name": run_name,
        "onnx": str(onnx_path),
        "output_dir": str(output_dir),
        "input_layers": input_layers,
        "output_layers": output_layers,
        "parsed_layers": [
            {"name": item.get("name"), "class_name": item.get("class_name")}
            for item in parsed_layers
        ],
        "config_layers": list(config["LayerName"].keys()),
        "patch_metadata": patch_metadata,
    }
    (output_dir / "hls4ml_generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def synthesize(run_name: str, part: str, clock_period: float, output_dir: Path, keep_project: bool) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    summary = write_project(run_name, part, clock_period, output_dir)
    preflight = run_preflight(ROOT)
    if not preflight["vitis_hls"]["available"]:
        raise RuntimeError("Vitis HLS is unavailable")
    env = os.environ.copy()
    env["PATH"] = str(Path(preflight["vitis_hls"]["path"]).parent) + os.pathsep + env.get("PATH", "")
    log_path = output_dir / "synthesis.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [preflight["vitis_hls"]["path"], "-f", "build_prj.tcl"],
            cwd=output_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Synthesis failed; see {log_path}")
    report_dir = output_dir / "myproject_prj" / "solution1" / "syn" / "report"
    xml = report_dir / "myproject_csynth.xml"
    rpt = report_dir / "myproject_csynth.rpt"
    report = parse_csynth_xml(xml)

    result_dir = ROOT / "results" / "synthesis" / f"{run_name}_hls4ml_v26_sigmoid"
    result_dir.mkdir(parents=True, exist_ok=True)
    copied_xml = result_dir / f"{run_name}_hls4ml_v26_sigmoid_csynth.xml"
    copied_rpt = result_dir / f"{run_name}_hls4ml_v26_sigmoid_csynth.rpt"
    shutil.copy2(xml, copied_xml)
    shutil.copy2(rpt, copied_rpt)
    shutil.copy2(log_path, result_dir / "synthesis.log")
    shutil.copy2(output_dir / "hls4ml_generation_summary.json", result_dir / "hls4ml_generation_summary.json")
    shutil.copy2(output_dir / "v26_patch_metadata.json", result_dir / "v26_patch_metadata.json")
    if keep_project:
        project_copy = result_dir / "hls4ml_project"
        if project_copy.exists():
            shutil.rmtree(project_copy)
        shutil.copytree(output_dir, project_copy, ignore=shutil.ignore_patterns("myproject_prj"))

    result = {
        "run_name": run_name,
        "variant": "hls4ml_parent_v26_patch_sigmoid",
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
        "implementation_boundary": "binary sigmoid output; no softmax",
        "generation": summary,
        "report_files": {
            "rpt": str(copied_rpt.relative_to(ROOT)),
            "xml": str(copied_xml.relative_to(ROOT)),
        },
    }
    (result_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="bitnet_binary_sigmoid_f7_fixed")
    parser.add_argument("--part", default="xcvu13p-flga2577-2-e")
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--keep-project", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or TMP_ROOT / f"{args.run_name}_hls4ml_v26"
    if args.write_only:
        print(json.dumps(write_project(args.run_name, args.part, args.clock_period, output_dir), indent=2))
    else:
        print(
            json.dumps(
                synthesize(args.run_name, args.part, args.clock_period, output_dir, args.keep_project),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
