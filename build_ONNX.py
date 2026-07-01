import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(ROOT))

from benchmark import ROOT, load_checkpoint, load_config
from model_registry import resolve_model_name


BITNET_MODELS = {
    "bitnet_mlp",
    "bitnet_topo",
    "bit158_mlp",
    "bit158_topo",
    "binary_large",
    "ternary_large",
}


def quantized_state_dict(model, sample):
    model.eval()
    with torch.no_grad():
        model(sample)
    state_dict = {}
    for name, layer in model.named_modules():
        if not hasattr(layer, "int_weight") or layer.int_weight is None:
            continue
        state_dict[f"{name}.int_weight"] = layer.int_weight.detach().cpu()
        if layer.beta_quant == "power2":
            state_dict[f"{name}.beta_shift"] = (
                layer.beta_shift.detach().cpu().to(torch.int32)
            )
        else:
            state_dict[f"{name}.beta_scale"] = layer.beta.detach().cpu()
        bias = layer.bias_q if layer.bias_q is not None else layer.bias
        if bias is not None:
            state_dict[f"{name}.bias"] = bias.detach().cpu()
    if not state_dict:
        raise ValueError("Model contains no exportable BitLinear layers")
    return state_dict


def layer_info_from_state_dict(state_dict):
    layer_info = {}
    for key, value in state_dict.items():
        layer_name, field = key.rsplit(".", 1)
        info = layer_info.setdefault(layer_name, {})
        if field == "int_weight":
            info[field] = value.numpy()
        elif field == "beta_scale":
            info["beta"] = float(value.item())
        elif field == "beta_shift":
            info[field] = int(value.item())
        elif field == "bias":
            info[field] = value.numpy()
    return layer_info, list(layer_info)


def build_explicit_model(layer_info, layers, input_dim, output_dim):
    graph_inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [None, input_dim]
        )
    ]
    initializers = []
    nodes = []
    previous = "input"

    for layer_name in layers:
        info = layer_info[layer_name]
        int_weight = np.asarray(info["int_weight"], dtype=np.float32)
        weight_name = f"{layer_name}_W"
        initializers.append(
            helper.make_tensor(
                weight_name,
                TensorProto.FLOAT,
                int_weight.T.shape,
                int_weight.T.flatten(),
            )
        )
        dot = f"{layer_name}_dot"
        nodes.append(
            helper.make_node(
                "MatMul",
                [previous, weight_name],
                [dot],
                name=f"{layer_name}_matmul",
            )
        )

        if "beta_shift" in info:
            shift = info["beta_shift"]
            beta = 2.0**shift
            beta_name = f"{layer_name}_beta_power2_shift_{shift}"
            mul_name = f"{layer_name}_mul_power2_shift_{shift}"
        else:
            beta = info["beta"]
            beta_name = f"{layer_name}_beta"
            mul_name = f"{layer_name}_mul"
        initializers.append(
            helper.make_tensor(
                beta_name,
                TensorProto.FLOAT,
                [1],
                np.asarray([beta], dtype=np.float32),
            )
        )
        scaled = f"{layer_name}_scaled"
        nodes.append(
            helper.make_node(
                "Mul", [dot, beta_name], [scaled], name=mul_name
            )
        )

        bias = np.asarray(info["bias"], dtype=np.float32)
        bias_name = f"{layer_name}_bias"
        initializers.append(
            helper.make_tensor(
                bias_name, TensorProto.FLOAT, bias.shape, bias.flatten()
            )
        )
        preactivation = f"{layer_name}_preact"
        nodes.append(
            helper.make_node(
                "Add",
                [scaled, bias_name],
                [preactivation],
                name=f"{layer_name}_add",
            )
        )
        if layer_name != layers[-1]:
            previous = f"{layer_name}_relu"
            nodes.append(
                helper.make_node(
                    "Relu",
                    [preactivation],
                    [previous],
                    name=f"{layer_name}_relu_node",
                )
            )
        else:
            previous = preactivation

    nodes.append(
        helper.make_node("Identity", [previous], ["logits"], name="logits_out")
    )
    graph_outputs = [
        helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, [None, output_dim]
        )
    ]
    graph = helper.make_graph(
        nodes,
        "BitNetDNN_ExplicitWeightsAndScale",
        graph_inputs,
        graph_outputs,
        initializers,
    )
    return helper.make_model(
        graph,
        producer_name="bitnet_to_onnx",
        opset_imports=[helper.make_opsetid("", 13)],
    )


def export_bitnet(config, output_dir):
    model, checkpoint, _ = load_checkpoint(config, torch.device("cpu"))
    input_dim = len(checkpoint["metadata"]["feature_names"])
    output_dim = int(
        checkpoint["metadata"].get("output_dim", len(checkpoint["metadata"]["class_names"]))
    )
    sample = torch.zeros(1, input_dim)
    state_dict = quantized_state_dict(model, sample)
    layer_info, layers = layer_info_from_state_dict(state_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{config['run_name']}_quantized.pt"
    output_path = output_dir / f"{config['run_name']}.onnx"
    torch.save(state_dict, state_path)
    onnx_model = build_explicit_model(
        layer_info, layers, input_dim, output_dim
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)

    print(f"Saved quantized state dictionary: {state_path}")
    print(f"Saved explicit BitNet ONNX model: {output_path}")
    for layer_name in layers:
        info = layer_info[layer_name]
        unique = np.unique(info["int_weight"]).astype(int).tolist()
        scale = (
            f"beta_shift={info['beta_shift']}"
            if "beta_shift" in info
            else f"beta={info['beta']:.8g}"
        )
        print(f"  {layer_name}: values={unique}, {scale}")


def export_traced(config, output_dir):
    model, checkpoint, _ = load_checkpoint(config, torch.device("cpu"))
    model.eval()
    input_dim = len(checkpoint["metadata"]["feature_names"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config['run_name']}.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, input_dim),
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=config["onnx"].get("opset", 17),
        dynamo=False,
    )
    print(f"Saved traced ONNX model: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export a saved DNN to ONNX.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        default="onnx/hardware",
        help="Output directory relative to this benchmark.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Use ordinary torch.onnx tracing instead of explicit BitNet export.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = ROOT / args.output_dir
    if resolve_model_name(config) in BITNET_MODELS and not args.trace:
        export_bitnet(config, output_dir)
    else:
        export_traced(config, output_dir)


if __name__ == "__main__":
    main()
