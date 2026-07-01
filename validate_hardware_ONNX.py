import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from benchmark import ROOT


def reference_output(state_dict, x):
    layers = [
        key[: -len(".int_weight")]
        for key in state_dict
        if key.endswith(".int_weight")
    ]
    output = torch.from_numpy(x)
    for index, layer_name in enumerate(layers):
        weight = state_dict[f"{layer_name}.int_weight"].float()
        if f"{layer_name}.beta_shift" in state_dict:
            beta = torch.pow(
                torch.tensor(2.0),
                state_dict[f"{layer_name}.beta_shift"].float(),
            )
        else:
            beta = state_dict[f"{layer_name}.beta_scale"].float()
        output = (
            output @ weight.T * beta + state_dict[f"{layer_name}.bias"]
        )
        if index != len(layers) - 1:
            output = torch.relu(output)
    return output.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Validate explicit BitNet ONNX files against quantized states."
    )
    parser.add_argument("--directory", default="onnx/hardware")
    args = parser.parse_args()
    directory = ROOT / args.directory
    paths = sorted(directory.glob("*.onnx"))
    if not paths:
        raise FileNotFoundError(f"No ONNX files under {directory}")

    rng = np.random.default_rng(42)
    for path in paths:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        input_dim = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        x = rng.normal(size=(31, input_dim)).astype(np.float32)
        state_path = path.with_name(f"{path.stem}_quantized.pt")
        state_dict = torch.load(
            state_path, map_location="cpu", weights_only=True
        )
        reference = reference_output(state_dict, x)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {"input": x})[0]
        maximum_error = float(np.max(np.abs(reference - actual)))
        if maximum_error > 1e-4:
            raise AssertionError(f"{path.name}: max error {maximum_error}")
        print(f"{path.name}: max_abs_error={maximum_error:.3e}")


if __name__ == "__main__":
    main()
