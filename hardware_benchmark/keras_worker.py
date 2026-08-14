import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from hardware_benchmark.metrics import compare_predictions, softmax
from benchmark import load_dataset


def _write_result(output_dir: Path, result: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conversion.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


def _load_run_config(root: Path, run_name: str) -> dict:
    path = root / "logs" / "run_configs" / f"{run_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _qkeras_quantizers(config: dict, run_name: str):
    from qkeras.quantizers import binary, quantized_bits, quantized_relu, ternary

    options = config.get("model", {}).get("qkeras", {})
    mode = options.get("quantizer", "quantized_bits")
    bits = options.get("bits", 7)
    integer_bits = options.get("integer_bits", 0)
    alpha = options.get("alpha", 1)
    activation_bits = options.get("activation_bits", bits)
    if mode == "quantized_bits":
        weight_quantizer = quantized_bits(bits, integer_bits, alpha=alpha)
        bias_quantizer = quantized_bits(bits, integer_bits, alpha=alpha)
    elif mode == "binary":
        weight_quantizer = binary(alpha=alpha)
        bias_quantizer = binary(alpha=alpha)
    elif mode == "ternary":
        weight_quantizer = ternary(alpha=alpha)
        bias_quantizer = ternary(alpha=alpha)
    else:
        raise ValueError(f"Unsupported QKeras quantizer: {mode}")
    return weight_quantizer, bias_quantizer, quantized_relu(activation_bits), {
        "bits": bits,
        "integer_bits": integer_bits,
        "activation_bits": activation_bits,
        "alpha": alpha,
        "quantizer": mode,
    }


def _load_qkeras_weights(model, path: Path):
    from qkeras import QDense

    dense_layers = [layer for layer in model.layers if isinstance(layer, QDense)]
    with h5py.File(path, "r") as handle:
        dependencies = handle["_layer_checkpoint_dependencies"]
        groups = [name for name in dependencies if name.startswith("q_dense")]
        groups.sort(
            key=lambda name: int(name.rsplit("_", 1)[-1])
            if name.rsplit("_", 1)[-1].isdigit()
            else 0
        )
        if len(groups) != len(dense_layers):
            raise ValueError(
                f"Expected {len(dense_layers)} QDense groups, found {len(groups)}"
            )
        for layer, group_name in zip(dense_layers, groups):
            values = dependencies[group_name]["vars"]
            layer.set_weights([values["0"][()], values["1"][()]])


def build_qkeras(run_name: str, weights: Path, config: dict):
    import tensorflow as tf
    from qkeras import QActivation, QDense

    hidden_dims = config.get("model", {}).get("hidden_dims", [64, 32, 32])
    output_dim = int(config.get("model", {}).get("output_dim", 5))
    kernel_quantizer, bias_quantizer, activation_quantizer, quantizer_metadata = (
        _qkeras_quantizers(config, run_name)
    )

    model = tf.keras.Sequential(name="qkeras_mlp")
    model.add(tf.keras.layers.Input(shape=(16,)))
    for width in hidden_dims:
        model.add(
            QDense(
                width,
                kernel_quantizer=kernel_quantizer,
                bias_quantizer=bias_quantizer,
            )
        )
        model.add(QActivation(activation_quantizer))
    model.add(
        QDense(
            output_dim,
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
        )
    )
    _load_qkeras_weights(model, weights)
    return model, {
        **quantizer_metadata,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "weight_restore": "explicit_hdf5_assignment",
    }


def _uses_binary_sigmoid(config: dict) -> bool:
    model = config.get("model", {})
    return int(model.get("output_dim", 5)) == 1 and model.get("output_mode") == "binary_sigmoid"


def _append_sigmoid(model):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=model.input_shape[1:], name="features")
    logits = model(inputs)
    outputs = tf.keras.layers.Activation("sigmoid", name="hardware_sigmoid")(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{model.name}_with_sigmoid")


def _load_hgq_weights(model, path: Path):
    with h5py.File(path, "r") as handle:
        if "layers" in handle:
            layers = handle["layers"]
        elif "_layer_checkpoint_dependencies" in handle:
            layers = handle["_layer_checkpoint_dependencies"]
        else:
            raise KeyError("No HGQ layer weights found in checkpoint")
        for layer in model.layers:
            if layer.name not in layers:
                continue
            variables = layers[layer.name]["vars"]
            layer.set_weights(
                [variables[str(index)][()] for index in range(len(variables))]
            )


def build_hgq(weights: Path, profile: np.ndarray, config: dict):
    import keras
    import tensorflow as tf
    from HGQ import to_proxy_model, trace_minmax
    from HGQ.layers import HDense, HQuantize

    options = config.get("model", {}).get("hgq", {})
    beta = options.get("beta", 3e-6)
    hidden_dims = config.get("model", {}).get("hidden_dims", [64, 32, 32])
    output_dim = int(config.get("model", {}).get("output_dim", 5))
    layers = [keras.layers.Input(shape=(16,)), HQuantize(beta=beta)]
    layers.extend(HDense(width, beta=beta, activation="relu") for width in hidden_dims)
    layers.append(HDense(output_dim, beta=beta))
    model = keras.models.Sequential(
        layers,
        name="hgq_mlp",
    )
    model(tf.zeros((1, 16)))
    _load_hgq_weights(model, weights)
    trace_minmax(model, profile, bsz=1024, verbose=False)
    proxy = to_proxy_model(model)
    return model, proxy, {
        "weight_restore": "explicit_hdf5_assignment",
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "beta": beta,
        "calibration_samples": int(len(profile)),
        "proxy_layers": len(proxy.layers),
    }


def convert_model(model, output_dir: Path):
    import hls4ml

    config = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vivado"
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        project_name="jet_classifier",
        backend="Vivado",
        part="xcvu13p-flga2577-2-e",
        clock_period=5.0,
        io_type="io_parallel",
    )
    hls_model.write()
    return len(hls_model.get_layers())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("qkeras", "hgq"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()

    config = _load_run_config(args.root, args.run_name)
    arrays = load_dataset(config)
    inputs = arrays["x_test"]
    labels = arrays["y_test"]
    reference = np.load(
        args.root
        / "data/synthesis/reference_predictions"
        / f"{args.run_name}.npy",
        mmap_mode="r",
    )
    limit = min(args.samples, len(inputs))

    if args.backend == "qkeras":
        model, metadata = build_qkeras(args.run_name, args.weights, config)
        conversion_model = _append_sigmoid(model) if _uses_binary_sigmoid(config) else model
    else:
        profile = arrays["x_train"]
        model, conversion_model, metadata = build_hgq(
            args.weights, profile[: max(limit, 4096)], config
        )
        if _uses_binary_sigmoid(config):
            conversion_model = _append_sigmoid(conversion_model)
    logits = np.asarray(model.predict(inputs[:limit], batch_size=1024, verbose=0))
    if logits.ndim == 1 or logits.shape[1] == 1:
        signal = 1.0 / (1.0 + np.exp(-logits.reshape(-1, 1)))
        probabilities = np.concatenate([1.0 - signal, signal], axis=1)
    else:
        probabilities = softmax(logits)
    validation = compare_predictions(
        probabilities, reference[:limit], labels[:limit]
    )
    layer_count = convert_model(conversion_model, args.output)
    result = {
        "status": "project_generated",
        "backend": args.backend,
        "run_name": args.run_name,
        "layers": layer_count,
        "compiled": False,
        "synthesized": False,
        "metadata": metadata,
        "validation": validation,
        "implementation_boundary": (
            "binary sigmoid output; no softmax"
            if _uses_binary_sigmoid(config)
            else "multiclass logits path; no softmax"
        ),
    }
    _write_result(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
