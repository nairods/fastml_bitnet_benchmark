import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort

from benchmark import (
    ROOT,
    compute_metrics,
    count_parameters,
    load_checkpoint,
    load_config,
    load_dataset,
    plot_evaluation,
    result_record,
    save_results,
    uses_binary_sigmoid,
)


def predict(session, x, batch_size):
    input_name = session.get_inputs()[0].name
    outputs = []
    for start in range(0, len(x), batch_size):
        logits = session.run(None, {input_name: x[start : start + batch_size]})[0]
        if logits.ndim == 1 or logits.shape[1] == 1:
            signal = 1.0 / (1.0 + np.exp(-logits.reshape(-1, 1)))
            outputs.append(np.concatenate([1.0 - signal, signal], axis=1))
        else:
            logits = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            outputs.append(probabilities / probabilities.sum(axis=1, keepdims=True))
    return np.concatenate(outputs)


def latency(session, input_dim, warmup, iterations):
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(0).normal(size=(1, input_dim)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: sample})
    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: sample})
    return (time.perf_counter() - start) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser(description="Validate and evaluate an ONNX DNN.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    arrays = load_dataset(config)
    model_path = ROOT / "onnx" / f"{config['run_name']}.onnx"
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    probabilities = predict(
        session, arrays["x_test"], config["evaluation"]["batch_size"]
    )
    class_names = arrays["metadata"]["class_names"]
    metrics = compute_metrics(arrays["y_test"], probabilities, class_names)
    run_name = f"{config['run_name']}_onnx"
    plot_evaluation(arrays["y_test"], probabilities, run_name, class_names)
    latency_config = config["evaluation"]["latency"]
    torch_model, _, _ = load_checkpoint(config, "cpu")
    latencies = {
        "cpu_latency_ms": latency(
            session,
            arrays["x_test"].shape[1],
            latency_config["warmup"],
            latency_config["iterations"],
        )
    }
    if "CUDAExecutionProvider" in ort.get_available_providers():
        cuda_session = ort.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider"]
        )
        latencies["gpu_latency_ms"] = latency(
            cuda_session,
            arrays["x_test"].shape[1],
            latency_config["warmup"],
            latency_config["iterations"],
        )
    result = result_record(
        config,
        "onnx",
        metrics,
        count_parameters(torch_model),
        model_path,
        latencies,
    )
    save_results(result)
    print(
        f"accuracy={result['accuracy']:.5f} "
        f"macro_auc={result['macro_auc']:.5f}"
    )


if __name__ == "__main__":
    main()
