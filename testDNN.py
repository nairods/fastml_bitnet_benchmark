import argparse

import torch

from benchmark import (
    compute_metrics,
    count_parameters,
    load_checkpoint,
    load_config,
    load_dataset,
    measure_torch_latency,
    predict_torch,
    result_record,
    save_results,
    select_device,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved PyTorch DNN.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"], config["training"].get("num_threads", 16))
    arrays = load_dataset(config)
    device = select_device(config["training"].get("device", "auto"))
    model, _, model_path = load_checkpoint(config, device)
    probabilities = predict_torch(
        model,
        arrays["x_test"],
        config["evaluation"]["batch_size"],
        device,
    )
    class_names = arrays["metadata"]["class_names"]
    metrics = compute_metrics(arrays["y_test"], probabilities, class_names)

    latency = config["evaluation"]["latency"]
    cpu_model, _, _ = load_checkpoint(config, torch.device("cpu"))
    latencies = {
        "cpu_latency_ms": measure_torch_latency(
            cpu_model,
            arrays["x_test"].shape[1],
            torch.device("cpu"),
            latency["warmup"],
            latency["iterations"],
        )
    }
    if torch.cuda.is_available():
        gpu_model, _, _ = load_checkpoint(config, torch.device("cuda"))
        latencies["gpu_latency_ms"] = measure_torch_latency(
            gpu_model,
            arrays["x_test"].shape[1],
            torch.device("cuda"),
            latency["warmup"],
            latency["iterations"],
        )

    result = result_record(
        config,
        "pytorch",
        metrics,
        count_parameters(model),
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
