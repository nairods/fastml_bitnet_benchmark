import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from benchmark import (
    ROOT,
    compute_metrics,
    load_config,
    load_dataset,
    plot_evaluation,
    result_record,
    save_results,
    set_seed,
)


def load_checkpoint(config):
    path = ROOT / "models" / f"{config['run_name']}.pkl"
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], payload, path


def predict_probabilities(model, x):
    probabilities = model.predict_proba(x)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
    return probabilities


def measure_latency_ms(model, input_dim, warmup, iterations):
    sample = np.random.randn(1, input_dim).astype(np.float32)
    for _ in range(warmup):
        model.predict_proba(sample)
    start = time.perf_counter()
    for _ in range(iterations):
        model.predict_proba(sample)
    return (time.perf_counter() - start) * 1000.0 / iterations


def count_tree_nodes(model):
    frame = model.get_booster().trees_to_dataframe()
    return int(len(frame))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved XGBoost BDT.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    arrays = load_dataset(config)
    model, _, model_path = load_checkpoint(config)
    probabilities = predict_probabilities(model, arrays["x_test"])
    class_names = arrays["metadata"]["class_names"]
    metrics = compute_metrics(arrays["y_test"], probabilities, class_names)
    plot_evaluation(arrays["y_test"], probabilities, config["run_name"], class_names)

    latency = config["evaluation"]["latency"]
    latencies = {
        "cpu_latency_ms": measure_latency_ms(
            model,
            arrays["x_test"].shape[1],
            latency["warmup"],
            latency["iterations"],
        )
    }
    result = result_record(
        config,
        "xgboost",
        metrics,
        count_tree_nodes(model),
        Path(model_path),
        latencies,
    )
    save_results(result)
    print(
        f"accuracy={result['accuracy']:.5f} "
        f"macro_auc={result['macro_auc']:.5f}"
    )


if __name__ == "__main__":
    main()
