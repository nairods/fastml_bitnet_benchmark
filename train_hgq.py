import argparse
import json

from benchmark import ROOT, artifact_path, ensure_output_dirs, load_config, load_dataset, plot_training
from hgq_benchmark import build_hgq_model, train_hgq
from qkeras_benchmark import set_qkeras_seed


def main():
    parser = argparse.ArgumentParser(description="Train the configured HGQ model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    set_qkeras_seed(config["seed"], config["training"].get("device", "cpu"))
    arrays = load_dataset(config)
    output_dim = int(
        config.get("model", {}).get(
            "output_dim", len(arrays["metadata"]["class_names"])
        )
    )
    model = build_hgq_model(config, arrays["x_train"].shape[1], output_dim)
    history, model_path = train_hgq(model, arrays, config)
    with open(
        artifact_path(config, "logs", f"{config['run_name']}_history.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(history, handle, indent=2)
    plot_training(history, config["run_name"], config)
    print(f"Saved HGQ weights: {model_path}")


if __name__ == "__main__":
    main()
