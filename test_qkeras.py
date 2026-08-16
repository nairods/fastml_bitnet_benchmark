import argparse

from benchmark import artifact_path, load_config, load_dataset
from qkeras_benchmark import (
    build_qkeras_model,
    evaluate_qkeras,
    load_qkeras_weights,
    set_qkeras_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved QKeras model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_qkeras_seed(config["seed"], config["training"].get("device", "cpu"))
    arrays = load_dataset(config)
    output_dim = int(
        config.get("model", {}).get(
            "output_dim", len(arrays["metadata"]["class_names"])
        )
    )
    model = build_qkeras_model(config, arrays["x_train"].shape[1], output_dim)
    model_path = artifact_path(config, "models", f"{config['run_name']}.weights.h5")
    load_qkeras_weights(model, model_path)
    result = evaluate_qkeras(model, arrays, config, model_path)
    print(
        f"accuracy={result['accuracy']:.5f} "
        f"macro_auc={result['macro_auc']:.5f}"
    )


if __name__ == "__main__":
    main()
