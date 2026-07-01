import argparse
import json

from benchmark import (
    ROOT,
    build_model,
    ensure_output_dirs,
    load_config,
    load_dataset,
    plot_training,
    save_checkpoint,
    select_device,
    set_seed,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train a configured PyTorch DNN.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs()
    set_seed(config["seed"])
    arrays = load_dataset(config)
    device = select_device(config["training"].get("device", "auto"))
    model = build_model(config, arrays["x_train"].shape[1]).to(device)
    history = train_model(model, arrays, config, device)
    model_path = save_checkpoint(model, config, arrays["metadata"], history)
    plot_training(history, config["run_name"])
    with open(
        ROOT / "logs" / f"{config['run_name']}_history.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(history, handle, indent=2)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
