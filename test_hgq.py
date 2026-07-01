import argparse

from benchmark import ROOT, load_config, load_dataset
from hgq_benchmark import build_hgq_model, evaluate_hgq
from qkeras_benchmark import set_qkeras_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved HGQ model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_qkeras_seed(config["seed"])
    arrays = load_dataset(config)
    output_dim = int(
        config.get("model", {}).get(
            "output_dim", len(arrays["metadata"]["class_names"])
        )
    )
    model = build_hgq_model(config, arrays["x_train"].shape[1], output_dim)
    model_path = ROOT / "models" / f"{config['run_name']}.weights.h5"
    model.load_weights(model_path)
    result = evaluate_hgq(model, arrays, config, model_path)
    print(
        f"accuracy={result['accuracy']:.5f} "
        f"macro_auc={result['macro_auc']:.5f}"
    )


if __name__ == "__main__":
    main()
