import argparse
import json
import pickle

from benchmark import ROOT, ensure_output_dirs, load_config, load_dataset, set_seed


def build_classifier(config, class_count):
    from xgboost import XGBClassifier

    options = dict(config.get("model", {}).get("xgboost", {}))
    early_stopping_rounds = options.pop("early_stopping_rounds", None)
    if options.get("objective") is None:
        options["objective"] = "multi:softprob" if class_count > 2 else "binary:logistic"
    if class_count > 2:
        options.setdefault("num_class", class_count)
    options.setdefault("n_estimators", 400)
    options.setdefault("max_depth", 6)
    options.setdefault("learning_rate", 0.05)
    options.setdefault("subsample", 0.8)
    options.setdefault("colsample_bytree", 0.8)
    options.setdefault("tree_method", "hist")
    options.setdefault("eval_metric", ["mlogloss", "merror"] if class_count > 2 else ["logloss", "error"])
    options.setdefault("random_state", int(config["seed"]))
    options.setdefault("n_jobs", 1)
    if early_stopping_rounds is not None:
        options.setdefault("early_stopping_rounds", int(early_stopping_rounds))
    return XGBClassifier(**options), early_stopping_rounds


def history_from_evals(result):
    if not result:
        return {}
    history = {}
    for split_name, metrics in result.items():
        for metric_name, values in metrics.items():
            history[f"{split_name}_{metric_name}"] = values
    return history


def main():
    parser = argparse.ArgumentParser(description="Train a configured XGBoost BDT.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs()
    set_seed(config["seed"])
    arrays = load_dataset(config)
    class_count = len(arrays["metadata"]["class_names"])
    model, early_stopping_rounds = build_classifier(config, class_count)

    fit_kwargs = {
        "X": arrays["x_train"],
        "y": arrays["y_train"],
        "eval_set": [
            (arrays["x_train"], arrays["y_train"]),
            (arrays["x_validation"], arrays["y_validation"]),
        ],
        "verbose": True,
    }
    model.fit(**fit_kwargs)

    history = history_from_evals(model.evals_result())
    payload = {
        "model": model,
        "config": {key: value for key, value in config.items() if key != "_config_path"},
        "metadata": arrays["metadata"],
        "history": history,
    }
    model_path = ROOT / "models" / f"{config['run_name']}.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        ROOT / "logs" / f"{config['run_name']}_history.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(history, handle, indent=2)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
