import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from model_registry import build_registered_model, resolve_model_name


ROOT = Path(__file__).resolve().parent
OUTPUT_DIRS = ("data", "models", "plots", "logs", "results")
MULTICLASS_LABELS = ["g", "q", "w", "z", "t"]
MULTICLASS_NAMES = ["gluon", "quark", "W", "Z", "top"]
MULTICLASS_INDEX_BY_LABEL = {label: index for index, label in enumerate(MULTICLASS_LABELS)}


def ensure_output_dirs():
    for name in OUTPUT_DIRS:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "splits").mkdir(parents=True, exist_ok=True)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(Path(path).resolve())
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    thread_count = int(
        os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", 0))
        or 0
    )
    if thread_count > 0:
        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(max(1, min(thread_count, 4)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _dataset_key(config):
    dataset = config["dataset"]
    max_samples = dataset.get("max_samples")
    sample_tag = "all" if max_samples is None else str(max_samples)
    return f"openml_{dataset['id']}_n{sample_tag}"


def _split_key(config):
    split = config["split"]
    split_seed = int(split.get("seed", config.get("split_seed", 42)))
    class_mode = config.get("dataset", {}).get("classification", {}).get(
        "mode", "multiclass"
    )
    return (
        f"{_dataset_key(config)}_splitseed{split_seed}"
        f"_class{class_mode}"
        f"_train{split['train']}_val{split['validation']}_test{split['test']}"
    ).replace(".", "p")


def _classification_spec(config):
    classification = config.get("dataset", {}).get("classification", {})
    mode = classification.get("mode", "multiclass")
    if mode == "binary_qg_vs_wzt":
        return {
            "mode": mode,
            "raw_labels": MULTICLASS_LABELS,
            "class_labels": ["qg", "wzt"],
            "class_names": ["quark/gluon", "W/Z/top"],
            "label_map": {
                "g": 0,
                "q": 0,
                "w": 1,
                "z": 1,
                "t": 1,
            },
        }
    if mode == "binary_top_vs_qg":
        return {
            "mode": mode,
            "raw_labels": MULTICLASS_LABELS,
            "class_labels": ["qg", "top"],
            "class_names": ["quark/gluon", "top"],
            "label_map": {
                "g": 0,
                "q": 0,
                "t": 1,
            },
            "drop_labels": {"w", "z"},
        }
    return {
        "mode": "multiclass",
        "raw_labels": MULTICLASS_LABELS,
        "class_labels": MULTICLASS_LABELS,
        "class_names": MULTICLASS_NAMES,
        "label_map": MULTICLASS_INDEX_BY_LABEL,
    }


def load_dataset(config):
    ensure_output_dirs()
    class_spec = _classification_spec(config)
    processed_path = ROOT / "data" / "cache" / f"{_split_key(config)}.npz"
    metadata_path = ROOT / "data" / "cache" / f"{_split_key(config)}.json"
    split_path = ROOT / "data" / "splits" / f"{_split_key(config)}.npz"
    raw_path = ROOT / "data" / "cache" / f"{_dataset_key(config)}_raw.npz"
    raw_metadata_path = ROOT / "data" / "cache" / f"{_dataset_key(config)}_raw.json"
    use_cache = config["dataset"].get("cache", True)

    if use_cache and processed_path.exists() and metadata_path.exists():
        cached = np.load(processed_path)
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("class_mode") == class_spec["mode"]
            and metadata.get("class_names") == class_spec["class_names"]
        ):
            arrays = {key: cached[key] for key in cached.files}
            arrays["metadata"] = metadata
            return arrays
        processed_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        split_path.unlink(missing_ok=True)

    if use_cache and raw_path.exists() and raw_metadata_path.exists():
        raw = np.load(raw_path)
        if "raw_labels" not in raw.files:
            raw_path.unlink()
            raw_metadata_path.unlink()
            return load_dataset(config)
        x = raw["x"]
        raw_labels = raw["raw_labels"].astype(str)
        with open(raw_metadata_path, encoding="utf-8") as handle:
            raw_metadata = json.load(handle)
    else:
        kwargs = {
            "data_id": config["dataset"]["id"],
            "data_home": str(ROOT / "data" / "openml"),
            "as_frame": False,
        }
        if "parser" in inspect.signature(fetch_openml).parameters:
            kwargs["parser"] = "auto"
        dataset = fetch_openml(**kwargs)
        x = np.asarray(dataset.data, dtype=np.float32)
        raw_labels = np.asarray(dataset.target).astype(str)
        raw_metadata = {
            "dataset_id": config["dataset"]["id"],
            "dataset_name": dataset.details.get("name", "hls4ml_lhc_jets_hlf"),
            "feature_names": list(dataset.feature_names),
        }
        max_samples = config["dataset"].get("max_samples")
        if max_samples is not None and max_samples < len(raw_labels):
            sample_seed = int(config["dataset"].get("sample_seed", 42))
            indices, _ = train_test_split(
                np.arange(len(raw_labels)),
                train_size=max_samples,
                random_state=sample_seed,
                stratify=raw_labels,
            )
            x, raw_labels = x[indices], raw_labels[indices]
        if use_cache:
            np.savez_compressed(raw_path, x=x, raw_labels=raw_labels)
            with open(raw_metadata_path, "w", encoding="utf-8") as handle:
                json.dump(raw_metadata, handle, indent=2)

    unknown = sorted(set(raw_labels) - set(class_spec["raw_labels"]))
    if unknown:
        raise ValueError(f"Unexpected OpenML target labels: {unknown}")
    drop_labels = set(class_spec.get("drop_labels", set()))
    if drop_labels:
        keep = np.asarray([label not in drop_labels for label in raw_labels], dtype=bool)
        x = x[keep]
        raw_labels = raw_labels[keep]
    y = np.asarray([class_spec["label_map"][label] for label in raw_labels], dtype=np.int64)

    split = config["split"]
    train_fraction = float(split["train"])
    validation_fraction = float(split["validation"])
    test_fraction = float(split["test"])
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("split train/validation/test fractions must sum to 1")

    split_seed = int(split.get("seed", config.get("split_seed", 42)))
    if split_path.exists():
        saved_split = np.load(split_path)
        train_indices = saved_split["train_indices"]
        validation_indices = saved_split["validation_indices"]
        test_indices = saved_split["test_indices"]
    else:
        all_indices = np.arange(len(y))
        train_indices, remainder_indices = train_test_split(
            all_indices,
            train_size=train_fraction,
            random_state=split_seed,
            stratify=y,
        )
        relative_validation = validation_fraction / (
            validation_fraction + test_fraction
        )
        validation_indices, test_indices = train_test_split(
            remainder_indices,
            train_size=relative_validation,
            random_state=split_seed,
            stratify=y[remainder_indices],
        )
        np.savez_compressed(
            split_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_indices]).astype(np.float32)
    x_validation = scaler.transform(x[validation_indices]).astype(np.float32)
    x_test = scaler.transform(x[test_indices]).astype(np.float32)
    arrays = {
        "x_train": x_train,
        "y_train": y[train_indices],
        "x_validation": x_validation,
        "y_validation": y[validation_indices],
        "x_test": x_test,
        "y_test": y[test_indices],
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
    }
    metadata = {
        "dataset_id": config["dataset"]["id"],
        "dataset_name": raw_metadata["dataset_name"],
        "feature_names": raw_metadata["feature_names"],
        "class_mode": class_spec["mode"],
        "class_labels": class_spec["class_labels"],
        "class_names": class_spec["class_names"],
        "split_seed": split_seed,
        "split_file": str(split_path.relative_to(ROOT)),
        "preprocessing": "StandardScaler fitted on fixed training indices only",
        "sizes": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
    }
    if use_cache:
        np.savez_compressed(processed_path, **arrays)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    arrays["metadata"] = metadata
    return arrays


def make_loader(x, y, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def build_model(config, input_dim, output_dim=None):
    if output_dim is None:
        output_dim = int(
            config.get("model", {}).get(
                "output_dim",
                1
                if config.get("model", {}).get("output_mode") == "binary_sigmoid"
                else len(_classification_spec(config)["class_names"]),
            )
        )
    return build_registered_model(config, input_dim, output_dim)


def uses_binary_sigmoid(config):
    return config.get("model", {}).get("output_mode") == "binary_sigmoid"


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def train_model(model, arrays, config, device):
    training = config["training"]
    train_loader = make_loader(
        arrays["x_train"],
        arrays["y_train"],
        training["batch_size"],
        True,
        config["seed"],
    )
    validation_loader = make_loader(
        arrays["x_validation"],
        arrays["y_validation"],
        training["batch_size"],
        False,
        config["seed"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
    )
    loss_function = nn.BCEWithLogitsLoss() if uses_binary_sigmoid(config) else nn.CrossEntropyLoss()
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
    best_state = None
    best_validation_loss = float("inf")

    for epoch in range(training["epochs"]):
        model.train()
        train_loss = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if uses_binary_sigmoid(config):
                loss = loss_function(logits.squeeze(1), targets.float())
            else:
                loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            train_count += len(targets)

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        validation_correct = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                if uses_binary_sigmoid(config):
                    loss = loss_function(logits.squeeze(1), targets.float())
                    validation_correct += (
                        (torch.sigmoid(logits.squeeze(1)) >= 0.5).long() == targets
                    ).sum().item()
                else:
                    loss = loss_function(logits, targets)
                    validation_correct += (logits.argmax(dim=1) == targets).sum().item()
                validation_loss += loss.item() * len(targets)
                validation_count += len(targets)

        epoch_train_loss = train_loss / train_count
        epoch_validation_loss = validation_loss / validation_count
        epoch_validation_accuracy = validation_correct / validation_count
        history["train_loss"].append(epoch_train_loss)
        history["validation_loss"].append(epoch_validation_loss)
        history["validation_accuracy"].append(epoch_validation_accuracy)
        print(
            f"Epoch {epoch + 1}/{training['epochs']} "
            f"train_loss={epoch_train_loss:.5f} "
            f"validation_loss={epoch_validation_loss:.5f} "
            f"validation_accuracy={epoch_validation_accuracy:.5f}"
        )
        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return history


def predict_torch(model, x, batch_size, device):
    model.eval()
    probabilities = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (inputs,) in loader:
            logits = model(inputs.to(device))
            if logits.shape[1] == 1:
                signal = torch.sigmoid(logits).cpu().numpy()
                probabilities.append(np.concatenate([1.0 - signal, signal], axis=1))
            else:
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def measure_torch_latency(model, input_dim, device, warmup, iterations):
    model.eval()
    sample = torch.randn(1, input_dim, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def compute_metrics(y_true, probabilities, class_names=None):
    class_count = probabilities.shape[1]
    if class_names is None:
        class_names = [str(index) for index in range(class_count)]
    predictions = probabilities.argmax(axis=1)
    binary_targets = np.eye(class_count, dtype=np.int8)[y_true]
    per_class_auc = {
        name: float(roc_auc_score(binary_targets[:, index], probabilities[:, index]))
        for index, name in enumerate(class_names)
    }
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_auc": float(
            roc_auc_score(binary_targets, probabilities, average="macro")
        ),
        "per_class_auc": per_class_auc,
        "cross_entropy": float(
            log_loss(y_true, probabilities, labels=np.arange(class_count))
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=np.arange(class_count)
        ).tolist(),
    }
    if class_count == 2:
        scores = probabilities[:, 1]
        background = scores[np.asarray(y_true) == 0]
        signal = scores[np.asarray(y_true) == 1]
        threshold = np.quantile(background, 0.99, method="higher")
        if np.mean(background >= threshold) > 0.01:
            threshold = np.nextafter(threshold, np.inf)
        result["signal_eff_at_1pct_fpr"] = float(np.mean(signal >= threshold))
    return result


def plot_training(history, run_name):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["validation_loss"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend()
    axes[1].plot(history["validation_accuracy"])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    figure.tight_layout()
    figure.savefig(ROOT / "plots" / f"{run_name}_training.png", dpi=150)
    plt.close(figure)


def save_checkpoint(model, config, metadata, history):
    path = ROOT / "models" / f"{config['run_name']}.pt"
    saved_metadata = {
        **metadata,
        "output_dim": int(config.get("model", {}).get("output_dim", 1 if uses_binary_sigmoid(config) else len(metadata["class_names"]))),
        "output_mode": config.get("model", {}).get("output_mode", "multiclass_logits"),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {key: value for key, value in config.items() if key != "_config_path"},
            "metadata": saved_metadata,
            "history": history,
        },
        path,
    )
    return path


def load_checkpoint(config, device):
    path = ROOT / "models" / f"{config['run_name']}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["config"],
        input_dim=len(checkpoint["metadata"]["feature_names"]),
        output_dim=int(
            checkpoint["metadata"].get(
                "output_dim", len(checkpoint["metadata"]["class_names"])
            )
        ),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint, path


def save_results(result):
    ensure_output_dirs()
    stem = f"{result['run_name']}_{result['backend']}"
    with open(ROOT / "results" / f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def result_record(config, backend, metrics, parameter_count, model_path, latencies):
    return {
        "run_name": config["run_name"],
        "base_run_name": config.get("base_run_name", config["run_name"]),
        "backend": backend,
        "dataset_id": config["dataset"]["id"],
        "model_name": resolve_model_name(config),
        "model_config": config["model"],
        "seed": config["seed"],
        "split_seed": int(
            config["split"].get("seed", config.get("split_seed", 42))
        ),
        "accuracy": metrics["accuracy"],
        "macro_auc": metrics["macro_auc"],
        "per_class_auc": metrics["per_class_auc"],
        "cross_entropy": metrics["cross_entropy"],
        "signal_eff_at_1pct_fpr": metrics.get("signal_eff_at_1pct_fpr"),
        "confusion_matrix": metrics["confusion_matrix"],
        "parameter_count": parameter_count,
        "model_size_bytes": os.path.getsize(model_path),
        **latencies,
    }
