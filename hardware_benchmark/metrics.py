import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def as_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 predictions, got shape {values.shape}")
    if values.shape[1] == 1:
        signal = (
            values
            if np.all(values >= -1e-7) and np.all(values <= 1.0 + 1e-7)
            else 1.0 / (1.0 + np.exp(-values))
        )
        return np.concatenate([1.0 - signal, signal], axis=1)
    row_sums = values.sum(axis=1)
    is_probability = (
        np.all(values >= -1e-7)
        and np.all(values <= 1.0 + 1e-7)
        and np.allclose(row_sums, 1.0, atol=2e-5)
    )
    return values if is_probability else softmax(values)


def compare_predictions(candidate, reference, labels):
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    candidate = as_probabilities(candidate)
    reference = as_probabilities(reference)
    labels = np.asarray(labels)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Prediction shape mismatch: {candidate.shape} != {reference.shape}"
        )
    if len(labels) != len(candidate):
        raise ValueError(f"Label count mismatch: {len(labels)} != {len(candidate)}")

    difference = np.abs(candidate - reference)
    candidate_class = candidate.argmax(axis=1)
    reference_class = reference.argmax(axis=1)
    accuracy = float(accuracy_score(labels, candidate_class))
    reference_accuracy = float(accuracy_score(labels, reference_class))
    if candidate.shape[1] == 2:
        macro_auc = float(roc_auc_score(labels, candidate[:, 1]))
        reference_macro_auc = float(roc_auc_score(labels, reference[:, 1]))
    else:
        binary = label_binarize(labels, classes=np.arange(candidate.shape[1]))
        macro_auc = float(roc_auc_score(binary, candidate, average="macro"))
        reference_macro_auc = float(roc_auc_score(binary, reference, average="macro"))
    return {
        "samples": int(len(labels)),
        "finite": bool(np.isfinite(candidate).all()),
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "class_agreement": float(np.mean(candidate_class == reference_class)),
        "accuracy": accuracy,
        "reference_accuracy": reference_accuracy,
        "accuracy_delta": accuracy - reference_accuracy,
        "macro_auc": macro_auc,
        "reference_macro_auc": reference_macro_auc,
        "macro_auc_delta": macro_auc - reference_macro_auc,
    }
