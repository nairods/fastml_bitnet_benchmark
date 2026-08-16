import random
import time

import numpy as np

from benchmark import (
    ROOT,
    artifact_path,
    compute_metrics,
    result_record,
    save_results,
    uses_binary_sigmoid,
)


def set_qkeras_seed(seed, device="cpu"):
    import tensorflow as tf

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _qkeras_quantizers(config):
    from qkeras.quantizers import binary, quantized_bits, quantized_relu, ternary

    options = config["model"].get("qkeras", {})
    mode = options.get("quantizer", "quantized_bits")
    bits = options.get("bits", 6)
    integer_bits = options.get("integer_bits", 0)
    alpha = options.get("alpha", 1)
    activation_bits = options.get("activation_bits", bits)
    if mode == "quantized_bits":
        weight_quantizer = quantized_bits(bits, integer_bits, alpha=alpha)
        bias_quantizer = quantized_bits(bits, integer_bits, alpha=alpha)
    elif mode == "binary":
        weight_quantizer = binary(alpha=alpha)
        bias_quantizer = binary(alpha=alpha)
    elif mode == "ternary":
        weight_quantizer = ternary(alpha=alpha)
        bias_quantizer = ternary(alpha=alpha)
    else:
        raise ValueError(f"Unsupported QKeras quantizer: {mode}")
    activation_quantizer = quantized_relu(activation_bits)
    return weight_quantizer, bias_quantizer, activation_quantizer


def build_qkeras_model(config, input_dim, output_dim):
    import tensorflow as tf
    from qkeras import QActivation, QDense

    hidden_dims = config["model"].get("hidden_dims", [64, 32, 32])
    kernel_quantizer, bias_quantizer, activation_quantizer = _qkeras_quantizers(config)

    model = tf.keras.Sequential(name="qkeras_mlp")
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for index, width in enumerate(hidden_dims):
        model.add(
            QDense(
                width,
                name=f"fc{index + 1}",
                kernel_quantizer=kernel_quantizer,
                bias_quantizer=bias_quantizer,
                kernel_initializer="lecun_uniform",
            )
        )
        model.add(QActivation(activation_quantizer, name=f"relu{index + 1}"))
    model.add(
        QDense(
            output_dim,
            name="output",
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
            kernel_initializer="lecun_uniform",
        )
    )
    return model


def load_qkeras_weights(model, path):
    """Restore QDense weights independent of saved layer-name conventions."""
    import h5py
    from qkeras import QDense

    dense_layers = [layer for layer in model.layers if isinstance(layer, QDense)]
    with h5py.File(path, "r") as handle:
        dependencies = handle["_layer_checkpoint_dependencies"]
        groups = [name for name in dependencies if name.startswith("q_dense")]
        groups.sort(
            key=lambda name: int(name.rsplit("_", 1)[-1])
            if name.rsplit("_", 1)[-1].isdigit()
            else 0
        )
        if len(groups) != len(dense_layers):
            raise ValueError(f"Expected {len(dense_layers)} QDense groups, found {len(groups)}")
        for layer, group_name in zip(dense_layers, groups):
            values = dependencies[group_name]["vars"]
            layer.set_weights([values["0"][()], values["1"][()]])


def train_qkeras(model, arrays, config, extra_callbacks=None):
    import tensorflow as tf

    training = config["training"]
    if training.get("optimizer", "adam") != "adam":
        raise ValueError("QKeras benchmark profiles require the Adam optimizer")
    if training.get("schedule", "constant") != "constant":
        raise ValueError("QKeras benchmark profiles currently support only a constant learning rate")
    binary = uses_binary_sigmoid(config)
    if binary:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)]
        y_train = arrays["y_train"].astype("float32")
        y_validation = arrays["y_validation"].astype("float32")
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
        y_train = arrays["y_train"]
        y_validation = arrays["y_validation"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=training["learning_rate"],
            weight_decay=training.get("weight_decay", 0.0),
        ),
        loss=loss,
        metrics=metrics,
    )
    checkpoint_path = artifact_path(config, "models", f"{config['run_name']}.weights.h5")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ]
    callbacks.extend(extra_callbacks or [])
    history = model.fit(
        arrays["x_train"],
        y_train,
        validation_data=(arrays["x_validation"], y_validation),
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        shuffle=bool(training.get("shuffle", True)),
        callbacks=callbacks,
        verbose=2,
    ).history
    model.load_weights(checkpoint_path)
    normalized = {
        "train_loss": history["loss"],
        "validation_loss": history["val_loss"],
        "validation_accuracy": history["val_accuracy"],
        "selected_epoch": int(np.argmin(history["val_loss"])) + 1,
        "selection_metric": "validation_loss",
        "max_epochs": int(training["epochs"]),
    }
    return normalized, checkpoint_path


def predict_qkeras(model, x, batch_size):
    import tensorflow as tf

    logits = np.asarray(model.predict(x, batch_size=batch_size, verbose=0))
    if logits.ndim == 1 or logits.shape[1] == 1:
        signal = tf.nn.sigmoid(logits.reshape(-1, 1)).numpy()
        return np.concatenate([1.0 - signal, signal], axis=1)
    return tf.nn.softmax(logits, axis=1).numpy()


def measure_qkeras_latency(model, input_dim, warmup, iterations):
    import tensorflow as tf

    sample = tf.zeros((1, input_dim), dtype=tf.float32)
    function = tf.function(model)
    for _ in range(warmup):
        function(sample)
    start = time.perf_counter()
    for _ in range(iterations):
        function(sample)
    return (time.perf_counter() - start) * 1000.0 / iterations


def evaluate_qkeras(model, arrays, config, model_path, backend="qkeras"):
    probabilities = predict_qkeras(
        model, arrays["x_test"], config["evaluation"]["batch_size"]
    )
    class_names = arrays["metadata"]["class_names"]
    metrics = compute_metrics(arrays["y_test"], probabilities, class_names)
    latency = config["evaluation"]["latency"]
    result = result_record(
        config,
        backend,
        metrics,
        model.count_params(),
        model_path,
        {
            "cpu_latency_ms": measure_qkeras_latency(
                model,
                arrays["x_test"].shape[1],
                latency["warmup"],
                latency["iterations"],
            )
        },
    )
    save_results(result)
    return result
