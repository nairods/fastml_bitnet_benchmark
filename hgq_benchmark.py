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


def build_hgq_model(config, input_dim, output_dim):
    import keras
    from HGQ.layers import HDense, HQuantize

    options = config["model"].get("hgq", {})
    beta = options.get("beta", 3e-6)
    hidden_dims = config["model"].get("hidden_dims", [64, 32, 32])
    layers = [keras.layers.Input(shape=(input_dim,)), HQuantize(beta=beta)]
    layers.extend(
        HDense(width, beta=beta, activation="relu") for width in hidden_dims
    )
    layers.append(HDense(output_dim, beta=beta))
    return keras.models.Sequential(layers, name="hgq_mlp")


def train_hgq(model, arrays, config):
    import keras
    from HGQ import FreeBOPs, ResetMinMax

    training = config["training"]
    if training.get("optimizer", "adam") != "adam":
        raise ValueError("HGQ benchmark profiles require the Adam optimizer")
    if training.get("schedule", "cosine_decay") != "cosine_decay":
        raise ValueError("HGQ benchmark profiles require full-profile cosine decay")
    binary = uses_binary_sigmoid(config)
    if binary:
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)]
        y_train = arrays["y_train"].astype("float32")
        y_validation = arrays["y_validation"].astype("float32")
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
        y_train = arrays["y_train"]
        y_validation = arrays["y_validation"]
    decay_steps = training.get("cosine_decay_steps", training["epochs"])
    schedule = keras.optimizers.schedules.CosineDecay(
        training["learning_rate"], decay_steps
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0),
        loss=loss,
        metrics=metrics,
    )
    checkpoint_path = artifact_path(config, "models", f"{config['run_name']}.weights.h5")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ResetMinMax(),
        FreeBOPs(),
        keras.callbacks.LearningRateScheduler(schedule),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
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
    for key in ("bops", "lr", "learning_rate"):
        if key in history:
            normalized[key] = [float(value) for value in history[key]]
    return normalized, checkpoint_path


def predict_hgq(model, x, batch_size):
    import numpy as np
    import tensorflow as tf

    logits = np.asarray(model.predict(x, batch_size=batch_size, verbose=0))
    if logits.ndim == 1 or logits.shape[1] == 1:
        signal = tf.nn.sigmoid(logits.reshape(-1, 1)).numpy()
        return np.concatenate([1.0 - signal, signal], axis=1)
    return tf.nn.softmax(logits, axis=1).numpy()


def measure_hgq_latency(model, input_dim, warmup, iterations):
    import tensorflow as tf

    sample = tf.zeros((1, input_dim), dtype=tf.float32)
    function = tf.function(model)
    for _ in range(warmup):
        function(sample)
    start = time.perf_counter()
    for _ in range(iterations):
        function(sample)
    return (time.perf_counter() - start) * 1000.0 / iterations


def evaluate_hgq(model, arrays, config, model_path):
    probabilities = predict_hgq(
        model, arrays["x_test"], config["evaluation"]["batch_size"]
    )
    class_names = arrays["metadata"]["class_names"]
    metrics = compute_metrics(arrays["y_test"], probabilities, class_names)
    latency = config["evaluation"]["latency"]
    result = result_record(
        config,
        "hgq",
        metrics,
        model.count_params(),
        model_path,
        {
            "cpu_latency_ms": measure_hgq_latency(
                model,
                arrays["x_test"].shape[1],
                latency["warmup"],
                latency["iterations"],
            )
        },
    )
    save_results(result)
    return result
