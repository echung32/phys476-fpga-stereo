from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras


def build_patch_student(
    *,
    window_size: int = 5,
    conv_filters: int = 8,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(window_size, window_size, 2), name="patch_pair")
    x = keras.layers.Conv2D(conv_filters, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.Conv2D(conv_filters, (3, 3), padding="valid", activation="relu", name="conv2")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense1")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="match_score")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="stereo_patch_student")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model