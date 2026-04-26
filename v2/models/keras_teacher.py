from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras


def build_patch_teacher(
    *,
    window_size: int = 5,
    conv_filters: int = 32,
    dense_units: int = 128,
    learning_rate: float = 5e-4,
) -> keras.Model:
    inputs = keras.Input(shape=(window_size, window_size, 2), name="patch_pair")
    x = keras.layers.Conv2D(conv_filters, (3, 3), padding="same", activation="relu", name="teacher_conv1")(inputs)
    x = keras.layers.Conv2D(conv_filters, (3, 3), padding="same", activation="relu", name="teacher_conv2")(x)
    x = keras.layers.Conv2D(conv_filters // 2, (3, 3), padding="valid", activation="relu", name="teacher_conv3")(x)
    x = keras.layers.Flatten(name="teacher_flatten")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="teacher_dense1")(x)
    x = keras.layers.Dense(dense_units // 2, activation="relu", name="teacher_dense2")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="teacher_match_score")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="stereo_patch_teacher")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model