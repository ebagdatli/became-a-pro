from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    LeakyReLU,
    MaxPool2D,
    Flatten,
)


def build_model(input_shape=(96, 96, 1)) -> tf.keras.Model:
    """
    CNN model closely matching the architecture from the original notebook.
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", use_bias=False, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding="same", use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(30))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    return model


def train_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    epochs: int = 50,
    batch_size: int = 256,
    validation_split: float = 0.2,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train the CNN model and return (model, history).
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    return model, history

