"""
Model for push-up video classification (correct vs incorrect).
Uses Conv3D for sequence classification.
"""
from typing import Tuple, Any

import numpy as np


def build_model(input_shape: tuple, num_classes: int = 2):
    """Build Conv3D model for video classification."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Conv3D(32, (3, 3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.25),
        Conv3D(64, (3, 3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 4,
) -> Tuple[Any, float]:
    """Train model. Returns (model, val_accuracy)."""
    model = build_model(X_train.shape[1:], num_classes=len(np.unique(y_train)))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    _, acc = model.evaluate(X_val, y_val)
    return model, float(acc)
