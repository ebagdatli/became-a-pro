"""
Preprocessing for exercise classification from joint angles.
One-hot encode Side, scale features, encode labels, train/val split.
"""
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


ANGLE_COLS = [
    "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
    "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
    "Knee_Ground_Angle", "Ankle_Ground_Angle",
]


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, MinMaxScaler, List[str], List[str]]:
    """
    Prepare X, y and split.
    One-hot encodes Side, uses angle columns, encodes Label.
    Returns (X_train, X_val, y_train, y_val, encoder, scaler, class_names, feature_cols).
    """
    df = df.copy()
    # One-hot encode Side (ensure consistent column order)
    side_dummies = pd.get_dummies(df["Side"], prefix="Side")
    for col in ["Side_left", "Side_right"]:
        if col not in side_dummies.columns:
            side_dummies[col] = 0
    side_dummies = side_dummies[["Side_left", "Side_right"]]
    df = pd.concat([df.drop(columns=["Side"]), side_dummies], axis=1)

    feature_cols = ANGLE_COLS + ["Side_left", "Side_right"]
    X = df[feature_cols].astype(np.float64).values
    y_raw = df["Label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = list(encoder.classes_)

    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, shuffle=True, random_state=random_state
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=random_state
        )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, encoder, scaler, class_names, feature_cols


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    return compute_class_weight(
        class_weight="balanced", classes=classes, y=y
    ).astype(np.float32)
