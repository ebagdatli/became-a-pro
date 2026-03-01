"""
Preprocessing for push-up video classification.
Encode labels, train/val split.
"""
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, List[str]]:
    """
    Encode labels, split. Returns (X_train, X_val, y_train, y_val, encoder, class_names).
    """
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    class_names = list(encoder.classes_)

    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=test_size, stratify=y_enc, shuffle=True, random_state=random_state
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=test_size, shuffle=True, random_state=random_state
        )
    return X_train, X_val, y_train, y_val, encoder, class_names
