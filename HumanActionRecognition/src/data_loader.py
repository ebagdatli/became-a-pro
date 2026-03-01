"""
Load Human Action Recognition CSV and resolve data directory.
Supports both data/raw and data/Human Action Recognition layout.
"""
import os
from pathlib import Path
from typing import Tuple

import pandas as pd


def get_data_dir(base_dir: str = ".") -> Path:
    """Return the directory containing Training_set.csv and train/test folders."""
    base = Path(base_dir)
    candidates = [
        base / "data" / "Human Action Recognition",
        base / "data" / "raw",
    ]
    for d in candidates:
        train_csv = d / "Training_set.csv"
        if train_csv.exists():
            return d
    raise FileNotFoundError(
        "Could not find 'Training_set.csv' in 'data/Human Action Recognition' or 'data/raw'. "
        "Please place the dataset there (Training_set.csv, Testing_set.csv, train/, test/)."
    )


def load_raw_data(base_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Load Training_set.csv and Testing_set.csv.
    Returns (train_df, test_df, data_dir).
    """
    data_dir = get_data_dir(base_dir)
    train_path = data_dir / "Training_set.csv"
    test_path = data_dir / "Testing_set.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Not found: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df, data_dir


def ensure_directories(base_dir: str = ".") -> None:
    """Ensure data/raw, data/processed, models exist."""
    base = Path(base_dir)
    for sub in ("data/raw", "data/processed", "models"):
        (base / sub).mkdir(parents=True, exist_ok=True)
