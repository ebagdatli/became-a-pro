"""
Load exercise angles CSV from data/raw.
Expects exercise_angles.csv with Side, angle columns, and Label.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd


DATA_FILE = "exercise_angles.csv"


def get_data_dir(base_dir: str = ".") -> Path:
    """Return the directory containing exercise_angles.csv."""
    base = Path(base_dir)
    data_dir = base / "data" / "raw"
    if not (data_dir / DATA_FILE).exists():
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}' in data/raw/. "
            "Place exercise_angles.csv in SmartAiCoach/data/raw/"
        )
    return data_dir


def load_raw_data(base_dir: str = ".") -> Tuple[pd.DataFrame, Path]:
    """
    Load exercise_angles.csv.
    Returns (df, data_dir).
    """
    data_dir = get_data_dir(base_dir)
    df = pd.read_csv(data_dir / DATA_FILE)
    return df, data_dir


def ensure_directories(base_dir: str = ".") -> None:
    """Ensure data/raw, data/processed, models exist."""
    base = Path(base_dir)
    for sub in ("data/raw", "data/processed", "models"):
        (base / sub).mkdir(parents=True, exist_ok=True)
