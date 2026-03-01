import os
from typing import Tuple

import pandas as pd


def load_raw_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Kaggle CSVs from the local data directory.

    Expected files:
      - training.csv
      - test.csv
      - IdLookupTable.csv (either in data/raw/ or data/)
    """
    train_path = os.path.join(data_dir, "training.csv")
    test_path = os.path.join(data_dir, "test.csv")

    lookup_candidates = [
        os.path.join(data_dir, "IdLookupTable.csv"),
        os.path.join("data", "IdLookupTable.csv"),
    ]

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Expected 'training.csv' and 'test.csv' under 'data/raw/'. "
            "Please download the Kaggle Facial Keypoints Detection data and "
            "place the CSVs into 'FacialKeypointsDetection/data/raw/'."
        )

    lookup_path = next((p for p in lookup_candidates if os.path.exists(p)), None)
    if lookup_path is None:
        raise FileNotFoundError(
            "Could not find 'IdLookupTable.csv' in either 'data/raw/' or 'data/'. "
            "Please copy it from the Kaggle dataset."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    lookup_df = pd.read_csv(lookup_path)

    return train_df, test_df, lookup_df


def ensure_directories(base_dir: str = ".") -> None:
    """
    Ensure that standard competition subdirectories exist.
    """
    for sub in ("data/raw", "data/processed", "models"):
        full = os.path.join(base_dir, sub)
        os.makedirs(full, exist_ok=True)

