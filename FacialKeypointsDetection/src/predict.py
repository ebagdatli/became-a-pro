import os
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump


def create_submission(
    predictions: np.ndarray,
    lookup_df: pd.DataFrame,
    output_path: str = "face_key_detection_submission.csv",
) -> pd.DataFrame:
    """
    Create a submission DataFrame and CSV using the Kaggle IdLookupTable.
    """
    lookid_list = list(lookup_df["FeatureName"])
    image_ids = list(lookup_df["ImageId"] - 1)
    pre_list = list(predictions)

    feature_indices = [lookid_list.index(f) for f in lookup_df["FeatureName"]]

    preded = []
    for x, y in zip(image_ids, feature_indices):
        preded.append(pre_list[x][y])

    submission = pd.DataFrame(
        {
            "RowId": lookup_df["RowId"],
            "Location": preded,
        }
    )

    submission.to_csv(output_path, index=False)
    return submission


def save_model(
    model,
    history,
    models_dir: str = "models",
    filename: Optional[str] = None,
) -> str:
    """
    Save the trained model as a .pkl file under models/.

    The filename includes the best validation MAE when available to match the
    MASTER_SPEC convention.
    """
    os.makedirs(models_dir, exist_ok=True)

    best_val_mae = None
    if history is not None and hasattr(history, "history"):
        mae_key = None
        for key in ("val_mean_absolute_error", "val_mae"):
            if key in history.history:
                mae_key = key
                break
        if mae_key is not None:
            best_val_mae = float(min(history.history[mae_key]))

    metric_part = f"{best_val_mae:.4f}" if best_val_mae is not None else "unknown"

    if filename is None:
        filename = f"model_cnn_mae_{metric_part}.pkl"

    path = os.path.join(models_dir, filename)
    dump(model, path)

    return path

