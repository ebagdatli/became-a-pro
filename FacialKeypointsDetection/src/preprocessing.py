from typing import Tuple

import numpy as np
import pandas as pd


def fill_missing(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill missing keypoint values as in the original notebook.
    """
    return train_df.fillna(method="ffill")


def images_to_array(image_series: pd.Series) -> np.ndarray:
    """
    Convert the Image string column into a 4D numpy array of shape
    (n_samples, 96, 96, 1).
    """
    imag = []
    for img_str in image_series:
        pixels = img_str.split(" ")
        pixels = ["0" if x == "" else x for x in pixels]
        imag.append(pixels)

    image_list = np.asarray(imag, dtype="float32")
    X = image_list.reshape(-1, 96, 96, 1)
    # Optional normalization
    X /= 255.0
    return X


def prepare_training_data(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X_train and y_train from the raw training DataFrame.
    """
    train_df = fill_missing(train_df)
    X = images_to_array(train_df["Image"])

    labels = train_df.drop(columns=["Image"])
    y = labels.values.astype("float32")

    return X, y


def prepare_test_data(test_df: pd.DataFrame) -> np.ndarray:
    """
    Prepare X_test from the raw test DataFrame.
    """
    return images_to_array(test_df["Image"])

