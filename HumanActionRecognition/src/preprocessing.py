"""
Preprocessing and dataset building for Human Action Recognition.
"""
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import tensorflow as tf


def prepare_dataframes(
    train_df: pd.DataFrame, test_df: pd.DataFrame, data_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Add filepath column and encode labels. Returns (train_df, test_df, categories).
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train"
    train_df = train_df.copy()
    train_df["filepath"] = train_df["filename"].apply(lambda x: str(train_path / x))
    train_df["label"] = train_df["label"].astype("category")
    categories = list(train_df["label"].cat.categories)
    train_df["label"] = train_df["label"].cat.codes
    test_df = test_df.copy()
    test_df["filepath"] = test_df["filename"].apply(
        lambda x: str(data_dir / "test" / x)
    )
    return train_df, test_df, categories


def load_image(filepath: tf.Tensor, label: tf.Tensor, img_size: int = 128):
    """Decode and resize image for the model."""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = image / 255.0
    return image, label


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 32,
    img_size: int = 128,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build train and validation tf.data.Datasets."""
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_df["filepath"].values, train_df["label"].values)
    )
    train_ds = (
        train_ds.map(lambda fp, l: load_image(fp, l, img_size))
        .batch(batch_size)
        .shuffle(buffer_size=len(train_df))
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_df["filepath"].values, val_df["label"].values)
    )
    val_ds = val_ds.map(lambda fp, l: load_image(fp, l, img_size)).batch(batch_size)
    return train_ds, val_ds
