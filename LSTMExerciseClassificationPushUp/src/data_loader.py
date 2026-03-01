"""
Load push-up video data from data/raw.
Expects: data/raw/Correct sequence/*.mp4, data/raw/Wrong sequence/*.mp4
"""
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2


FRAME_LIMIT = 30
FRAME_SIZE = (112, 112)
CORRECT_DIR = "Correct sequence"
WRONG_DIR = "Wrong sequence"


def get_data_dir(base_dir: str = ".") -> Path:
    """Return data/raw path. On Kaggle, uses /kaggle/input/pushup/."""
    base = Path(base_dir)
    kaggle_path = Path("/kaggle/input/pushup")
    if kaggle_path.exists():
        return kaggle_path
    data_dir = base / "data" / "raw"
    correct_path = data_dir / CORRECT_DIR
    wrong_path = data_dir / WRONG_DIR
    if not correct_path.exists():
        raise FileNotFoundError(
            f"Could not find '{CORRECT_DIR}' in data/raw/. "
            "Download from Kaggle: https://www.kaggle.com/datasets/mohamadashrafsalama/pushup/data "
            "and extract to LSTMExerciseClassificationPushUp/data/raw/"
        )
    if not wrong_path.exists():
        raise FileNotFoundError(
            f"Could not find '{WRONG_DIR}' in data/raw/. "
            "Download from Kaggle and extract."
        )
    return data_dir


def extract_frames(video_path: Path, frame_limit: int = FRAME_LIMIT, frame_size: tuple = FRAME_SIZE) -> np.ndarray:
    """Extract frames from video. Returns (frame_limit, H, W, 3) or None if insufficient frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    while cap.isOpened() and count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        count += 1
    cap.release()
    if len(frames) < frame_limit:
        return None
    return np.array(frames[:frame_limit], dtype=np.float32)


def load_raw_data(base_dir: str = ".") -> Tuple[np.ndarray, np.ndarray, Path]:
    """
    Load videos, extract frames, build X and y.
    Returns (X, y, data_dir). X shape: (n_samples, frame_limit, H, W, 3).
    """
    data_dir = get_data_dir(base_dir)
    data = []
    labels = []

    correct_path = data_dir / CORRECT_DIR
    for f in sorted(correct_path.glob("*.mp4")):
        frames = extract_frames(f)
        if frames is not None:
            data.append(frames)
            labels.append("correct")

    wrong_path = data_dir / WRONG_DIR
    for f in sorted(wrong_path.glob("*.mp4")):
        frames = extract_frames(f)
        if frames is not None:
            data.append(frames)
            labels.append("incorrect")

    if len(data) == 0:
        raise ValueError("No valid videos found. Ensure .mp4 files exist in Correct sequence and Wrong sequence.")
    X = np.array(data, dtype=np.float32)
    y = np.array(labels)
    return X, y, data_dir


def extract_frames_for_prediction(
    video_path: str | Path,
    frame_limit: int = FRAME_LIMIT,
    frame_size: tuple = FRAME_SIZE,
) -> np.ndarray:
    """Extract frames for inference. Pads with zeros if video has fewer frames."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    while cap.isOpened() and count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        count += 1
    cap.release()
    while len(frames) < frame_limit:
        frames.append(np.zeros((*frame_size, 3), dtype=np.float32))
    return np.array(frames[:frame_limit], dtype=np.float32)


def ensure_directories(base_dir: str = ".") -> None:
    """Ensure data/raw, data/processed, models exist."""
    base = Path(base_dir)
    for sub in ("data/raw", "data/processed", "models"):
        (base / sub).mkdir(parents=True, exist_ok=True)
