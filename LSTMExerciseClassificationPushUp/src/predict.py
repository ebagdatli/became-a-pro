"""
Save and load model artifacts for push-up classification.
"""
import json
from pathlib import Path
from typing import List, Any


def save_model_and_artifacts(
    model: Any,
    encoder: Any,
    val_accuracy: float,
    models_dir: str = "models",
    input_shape: tuple = None,
) -> str:
    """Save Keras model, encoder, metadata."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "final_model.keras"
    model.save(model_path)

    import joblib
    joblib.dump(encoder, models_dir / "encoder.pkl")
    joblib.dump(encoder.classes_.tolist(), models_dir / "categories.pkl")

    meta = {
        "model_path": str(model_path),
        "model_type": "keras",
        "categories": encoder.classes_.tolist(),
        "val_accuracy": val_accuracy,
        "input_shape": list(input_shape) if input_shape else None,
    }
    joblib.dump(meta, models_dir / "meta.pkl")

    metadata_json = {"input_shape": list(input_shape) if input_shape else None}
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    return str(models_dir / "meta.pkl")


def load_model_and_artifacts(models_dir: str = "models"):
    """Load model, encoder, categories from models/."""
    models_dir = Path(models_dir)
    import joblib
    meta = joblib.load(models_dir / "meta.pkl")
    encoder = joblib.load(models_dir / "encoder.pkl")
    categories = meta.get("categories", joblib.load(models_dir / "categories.pkl"))

    from tensorflow.keras.models import load_model
    model_path = meta.get("model_path")
    if model_path and Path(model_path).is_absolute():
        model_file = Path(model_path)
    else:
        model_file = models_dir / "final_model.keras"
    model = load_model(model_file)
    return model, encoder, categories, "keras"


def predict_video(model, encoder, categories, frames: "np.ndarray") -> str:
    """Predict class for video frames. frames shape: (1, 30, H, W, 3) or (30, H, W, 3)."""
    import numpy as np
    if frames.ndim == 4:
        frames = np.expand_dims(frames, axis=0)
    probs = model.predict(frames, verbose=0)
    pred_idx = int(np.argmax(probs[0]))
    return categories[pred_idx]
