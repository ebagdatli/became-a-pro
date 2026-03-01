"""
Save model and categories for deployment.
"""
import os
from pathlib import Path

from joblib import dump


def save_model_and_artifacts(model, categories: list, history, models_dir: str = "models"):
    """
    Save Keras model as .keras and categories as .pkl.
    Also save a small final_model.pkl (metadata) so run_competition sees a .pkl and the app can load.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "final_model.keras"
    model.save(model_path)

    categories_path = models_dir / "categories.pkl"
    dump(categories, categories_path)

    # Metadata pkl for spec and for Streamlit (path + categories)
    meta = {"model_path": str(model_path), "categories": categories}
    if history is not None and hasattr(history, "history") and "val_accuracy" in history.history:
        meta["val_accuracy"] = float(max(history.history["val_accuracy"]))
    meta_path = models_dir / "final_model.pkl"
    dump(meta, meta_path)

    return str(meta_path)
