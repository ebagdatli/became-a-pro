import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# Ensure we can import the local `src` package when running via
# `streamlit run FacialKeypointsDetection/app/streamlit_app.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]  # FacialKeypointsDetection/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.preprocessing import images_to_array


def load_trained_model(models_dir=None):
    """
    Load a trained model from the models directory.
    Prefers 'final_model.pkl' if it exists, otherwise the first .pkl file found.
    """
    if models_dir is None:
        models_dir = ROOT / "models"
    models_dir = Path(models_dir)
    if not models_dir.exists():
        st.error(
            "Models directory not found. Please train a model first by running: "
            "python run_competition.py FacialKeypointsDetection"
        )
        return None

    candidates = []
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl"):
            candidates.append(str(models_dir / fname))

    if not candidates:
        st.error("No .pkl model files found in 'models/'.")
        return None

    preferred = None
    for path in candidates:
        if os.path.basename(path) == "final_model.pkl":
            preferred = path
            break

    model_path = preferred or candidates[0]
    st.info(f"Loading model from: {model_path}")
    return load(str(model_path))


def main():
    st.title("Facial Keypoints Detection")
    st.write(
        "This app loads a trained CNN model and predicts facial keypoints "
        "for a single 96×96 grayscale face image."
    )

    model = load_trained_model()
    if model is None:
        return

    st.subheader("Input")
    st.markdown(
        "Paste the **space-separated pixel values** for a single 96×96 grayscale "
        "image (exactly 9,216 numbers). This matches the Kaggle `Image` column format."
    )

    image_str = st.text_area("Image pixel string", height=150)

    if st.button("Predict keypoints"):
        if not image_str.strip():
            st.warning("Please paste an image pixel string first.")
            return

        try:
            # Wrap into a one-row DataFrame to reuse preprocessing helper.
            df = pd.DataFrame({"Image": [image_str.strip()]})
            X = images_to_array(df["Image"])
        except Exception as e:
            st.error(f"Failed to parse image string: {e}")
            return

        try:
            preds = model.predict(X)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return

        coords = preds[0]
        st.subheader("Predicted keypoints")
        st.write("30 coordinates (15 (x, y) pairs):")
        st.write(coords)


if __name__ == "__main__":
    main()

