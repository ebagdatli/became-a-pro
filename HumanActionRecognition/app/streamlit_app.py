import sys
from pathlib import Path

import streamlit as st
from joblib import load
import tensorflow as tf
import numpy as np

# Run from repo root: streamlit run HumanActionRecognition/app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
IMG_SIZE = 128


def load_model_and_categories():
    """Load metadata from final_model.pkl and the Keras model."""
    meta_path = MODELS_DIR / "final_model.pkl"
    if not meta_path.exists():
        st.error(
            "Models not found. Train first: python run_competition.py HumanActionRecognition"
        )
        return None, None
    meta = load(meta_path)
    categories = meta.get("categories", [])
    model_path = meta.get("model_path")
    if not model_path:
        st.error("final_model.pkl missing 'model_path'.")
        return None, None
    full_path = Path(model_path)
    if not full_path.is_absolute():
        full_path = MODELS_DIR / full_path.name
    if not full_path.exists():
        st.error(f"Model file not found: {full_path}")
        return None, None
    model = tf.keras.models.load_model(full_path)
    return model, categories


def preprocess_image(uploaded_file):
    """Decode and resize uploaded image to (1, IMG_SIZE, IMG_SIZE, 3)."""
    raw = uploaded_file.read()
    img = tf.io.decode_image(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return np.expand_dims(img.numpy(), axis=0)


def main():
    st.title("Human Action Recognition")
    st.write("Upload an image to classify the action (15 classes).")

    model, categories = load_model_and_categories()
    if model is None:
        return

    st.subheader("Upload image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    if st.button("Predict"):
        try:
            X = preprocess_image(uploaded)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            return
        try:
            logits = model.predict(X)
            pred_idx = int(np.argmax(logits[0]))
            pred_label = categories[pred_idx] if pred_idx < len(categories) else f"Class_{pred_idx}"
            conf = float(np.max(logits[0]))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        st.subheader("Prediction")
        st.success(f"**{pred_label}** (confidence: {conf:.2%})")
        st.write("All classes:", ", ".join(categories))


if __name__ == "__main__":
    main()
