"""
Streamlit app for push-up video classification (correct vs incorrect).
Upload a video file to get prediction.
"""
import sys
from pathlib import Path

import streamlit as st
import numpy as np

# Run from project root: streamlit run LSTMExerciseClassificationPushUp/app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"


def load_model_and_artifacts():
    """Load model and artifacts from models/."""
    meta_path = MODELS_DIR / "meta.pkl"
    if not meta_path.exists():
        st.error(
            "Models not found. Train first: python run_competition.py LSTMExerciseClassificationPushUp"
        )
        return None, None, None

    from src.predict import load_model_and_artifacts as load_artifacts
    model, encoder, categories, _ = load_artifacts(str(MODELS_DIR))
    return model, encoder, categories


def main():
    st.title("Push-up Classification (Correct vs Incorrect)")
    st.write(
        "Upload a push-up video to classify whether the exercise is performed correctly or incorrectly."
    )

    model, encoder, categories = load_model_and_artifacts()
    if model is None:
        st.error("Models not found. Train first: python run_competition.py LSTMExerciseClassificationPushUp")
        return

    uploaded = st.file_uploader("Choose a video file (.mp4)", type=["mp4"])
    if uploaded is None:
        st.stop()

    with st.spinner("Extracting frames and predicting..."):
        import tempfile
        from src.data_loader import extract_frames_for_prediction
        from src.predict import predict_video

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            frames = extract_frames_for_prediction(tmp_path)
            pred = predict_video(model, encoder, categories, frames)
            st.subheader("Prediction")
            st.success(f"**{pred.upper()}**")
            st.write("Classes:", ", ".join(categories))
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
