"""
Streamlit app for SmartAiCoach exercise classification.
Upload CSV with Side and angle columns to get predictions.
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Run from SmartAiCoach: streamlit run app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

# Turkce egzersiz karsiliklari
LABEL_TO_TURKISH = {
    "Jumping Jacks": "Ziplama",
    "Push-ups": "Sinav",
    "Pull-ups": "Barfiks",
    "Squats": "Squat",
    "Russian Twists": "Rus Donusu",
}


def load_model_and_artifacts():
    """Load model and artifacts from models/."""
    meta_path = MODELS_DIR / "meta.pkl"
    if not meta_path.exists():
        st.error(
            "Models not found. Train first: python run_competition.py SmartAiCoach"
        )
        return None, None, None, None, None, None

    meta = load(meta_path)
    encoder = load(MODELS_DIR / "encoder.pkl")
    scaler = load(MODELS_DIR / "scaler.pkl")
    categories = meta.get("categories", load(MODELS_DIR / "categories.pkl"))
    model_type = meta.get("model_type", "xgboost")
    feature_columns = None
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, encoding="utf-8") as f:
            feature_columns = json.load(f).get("feature_columns", [])

    model_path = meta.get("model_path")
    if model_path and not Path(model_path).is_absolute():
        model_path = MODELS_DIR / Path(model_path).name

    if model_type == "xgboost":
        model = load(model_path)
    else:
        import torch
        from src.train import build_pytorch_model
        input_size = meta.get("input_size", 12)
        num_classes = meta.get("num_classes", len(categories))
        model = build_pytorch_model(input_size, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    return model, encoder, scaler, categories, model_type, feature_columns


def prepare_uploaded_df(df: pd.DataFrame, feature_columns: list) -> np.ndarray:
    """Apply same preprocessing as training: one-hot Side, select features."""
    df = df.copy()
    if "Side" in df.columns:
        side_dummies = pd.get_dummies(df["Side"], prefix="Side")
        for col in ["Side_left", "Side_right"]:
            if col not in side_dummies.columns:
                side_dummies[col] = 0
        side_dummies = side_dummies[["Side_left", "Side_right"]]
        df = pd.concat([df.drop(columns=["Side"]), side_dummies], axis=1)
    X = df[feature_columns].astype(np.float64).values
    return X


def predict(model, encoder, scaler, categories, model_type, X: np.ndarray):
    """Run prediction on feature matrix X."""
    X_scaled = scaler.transform(X)
    if model_type == "xgboost":
        pred_indices = model.predict(X_scaled)
    else:
        import torch
        with torch.no_grad():
            X_t = torch.from_numpy(X_scaled.astype(np.float32))
            outputs = model(X_t)
            _, pred_indices = torch.max(outputs, 1)
            pred_indices = pred_indices.numpy()
    return [encoder.inverse_transform([i])[0] for i in pred_indices]


def main():
    st.title("SmartAiCoach – Egzersiz Siniflandirma")
    st.write(
        "Side, Shoulder_Angle, Elbow_Angle, Hip_Angle, Knee_Angle, Ankle_Angle, "
        "Shoulder_Ground_Angle, Elbow_Ground_Angle, Hip_Ground_Angle, "
        "Knee_Ground_Angle, Ankle_Ground_Angle sutunlarina sahip CSV yukleyin."
    )

    model, encoder, scaler, categories, model_type, feature_columns = load_model_and_artifacts()
    if model is None:
        return
    if feature_columns is None:
        st.error("metadata.json or feature_columns not found in meta.")
        return

    st.subheader("CSV Yukle")
    uploaded = st.file_uploader("CSV dosyasi secin", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV yuklenemedi: {e}")
        return

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        st.error(f"Eksik sutunlar: {', '.join(missing)}")
        return

    try:
        X = prepare_uploaded_df(df, feature_columns)
    except Exception as e:
        st.error(f"Veri hazirlama hatasi: {e}")
        return

    if X.shape[1] != scaler.n_features_in_:
        st.error(
            f"Beklenen {scaler.n_features_in_} ozellik, gelen {X.shape[1]}. "
            "CSV sutunlari exercise_angles.csv ile uyumlu olmali."
        )
        return

    if st.button("Tahmin Et"):
        preds = predict(model, encoder, scaler, categories, model_type, X)
        df_result = df.copy()
        df_result["predicted_exercise"] = preds
        df_result["predicted_turkce"] = [LABEL_TO_TURKISH.get(p, p) for p in preds]
        st.subheader("Tahminler")
        st.dataframe(df_result)
        st.write("Siniflar:", ", ".join(categories))
        st.write("Turkce:", ", ".join(LABEL_TO_TURKISH.get(c, c) for c in categories))


if __name__ == "__main__":
    main()
