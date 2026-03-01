# SmartAiCoach

Exercise classification from joint angles. Uses MediaPipe-extracted angles (Shoulder, Elbow, Hip, Knee, Ankle, and ground angles) to classify exercises: Jumping Jacks, Push-ups, Pull-ups, Squats, Russian Twists.

## Data

Place `exercise_angles.csv` in `data/raw/`. Columns: Side, Shoulder_Angle, Elbow_Angle, Hip_Angle, Knee_Angle, Ankle_Angle, Shoulder_Ground_Angle, Elbow_Ground_Angle, Hip_Ground_Angle, Knee_Ground_Angle, Ankle_Ground_Angle, Label.

## Project structure

```
SmartAiCoach/
├── data/
│   ├── raw/          # exercise_angles.csv
│   └── processed/
├── models/           # final_model.pkl or .pt, encoder, scaler, meta
├── notebooks/
│   └── main.ipynb    # load → train (XGBoost + PyTorch) → save best
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
└── requirements.txt
```

## Setup

```bash
cd SmartAiCoach
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\pip install ipykernel
venv\Scripts\python -m ipykernel install --user --name=smart-ai-coach --display-name="Python (SmartAiCoach)"
```

## Run

```bash
# From repo root
python run_competition.py SmartAiCoach
```

Streamlit (CSV upload for predictions):

```bash
cd SmartAiCoach
venv\Scripts\python -m streamlit run app/streamlit_app.py
```

## Turkish labels

| Label          | Turkce      |
|----------------|-------------|
| Jumping Jacks  | Ziplama     |
| Push-ups       | Sinav      |
| Pull-ups       | Barfiks     |
| Squats         | Squat       |
| Russian Twists | Rus Donusu  |
