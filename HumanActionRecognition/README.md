# Human Action Recognition

**Kaggle**: [Human Action Recognition HAR Dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition)

Image classification: 15 action classes. Metric: Accuracy.

## Getting the data

Download from Kaggle and place under `data/Human Action Recognition/` or `data/raw/`:

- `Training_set.csv`, `Testing_set.csv`
- `train/` – training images
- `test/` – test images

## Project structure

```
HumanActionRecognition/
├── data/
│   ├── raw/          # or Human Action Recognition/ – CSVs + train/, test/
│   └── processed/    # cleaned/transformed artifacts
├── models/           # final_model.keras, final_model.pkl, categories.pkl
├── notebooks/
│   └── main.ipynb    # final pipeline (load → train → predict)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
└── requirements.txt
```

## Run

```bash
# From repo root
python run_competition.py HumanActionRecognition
```

Streamlit demo:

```bash
streamlit run HumanActionRecognition/app/streamlit_app.py
```
