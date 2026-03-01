# Facial Keypoints Detection

**Kaggle**: [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection)

Regression task: predict 30 facial keypoint coordinates (15 keypoints × 2) from 96×96 grayscale images. Metric: RMSE.

## Getting the data

Download from Kaggle and place in `data/raw/`:

- `training.csv`, `test.csv`, `IdLookupTable.csv`

## Project structure

```
FacialKeypointsDetection/
├── data/
│   ├── raw/          # training.csv, test.csv, IdLookupTable.csv
│   └── processed/    # cleaned/transformed artifacts
├── models/           # saved models (e.g. final_model.pkl)
├── notebooks/
│   └── main.ipynb    # final pipeline (load → train → predict → submission)
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
python run_competition.py FacialKeypointsDetection
```

Then run the Streamlit app:

```bash
streamlit run FacialKeypointsDetection/app/streamlit_app.py
```
