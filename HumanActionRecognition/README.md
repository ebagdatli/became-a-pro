# Human Action Recognition

**Problem type**: Image classification (15 action classes)  
**Target**: Predict action label from a single image  
**Metric**: Accuracy  
**Dataset**: Training_set.csv (filename, label) + train/ and test/ image folders

### Getting the data

Data and models are not in the repo. Download from Kaggle and place files under `data/Human Action Recognition/` or `data/raw/`:

- **Kaggle**: Search **"Human Action Recognition"** or use [Human Action Recognition HAR Dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition) — place `Training_set.csv`, `Testing_set.csv`, and the `train/` and `test/` image folders in the same directory.

## Data layout

Place the full dataset under one of:

- `data/Human Action Recognition/`  
  - `Training_set.csv`, `Testing_set.csv`  
  - **`train/`** – folder of training images (e.g. `Image_1.jpg`, …)  
  - **`test/`** – folder of test images  
- or `data/raw/` with the same four items

If you only have the CSVs, add the `train/` and `test/` image folders from the Kaggle dataset (e.g. *Human Action Recognition HAR Dataset*) into the same directory.

## Run pipeline

From repo root:

```bash
python run_competition.py HumanActionRecognition
```

This runs `notebooks/main.ipynb`: loads data, trains a CNN, saves the model under `models/`.

## Streamlit app

```bash
streamlit run HumanActionRecognition/app/streamlit_app.py
```

Upload an image to get the predicted action class.

## Project structure (MASTER_SPEC)

- `data/raw/`, `data/processed/` – data
- `notebooks/main.ipynb` – final pipeline
- `models/` – `final_model.keras`, `final_model.pkl`, `categories.pkl`
- `src/` – `data_loader`, `preprocessing`, `train`, `predict`
- `app/streamlit_app.py` – demo UI
