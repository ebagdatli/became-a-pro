## Facial Keypoints Detection – Competition Overview

**Competition name**: Kaggle Facial Keypoints Detection  
**Problem type**: Regression (multi-output; predict \((x, y)\) coordinates of facial keypoints)  
**Target variables**: 30 continuous coordinates (15 keypoints × 2 coordinates)  
**Evaluation metric**: Root Mean Squared Error (RMSE) between predicted and true keypoint locations  

### Getting the data

Data and models are not in the repo (too large). Download the dataset from Kaggle and put the CSVs into `data/raw/`:

- **Kaggle**: [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection) — download and place `training.csv`, `test.csv`, `IdLookupTable.csv` in `FacialKeypointsDetection/data/raw/`.

### Dataset

- **training.csv** (expected in `data/raw/`):
  - 7,049 rows.
  - Columns:
    - 30 numeric columns with facial keypoint coordinates.
    - 1 `Image` column: 96×96 grayscale image flattened into a space-separated string of 9,216 pixel values.
- **test.csv** (expected in `data/raw/`):
  - Same `Image` format as training but without keypoint columns.
- **IdLookupTable.csv**:
  - Maps `RowId` to (`ImageId`, `FeatureName`) pairs.
  - Used to reconstruct the submission file.

### Submission Format

Submission file must have the following columns:

- `RowId` – as provided by `IdLookupTable.csv`.
- `Location` – predicted coordinate for the corresponding (`ImageId`, `FeatureName`) pair.

File is saved as `face_key_detection_submission.csv`.

### Key Challenges

- Many missing labels for some keypoints.
- Small, low-resolution images (96×96) with significant variation in pose, occlusion, and lighting.
- Risk of overfitting on a relatively small dataset.
- Need for careful preprocessing of the `Image` string column into numeric pixel arrays.

---

## Solution Insights (Summary)

Based on common public solutions and discussions for this competition:

- **Model types**:
  - Convolutional neural networks (CNNs) are the dominant approach.
  - Often shallow-to-moderate depth CNNs with batch normalization and dropout.
- **Feature engineering / preprocessing**:
  - Normalize pixel values to \([0, 1]\) or standardize to zero mean / unit variance.
  - Fill missing coordinate labels with forward-fill or drop rows with too many missing values.
  - Optional: data augmentation (horizontal flips, small rotations, slight zooms).
- **Validation strategy**:
  - Train/validation split with hold-out (e.g. 80/20).
  - Sometimes K-fold cross-validation to get more stable estimates of performance.
- **Common mistakes**:
  - Forgetting to normalize pixel values.
  - Training for too many epochs without regularization (overfitting).
  - Inconsistent preprocessing between training and test data.

---

## Project Structure (per MASTER_SPEC)

This competition folder follows the standard layout:

- `data/raw/` – raw CSV files downloaded from Kaggle (`training.csv`, `test.csv`, `IdLookupTable.csv`).
- `data/processed/` – any cleaned or transformed data artifacts.
- `notebooks/` – experimentation notebooks; `main.ipynb` serves as the final pipeline entrypoint.
- `models/` – serialized model artifacts (e.g. `model_cnn_mae_<score>.pkl`, `final_model.pkl`).
- `src/` – reusable Python modules:
  - `data_loader.py` – functions to load raw data.
  - `preprocessing.py` – image and label preprocessing utilities.
  - `train.py` – model definition and training helpers.
  - `predict.py` – submission creation and model saving helpers.
- `app/streamlit_app.py` – minimal Streamlit app to run predictions with a trained model.
- `requirements.txt` – Python dependencies for this competition.

---

## Final Pipeline (notebooks/main.ipynb)

The `notebooks/main.ipynb` notebook implements the final pipeline:

1. Load raw data from `data/raw/`.
2. Fill missing values and preprocess images into \((96, 96, 1)\) arrays.
3. Train a CNN model with Keras/TensorFlow.
4. Evaluate on a validation split (report MAE).
5. Generate predictions for the test set.
6. Create `face_key_detection_submission.csv` using `IdLookupTable.csv`.
7. Save the trained model to `models/` as a `.pkl` file using `joblib`.

This notebook is executed automatically by `run_competition.py` and is expected to run top-to-bottom without errors.

---

## How to Run Locally

1. Place the Kaggle data files into `data/raw/`:
   - `training.csv`
   - `test.csv`
   - `IdLookupTable.csv`
2. Create a virtual environment and install dependencies:
   - `pip install -r requirements.txt`
3. From the repository root, run:
   - `python run_competition.py FacialKeypointsDetection`
4. After successful execution:
   - A model file will be available under `FacialKeypointsDetection/models/`.
   - A submission file `face_key_detection_submission.csv` will be created in `FacialKeypointsDetection/notebooks/` (or current working directory of the notebook).

