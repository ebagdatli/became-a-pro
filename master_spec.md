# BECOMEAPRO – Kaggle 20 Competition Execution Specification

## 1. Purpose

This document defines the mandatory execution framework for completing 20 Kaggle competitions locally under the `BECOMEAPRO` directory.

The objective is not only to run notebooks, but to:

* Deeply understand each competition
* Reproduce EDA and baseline kernels
* Study solution discussions
* Improve and retrain models
* Standardize pipelines
* Save trained artifacts
* Deploy via Streamlit
* Publish via Hugging Face

Completion of all 20 competitions under this specification equals a production-grade ML portfolio.

---

# 2. Root Directory Structure

All competitions must live under:

```
BECOMEAPRO/
│
├── competition-tracker.xlsx
├── MASTER_SPEC.md
│
├── CompetitionName01/
├── CompetitionName02/
...
└── CompetitionName20/
```

---

# 3. Standard Competition Folder Structure

Each competition folder must follow this structure:

```
CompetitionName/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_reproduction.ipynb
│   ├── 03_improved_training.ipynb
│   └── 04_final_pipeline.ipynb
│
├── models/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── README.md
└── requirements.txt
```

If an original Kaggle notebook exists, it must be moved into `/notebooks/` and renamed clearly.

---

# 4. Mandatory Workflow Per Competition

Each competition must satisfy ALL steps below.

---

## STEP 1 — Competition Understanding

Inside `README.md`, document:

* Competition name
* Problem type (classification, regression, CV, NLP, etc.)
* Target variable
* Evaluation metric
* Dataset description
* Submission format
* Key challenges

This must be written manually after reading Kaggle description.

---

## STEP 2 — Dataset Handling

* Download dataset via Kaggle CLI
* Store untouched data inside:

```
data/raw/
```

* Any cleaned or transformed dataset must go into:

```
data/processed/
```

Raw files must never be modified.

---

## STEP 3 — Reproduce EDA

Notebook: `01_eda.ipynb`

Must include:

* Dataset shape and info
* Missing value analysis
* Target distribution
* Feature distributions
* Correlation analysis (if applicable)
* Visualizations
* Observations written in markdown

EDA must be reproduced manually (not copied blindly).

---

## STEP 4 — Reproduce Baseline Kernel

Notebook: `02_baseline_reproduction.ipynb`

Requirements:

* Re-implement a public "Getting Started" notebook
* Rewrite code in your own structure
* Confirm baseline metric
* Document baseline score in README

---

## STEP 5 — Study Solutions

Before improving model:

* Read top solution discussions
* Extract:

  * Model types used
  * Feature engineering strategies
  * Validation strategy
  * Common mistakes

Must document findings inside README under:

```
## Solution Insights
```

Ignore:

* Data leaks
* Magic features
* Extreme ensemble tricks

Focus on learning patterns and reasoning.

---

## STEP 6 — Improved Model Training

Notebook: `03_improved_training.ipynb`

Must include:

* Proper cross-validation
* At least 3 different models
* Hyperparameter tuning attempt
* Feature engineering improvements
* Clear metric logging

Comparison table must be included:

| Model | CV Score | Notes |

---

## STEP 7 — Final Pipeline

Notebook: `04_final_pipeline.ipynb`

This notebook MUST:

1. Load raw data
2. Apply preprocessing
3. Apply feature engineering
4. Train final model
5. Evaluate using correct metric
6. Save trained model to:

```
models/final_model.pkl
```

7. Optionally generate submission file

This notebook must run top-to-bottom without error.

If it does not execute cleanly, the competition is NOT complete.

---

# 5. Model Saving Rules

All models must be saved using joblib or pickle.

Naming convention:

```
model_<modeltype>_<metric>.pkl
```

Example:

```
model_xgboost_auc_0.8731.pkl
```

At least one final model must exist in `/models/`.

---

# 6. Streamlit Deployment

Inside `/app/streamlit_app.py`:

The app must:

* Load saved model
* Accept input
* Apply preprocessing
* Output prediction

App must run locally using:

```
streamlit run streamlit_app.py
```

---

# 7. Hugging Face Deployment

For each competition:

* Create Hugging Face Space
* Upload:

  * streamlit_app.py
  * requirements.txt
  * trained model
* Ensure public deployment works

Deployment link must be recorded in README.

---

# 8. Competition Tracker Requirements

File: `competition-tracker.xlsx`

Columns:

* Competition Name
* Problem Type
* Metric
* Baseline Score
* Best CV Score
* Public LB Score
* Techniques Used
* Streamlit Ready (Yes/No)
* Hugging Face Deployed (Yes/No)
* Status (In Progress / Done)

---

# 9. Definition of DONE

A competition is complete ONLY if:

* README fully documented
* EDA notebook exists
* Baseline reproduced
* Improved training completed
* Final pipeline runs cleanly
* Model saved
* Streamlit app working
* Hugging Face deployed
* Tracker updated

If any of these are missing → status remains IN PROGRESS.

---

# 10. Execution Order Rule

Competitions must be completed sequentially.

Competition 01 must be DONE before Competition 02 begins.

No parallel unfinished competitions.

---

# 11. Expected Final Output After 20 Competitions

* 20 reproducible ML pipelines
* 20 trained models
* 20 Streamlit demos
* 20 Hugging Face deployments
* Deep understanding of competition ML workflows

This specification defines the operating system of the BECOMEAPRO Kaggle Program.
