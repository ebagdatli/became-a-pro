---

# NoPainNoGain

### AI-Powered Real-Time Exercise Tracking Platform

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ActiMetric--AI-blue)](https://huggingface.co/spaces/bagdatli/ActiMetric-AI)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## About the Project

An **AI-powered real-time exercise recognition and analysis platform**.
Using your camera, the system analyzes body movements instantly and detects different exercise types.

### Current Features

* **Exercise Recognition**
  Automatically detects exercises such as **push-ups, sit-ups, squats, pull-ups, and jumping** using real-time camera input.

### Planned Features (v2)

* **Repetition Counting**
  Automatically counts the number of repetitions for each detected exercise.

* **Calorie Estimation**
  Estimates the calories burned based on the **exercise type, duration, and repetition count**.

---

# Approach

The project is designed as a **single integrated system**.

To achieve this goal:

* Projects that perform well in **sit-up detection**
* Projects that perform well in **push-up detection**
* Projects that perform well in **pull-up detection**
* Projects that perform well in **general action recognition**

were analyzed, their datasets were studied, and models were developed.

These subprojects will eventually be **combined into a single unified system**.

---

# Kaggle Exercise & Computer Vision Dataset Summary

| Topic                   | Competition / Dataset      | Data Content (Summary)                                                                        | Link   |
| ----------------------- | -------------------------- | --------------------------------------------------------------------------------------------- | ------ |
| **Calorie Expenditure** | Playground Series S5E5     | **Tabular:** Calorie estimation using pulse, duration, and body type data.                    | Kaggle |
| **Facial Keypoints**    | Facial Keypoints Detection | **Image:** $(x, y)$ coordinates of 15 facial landmarks.                                       | Kaggle |
| **Action Recognition**  | Human Action Recognition   | **Image:** 15 daily activity labels (running, walking, etc.).                                 | Kaggle |
| **Push-Up (LSTM)**      | Pushup Pose Detection      | **Time Series:** Joint coordinates extracted from push-up videos.                             | Kaggle |
| **Yoga Classification** | Yoga Pose Classification   | **Image:** Labeled images of 5 fundamental yoga poses.                                        | Kaggle |
| **Smart AI Coach**      | Exercise Recognition       | **Coordinate Data:** Motion data of 33 MediaPipe body landmarks.                              | Kaggle |
| **Exercise Prediction** | Multi-Class Exercise Poses | **Tabular:** 10 exercise poses (push-up, pull-up, sit-up, etc.) using MediaPipe 33 landmarks. | Kaggle |

---
### UI

|                                     |                                               |
| :---------------------------------: | :-------------------------------------------: |
| ![Hero](assets/actimetric_hero.png) | ![Exercises](assets/actimetric_exercises.png) |
|       Modern UI – Hero Section      |              Supported Exercises              |

---

```
## Folder Structure

```
BecomeAPro/
├── ExercisePrediction/              # Ana egzersiz tanimlama projesi
│   ├── app/
│   │   └── streamlit_app.py         # Lokal Streamlit UI (WebRTC kamera)
│   ├── src/
│   │   ├── camera_demo.py           # OpenCV kamera demo
│   │   ├── data_loader.py           # Veri yukleme
│   │   ├── preprocessing.py         # Onisleme pipeline
│   │   ├── train.py                 # Model egitimi
│   │   └── predict.py               # Tahmin fonksiyonlari
│   ├── hf_space/                    # HuggingFace Space deployment
│   │   ├── app.py                   # WebRTC + Streamlit
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── notebooks/
│   │   └── main.ipynb               # Ana egitim pipeline
│   ├── models/                      # Egitilmis modeller
│   ├── data/                        # Veri setleri
│   └── requirements.txt
```

### Model Run

```bash
# Tum pipeline'i bastan calistir (notebook uzerinden)
python run_competition.py ExercisePrediction

# Lokal kamera demo (OpenCV penceresi)
cd ExercisePrediction
python -m src.camera_demo

# Streamlit web arayuzu (tarayici icinde WebRTC)
cd ExercisePrediction
streamlit run app/streamlit_app.py
```

---

## Technologies Used

### Core ML & AI

| Teknoloji | 
|-----------|
| **MediaPipe** | 
| **XGBoost** |
| **PyTorch** |
| **TensorFlow / Keras** | 
| **scikit-learn** | 

> The **ExercisePrediction** project is also being developed as an independent repository:
> [https://github.com/ebagdatli/no-pain-no-gain](https://github.com/ebagdatli/no-pain-no-gain)

---
