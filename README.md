🏋️‍♂️ NoPainNoGain
Next-Generation AI Exercise Analytics
<p align="center">
<img src="assets/actimetric_hero.png" width="850" alt="NoPainNoGain Hero">
</p>

<p align="center">
<a href="https://huggingface.co/spaces/bagdatli/ActiMetric-AI">
<img src="https://img.shields.io/badge/%F0%9F%9A%80%20LIVE%20DEMO-TRY%20ON%20HUGGING%20FACE-FFA000?style=for-the-badge&logo=huggingface&logoColor=white" height="45">
</a>
</p>

<p align="center">
<b>Transform your webcam into a personal AI trainer.</b>


<i>Real-time pose estimation, exercise classification, and movement analytics.</i>
</p>

<p align="center">
<img src="https://img.shields.io/github/stars/ebagdatli/no-pain-no-gain?style=flat-square&color=6e5494" alt="Stars">
<img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Framework-MediaPipe-00B0FF?style=flat-square" alt="MediaPipe">
<img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

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

## 🛣️ Development Roadmap (v2 & Beyond)

| Feature | Status | Tech Stack |
| :--- | :---: | :--- |
| **Real-time Recognition** | ✅ | MediaPipe + XGBoost |
| **Repetition Counting** | 🏗️ | Temporal Smoothing / Peak Detection |
| **Calorie Estimation** | 📋 | MET Value Integration |
| **Form Analysis (AI Coach)** | 📋 | Angle Consistency Check |
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

## UI

|                                     |                                               |
| :---------------------------------: | :-------------------------------------------: |
| ![Hero](assets/actimetric_hero.png) | ![Exercises](assets/actimetric_exercises.png) |
|       Modern UI – Hero Section      |              Supported Exercises              |

---

## Folder Structure

```
BecameAPro/
├── CalorieExpenditurePrediction/
├── ExercisePrediction/          
│   ├── app/                
│   ├── src/                
│   ├── hf_space/           # Hugging Face Space deployment
│   └── models/             
├── FacialKeypointsDetection/
├── HumanActionRecognition/
├── LSTMExerciseClassificationPushUp/
├── SmartAICoach/
├── Yoga Pose Classification/
├── run_competition.py     
└── README.md
```

---

## Model Run

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

---

### 🧠 Core ML & AI Stack

| Technology | Role in Project | Category |
| --- | --- | --- |
| <img src="[https://raw.githubusercontent.com/google/mediapipe/master/mediapipe_logo.png](https://www.google.com/search?q=https://raw.githubusercontent.com/google/mediapipe/master/mediapipe_logo.png)" width="18"/> **MediaPipe** | 33-landmark real-time pose estimation & tracking. | **Computer Vision** |
| <img src="[https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/logo/xgboost-logo.png](https://www.google.com/search?q=https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/logo/xgboost-logo.png)" width="18"/> **XGBoost** | High-performance gradient boosting for exercise classification. | **Machine Learning** |
| <img src="[https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg](https://www.google.com/search?q=https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg)" width="18"/> **PyTorch** | Deep learning research and custom model architecture. | **Deep Learning** |
| <img src="[https://www.gstatic.com/devrel-devsite/prod/v2385106516327668630712398553648115664326/tensorflow/images/favicon.png](https://www.google.com/search?q=https://www.gstatic.com/devrel-devsite/prod/v2385106516327668630712398553648115664326/tensorflow/images/favicon.png)" width="18"/> **TensorFlow** | Pose-based sequence modeling and Keras integration. | **Deep Learning** |
| <img src="[https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg](https://www.google.com/search?q=https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)" width="18"/> **scikit-learn** | Data preprocessing, pipeline scaling, and evaluation metrics. | **Data Science** |

### 💻 Infrastructure & Web

* **Streamlit:** Powers the interactive web dashboard and real-time camera feedback.
* **Hugging Face Spaces:** Handles global deployment and model hosting.
* **Python 3.10+:** The core language driving the entire data pipeline.
---

> The **ExercisePrediction** project is also being developed as an independent repository:
> https://github.com/ebagdatli/no-pain-no-gain
