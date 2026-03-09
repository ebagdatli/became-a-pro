<div align="center">

# BecomeAPro

### AI-Powered Real-Time Exercise Tracking Platform

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ActiMetric--AI-blue)](https://huggingface.co/spaces/bagdatli/ActiMetric-AI)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Kameraniz araciligiyla vucut hareketlerinizi anlik olarak analiz eden,
tekrarlari sayan ve performansinizi takip eden yapay zeka platformu.

[**Canli Demo**](https://huggingface.co/spaces/bagdatli/ActiMetric-AI) · [**ExercisePrediction Repo**](https://github.com/ebagdatli/no-pain-no-gain)

</div>

---

## Proje Hakkinda

**BecomeAPro**, birden fazla Kaggle veri seti ve yaklasimi birlestirerek tek bir butunlesik egzersiz takip sistemi olusturmayi hedefleyen bir AI platformudur. Her alt proje farkli bir modaliteyi (goruntu, video, zaman serisi, tablo) ele alir ve nihayetinde tek bir urun olarak birlestirilecektir.

| Ozellik | Durum |
|---------|-------|
| Gercek zamanli egzersiz tanimlama (5 hareket) | Done |
| Tarayici icinde WebRTC kamera | Done |
| Otomatik tekrar sayimi | Done |
| Tahmini kalori hesabi | Done |
| HuggingFace Spaces deployment | Done |
| Coklu model birlestirme (ensemble) | Planned |

---

## Model Pipeline

Proje uctan uca bir ML pipeline'i izler. Her alt proje ayni yapiyi kullanir:

```
                                    INFERENCE
DATA                 TRAINING       (Real-time)
 |                      |               |
 v                      v               v
┌──────────┐    ┌──────────────┐    ┌──────────────────┐
│  Kaggle  │    │  Notebook    │    │  Streamlit UI    │
│  Dataset │───>│  (main.ipynb)│───>│  + WebRTC Camera │
└──────────┘    └──────┬───────┘    └────────┬─────────┘
                       │                     │
                       v                     v
              ┌────────────────┐    ┌────────────────────┐
              │  Feature Eng.  │    │  MediaPipe Pose    │
              │  - MediaPipe   │    │  Landmark Detection│
              │  - 33 Landmark │    │  (33 keypoints)    │
              │  - x, y, z    │    └────────┬───────────┘
              └────────┬───────┘             │
                       │                     v
                       v            ┌────────────────────┐
              ┌────────────────┐    │  Preprocessing     │
              │  Model Train   │    │  - Scaling (x100)  │
              │  - XGBoost     │    │  - StandardScaler  │
              │  - PyTorch     │    └────────┬───────────┘
              │  - TensorFlow  │             │
              └────────┬───────┘             v
                       │            ┌────────────────────┐
                       v            │  Classification    │
              ┌────────────────┐    │  - 10 pose classes │
              │  models/       │    │  - Smoothing buffer│
              │  - model.pkl   │───>│  - Rep counting    │
              │  - scaler.pkl  │    │  - Calorie est.    │
              │  - encoder.pkl │    └────────────────────┘
              └────────────────┘
```

### Detayli Pipeline Adimlari

| Adim | Aciklama | Araclar |
|------|----------|---------|
| **1. Veri Toplama** | Kaggle veri setlerinden egzersiz goruntuleri ve landmark verileri | Kaggle API |
| **2. Onisleme** | MediaPipe ile 33 vucut noktasi cikarimi, x/y/z koordinat normalizasyonu | MediaPipe, NumPy |
| **3. Feature Engineering** | Koordinatlarin olceklenmesi (XY: x100, Z: x200), ozellik secimi | Pandas, scikit-learn |
| **4. Model Egitimi** | XGBoost (ana model) veya PyTorch/TensorFlow alternatifi | XGBoost, PyTorch, TensorFlow |
| **5. Degerlendirme** | Cross-validation, classification report, confusion matrix | scikit-learn |
| **6. Inference** | Gercek zamanli kamera goruntusu -> pose algilama -> siniflandirma -> tekrar sayimi | MediaPipe, OpenCV, Streamlit |

### Model Calistirma

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

## Alt Projeler

Her alt proje farkli bir yaklasim ve veri seti ile egzersiz analizinin bir yonunu ele alir:

### Ana Proje

| Proje | Veri Turu | Model | Aciklama |
|-------|-----------|-------|----------|
| **ExercisePrediction** | MediaPipe 33 Landmark (tabular) | XGBoost / PyTorch | 5 egzersiz x 2 poz = 10 sinif tanimlama. Ana uretim modeli. |

### Yardimci Projeler

| Proje | Veri Turu | Model | Aciklama |
|-------|-----------|-------|----------|
| **SmartAICoach** | Eklem acilari (zaman serisi) | XGBoost / PyTorch | MediaPipe eklem acilarindan egzersiz siniflandirma |
| **LSTMExerciseClassificationPushUp** | Video keypoints (.npy) | LSTM (TensorFlow) | Sinav dogru/yanlis form tespiti |
| **HumanActionRecognition** | Goruntu (15 sinif) | CNN (TensorFlow) | Genel insan aktivitesi tanimlama |
| **FacialKeypointsDetection** | 96x96 grayscale goruntu | CNN (TensorFlow/Keras) | Yuz anahtar noktasi regresyonu |
| **CalorieExpenditurePrediction** | Tabular (nabiz, sure, vucut tipi) | CatBoost | Kalori harcamasi tahmini |

---

## Technologies Used

### Core ML & AI

| Teknoloji | Kullanim Alani |
|-----------|---------------|
| **MediaPipe** | Gercek zamanli 33 vucut noktasi (pose landmark) algilama |
| **XGBoost** | Ana siniflandirma modeli (ExercisePrediction, SmartAICoach) |
| **PyTorch** | Alternatif derin ogrenme modeli |
| **TensorFlow / Keras** | CNN ve LSTM modelleri (HAR, Facial Keypoints, Push-up) |
| **scikit-learn** | Onisleme, olcekleme, degerlendirme metrikleri |

### Web & Deployment

| Teknoloji | Kullanim Alani |
|-----------|---------------|
| **Streamlit** | Web arayuzu ve interaktif dashboard |
| **streamlit-webrtc** | Tarayici icinde gercek zamanli kamera erisimi (WebRTC) |
| **Twilio TURN** | NAT/firewall arkasinda kamera baglantisi icin relay sunucu |
| **Hugging Face Spaces** | Canli demo deployment (Docker) |
| **Docker** | Konteyner tabanli deployment (Python 3.10-slim) |

### Veri Isleme & Gorsellestirme

| Teknoloji | Kullanim Alani |
|-----------|---------------|
| **OpenCV** | Goruntu isleme, iskelet cizimi, video overlay |
| **NumPy / Pandas** | Veri manipulasyonu ve sayisal islemler |
| **Matplotlib** | Egitim grafikleri ve analiz gorselleri |
| **Joblib** | Model serializasyonu (.pkl) |

---

## Proje Yapisi

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
│
├── SmartAICoach/                    # Eklem acisi tabanli siniflandirma
├── LSTMExerciseClassificationPushUp/# LSTM ile sinav form analizi
├── HumanActionRecognition/          # CNN ile aksiyon tanimlama
├── FacialKeypointsDetection/        # Yuz anahtar noktasi algilama
├── CalorieExpenditurePrediction/    # Kalori harcamasi tahmini
├── ActiMetric-AI/                   # HF Space deployment mirror
│
├── run_competition.py               # Alt proje notebook orkestratoru
└── README.md
```

---

## Kurulum ve Calistirma

### Gereksinimler

- Python 3.10+
- Web kamerasi (gercek zamanli demo icin)

### Hizli Baslangic

```bash
# 1. Repoyu klonlayin
git clone https://github.com/ebagdatli/no-pain-no-gain.git
cd no-pain-no-gain

# 2. Sanal ortam olusturun
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Bagimliliklari yukleyin
cd ExercisePrediction
pip install -r requirements.txt

# 4. Modeli egitin (ilk seferde)
cd ..
python run_competition.py ExercisePrediction

# 5. Uygulamayi baslatin
cd ExercisePrediction
streamlit run app/streamlit_app.py
```

### HuggingFace Spaces Deployment

HuggingFace'te kamera baglantisi icin Twilio TURN sunucu gereklidir:

1. [twilio.com](https://www.twilio.com/try-twilio) adresinden ucretsiz hesap olusturun
2. Account SID ve Auth Token alin
3. HF Space Settings > Secrets bolumune ekleyin:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`

---

## Kaggle Veri Setleri

| Konu | Yarisma / Veri Seti | Veri Icerigi | Link |
|------|---------------------|-------------|------|
| **Exercise Prediction** | Multi-Class Exercise Poses | 10 poz, MediaPipe 33 landmark (x,y,z) | [Kaggle](https://www.kaggle.com/datasets/dp5995/gym-exercise-mediapipe-33-landmarks) |
| **Smart AI Coach** | Exercise Recognition | 33 MediaPipe eklem noktasi hareket verisi | [Kaggle](https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition-time-series) |
| **Push-Up (LSTM)** | Pushup Pose Detection | Sinav videolarindan cikarilmis eklem koordinatlari | [Kaggle](https://www.kaggle.com/datasets/mohamadashrafsalama/pushup) |
| **Action Recognition** | Human Action Recognition | 15 farkli gunluk aktivite etiketi | [Kaggle](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) |
| **Facial Keypoints** | Facial Keypoints Detection | Yuzdeki 15 kilit noktanin (x,y) koordinatlari | [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection) |
| **Calorie Expenditure** | Playground Series S5E5 | Nabiz, sure ve vucut tipi ile kalori tahmini | [Kaggle](https://www.kaggle.com/competitions/playground-series-s5e5) |

---

## Demo

**ActiMetric AI** -- HuggingFace Spaces uzerinde canli demo:

[**https://huggingface.co/spaces/bagdatli/ActiMetric-AI**](https://huggingface.co/spaces/bagdatli/ActiMetric-AI)

| | |
|:---:|:---:|
| ![Hero](assets/actimetric_hero.png) | ![Exercises](assets/actimetric_exercises.png) |
| Modern UI - Hero Section | Desteklenen Egzersizler |

---

## Future Work

### Kisa Vadeli (v2)

- [ ] **Form Analizi** -- LSTM modeli ile sinav, squat gibi hareketlerde dogru/yanlis form tespiti ve geri bildirim
- [ ] **Coklu Model Ensemble** -- ExercisePrediction + SmartAICoach modellerini birlestirerek daha yuksek dogruluk
- [ ] **Egzersiz Gecmisi** -- Kullanici bazli antrenman gecmisi kaydi ve ilerleme grafikleri
- [ ] **Sesli Geri Bildirim** -- Tekrar sayimi ve form duzeltmeleri icin sesli komutar

### Orta Vadeli (v3)

- [ ] **Kisisellestirilmis Antrenman Plani** -- Kullanicinin performansina gore AI tabanli program onerisi
- [ ] **Video Analizi** -- Canli kamera yerine yuklenen video dosyalarinin analizi
- [ ] **Mobil Uygulama** -- Flutter veya React Native ile mobil versiyon
- [ ] **Kalori Modeli Entegrasyonu** -- CalorieExpenditurePrediction modelinin gercek zamanli sisteme eklenmesi

### Uzun Vadeli

- [ ] **Transformer Modeli** -- Zaman serisi tabanli hareket tanimlama icin Transformer mimarisi
- [ ] **3D Poz Tahmini** -- Derinlik kamerasi veya stereo goruntu ile 3 boyutlu poz analizi
- [ ] **Coklu Kullanici** -- Ayni anda birden fazla kisinin egzersiz takibi
- [ ] **Giyilebilir Cihaz Entegrasyonu** -- Akilli saat ve fitness tracker verileriyle birlesik analiz

---

<div align="center">

**BecomeAPro** -- AI-Powered Exercise Tracker

MediaPipe · XGBoost · PyTorch · TensorFlow · Streamlit · WebRTC

Made with dedication by [@ebagdatli](https://github.com/ebagdatli)

</div>
