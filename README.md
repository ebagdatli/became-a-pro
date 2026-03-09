# NoPainNoGain

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ActiMetric--AI-blue)](https://huggingface.co/spaces/bagdatli/ActiMetric-AI)

> **Canli Demo:** [https://huggingface.co/spaces/bagdatli/ActiMetric-AI](https://huggingface.co/spaces/bagdatli/ActiMetric-AI)

## Proje Hakkinda

Yapay zeka destekli, gercek zamanli egzersiz tanimlama ve analiz platformu. Kameraniz araciligiyla vucut hareketlerinizi anlik olarak analiz eder.

- [x] **Egzersiz Tanimlama** -- Gercek zamanli kamera goruntusuyle sinav, mekik, squat, barfiks ve ziplama hareketlerini otomatik olarak tanir
- [ ] **Tekrar Sayimi** -- Her egzersiz icin tekrar sayisini otomatik hesaplar *(v2)*
- [ ] **Kalori Tahmini** -- Yapilan egzersiz turune ve tekrar sayisina gore yakilaN kaloriyi hesaplar *(v2)*

---

## Yaklaşım

Proje, tek bir bütünleşik sistem oluşturmak üzere tasarlanmıştır. Bu amaçla:

- Mekik konusunda iyi çalışan
- Şınav konusunda iyi çalışan
- Barfiks konusunda iyi çalışan
- Tüm hareketleri tanımada iyi olan

birden fazla referans proje incelenmiş, veri setleri kullanılmış ve modeller geliştirilmiştir. Bu alt projeler daha sonra **tek bir proje amacıyla birleştirilecektir**.

---

## Kaggle Egzersiz ve Görüntü İşleme Veri Özetleri

| Konu Başlığı | Yarışma / Veri Seti | Veri İçeriği (Özet) | Link |
| --- | --- | --- | --- |
| **Calorie Expenditure** | [Playground Series S5E5](https://www.kaggle.com/competitions/playground-series-s5e5) | **Tabular:** Nabız, süre ve vücut tipi verisiyle kalori tahmini. | [Kaggle](https://www.kaggle.com/competitions/playground-series-s5e5) |
| **Facial Keypoints** | [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection) | **Görüntü:** Yüzdeki 15 kilit noktanın $(x, y)$ koordinatları. | [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection) |
| **Action Recognition** | [Human Action Recognition](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) | **Görüntü:** 15 farklı günlük aktivite etiketi (koşma, yürüme vb.). | [Kaggle](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) |
| **Push-Up (LSTM)** | [Pushup Pose Detection](https://www.kaggle.com/datasets/mohamadashrafsalama/pushup) | **Zaman Serisi:** Şınav videolarından çıkarılmış eklem koordinatları. | [Kaggle](https://www.kaggle.com/datasets/mohamadashrafsalama/pushup) |
| **Yoga Classification** | [Yoga Pose Classification](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification) | **Görüntü:** 5 temel yoga pozunun sınıflandırılmış fotoğrafları. | [Kaggle](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification) |
| **Smart AI Coach** | [Exercise Recognition](https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition-time-series) | **Koordinat:** 33 farklı MediaPipe eklem noktasının hareket verisi. | [Kaggle](https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition-time-series) |
| **Exercise Prediction** | [Multi-Class Exercise Poses](https://www.kaggle.com/datasets/dp5995/gym-exercise-mediapipe-33-landmarks) | **Tabular:** 10 egzersiz pozu (şınav, barfiks, mekik vb.) MediaPipe 33 landmark. | [Kaggle](https://www.kaggle.com/datasets/dp5995/gym-exercise-mediapipe-33-landmarks) |

---

## Demo

**ActiMetric AI** -- Yapay zeka destekli gercek zamanli egzersiz takip uygulamasi Hugging Face Spaces uzerinde yayinda:

[**ActiMetric AI - Hugging Face Space**](https://huggingface.co/spaces/bagdatli/ActiMetric-AI)

| | |
|:---:|:---:|
| ![Hero](assets/actimetric_hero.png) | ![Exercises](assets/actimetric_exercises.png) |
| Modern UI - Hero Section | Desteklenen Egzersizler |

---

## Proje Yapisi

```
NoPainNoGain/
├── CalorieExpenditurePrediction/
├── ExercisePrediction/          -> github.com/ebagdatli/no-pain-no-gain
│   ├── app/                # Yerel Streamlit UI
│   ├── src/                # Model egitimi & kamera demo
│   ├── hf_space/           # Hugging Face Space deployment
│   └── models/             # Egitilmis model dosyalari
├── FacialKeypointsDetection/
├── HumanActionRecognition/
├── LSTMExerciseClassificationPushUp/
├── SmartAICoach/
├── Yoga Pose Classification/
├── run_competition.py      # Alt projelerin notebook'larini calistirir
└── README.md
```

> **ExercisePrediction** projesi bagimsiz bir repo olarak da gelistirilmektedir:
> [github.com/ebagdatli/no-pain-no-gain](https://github.com/ebagdatli/no-pain-no-gain)

