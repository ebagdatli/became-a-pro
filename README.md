# BecomeAPro

## Ana Amaç

Bu projenin temel hedefi, **anlık (real-time) kamera görüntüleri** ile hareketleri algılayıp:

1. **Hangi egzersizlerin** yapıldığını tespit etmek (şınav, mekik, barfiks vb.)
2. **Kaç tekrar** yapıldığını saymak
3. Bu hareketlere dayanarak **yakılan kalori** miktarını hesaplamak

olacak şekilde entegre bir sistem kurmaktır.

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

## Proje Yapısı

```
BecomeAPro/
├── CalorieExpenditurePrediction/
├── ExercisePrediction/
├── FacialKeypointsDetection/
├── HumanActionRecognition/
├── LSTMExerciseClassificationPushUp/
├── SmartAICoach/
├── Yoga Pose Classification/
├── run_competition.py    # Alt projelerin notebook'larını çalıştırır
└── README.md
```

---

## Çalıştırma

Belirli bir alt projeyi eğitmek için:

```bash
python run_competition.py <ProjeKlasörAdı>
```

Örnek:
```bash
python run_competition.py ExercisePrediction
python run_competition.py LSTMExerciseClassificationPushUp
```
