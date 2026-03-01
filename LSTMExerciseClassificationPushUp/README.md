# LSTM Exercise Classification – Push-up (Correct vs Incorrect)

**Kaggle**: [Push-up Dataset](https://www.kaggle.com/datasets/mohamadashrafsalama/pushup/data)

Bu proje **sadece şınav hareketleri** ile ilgilidir. Videoları **doğru** (correct) veya **yanlış** (incorrect) olarak sınıflandırır. Conv3D tabanlı video sınıflandırma modeli kullanılır.

## Veri

Kaggle'dan indirip `data/raw/` altına çıkarın:

- `Correct sequence/` – doğru şınav videoları (.mp4)
- `Wrong sequence/` – yanlış şınav videoları (.mp4)

Kaggle ortamında: `/kaggle/input/pushup/`

## Proje yapısı

```
LSTMExerciseClassificationPushUp/
├── data/
│   ├── raw/              # Correct sequence, Wrong sequence
│   └── processed/
├── models/               # final_model.keras, encoder.pkl, meta.pkl
├── notebooks/
│   └── main.ipynb       # pipeline: load → train → save
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
└── requirements.txt
```

## Kurulum

```bash
cd LSTMExerciseClassificationPushUp
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\pip install ipykernel
venv\Scripts\python -m ipykernel install --user --name=lstm-pushup --display-name="Python (LSTMExerciseClassificationPushUp)"
```

## Çalıştırma

```bash
# Repo kökünden
python run_competition.py LSTMExerciseClassificationPushUp
```

Streamlit (video yükleyerek tahmin):

```bash
cd LSTMExerciseClassificationPushUp
venv\Scripts\python -m streamlit run app/streamlit_app.py
```
