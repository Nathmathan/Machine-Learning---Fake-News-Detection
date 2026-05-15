# Fake News Detection

## Project Description

This project builds a machine learning system to classify news articles as **real or fake**.

Two approaches are implemented:
- **Baseline model:** TF-IDF + Logistic Regression  
- **Advanced model:** DistilBERT (Transformer-based NLP model)

The goal is to compare traditional NLP methods with modern deep learning approaches.

---

## How to run

1. Install dependencies:
pip install -r requirements.txt

2. Prepare data:
python src/prepare_data.py

3. Train model:
python src/train.py

4. Run demo:
streamlit run app.py

## Dataset
This project uses the Kaggle Fake News dataset:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

You can either:
- Run prepare_data.py to automatically process the dataset
- OR manually download and place files in data/

## Pretrained Model

Download the trained model here:
https://drive.google.com/drive/folders/1u3eJDyw7ytkhFEBjcdZC6w3kwj8qccRv?usp=sharing

Unzip into:
model/

## Dependencies
- transformers
- torch
- datasets
- scikit-learn
- streamlit

## Code structure
- src/train.py → trains DistilBERT
- src/baseline.py → TF-IDF baseline
- app.py → UI demo