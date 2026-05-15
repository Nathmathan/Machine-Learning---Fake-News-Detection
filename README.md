# Fake News Detection

## How to run

1. Install dependencies:
pip install -r requirements.txt

2. Train model:
python src/train.py

3. Run demo:
streamlit run app.py

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