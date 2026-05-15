# Fake News Detection

## How to run

1. Install dependencies:
pip install -r requirements.txt

2. Train model:
python src/train.py

3. Run demo:
streamlit run app.py

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