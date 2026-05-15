import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "model/"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    return "Fake" if prediction == 1 else "Real"


if __name__ == "__main__":
    sample = input("Enter news text: ")
    print("Prediction:", predict(sample))