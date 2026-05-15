import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
def load_data(path):
    df = pd.read_csv(path)
    df = df[['text', 'label']]
    return Dataset.from_pandas(df)

# Tokenization
def tokenize(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def main():
    # Load data
    dataset = load_data("data/train.csv")

    # Train/test split
    dataset = dataset.train_test_split(test_size=0.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenized = tokenize(dataset, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_dir="./logs",
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model
    trainer.save_model("model/")
    tokenizer.save_pretrained("model/")

if __name__ == "__main__":
    main()