import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

def main():
    df = pd.read_csv("data/train.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))

    # Save
    joblib.dump(model, "model/baseline_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

if __name__ == "__main__":
    main()