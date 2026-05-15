import kagglehub
import pandas as pd
import os

def download_data():
    path = kagglehub.dataset_download(
        "clmentbisaillon/fake-and-real-news-dataset"
    )
    return path

def prepare():
    path = download_data()

    fake = pd.read_csv(os.path.join(path, "Fake.csv"))
    true = pd.read_csv(os.path.join(path, "True.csv"))

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true])
    df = df[["text", "label"]]

    df.to_csv("data/train.csv", index=False)

if __name__ == "__main__":
    prepare()