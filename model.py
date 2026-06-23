import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

DATASET_PATH = "dataset/spam.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def load_dataset(path):
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, [0, 1]]
    df.columns = ["label", "text"]
    return df


def train():
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"Samples: {len(df)} | Ham: {(df.label=='ham').sum()} | Spam: {(df.label=='spam').sum()}")

    print("Cleaning text...")
    df["clean"] = df["text"].apply(clean_text)
    df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label_enc"], test_size=0.20, random_state=42, stratify=df["label_enc"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved successfully!")


if __name__ == "__main__":
    train()