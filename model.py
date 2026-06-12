# =============================================================================
# model.py
# Part 1 Author : Tayyab Jadoon (Roll No. 050) — Data Loading & Preprocessing
# Part 2 Author : M. Arqam (Roll No. 027) — Vectorization, Training, Saving
# Purpose       : Load dataset, preprocess text, train Naive Bayes classifier,
#                 evaluate performance, and save the trained model and vectorizer.
# =============================================================================

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

# ---------------------------------------------------------------------------
# NLTK Resource Setup
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset/spam.csv"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "spam_model.pkl")
VEC_PATH     = os.path.join(MODEL_DIR, "vectorizer.pkl")
TEST_SIZE    = 0.20
RANDOM_STATE = 42
MAX_FEATURES = 5000
NGRAM_RANGE  = (1, 2)
ALPHA        = 0.1        # Laplace smoothing for Naive Bayes


# ---------------------------------------------------------------------------
# Helper: clean_text
# Applies a standard NLP preprocessing pipeline to a single string.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Preprocess raw email/SMS text for TF-IDF vectorization.

    Steps applied in order:
        1. Lowercase conversion
        2. URL removal
        3. Digit removal
        4. Special character removal (keep only a-z and spaces)
        5. Stopword removal and short-word filtering (len <= 2)

    Parameters
    ----------
    text : str
        Raw input message text.

    Returns
    -------
    str
        Cleaned, space-separated token string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)             # remove URLs
    text = re.sub(r"\d+", "", text)                  # remove digits
    text = re.sub(r"[^a-z\s]", "", text)             # remove special chars
    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS and len(word) > 2  # filter stopwords & short tokens
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Step 1 — Load Dataset
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the SMS Spam Collection CSV and return a clean two-column DataFrame.

    Parameters
    ----------
    path : str
        Relative or absolute path to spam.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['label', 'text'].
    """
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    return df


# ---------------------------------------------------------------------------
# Step 2 — Preprocess
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning and encode labels as integers.

    Encoding: ham -> 0, spam -> 1

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 'label' and 'text' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'clean_text' and 'label_encoded' columns.
    """
    df = df.copy()
    df["clean_text"]    = df["text"].apply(clean_text)
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# ---------------------------------------------------------------------------
# Step 3 — Train / Evaluate / Save
# Author: M. Arqam (Roll No. 027)
# ---------------------------------------------------------------------------
def train_and_save(df: pd.DataFrame) -> None:
    """
    Split data, fit TF-IDF vectorizer and Multinomial Naive Bayes model,
    print evaluation metrics, then persist both artifacts to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing 'clean_text' and 'label_encoded'.
    """
    # --- Train / Test Split (stratified to preserve class ratio) -----------
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label_encoded"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_encoded"],
    )

    # --- TF-IDF Vectorization ---------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )
    X_train_vec = vectorizer.fit_transform(X_train)  # fit on training data only
    X_test_vec  = vectorizer.transform(X_test)        # transform test data

    # --- Model Training ---------------------------------------------------
    model = MultinomialNB(alpha=ALPHA)
    model.fit(X_train_vec, y_train)

    # --- Evaluation ---------------------------------------------------------
    predictions = model.predict(X_test_vec)
    accuracy    = accuracy_score(y_test, predictions)

    print(f"[RESULT] Accuracy : {accuracy * 100:.2f}%")
    print()
    print(classification_report(
        y_test, predictions,
        target_names=["Ham (Normal)", "Spam"]
    ))

    # --- Save Artifacts -----------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"[SAVED] Model      -> {MODEL_PATH}")
    print(f"[SAVED] Vectorizer -> {VEC_PATH}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"[INFO] Total samples : {len(df)}  |  Ham: {(df.label == 'ham').sum()}  |  Spam: {(df.label == 'spam').sum()}")

    print("[INFO] Preprocessing text...")
    df = preprocess(df)

    print("[INFO] Training Multinomial Naive Bayes model...\n")
    train_and_save(df)

    print("\n[DONE] Training complete.")
