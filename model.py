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
# Entry Point (Part 2 — training logic — will be added by Arqam)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"[INFO] Total samples : {len(df)}  |  Ham: {(df.label == 'ham').sum()}  |  Spam: {(df.label == 'spam').sum()}")

    print("[INFO] Preprocessing text...")
    df = preprocess(df)
    print("[INFO] Preprocessing complete. Sample cleaned text:")
    print(df[["text", "clean_text"]].head(3))

    # TODO (Arqam): TF-IDF vectorization, Naive Bayes training, evaluation, and saving
