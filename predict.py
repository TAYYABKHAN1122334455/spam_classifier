# =============================================================================
# predict.py
# Author  : M. Arqam (Roll No. 027)
# Purpose : Load the saved Naive Bayes model and TF-IDF vectorizer, apply
#           the same text preprocessing used during training, and return a
#           classification result with a confidence percentage.
#           This module is imported by both gui.py and test_cases.py.
# =============================================================================

import re
import pickle
import sys

import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# NLTK Resource Setup
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Constants — paths must match what model.py wrote
# ---------------------------------------------------------------------------
MODEL_PATH = "model/spam_model.pkl"
VEC_PATH   = "model/vectorizer.pkl"

# ---------------------------------------------------------------------------
# Model Loading
# Loaded once at import time so every call to predict_spam() is instant.
# ---------------------------------------------------------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        _vectorizer = pickle.load(f)
except FileNotFoundError:
    print("[ERROR] Model files not found. Run model.py first.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: _clean_text
# Must be identical to the clean_text function used in model.py so that
# the vectorizer receives consistently formatted input at inference time.
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """
    Apply the standard NLP preprocessing pipeline to a single string.

    Steps:
        1. Lowercase conversion
        2. URL removal
        3. Digit removal
        4. Special character removal (keep only a-z and spaces)
        5. Stopword removal and short-word filtering (len <= 2)

    Parameters
    ----------
    text : str
        Raw input text from the user.

    Returns
    -------
    str
        Cleaned, space-separated token string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)             # remove URLs / links
    text = re.sub(r"\d+", "", text)                  # remove digits
    text = re.sub(r"[^a-z\s]", "", text)             # remove special characters
    tokens = [
        word for word in text.split()
        if word not in STOP_WORDS and len(word) > 2   # drop stopwords & short tokens
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Public API: predict_spam
# ---------------------------------------------------------------------------
def predict_spam(email_text: str) -> tuple[str, float]:
    """
    Classify an email or SMS message as SPAM or NOT SPAM.

    Processing pipeline:
        1. Validate that input is non-empty
        2. Clean text using the standard preprocessing pipeline
        3. Vectorize using the fitted TF-IDF vectorizer
        4. Predict class using the trained Naive Bayes model
        5. Extract confidence probability for the predicted class

    Parameters
    ----------
    email_text : str
        Raw email or SMS text entered by the user.

    Returns
    -------
    tuple[str, float]
        A tuple of (label, confidence) where:
            label      : "SPAM", "NOT SPAM", or "INVALID"
            confidence : percentage rounded to 2 decimal places (0.0 if invalid)
    """
    # --- Input Validation -------------------------------------------------
    if not email_text or not email_text.strip():
        return "INVALID", 0.0

    # --- Preprocessing -------------------------------------------------------
    cleaned = _clean_text(email_text)

    # --- Vectorization -------------------------------------------------------
    vectorized = _vectorizer.transform([cleaned])

    # --- Inference -------------------------------------------------------
    prediction  = _model.predict(vectorized)[0]
    probability = _model.predict_proba(vectorized)[0]

    # prediction == 1 -> SPAM, prediction == 0 -> NOT SPAM
    if prediction == 1:
        confidence = round(probability[1] * 100, 2)
        return "SPAM", confidence
    else:
        confidence = round(probability[0] * 100, 2)
        return "NOT SPAM", confidence
