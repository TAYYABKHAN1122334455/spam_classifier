import re
import pickle
import os
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

MODEL_PATH = "model/spam_model.pkl"
VEC_PATH = "model/vectorizer.pkl"

_model = None
_vectorizer = None


def _load_model():
    global _model, _vectorizer
    
    if _model is not None:
        return  # Already loaded

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        raise FileNotFoundError(
            "Model not found! Please run:\npython model.py"
        )

    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(VEC_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def predict_spam(email_text):
    if not email_text or not email_text.strip():
        return "INVALID", 0.0

    try:
        _load_model()
    except Exception as e:
        print(f"Error: {e}")
        return "ERROR", 0.0

    cleaned = clean_text(email_text)
    vectorized = _vectorizer.transform([cleaned])
    
    prediction = _model.predict(vectorized)[0]
    probability = _model.predict_proba(vectorized)[0]

    if prediction == 1:
        return "SPAM", round(probability[1] * 100, 2)
    else:
        return "NOT SPAM", round(probability[0] * 100, 2)