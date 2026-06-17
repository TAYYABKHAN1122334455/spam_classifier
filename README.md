# Email Spam Classifier

AI course project — Email Spam Classifier using Naive Bayes and Natural Language Processing.

## Team

| Roll No. | Name | Role | Files |
|---|---|---|---|
| 050 | Tayyab Jadoon | GUI Developer + Model Training (Part 1) | gui.py, model.py (half) |
| 032 | Jamal Raja | Testing & Validation | test_cases.py |
| 027 | M. Arqam | Model Training (Part 2) + Prediction Logic | model.py (half), predict.py |

## Project Structure

```
spam_classifier/
├── dataset/
│   └── spam.csv          (download from Kaggle, see dataset/README.md)
├── model/
│   ├── spam_model.pkl     (generated after running model.py)
│   └── vectorizer.pkl     (generated after running model.py)
├── model.py
├── predict.py
├── gui.py
├── test_cases.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python model.py          # trains and saves the model
python test_cases.py     # runs the validation suite
python gui.py             # launches the desktop app
```

## Results

- Model accuracy: ~97.8% on the held-out test set
- Test suite: 10/10 test cases passed

## Status

✅ Complete.

