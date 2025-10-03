import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt

def basic_clean(text):
    URL_RE   = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    HTML_RE  = re.compile(r'<.*?>')
    NUM_RE   = re.compile(r'\d+')
    # PUNCT_RE = re.compile(r"[^A-Za-z']+")
    STOPWORDS = ENGLISH_STOP_WORDS

    s = str(text).lower()
    s = HTML_RE.sub(' ', s)
    s = URL_RE.sub(' ', s)
    s = NUM_RE.sub(' num ', s)

    tokens = re.findall(r"[A-Za-z']+", s)
    tokens = [t for t in tokens if t not in STOPWORDS]

    return ' '.join(tokens)

def stratified_splits(df, val_size=0.1, test_size=0.2, seed=42):
    """
    Stratified train/val/test split.
    Ensures that all splits have the same class distribution as the original dataset."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['sentiment'])
    y=train['sentiment']; n=y.nunique(); 
    vs = max(int(round(len(train)*val_size)), n)
    vf = max(min(vs/len(train),0.5), 1.0/len(train))
    train, val = train_test_split(train, test_size=vf, random_state=seed, stratify=y)
    return train, val, test

def save_report_and_cm(y_true, y_pred, name: str, out_metrics: Path, out_fig: Path) -> dict:
    """
    Save a classification report (JSON) and a confusion matrix (PNG) for a single model.
    Returns a compact summary row for cross-model comparison.
    """
    rep = classification_report(y_true, y_pred, zero_division=0, labels=["negative", "positive"], output_dict=True)

    # Classification report (precision/recall/f1 by class, plus averages)
    with open(out_metrics / f"{name}_classification_report.json", "w") as f:
        json.dump(rep, f, indent=2)

    # Confusion matrix (rows = true, cols = predicted)
    cm = confusion_matrix(y_true, y_pred, labels=["negative", "positive"])
    fig = plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix — {name}")
    plt.colorbar(im)
    plt.xticks([0, 1], ["negative", "positive"])
    plt.yticks([0, 1], ["negative", "positive"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.tight_layout()
    plt.savefig(out_fig / f"{name}_confusion_matrix.png", dpi=140)
    plt.close(fig)

    # Compact summary row (used to build the comparison table)
    row = {
        "model": name,
        "accuracy": rep["accuracy"],
        "macro_precision": rep["macro avg"]["precision"],
        "macro_recall": rep["macro avg"]["recall"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
    }
    return row