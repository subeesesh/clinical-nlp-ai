"""
classifier.py
-------------
Trains and loads a TF-IDF + Logistic Regression disease classifier.
Saves/loads the model as a pickle file for fast inference.

Usage (standalone training):
    python classifier.py
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Resolve paths relative to this file so the module works
# whether called from /backend or the project root
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "..", "models", "trained_classifier.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "medical_notes_dataset.csv")

# Add backend dir to path so sibling imports work
sys.path.insert(0, BASE_DIR)
from preprocess import preprocess_batch


# ── Training ─────────────────────────────────────────────────────────────────

def train_model() -> Pipeline:
    """
    Train a TF-IDF + Logistic Regression pipeline on the medical notes dataset.
    Saves the model to disk and returns it.
    """
    print("[INFO] Loading dataset from:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    # Strip surrounding quotes that CSV may carry
    df["text"]  = df["text"].str.strip('"').str.strip("'")
    df["label"] = df["label"].str.strip()
    df.dropna(inplace=True)

    X = preprocess_batch(df["text"].tolist())
    y = df["label"].tolist()

    print(f"[INFO] Dataset size: {len(X)} samples, {len(set(y))} classes")
    print("[INFO] Classes:", sorted(set(y)))

    # Use stratified split only when every class has ≥2 samples
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print("[INFO] Training TF-IDF + Logistic Regression pipeline …")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.5,
            solver="lbfgs",
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[RESULTS] Test Accuracy: {acc:.2%}")
    print("[RESULTS] Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[INFO] Model saved → {MODEL_PATH}")
    return pipeline


# ── Load ──────────────────────────────────────────────────────────────────────

def load_model() -> Pipeline:
    """Load a pre-trained model from disk; train one if not found."""
    if os.path.exists(MODEL_PATH):
        print("[INFO] Loading existing classifier …")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        print("[INFO] No saved model found — training now …")
        return train_model()


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model: Pipeline, text: str) -> dict:
    """
    Run inference on a single clinical note.

    Args:
        model: Trained sklearn Pipeline
        text:  Raw clinical note string

    Returns:
        dict with predicted disease, confidence score, and full probability map.
    """
    from preprocess import clean_text

    cleaned      = clean_text(text)
    prediction   = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    classes      = model.classes_
    confidence   = float(max(probabilities))
    all_probs    = {
        cls: round(float(prob), 4)
        for cls, prob in zip(classes, probabilities)
    }

    return {
        "diagnosis":        prediction,
        "confidence":       round(confidence, 4),
        "all_probabilities": all_probs,
    }


# ── Standalone entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model()
