"""
train_model.py
--------------
Train and compare four ML models on the sentiment dataset, then
save the best-performing model and its TF-IDF vectoriser to disk.

Usage
-----
    python train_model.py

Outputs
-------
    model.pkl       – best-performing sklearn classifier
    vectorizer.pkl  – fitted TfidfVectorizer
    label_encoder.pkl – fitted LabelEncoder
"""

import os
import pickle
import warnings

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocess import preprocess_pipeline

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join('dataset', 'sentiment_dataset.csv')
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
ENCODER_PATH = 'label_encoder.pkl'
TFIDF_MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset, drop nulls and duplicates."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] After cleaning: {df.shape[0]} rows")

    return df


def detect_columns(df: pd.DataFrame):
    """
    Auto-detect text and label columns.
    Tries common naming conventions found in social-media datasets.
    """
    text_candidates = ['text', 'tweet', 'post', 'content', 'message', 'Text']
    label_candidates = ['label', 'sentiment', 'Sentiment', 'target', 'Label']

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect text/label columns. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Using text column: '{text_col}', label column: '{label_col}'")
    return text_col, label_col


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Apply the full cleaning pipeline to the text column."""
    print("[INFO] Preprocessing text …")
    df['clean_text'] = df[text_col].apply(preprocess_pipeline)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def vectorise(X_train, X_test):
    """Fit TF-IDF on training data and transform both splits."""
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return vectorizer, X_train_tfidf, X_test_tfidf


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """Return a dict of model name → unfitted estimator."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        'Support Vector Machine': SVC(
            kernel='linear', random_state=RANDOM_STATE, probability=True
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        'Multinomial Naïve Bayes': MultinomialNB(),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    models: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    label_names,
) -> pd.DataFrame:
    """
    Train each model, compute metrics, print reports, and return a
    comparison DataFrame.
    """
    results = []

    for name, model in models.items():
        print(f"\n{'=' * 55}")
        print(f"  Training: {name}")
        print('=' * 55)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

        print("  Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'estimator': model,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Saving artefacts
# ---------------------------------------------------------------------------

def save_artefacts(model, vectorizer, encoder):
    """Persist model, vectoriser, and encoder to disk."""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n[INFO] Best model saved → {MODEL_PATH}")

    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] TF-IDF vectoriser saved → {VECTORIZER_PATH}")

    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"[INFO] Label encoder saved → {ENCODER_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load data
    df = load_dataset(DATASET_PATH)
    text_col, label_col = detect_columns(df)

    # 2. Preprocess text
    df = preprocess_dataframe(df, text_col)

    # 3. Encode labels
    encoder = LabelEncoder()
    df['encoded_label'] = encoder.fit_transform(df[label_col])
    print(f"[INFO] Classes: {list(encoder.classes_)}")

    # 4. Prepare features and split
    X = df['clean_text']
    y = df['encoded_label']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 5. TF-IDF vectorisation
    vectorizer, X_train, X_test = vectorise(X_train_raw, X_test_raw)
    print(f"[INFO] TF-IDF matrix shapes → train: {X_train.shape}, test: {X_test.shape}")

    # 6. Train and evaluate all models
    models = get_models()
    results_df = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        label_names=list(encoder.classes_),
    )

    # 7. Print comparison table
    comparison = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
    print(f"\n{'=' * 65}")
    print("  MODEL COMPARISON")
    print('=' * 65)
    print(comparison.to_string(index=False))

    # 8. Identify best model
    best_row = results_df.loc[results_df['Accuracy'].idxmax()]
    best_model = best_row['estimator']
    print(f"\n[INFO] Best model: {best_row['Model']} "
          f"(Accuracy = {best_row['Accuracy']:.4f})")

    # 9. Save artefacts
    save_artefacts(best_model, vectorizer, encoder)

    print("\n[INFO] Training complete. You can now run:  python app.py")


if __name__ == '__main__':
    main()
