"""
prediction.py
-------------
Load saved artefacts and expose a predict() function used by the
Flask application.
"""

import os
import pickle
import numpy as np
from preprocess import preprocess_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# ---------------------------------------------------------------------------
# Lazy-load artefacts (loaded once, reused across requests)
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_encoder = None


def _load_artefacts():
    """Load model, vectoriser, and encoder from disk if not yet loaded."""
    global _model, _vectorizer, _encoder

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Please run  python train_model.py  first."
            )
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)

    if _vectorizer is None:
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Vectoriser file not found at '{VECTORIZER_PATH}'. "
                "Please run  python train_model.py  first."
            )
        with open(VECTORIZER_PATH, 'rb') as f:
            _vectorizer = pickle.load(f)

    if _encoder is None:
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                f"Encoder file not found at '{ENCODER_PATH}'. "
                "Please run  python train_model.py  first."
            )
        with open(ENCODER_PATH, 'rb') as f:
            _encoder = pickle.load(f)


def predict(text: str) -> dict:
    """
    Preprocess, vectorise, and predict the sentiment of a text string.

    Parameters
    ----------
    text : str
        Raw user input (social-media post or any text).

    Returns
    -------
    dict with keys:
        sentiment   : str   – human-readable label (e.g. 'Positive')
        confidence  : float – probability of the predicted class (0–1)
        label_index : int   – encoded integer label
        all_probs   : dict  – {label: probability} for every class
    """
    _load_artefacts()

    if not text or not text.strip():
        return {
            'sentiment': 'Unknown',
            'confidence': 0.0,
            'label_index': -1,
            'all_probs': {},
        }

    # Preprocessing
    clean = preprocess_pipeline(text)

    # TF-IDF vectorisation
    X = _vectorizer.transform([clean])

    # Prediction
    label_index = int(_model.predict(X)[0])
    sentiment = _encoder.inverse_transform([label_index])[0]

    # Confidence / probabilities
    all_probs = {}
    confidence = 1.0  # default when predict_proba is unavailable

    if hasattr(_model, 'predict_proba'):
        proba = _model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        all_probs = {
            str(_encoder.inverse_transform([i])[0]): round(float(p), 4)
            for i, p in enumerate(proba)
        }
    elif hasattr(_model, 'decision_function'):
        # SVM without probability=True – use softmax of decision scores
        scores = _model.decision_function(X)[0]
        if scores.ndim == 0:
            scores = np.array([scores])
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()
        confidence = float(np.max(proba))
        all_probs = {
            str(_encoder.inverse_transform([i])[0]): round(float(p), 4)
            for i, p in enumerate(proba)
        }

    return {
        'sentiment': str(sentiment).capitalize(),
        'confidence': round(confidence * 100, 1),
        'label_index': label_index,
        'all_probs': all_probs,
    }
