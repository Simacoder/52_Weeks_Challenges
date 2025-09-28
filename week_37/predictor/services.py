# predictor/services.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np

_MODEL_CACHE: Dict[str, Any] = {}

def get_model_bundle():
    """
    Loads and caches the trained model bundle:
    {
      "estimator": RandomForestClassifier,
      "target_names": ndarray[str],
      "feature_names": list[str],
    }
    """
    global _MODEL_CACHE
    if "bundle" not in _MODEL_CACHE:
        model_path = Path(__file__).resolve().parent / "model" / "iris_rf.joblib"
        _MODEL_CACHE["bundle"] = joblib.load(model_path)
    return _MODEL_CACHE["bundle"]

def predict_iris(features):
    """
    features: list[float] of length 4 (sepal_length, sepal_width, petal_length, petal_width)
    Returns dict with class_name and probabilities.
    """
    bundle = get_model_bundle()
    clf = bundle["estimator"]
    target_names = bundle["target_names"]

    X = np.array([features], dtype=float)
    proba = clf.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return {
        "class_index": idx,
        "class_name": str(target_names[idx]),
        "probabilities": {str(name): float(p) for name, p in zip(target_names, proba)},
    }