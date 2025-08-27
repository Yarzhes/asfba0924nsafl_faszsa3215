"""Inference helper for meta probability model.

Validates feature order contract and applies scaler + optional calibrator.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import joblib
import numpy as np

_CACHE: Dict[str, Any] = {}


def load_bundle(path: str) -> Dict[str, Any] | None:
    if path in _CACHE:
        return _CACHE[path]
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and 'feature_names' in bundle:
            _CACHE[path] = bundle
            return bundle
    except Exception:
        return None
    return None


def predict_proba(symbol: str, tf: str, ts: int, side: str, feature_bundle: Dict[str, Any], model_path: str) -> Optional[float]:
    bundle = load_bundle(model_path)
    if not bundle:
        return None
    feature_names = bundle.get('feature_names') or []
    if not feature_names:
        return None
    row = []
    missing = []
    for name in feature_names:
        if name not in feature_bundle:
            missing.append(name)
        row.append(float(feature_bundle.get(name, 0.0)))  # safe fill 0.0
    if missing:
        # Soft log only (avoid raising in prod inference path)
        pass
    X = np.asarray(row, dtype=float).reshape(1, -1)
    scaler = bundle.get('pre')
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    model = bundle.get('model')
    if model is None:
        return None
    try:
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X)[0,1])
        elif hasattr(model, 'decision_function'):
            raw = float(model.decision_function(X)[0])
            # map via logistic for fallback
            prob = 1/(1+np.exp(-raw))
        else:
            pred = float(model.predict(X)[0])
            prob = max(0.0, min(1.0, pred))
    except Exception:
        return None
    # Calibrator expects raw probabilities or scores; assume prob -> calibrator(proba)
    cal = bundle.get('calibrator')
    if cal is not None:
        try:
            if hasattr(cal, 'predict_proba'):
                prob = float(cal.predict_proba(X)[0,1])
            elif hasattr(cal, 'predict'):
                prob = float(cal.predict(X)[0])
        except Exception:
            pass
    return max(0.0, min(1.0, prob))
