"""Thin wrappers for tabular base learners exposing predict_proba(X).

Wrappers use lazy imports so the package can be imported without heavy deps.
Each wrapper accepts a fitted model object and a feature list; predict_proba
will align input DataFrame columns and handle missing features gracefully.
"""
from typing import Any, List
import numpy as np
import pandas as pd


class BaseWrapper:
    def __init__(self, model: Any, features: List[str]):
        self.model = model
        self.features = list(features)

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        # ensure columns exist; missing columns filled with zeros
        cols = [c for c in self.features]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = 0.0
        return X[cols].to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class SklearnLogisticWrapper(BaseWrapper):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        arr = self._align(X)
        probs = self.model.predict_proba(arr)
        # assume binary -> return probability of positive class
        return probs[:, 1]


class XGBoostWrapper(BaseWrapper):
    """Wrapper for XGBoost models.

    Accepts either the sklearn-style XGBClassifier/XGBRegressor or a low-level
    xgboost.Booster. Returns a 1-D numpy array of probabilities for the
    positive class when available; falls back to raw model prediction.
    """
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            import xgboost as xgb  # lazy
        except Exception as e:
            raise RuntimeError('xgboost is required for XGBoostWrapper') from e
        arr = self._align(X)
        # sklearn API
        if hasattr(self.model, 'predict_proba'):
            out = self.model.predict_proba(arr)
            return np.asarray(out)[:, 1]
        # raw Booster
        if isinstance(self.model, xgb.Booster):
            dmat = xgb.DMatrix(arr, feature_names=self.features)
            out = self.model.predict(dmat)
            return np.asarray(out).reshape(-1)
        # fallback to predict
        out = self.model.predict(arr)
        return np.asarray(out).reshape(-1)


class LightGBMWrapper(BaseWrapper):
    """Wrapper for LightGBM models (sklearn API or lgb.Booster)."""
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            import lightgbm as lgb
        except Exception as e:
            raise RuntimeError('lightgbm is required for LightGBMWrapper') from e
        arr = self._align(X)
        # sklearn API
        if hasattr(self.model, 'predict_proba'):
            return np.asarray(self.model.predict_proba(arr))[:, 1]
        # raw Booster
        if isinstance(self.model, lgb.Booster):
            out = self.model.predict(arr)
            return np.asarray(out).reshape(-1)
        # fallback
        out = self.model.predict(arr)
        return np.asarray(out).reshape(-1)
