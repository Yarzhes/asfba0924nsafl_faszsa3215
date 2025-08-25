"""Model Registry & Per-Regime Scoring (Sprint 13)
===================================================
Goal
----
Provide a SIMPLE, file-based model registry and helpers to:
  1. Train per-regime ML scorers (e.g. Gradient Boosting) using engineered labels.
  2. Save & load models keyed by regime profile name.
  3. Produce offline evaluation metrics per regime (accuracy, precision, recall, f1, confusion matrix).

IMPORTANT: These models are *scorers only*; they DO NOT place trades directly.
They output a probability of LONG vs SHORT vs NO-TRADE (multiclass). The live
engine can (in a later sprint) incorporate these via the ensemble by mapping
model probabilities into a synthetic `SubSignal`.

Design Choices
--------------
* Simplicity over complexity: basic scikit-learn GradientBoostingClassifier.
* One model per regime profile ("trend", "mean_revert", "chop").
* Registry path default: `models/` under project root.
* Feature selection: user passes a DataFrame of features already aligned with labels.
  - We automatically drop leakage columns like label itself from X.
* Multiclass target: values in {-1,0,1}. We remap to {0,1,2} internally for sklearn.

Usage (Offline Notebook / Script)
---------------------------------
    from ultra_signals.analytics.labeling import compute_vol_scaled_labels
    from ultra_signals.analytics.model_registry import (
        PerRegimeTrainer, ModelRegistry, default_feature_filter
    )

    # 1) Build features_df (your pipeline) and labels_df
    data = features_df.join(labels_df)  # index aligned
    data = data.dropna(subset=['label'])
    # 2) Instantiate registry & trainer
    reg = ModelRegistry(base_path="models")
    trainer = PerRegimeTrainer(registry=reg)
    # 3) Train
    result = trainer.train_all(data, feature_filter=default_feature_filter)
    # 4) Inspect metrics
    print(result.metrics_per_regime)
    # 5) Later load model for scoring
    model = reg.load('trend')

Leakage Defenses
----------------
* User supplies already time-aligned features; we require NO future columns.
* We never peek forward; labeling module already ensures safe targets.
* Train/test split is random stratified by label (for demonstration). For more
  robust evaluation use walk-forward or time-based split (future sprint).

Extensibility
-------------
Future sprints can add:
* Probabilistic calibration per regime.
* SHAP value export.
* Time-based cross-validation.
* Integration adapter to plug these probabilities into ensemble.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

REGIMES = ["trend", "mean_revert", "chop"]


# ----------------------------- Registry ---------------------------------

class ModelRegistry:
    """File-system backed registry (very simple)."""
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _path(self, regime: str) -> str:
        return os.path.join(self.base_path, f"model_{regime}.joblib")

    def save(self, regime: str, model) -> str:
        path = self._path(regime)
        joblib.dump(model, path)
        return path

    def load(self, regime: str):
        path = self._path(regime)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No model stored for regime '{regime}' at {path}")
        return joblib.load(path)

    def list_models(self) -> Dict[str, bool]:
        return {r: os.path.isfile(self._path(r)) for r in REGIMES}


# ----------------------------- Utilities --------------------------------

def default_feature_filter(columns: List[str]) -> List[str]:
    """Return a filtered feature column list (drop target & obviously bad columns)."""
    blacklist = {"label", "mfe_pct", "mae_pct", "fwd_close_ret_pct", "regime_profile"}
    return [c for c in columns if c not in blacklist]


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_raw = df["label"].astype(int).values  # -1,0,1
    # map -1->0, 0->1, 1->2 for multiclass stable ordering
    mapping = {-1: 0, 0: 1, 1: 2}
    y = np.array([mapping[v] for v in y_raw])
    return X, y


def _inverse_label_map(y_pred_enc: np.ndarray) -> np.ndarray:
    inv_map = {0: -1, 1: 0, 2: 1}
    return np.array([inv_map[int(v)] for v in y_pred_enc])


@dataclass
class RegimeModelMetrics:
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    confusion: List[List[int]]


@dataclass
class TrainResult:
    metrics_per_regime: Dict[str, RegimeModelMetrics] = field(default_factory=dict)
    feature_cols_used: Dict[str, List[str]] = field(default_factory=dict)
    models: Dict[str, object] = field(default_factory=dict)


class PerRegimeTrainer:
    """Handles training of one model per regime profile."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def train_all(
        self,
        dataset: pd.DataFrame,
        feature_filter: Callable[[List[str]], List[str]] = default_feature_filter,
        test_size: float = 0.25,
        random_state: int = 42,
        min_samples: int = 200,
        regime_column: str = "regime_profile",
        gb_params: Optional[Dict] = None,
    ) -> TrainResult:
        """Train a model for each regime.

        dataset MUST contain: 'label' and `regime_column` giving regime profile per row.
        """
        if "label" not in dataset.columns:
            raise ValueError("Dataset missing 'label' column.")
        if regime_column not in dataset.columns:
            raise ValueError(f"Dataset missing regime column '{regime_column}'.")

        gb_params = gb_params or {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}

        result = TrainResult()

        for regime in REGIMES:
            subset = dataset[dataset[regime_column] == regime]
            if len(subset) < min_samples:
                # Skip if insufficient data; record placeholder metrics
                result.metrics_per_regime[regime] = RegimeModelMetrics(
                    accuracy=float("nan"), precision={}, recall={}, f1={}, confusion=[]
                )
                continue
            feature_cols = feature_filter(list(subset.columns))
            # Remove regime column if still present & enforce numeric-only
            if regime_column in feature_cols:
                feature_cols.remove(regime_column)
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(subset[c])]
            X, y = _prepare_xy(subset, feature_cols)
            # stratified split on encoded y
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError:
                # fallback: no stratification if a single class present
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

            model = GradientBoostingClassifier(**gb_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            # metrics per encoded class
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=[0, 1, 2], zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

            def pack_metrics(vals):
                return {"-1": vals[0], "0": vals[1], "+1": vals[2]}

            metrics = RegimeModelMetrics(
                accuracy=acc,
                precision=pack_metrics(prec),
                recall=pack_metrics(rec),
                f1=pack_metrics(f1),
                confusion=cm,
            )
            result.metrics_per_regime[regime] = metrics
            result.feature_cols_used[regime] = feature_cols
            result.models[regime] = model
            self.registry.save(regime, model)

        # Persist a registry manifest for reference
        manifest = {
            "regimes": REGIMES,
            "available": self.registry.list_models(),
            "metrics": {
                r: (
                    None
                    if not result.metrics_per_regime.get(r)
                    else {
                        "accuracy": result.metrics_per_regime[r].accuracy,
                        "precision": result.metrics_per_regime[r].precision,
                        "recall": result.metrics_per_regime[r].recall,
                        "f1": result.metrics_per_regime[r].f1,
                        "confusion": result.metrics_per_regime[r].confusion,
                    }
                )
                for r in REGIMES
            },
            "feature_cols_used": result.feature_cols_used,
        }
        with open(os.path.join(self.registry.base_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return result


def load_model_scores(registry: ModelRegistry, regime: str, feature_row: pd.Series) -> Dict[str, float]:
    """Return predicted class probabilities mapped back to {-1,0,1} keys.

    feature_row: a single row of features (same preprocessing as training) WITHOUT label.
    """
    model = registry.load(regime)
    arr = feature_row.values.astype(float).reshape(1, -1)
    probs = model.predict_proba(arr)[0]
    # encoded order [0->-1, 1->0, 2->+1]
    return {"-1": float(probs[0]), "0": float(probs[1]), "+1": float(probs[2])}


__all__ = [
    "ModelRegistry",
    "PerRegimeTrainer",
    "TrainResult",
    "RegimeModelMetrics",
    "default_feature_filter",
    "load_model_scores",
]
