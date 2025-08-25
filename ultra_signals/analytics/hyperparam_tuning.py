"""Bayesian Hyperparameter Tuning (Sprint 14)
================================================
Provides a lightweight Bayesian Optimization wrapper (scikit-optimize if available,
otherwise random fallback) to tune model hyperparameters PER REGIME.

Flow:
 1. User prepares a labeled & regime-tagged dataset (same as Sprint 13).
 2. Define a search space for GradientBoostingClassifier parameters.
 3. Optimize accuracy (or F1 for directional classes) via cross-validated scoring.
 4. Persist best params snapshot per regime -> JSON file (parameter pinning).

Outputs:
  models/hpt_params_<regime>.json   (stores best hyperparameters)
  Return object includes full trial history for audit.

This is OFFLINE tooling only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    _HAS_SKOPT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SKOPT = False

REGIMES = ["trend", "mean_revert", "chop"]


@dataclass
class HPTTrial:
    params: Dict[str, float]
    score: float


@dataclass
class HPTResult:
    best_params: Dict[str, Dict[str, float]] = field(default_factory=dict)  # regime -> params
    trials: Dict[str, List[HPTTrial]] = field(default_factory=dict)


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['label'].astype(int).values
    return X, y


def default_feature_filter(columns: List[str]) -> List[str]:
    blacklist = {"label", "regime_profile", "mfe_pct", "mae_pct", "fwd_close_ret_pct"}
    return [c for c in columns if c not in blacklist]


class BayesianRegimeTuner:
    def __init__(self, output_dir: str = "models"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _persist_params(self, regime: str, params: Dict[str, float]):
        path = os.path.join(self.output_dir, f"hpt_params_{regime}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    def tune(
        self,
        dataset: pd.DataFrame,
        regime: str,
        feature_filter: Callable[[List[str]], List[str]] = default_feature_filter,
        n_calls: int = 20,
        cv_folds: int = 3,
        min_samples: int = 250,
        random_state: int = 42,
    ) -> HPTResult:
        if regime not in REGIMES:
            raise ValueError("Unknown regime")
        subset = dataset[dataset['regime_profile'] == regime]
        if len(subset) < min_samples:
            raise ValueError(f"Not enough samples for regime {regime} (need {min_samples}).")
        feature_cols = feature_filter(list(subset.columns))
        feature_cols = [c for c in feature_cols if subset[c].dtype.kind in 'bifc']
        X, y = _prepare_xy(subset, feature_cols)

        # Search space definitions
        space_defs = {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.2),
            'max_depth': (2, 6),
            'subsample': (0.6, 1.0),
        }

        trials: List[HPTTrial] = []

        def evaluate(params_list):
            # Map list to dict according ordering below
            p = {
                'n_estimators': int(params_list[0]),
                'learning_rate': float(params_list[1]),
                'max_depth': int(params_list[2]),
                'subsample': float(params_list[3]),
            }
            # cross-val
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores = []
            for train_idx, test_idx in cv.split(X, y):
                model = GradientBoostingClassifier(
                    n_estimators=p['n_estimators'],
                    learning_rate=p['learning_rate'],
                    max_depth=p['max_depth'],
                    subsample=p['subsample'],
                )
                model.fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[test_idx])
                scores.append(accuracy_score(y[test_idx], pred))
            score = 1 - float(np.mean(scores))  # minimize
            trials.append(HPTTrial(params=p, score=1 - score))  # store accuracy
            return score

        if _HAS_SKOPT:
            space = [
                Integer(space_defs['n_estimators'][0], space_defs['n_estimators'][1]),
                Real(space_defs['learning_rate'][0], space_defs['learning_rate'][1], prior='log-uniform'),
                Integer(space_defs['max_depth'][0], space_defs['max_depth'][1]),
                Real(space_defs['subsample'][0], space_defs['subsample'][1]),
            ]
            gp_minimize(
                evaluate,
                dimensions=space,
                n_calls=n_calls,
                random_state=random_state,
                verbose=False,
            )
        else:  # random fallback
            rng = np.random.default_rng(random_state)
            for _ in range(n_calls):
                params_list = [
                    rng.integers(space_defs['n_estimators'][0], space_defs['n_estimators'][1] + 1),
                    rng.uniform(*space_defs['learning_rate']),
                    rng.integers(space_defs['max_depth'][0], space_defs['max_depth'][1] + 1),
                    rng.uniform(*space_defs['subsample']),
                ]
                evaluate(params_list)

        # Best trial
        best = max(trials, key=lambda t: t.score)
        self._persist_params(regime, best.params)
        result = HPTResult(best_params={regime: best.params}, trials={regime: trials})
        return result

    def load_params(self, regime: str) -> Optional[Dict[str, float]]:
        path = os.path.join(self.output_dir, f"hpt_params_{regime}.json")
        if not os.path.isfile(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


__all__ = [
    'BayesianRegimeTuner',
    'HPTResult',
    'HPTTrial',
    'default_feature_filter'
]
