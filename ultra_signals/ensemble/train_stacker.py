"""Train a stacker with OOF predictions, optional hyperparameter hook, and persistence.

API contract (simple):
- base_factories: dict[name -> factory], where factory(X_train, y_train) -> predictor_fn
  and predictor_fn(X) -> np.ndarray probabilities for positive class.
- hyperparam_search (optional): callable(oof_matrix, y) -> fitted_meta_model
- meta_factory (optional): callable() -> unfitted meta estimator with fit(X,y)

The function saves a joblib bundle containing: 'meta', 'calibrator', 'features',
'oof_preds' (DataFrame), and 'config'.
"""

from typing import Dict, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

from .oof import purged_kfold_indexes
from .calibration import platt_scaler


def train_stacker(
    X: pd.DataFrame,
    y: pd.Series,
    base_factories: Dict[str, Callable[[pd.DataFrame, pd.Series], Callable[[pd.DataFrame], np.ndarray]]],
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
    meta_factory: Optional[Callable[[], object]] = None,
    hyperparam_search: Optional[Callable[[np.ndarray, np.ndarray], object]] = None,
    out_path: str = 'stacker_bundle.joblib',
    save_oof_path: Optional[str] = None,
):
    n = len(X)
    # storage for OOF preds
    oof_preds = {name: np.full(n, np.nan, dtype=float) for name in base_factories}

    # run purged kfold to produce OOF predictions
    for train_idx, test_idx in purged_kfold_indexes(n, n_splits=n_splits, purge=purge, embargo=embargo):
        # guard against degenerate folds created by purge/embargo (no train samples)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        for name, factory in base_factories.items():
            # if the train split is empty or has only a single class, fall back to training on full data
            try:
                needs_fallback = (len(X_train) == 0) or (len(np.unique(y_train)) < 2)
            except Exception:
                needs_fallback = len(X_train) == 0
            if needs_fallback:
                predictor = factory(X, y)
            else:
                predictor = factory(X_train, y_train)
            preds = predictor(X_test)
            # ensure shape matches
            if len(preds) != len(X_test):
                raise ValueError(f'predictor for {name} returned {len(preds)} preds but expected {len(X_test)}')
            oof_preds[name][test_idx] = preds

    # convert to stacked matrix
    stacked = np.vstack([v for v in oof_preds.values()]).T

    # if a hyperparam_search hook is provided it returns a fitted meta model
    if hyperparam_search is not None:
        meta = hyperparam_search(stacked, y.values)
    else:
        meta = meta_factory() if meta_factory is not None else LogisticRegression(solver='lbfgs')
        meta.fit(stacked, y.values)

    # calibrate with Platt scaling
    probs = meta.predict_proba(stacked)[:, 1]
    calibrator = platt_scaler(probs, y.values)

    # persist artifacts
    bundle = {
        'meta': meta,
        'calibrator': calibrator,
        'features': list(X.columns),
        'oof_preds': pd.DataFrame(oof_preds, index=X.index),
        'config': {'n_splits': n_splits, 'purge': purge, 'embargo': embargo},
    }
    joblib.dump(bundle, out_path)
    if save_oof_path:
        joblib.dump(bundle['oof_preds'], save_oof_path)
    return meta, calibrator, bundle
