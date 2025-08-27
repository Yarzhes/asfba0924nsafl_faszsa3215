"""Sprint 31 Meta Model Trainer

Loads dataset (features.npy, labels.npy) and trains a probabilistic classifier
with time-series cross validation (simple chronological folds) + calibration.

We approximate purge/embargo by trimming overlap rows between folds. Since our
dataset rows correspond to trade entry times (not uniform bars) we sort by an
optional timestamp vector if provided in meta.json (future). For now we assume
row order is chronological.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import joblib
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore


def _load_dataset(data_dir: str):
    X = np.load(Path(data_dir)/'features.npy')
    y = np.load(Path(data_dir)/'labels.npy') if (Path(data_dir)/'labels.npy').exists() else np.zeros((len(X),),dtype=np.uint8)
    fn_path = Path(data_dir)/'feature_names.json'
    feature_names = json.loads(fn_path.read_text()) if fn_path.exists() else [f'f{i}' for i in range(X.shape[1])]
    # Optional timestamp arrays (per-symbol). Concatenate if multiple found.
    ts_arrays = []
    for p in Path(data_dir).glob('timestamps_*.npy'):
        try:
            ts_arrays.append(np.load(p))
        except Exception:
            pass
    timestamps = None
    if ts_arrays:
        try:
            timestamps = np.concatenate(ts_arrays)
            if len(timestamps) != len(X):
                timestamps = None  # size mismatch -> ignore
        except Exception:
            timestamps = None
    return X, y, feature_names, timestamps


def _build_model(kind: str) -> Any:
    if kind == 'logreg':
        return LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=None)
    if kind == 'xgboost':
        if XGBClassifier is None:
            raise RuntimeError('xgboost not installed')
        return XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=1,
        )
    raise ValueError(f'Unknown model kind {kind}')


def _chronological_folds(n: int, k: int) -> List[np.ndarray]:
    idx = np.arange(n)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
    folds = []
    start = 0
    for sz in fold_sizes:
        folds.append(idx[start:start+sz])
        start += sz
    return folds

def _purged_embargo_folds(timestamps: np.ndarray, k: int, embargo_frac: float = 0.01) -> List[np.ndarray]:
    """Create k sequential folds with purge+embargo.

    - timestamps: int64 epoch ns (or ms/s scaled) monotonic
    - embargo_frac: fraction of dataset length to embargo after each validation segment.
    """
    n = len(timestamps)
    base_folds = _chronological_folds(n, k)
    embargo = int(max(1, n * embargo_frac)) if embargo_frac > 0 else 0
    purged = []
    last_val_end = -1
    for fold in base_folds:
        # Purge overlap: ensure validation indices strictly after last_val_end
        val = fold
        # embargo previous validation region
        if last_val_end >= 0:
            start_allowed = last_val_end + embargo + 1
            val = val[val >= start_allowed]
        if len(val) == 0:  # skip empty -> fallback to original fold
            val = fold
        purged.append(val)
        if len(fold) > 0:
            last_val_end = max(fold)
    return purged


def train(data_dir: str, model_kind: str, out_path: str, report_dir: str, calibration: str = 'isotonic', n_splits: int = 5):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    X, y, feature_names, timestamps = _load_dataset(data_dir)
    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    if X.size == 0 or len(y) == 0:
        status = 'empty'
    elif len(np.unique(y)) < 2 or pos_count < 2:
        status = 'single_class'
    elif pos_count < 3 or pos_count < 2 * 1:  # extremely imbalanced tiny dataset
        status = 'too_small_for_cv'
    else:
        status = 'ok'
    if status != 'ok':
        logger.warning(f'Dataset not suitable for full training (status={status}, pos={pos_count}, neg={neg_count}); writing dummy bundle.')
        meta = {'status': status, 'rows': int(len(y)), 'pos_rate': float(y.mean()) if len(y)>0 else None}
        joblib.dump({'model': None, 'feature_names': feature_names, 'calibrator': None, 'pre': None, 'meta': meta}, out_path)
        (Path(report_dir)/'cv_metrics.json').write_text(json.dumps(meta, indent=2))
        return
    # Optional time sort (if timestamps available and unsorted)
    order = np.arange(len(X))
    if timestamps is not None:
        try:
            if not np.all(np.diff(timestamps) >= 0):
                order = np.argsort(timestamps)
                X = X[order]; y = y[order]
                timestamps = timestamps[order]
        except Exception:
            pass
    # Build folds with purge+embargo if timestamps present
    if timestamps is not None:
        folds = _purged_embargo_folds(timestamps, n_splits, embargo_frac=0.01)
    else:
        folds = _chronological_folds(len(X), n_splits)
    preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    metrics: List[Dict[str, Any]] = []
    oof_pred = np.zeros(len(X))
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2:
            logger.warning(f"Skipping fold {i}: training partition single-class (size={len(y_tr)})")
            metrics.append({'fold': i, 'skipped': True})
            continue
        preproc.fit(X_tr)
        X_tr_p = preproc.transform(X_tr)
        X_te_p = preproc.transform(X_te)
        clf = _build_model(model_kind)
        clf.fit(X_tr_p, y_tr)
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(X_te_p)[:,1]
        else:
            prob = clf.decision_function(X_te_p)
        oof_pred[test_idx] = prob
        # Metrics (guard for single-class y_te)
        try:
            auc_pr = float(average_precision_score(y_te, prob))
        except Exception:
            auc_pr = None
        try:
            auc_roc = float(roc_auc_score(y_te, prob))
        except Exception:
            auc_roc = None
        brier = float(brier_score_loss(y_te, prob)) if len(np.unique(y_te))==2 else None
        fold_metrics = {'fold': i, 'auc_pr': auc_pr, 'auc_roc': auc_roc, 'brier': brier, 'pos_rate': float(y_te.mean())}
        metrics.append(fold_metrics)
        logger.info(f"Fold {i}: AUC-PR={auc_pr} AUC-ROC={auc_roc} Brier={brier}")

    # Calibration fit on full data using chosen method
    if all(m.get('skipped') for m in metrics):
        logger.warning('All CV folds skipped due to single-class partitions; writing dummy bundle.')
        meta = {'status': 'cv_all_skipped', 'rows': int(len(y)), 'pos_rate': float(y.mean())}
        joblib.dump({'model': None, 'feature_names': feature_names, 'calibrator': None, 'pre': None, 'meta': meta}, out_path)
        (Path(report_dir)/'cv_metrics.json').write_text(json.dumps(meta, indent=2))
        return
    # Fit final model on all data
    preproc.fit(X)
    X_full = preproc.transform(X)
    clf_final = _build_model(model_kind)
    clf_final.fit(X_full, y)
    if calibration in ('isotonic','platt') and len(np.unique(y))==2:
        method = 'isotonic' if calibration=='isotonic' else 'sigmoid'
        calibrator = CalibratedClassifierCV(clf_final, method=method, cv='prefit')
        calibrator.fit(X_full, y)
    else:
        calibrator = None

    # Reliability / lift chart
    try:
        from ultra_signals.backtest.metrics import compute_reliability_bins
        rel = compute_reliability_bins(oof_pred, y, n_bins=10)
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        bins = rel['bins']
        ax[0].plot(bins['mean_predicted'], bins['fraction_positives'], marker='o')
        ax[0].plot([0,1],[0,1], '--', color='gray')
        ax[0].set_title('Calibration')
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('Empirical')
        # Lift (deciles)
        deciles = pd.qcut(oof_pred, 10, labels=False, duplicates='drop')
        lift = []
        for d in range(deciles.max()+1):
            mask = deciles==d
            if mask.sum()>0:
                lift.append({'decile': int(d), 'mean_p': float(oof_pred[mask].mean()), 'win_rate': float(y[mask].mean())})
        xs = [r['mean_p'] for r in lift]; ys = [r['win_rate'] for r in lift]
        ax[1].bar(range(len(lift)), ys)
        ax[1].set_title('Win-rate by predicted decile')
        ax[1].set_xlabel('Decile (low->high)')
        ax[1].set_ylabel('Win rate')
        fig.tight_layout()
        fig.savefig(Path(report_dir)/'calibration.png')
        fig.savefig(Path(report_dir)/'lift_by_decile.png')
    except Exception as e:  # pragma: no cover
        logger.warning(f'Plot generation failed: {e}')

    # Persist bundle
    # Aggregate overall metrics safely
    try:
        oof_auc_pr = float(average_precision_score(y, oof_pred))
    except Exception:
        oof_auc_pr = None
    try:
        oof_auc_roc = float(roc_auc_score(y, oof_pred))
    except Exception:
        oof_auc_roc = None
    try:
        oof_brier = float(brier_score_loss(y, oof_pred)) if len(np.unique(y))==2 else None
    except Exception:
        oof_brier = None
    bundle = {
        'model': clf_final,
        'pre': preproc,
        'calibrator': calibrator,
        'feature_names': feature_names,
        'meta': {
            'model_kind': model_kind,
            'calibration': calibration,
            'cv_metrics': metrics,
            'oof_auc_pr': oof_auc_pr,
            'oof_auc_roc': oof_auc_roc,
            'oof_brier': oof_brier,
            'rows': int(len(y)),
            'pos_rate': float(y.mean()),
            'purge_embargo': bool(timestamps is not None),
        }
    }
    joblib.dump(bundle, out_path)
    (Path(report_dir)/'cv_metrics.json').write_text(json.dumps(bundle['meta'], indent=2))
    logger.success(f"Model saved to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True, choices=['logreg','xgboost'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--calibration', default='isotonic', choices=['isotonic','platt','none'])
    ap.add_argument('--cv', type=int, default=5)
    args = ap.parse_args()
    train(args.data, args.model, args.out, args.report, calibration=args.calibration, n_splits=args.cv)
