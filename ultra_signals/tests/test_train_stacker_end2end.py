import os
import tempfile
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from ultra_signals.ensemble.train_stacker import train_stacker


def _factory_feature(col: str):
    def factory(X_train: pd.DataFrame, y_train: pd.Series):
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train[[col]], y_train)

        def predictor(X: pd.DataFrame):
            return model.predict_proba(X[[col]])[:, 1]

        return predictor

    return factory


def test_train_stacker_end2end(tmp_path):
    # synthetic data
    rng = np.random.RandomState(0)
    n = 200
    X = pd.DataFrame({'f1': rng.normal(size=n), 'f2': rng.normal(size=n)})
    y = (X['f1'] + 0.1 * rng.normal(size=n) > 0).astype(int)

    factories = {'f1_logreg': _factory_feature('f1'), 'f2_logreg': _factory_feature('f2')}

    out_path = tmp_path / 'bundle.joblib'
    meta, calibrator, bundle = train_stacker(X, y, factories, n_splits=4, purge=2, embargo=1, out_path=str(out_path))

    # artifact checks
    assert 'meta' in bundle and 'calibrator' in bundle and 'oof_preds' in bundle
    oof = bundle['oof_preds']
    assert isinstance(oof, (pd.DataFrame,))
    assert oof.shape[0] == n
    # meta must have predict_proba
    assert hasattr(meta, 'predict_proba')
    # calibrator must transform
    assert hasattr(calibrator, 'predict') or hasattr(calibrator, 'transform')