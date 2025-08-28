import numpy as np
import pandas as pd
import pytest

from ultra_signals.ensemble.tabular_wrappers import XGBoostWrapper, LightGBMWrapper, SklearnLogisticWrapper


def test_xgboost_wrapper_predict(tmp_path):
    xgboost = pytest.importorskip('xgboost')
    # build a tiny dataset
    X = pd.DataFrame({'a': [0.1, 0.2, 0.3], 'b': [1, 2, 3]})
    y = np.array([0, 1, 1])
    try:
        from xgboost import XGBClassifier
    except Exception:
        pytest.skip('xgboost sklearn API not available')
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X[['a']], y)
    wrapper = XGBoostWrapper(model, ['a'])
    preds = wrapper.predict_proba(X)
    assert preds.shape[0] == len(X)


def test_lightgbm_wrapper_predict(tmp_path):
    lgb = pytest.importorskip('lightgbm')
    X = pd.DataFrame({'a': [0.1, 0.2, 0.3], 'b': [1, 2, 3]})
    y = np.array([0, 1, 1])
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        pytest.skip('lightgbm sklearn API not available')
    model = LGBMClassifier()
    model.fit(X[['a']], y)
    wrapper = LightGBMWrapper(model, ['a'])
    preds = wrapper.predict_proba(X)
    assert preds.shape[0] == len(X)
