import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from ultra_signals.ensemble.tabular_wrappers import SklearnLogisticWrapper


def test_sklearn_wrapper_happy_path():
    X = pd.DataFrame({'f1': [0.1, 0.2, 0.3], 'f2': [1.0, 0.0, -1.0]})
    y = np.array([0, 1, 0])
    lr = LogisticRegression().fit(X.values, y)
    w = SklearnLogisticWrapper(lr, features=['f1', 'f2'])
    probs = w.predict_proba(X)
    assert probs.shape == (3,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_sklearn_wrapper_missing_feature():
    X = pd.DataFrame({'f1': [0.1, 0.2, 0.3]})
    y = np.array([1, 0, 1])
    # train model expecting f1 and f2
    X_train = pd.DataFrame({'f1': [0.1, 0.2, 0.3], 'f2': [0.0, 0.5, -0.5]})
    lr = LogisticRegression().fit(X_train.values, y)
    w = SklearnLogisticWrapper(lr, features=['f1', 'f2'])
    probs = w.predict_proba(X)
    assert probs.shape == (3,)
    assert np.all((probs >= 0) & (probs <= 1))
