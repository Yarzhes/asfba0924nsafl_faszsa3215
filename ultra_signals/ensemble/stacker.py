"""Stacker utilities: out-of-fold stacking, calibration helpers, and simple metrics."""
from typing import List, Dict, Callable, Any, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression


class Stacker:
    def __init__(self):
        self.meta_model = None
        self.calibrator = None

    def fit_meta(self, oof_preds: np.ndarray, y: np.ndarray, meta_fn: Callable):
        """Fit meta model on OOF predictions.

        oof_preds: (n_samples, n_models)
        y: (n_samples,)
        meta_fn: callable that returns fitted model when called with (X, y)
        """
        self.meta_model = meta_fn(oof_preds, y)
        return self.meta_model

    def fit_calibrator(self, probs: np.ndarray, y: np.ndarray, method: str = 'isotonic'):
        if method == 'isotonic':
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs, y)
            self.calibrator = iso
            return iso
        raise ValueError('unsupported calibrator')

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return probs
        return self.calibrator.transform(probs)
