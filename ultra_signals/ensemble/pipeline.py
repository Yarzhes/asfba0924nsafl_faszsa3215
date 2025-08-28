"""Ensemble pipeline orchestration.

Provides a simple, testable pipeline class that accepts base learners (callable
predict_proba), a meta-learner, and a calibrator. This is intentionally small to
integrate with the existing codebase and tests; further heavy ML logic will be
implemented separately.
"""
from typing import List, Callable, Dict, Any, Optional
import numpy as np


class EnsemblePipeline:
    def __init__(self, base_models: Dict[str, Callable], meta_model: Optional[Callable] = None, calibrator: Optional[Callable] = None):
        """base_models: name -> predict_fn(X) -> prob array (n_samples,)
        meta_model: callable taking stacked OOF preds and returning final prob
        calibrator: callable(prob_array) -> calibrated probs
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.calibrator = calibrator

    def predict_proba(self, X: Any) -> np.ndarray:
        # collect base probs
        probs = []
        for name, fn in self.base_models.items():
            p = fn(X)
            probs.append(np.asarray(p))
        if not probs:
            raise RuntimeError("no base models registered")
        stacked = np.vstack(probs).T  # (n_samples, n_models)
        if self.meta_model is not None:
            out = self.meta_model(stacked)
        else:
            # simple average
            out = np.mean(stacked, axis=1)
        if self.calibrator is not None:
            out = self.calibrator(out)
        return out


def simple_average_predictors(preds: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(preds), axis=0)
