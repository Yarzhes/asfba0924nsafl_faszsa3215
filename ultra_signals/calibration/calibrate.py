import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Calibrator = Union[IsotonicRegression, LogisticRegression]
RegimeCalibrator = Dict[str, Calibrator]

def fit_calibration_model(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    method: str = "isotonic",
    regimes: np.ndarray = None
) -> Union[Calibrator, RegimeCalibrator]:
    """
    Fits a calibration model to the data.
    
    Args:
        predictions: Raw model predictions (e.g., scores between 0 and 1).
        outcomes: True binary outcomes (0 or 1).
        method: 'isotonic' or 'platt'.
        regimes: Optional array of regime labels for per-regime calibration.

    Returns:
        A fitted calibration model or a dictionary of per-regime models.
    """
    if regimes is not None:
        return _fit_per_regime(predictions, outcomes, regimes, method)
    else:
        return _fit_single_model(predictions, outcomes, method)

def _fit_single_model(X, y, method):
    """Fits a single calibration model."""
    if method == 'isotonic':
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    elif method == 'platt':
        model = LogisticRegression()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    # Reshape for sklearn
    X_reshaped = np.asarray(X).reshape(-1, 1)
    y_reshaped = np.asarray(y).ravel()
    return model.fit(X_reshaped, y_reshaped)

def _fit_per_regime(X, y, regimes, method):
    """Fits one calibration model for each unique regime."""
    unique_regimes = np.unique(regimes)
    models = {}
    for regime in unique_regimes:
        mask = regimes == regime
        if np.sum(mask) > 1: # Need enough data to fit
            models[regime] = _fit_single_model(X[mask], y[mask], method)
    return models

def apply_calibration(
    model: Union[Calibrator, RegimeCalibrator],
    predictions: np.ndarray,
    regimes: np.ndarray = None
) -> np.ndarray:
    """Applies a fitted calibration model."""
    if isinstance(model, dict): # Per-regime model
        return _apply_per_regime(model, predictions, regimes)
    else: # Single model
        return _apply_single_model(model, predictions)

def _apply_single_model(model, X):
    X_reshaped = np.asarray(X).reshape(-1, 1)
    if isinstance(model, IsotonicRegression):
        return model.predict(X_reshaped)
    else: # LogisticRegression
        return model.predict_proba(X_reshaped)[:, 1]

def _apply_per_regime(models: RegimeCalibrator, X: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    """Applies the correct model for each sample based on its regime."""
    calibrated_preds = np.zeros_like(X, dtype=float)
    for regime, model in models.items():
        mask = regimes == regime
        if np.sum(mask) > 0:
            calibrated_preds[mask] = _apply_single_model(model, X[mask])
    return calibrated_preds

def save_model(model: Any, path: Path):
    """Saves a model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Path) -> Any:
    """Loads a model from disk."""
    return joblib.load(path)

def plot_reliability_diagram(ax, y_true, y_prob_raw, y_prob_cal, n_bins=10):
    """Plot reliability diagram."""
    from sklearn.calibration import calibration_curve

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Raw
    fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(y_true, y_prob_raw, n_bins=n_bins, strategy='uniform')
    ax.plot(mean_predicted_value_raw, fraction_of_positives_raw, "s-", label="Raw")

    # Calibrated
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_true, y_prob_cal, n_bins=n_bins, strategy='uniform')
    ax.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Calibrated")

    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")
    ax.set_title('Calibration plot')