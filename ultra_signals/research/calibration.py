"""
Lightweight Probability Calibration

This module provides logistic calibration for converting raw ensemble scores
to calibrated probabilities. Uses numpy least-squares for fitting calibration
coefficients without external dependencies.

Reference: Platt, J. (1999). Probabilistic outputs for support vector machines.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from loguru import logger


def fit_logistic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit logistic calibration coefficients using least squares.
    
    Args:
        x: Raw ensemble scores (1D array)
        y: Binary outcomes (1D array, 0 or 1)
        
    Returns:
        Tuple of (a, b) coefficients for p = 1/(1+e^(−(a+b*x)))
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    
    if len(x) < 10:
        logger.warning("Insufficient data for calibration, using default coefficients")
        return 0.0, 1.0
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 10:
        logger.warning("Insufficient valid data for calibration, using default coefficients")
        return 0.0, 1.0
    
    try:
        # Use logistic regression with least squares
        # Transform to logit space: log(p/(1-p)) = a + b*x
        # For y=0, use small positive value; for y=1, use large positive value
        eps = 1e-6
        y_transformed = np.where(y_clean == 0, eps, 1 - eps)
        logits = np.log(y_transformed / (1 - y_transformed))
        
        # Add constant term for intercept
        X = np.column_stack([np.ones(len(x_clean)), x_clean])
        
        # Solve least squares: (X^T X)^(-1) X^T y
        coeffs = np.linalg.lstsq(X, logits, rcond=None)[0]
        
        # Extract a and b coefficients
        a = coeffs[0]
        b = coeffs[1]
        
        # Validate coefficients
        if not np.isfinite(a) or not np.isfinite(b):
            logger.warning("Invalid calibration coefficients, using defaults")
            return 0.0, 1.0
        
        logger.info(f"Calibration fitted: a={a:.4f}, b={b:.4f}")
        return a, b
        
    except Exception as e:
        logger.warning(f"Calibration fitting failed: {e}, using default coefficients")
        return 0.0, 1.0


def apply_logistic(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Apply logistic calibration to raw scores.
    
    Args:
        x: Raw ensemble scores
        a: Intercept coefficient
        b: Slope coefficient
        
    Returns:
        Calibrated probabilities in [0, 1]
    """
    # Apply logistic transformation: p = 1/(1+e^(−(a+b*x)))
    logits = a + b * x
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Clip to valid probability range
    probs = np.clip(probs, 0.0, 1.0)
    
    return probs


def calibrate_ensemble_scores(
    scores: np.ndarray,
    outcomes: np.ndarray,
    validation_split: float = 0.2
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Calibrate ensemble scores using train/validation split.
    
    Args:
        scores: Raw ensemble scores
        outcomes: Binary outcomes (0 or 1)
        validation_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (calibrated_probs, (a, b) coefficients)
    """
    if len(scores) != len(outcomes):
        raise ValueError("scores and outcomes must have same length")
    
    # Split data
    n_train = int(len(scores) * (1 - validation_split))
    train_scores = scores[:n_train]
    train_outcomes = outcomes[:n_train]
    val_scores = scores[n_train:]
    val_outcomes = outcomes[n_train:]
    
    # Fit calibration on training data
    a, b = fit_logistic(train_scores, train_outcomes)
    
    # Apply calibration to all data
    calibrated_probs = apply_logistic(scores, a, b)
    
    # Validate on validation set
    if len(val_scores) > 0:
        val_probs = apply_logistic(val_scores, a, b)
        val_auc = calculate_auc(val_probs, val_outcomes)
        logger.info(f"Validation AUC: {val_auc:.4f}")
    
    return calibrated_probs, (a, b)


def calculate_auc(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate Area Under Curve (AUC) for calibration validation.
    
    Args:
        probs: Predicted probabilities
        outcomes: True outcomes (0 or 1)
        
    Returns:
        AUC score
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must have same length")
    
    # Sort by probabilities
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_outcomes = outcomes[sorted_indices]
    
    # Calculate TPR and FPR
    tp = np.cumsum(sorted_outcomes)
    fp = np.cumsum(1 - sorted_outcomes)
    
    # Normalize
    total_pos = np.sum(outcomes)
    total_neg = len(outcomes) - total_pos
    
    if total_pos == 0 or total_neg == 0:
        return 0.5  # No discrimination possible
    
    tpr = tp / total_pos
    fpr = fp / total_neg
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return auc


def evaluate_calibration(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate calibration quality using reliability diagram.
    
    Args:
        probs: Predicted probabilities
        outcomes: True outcomes
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary with calibration metrics
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must have same length")
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign samples to bins
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate reliability
    reliability = []
    confidence = []
    counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            avg_prob = np.mean(probs[mask])
            avg_outcome = np.mean(outcomes[mask])
            count = np.sum(mask)
            
            reliability.append(avg_outcome)
            confidence.append(avg_prob)
            counts.append(count)
        else:
            reliability.append(0.0)
            confidence.append(bin_centers[i])
            counts.append(0)
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(np.array(reliability) - np.array(confidence)))
    
    # Calculate Brier score
    brier_score = np.mean((probs - outcomes) ** 2)
    
    return {
        'calibration_error': calibration_error,
        'brier_score': brier_score,
        'reliability': reliability,
        'confidence': confidence,
        'counts': counts
    }


def save_calibration_coefficients(
    a: float,
    b: float,
    settings: Dict,
    key: str = 'calibration'
) -> None:
    """
    Save calibration coefficients to settings dictionary.
    
    Args:
        a: Intercept coefficient
        b: Slope coefficient
        settings: Settings dictionary to update
        key: Key under which to store coefficients
    """
    if 'ensemble' not in settings:
        settings['ensemble'] = {}
    
    settings['ensemble'][key] = {
        'a': float(a),
        'b': float(b)
    }


def load_calibration_coefficients(
    settings: Dict,
    key: str = 'calibration'
) -> Tuple[float, float]:
    """
    Load calibration coefficients from settings.
    
    Args:
        settings: Settings dictionary
        key: Key under which coefficients are stored
        
    Returns:
        Tuple of (a, b) coefficients, defaults to (0.0, 1.0) if not found
    """
    try:
        calib = settings.get('ensemble', {}).get(key, {})
        a = float(calib.get('a', 0.0))
        b = float(calib.get('b', 1.0))
        return a, b
    except Exception:
        logger.warning("Failed to load calibration coefficients, using defaults")
        return 0.0, 1.0



