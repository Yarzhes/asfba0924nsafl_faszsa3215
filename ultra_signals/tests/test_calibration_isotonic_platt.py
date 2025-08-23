import pytest
import numpy as np
from ultra_signals.calibration import calibrate

@pytest.fixture
def sample_calibration_data():
    """Generate sample prediction data that is poorly calibrated."""
    np.random.seed(42)
    # Predictions are systematically overconfident
    raw_preds = np.random.rand(100) * 0.5 + 0.25 # Centered around 0.5
    # True outcomes are less frequent than predicted
    outcomes = (raw_preds > np.random.rand(100)).astype(int)
    return raw_preds, outcomes

def test_fit_isotonic(sample_calibration_data):
    """Test fitting an Isotonic Regression model."""
    preds, outcomes = sample_calibration_data
    model = calibrate.fit_calibration_model(preds, outcomes, method="isotonic")
    
    assert model is not None
    # After fitting, the model should be able to make predictions
    calibrated_preds = calibrate.apply_calibration(model, preds)
    assert len(calibrated_preds) == len(preds)
    # Isotonic regression should produce monotonic output
    assert np.all(np.diff(calibrated_preds[np.argsort(preds)]) >= 0)

def test_fit_platt(sample_calibration_data):
    """Test fitting a Platt Scaling (Logistic Regression) model."""
    preds, outcomes = sample_calibration_data
    model = calibrate.fit_calibration_model(preds, outcomes, method="platt")
    
    assert model is not None
    calibrated_preds = calibrate.apply_calibration(model, preds)
    assert len(calibrated_preds) == len(preds)
    # Probabilities should be between 0 and 1
    assert np.all((calibrated_preds >= 0) & (calibrated_preds <= 1))

def test_per_regime_calibration(sample_calibration_data):
    """Test the per-regime calibration fitting and application."""
    preds, outcomes = sample_calibration_data
    regimes = np.array(['A'] * 50 + ['B'] * 50)
    
    # Fit one model per regime
    models = calibrate.fit_calibration_model(preds, outcomes, method="isotonic", regimes=regimes)
    assert isinstance(models, dict)
    assert "A" in models
    assert "B" in models
    
    # Apply per-regime models
    calibrated = calibrate.apply_calibration(models, preds, regimes=regimes)
    assert len(calibrated) == len(preds)
    
    # Check that predictions for regime 'A' differ from regime 'B'
    cal_A = calibrated[regimes == 'A']
    cal_B = calibrated[regimes == 'B']
    
    # Given the random data, it's highly unlikely they would be identical
    assert not np.allclose(cal_A, cal_B)