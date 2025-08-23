import pytest
import numpy as np
from ultra_signals.backtest.metrics import compute_reliability_bins, calculate_brier_score

def test_brier_score():
    """Test the Brier score calculation."""
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    # Expected: ((0.1-0)^2 + (0.9-1)^2 + (0.8-1)^2 + (0.3-0)^2) / 4
    # = (0.01 + 0.01 + 0.04 + 0.09) / 4 = 0.15 / 4 = 0.0375
    assert calculate_brier_score(y_true, y_prob) == pytest.approx(0.0375)

def test_reliability_bins():
    """Test the reliability binning logic."""
    # Predictions are well-calibrated in this example
    predictions = np.array([0.1, 0.25, 0.75, 0.9])
    outcomes = np.array([0, 0, 1, 1]) # Outcomes match probabilities
    
    report = compute_reliability_bins(predictions, outcomes, n_bins=5)
    
    # Check Brier score
    assert report['brier_score'] > 0

    # Check bin contents
    # Bin 0 (0.0-0.2): pred=0.1, outcome=0. Count=1, MeanPred=0.1, FracPos=0.0
    # Bin 1 (0.2-0.4): pred=0.25, outcome=0. Count=1, MeanPred=0.25, FracPos=0.0
    # Bin 3 (0.6-0.8): pred=0.75, outcome=1. Count=1, MeanPred=0.75, FracPos=1.0
    # Bin 4 (0.8-1.0): pred=0.9, outcome=1. Count=1, MeanPred=0.9, FracPos=1.0
    counts = report['bins']['counts']
    mean_preds = report['bins']['mean_predicted']
    frac_pos = report['bins']['fraction_positives']

    assert counts[0] == 1
    assert counts[1] == 1
    assert counts[2] == 0 # Empty bin
    assert counts[3] == 1
    assert counts[4] == 1
    
    assert mean_preds[0] == pytest.approx(0.1)
    assert frac_pos[0] == pytest.approx(0.0)
    
    assert mean_preds[3] == pytest.approx(0.75)
    assert frac_pos[3] == pytest.approx(1.0)