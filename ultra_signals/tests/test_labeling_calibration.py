"""
Unit tests for labeling and calibration modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from ultra_signals.research.labeling import (
    label_trades, label_trades_batch, calculate_outcome_statistics,
    validate_triple_barrier, TripleBarrierResult
)
from ultra_signals.research.calibration import (
    fit_logistic, apply_logistic, calibrate_ensemble_scores,
    calculate_auc, evaluate_calibration, save_calibration_coefficients,
    load_calibration_coefficients
)


class TestTripleBarrierLabeling:
    """Test triple-barrier labeling functionality."""
    
    def test_label_trades_take_profit(self):
        """Test labeling when take profit is hit first."""
        # Create price series with clear uptrend
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        
        # Entry at index 5 (price 105), TP at 107 (2% above), SL at 103 (2% below)
        result = label_trades(prices, entry_idx=5, pt_mult=2.0, sl_mult=1.0, max_horizon_bars=10)
        
        assert result.outcome == 1  # Win
        assert result.hit_barrier == "tp"
        assert result.horizon_bars == 3  # Hit TP after 3 bars (107.1 hit at price 108)
        assert result.exit_price == pytest.approx(107.1, rel=1e-3)
        assert result.return_pct == pytest.approx(0.02, rel=1e-3)
    
    def test_label_trades_stop_loss(self):
        """Test labeling when stop loss is hit first."""
        # Create price series with clear downtrend
        prices = pd.Series([110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100])
        
        # Entry at index 5 (price 105), TP at 107, SL at 103
        result = label_trades(prices, entry_idx=5, pt_mult=2.0, sl_mult=1.0, max_horizon_bars=10)
        
        assert result.outcome == -1  # Loss
        assert result.hit_barrier == "sl"
        assert result.horizon_bars == 2  # Hit SL after 2 bars
        assert result.exit_price == pytest.approx(103.95, rel=1e-3)
        assert result.return_pct == pytest.approx(-0.01, rel=1e-3)
    
    def test_label_trades_timeout(self):
        """Test labeling when neither TP nor SL is hit within horizon."""
        # Create price series that stays within bounds
        prices = pd.Series([100, 100.5, 100.3, 100.7, 100.2, 100.6, 100.4, 100.8, 100.1, 100.9])
        
        # Entry at index 2 (price 100.3), TP at 102.3, SL at 98.3, max horizon 5
        result = label_trades(prices, entry_idx=2, pt_mult=2.0, sl_mult=1.0, max_horizon_bars=5)
        
        assert result.outcome == 0  # Timeout
        assert result.hit_barrier == "timeout"
        assert result.horizon_bars == 5
        assert result.exit_price == pytest.approx(100.8, rel=1e-3)  # Last price in horizon
    
    def test_label_trades_batch(self):
        """Test batch labeling of multiple trades."""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        entry_indices = [2, 5, 8]  # Multiple entry points
        
        results = label_trades_batch(prices, entry_indices, pt_mult=2.0, sl_mult=1.0, max_horizon_bars=10)
        
        assert len(results) == 3
        assert all(isinstance(r, TripleBarrierResult) for r in results)
    
    def test_calculate_outcome_statistics(self):
        """Test calculation of outcome statistics."""
        # Create sample outcomes
        outcomes = [
            TripleBarrierResult(outcome=1, hit_barrier="tp", horizon_bars=3, exit_price=105, return_pct=0.05),
            TripleBarrierResult(outcome=-1, hit_barrier="sl", horizon_bars=2, exit_price=95, return_pct=-0.05),
            TripleBarrierResult(outcome=1, hit_barrier="tp", horizon_bars=4, exit_price=106, return_pct=0.06),
            TripleBarrierResult(outcome=0, hit_barrier="timeout", horizon_bars=10, exit_price=101, return_pct=0.01),
        ]
        
        stats = calculate_outcome_statistics(outcomes)
        
        assert stats['win_rate'] == 0.5  # 2 wins out of 4 trades
        assert stats['avg_return'] == pytest.approx(0.0175, rel=1e-3)  # Average return
        assert stats['avg_horizon'] == 4.75  # Average horizon
        assert stats['tp_rate'] == 0.5  # 2 TP hits
        assert stats['sl_rate'] == 0.25  # 1 SL hit
        assert stats['timeout_rate'] == 0.25  # 1 timeout
    
    def test_validate_triple_barrier(self):
        """Test validation of triple-barrier labeling."""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        
        # Should validate correctly
        assert validate_triple_barrier(prices, entry_idx=5, expected_outcome=1, pt_mult=2.0, sl_mult=1.0)
        
        # Should fail validation
        assert not validate_triple_barrier(prices, entry_idx=5, expected_outcome=-1, pt_mult=2.0, sl_mult=1.0)


class TestCalibration:
    """Test calibration functionality."""
    
    def test_fit_logistic(self):
        """Test logistic calibration fitting."""
        # Create synthetic separable data
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = (x > 0).astype(float)  # Perfect separation
        
        a, b = fit_logistic(x, y)
        
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert np.isfinite(a)
        assert np.isfinite(b)
        assert b > 0  # Should be positive slope for separable data
    
    def test_apply_logistic(self):
        """Test applying logistic calibration."""
        x = np.array([-2, -1, 0, 1, 2])
        a, b = 0.0, 1.0  # Standard logistic
        
        probs = apply_logistic(x, a, b)
        
        assert len(probs) == len(x)
        assert all(0 <= p <= 1 for p in probs)
        assert probs[0] < probs[1] < probs[2] < probs[3] < probs[4]  # Monotonic
    
    def test_calibrate_ensemble_scores(self):
        """Test ensemble score calibration."""
        np.random.seed(42)
        scores = np.random.normal(0, 1, 100)
        outcomes = (scores > 0).astype(float)
        
        calibrated_probs, (a, b) = calibrate_ensemble_scores(scores, outcomes)
        
        assert len(calibrated_probs) == len(scores)
        assert all(0 <= p <= 1 for p in calibrated_probs)
        assert isinstance(a, float)
        assert isinstance(b, float)
    
    def test_calculate_auc(self):
        """Test AUC calculation."""
        # Perfect classifier
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
        
        auc = calculate_auc(probs, outcomes)
        assert auc > 0.9  # Should be high for good separation
    
    def test_evaluate_calibration(self):
        """Test calibration evaluation."""
        np.random.seed(42)
        probs = np.random.uniform(0, 1, 100)
        outcomes = (probs > 0.5).astype(float)  # Perfect calibration
        
        metrics = evaluate_calibration(probs, outcomes, n_bins=10)
        
        assert 'calibration_error' in metrics
        assert 'brier_score' in metrics
        assert 'reliability' in metrics
        assert 'confidence' in metrics
        assert 'counts' in metrics
        assert len(metrics['reliability']) == 10
    
    def test_save_load_calibration_coefficients(self):
        """Test saving and loading calibration coefficients."""
        settings = {}
        a, b = 0.5, 1.2
        
        # Save coefficients
        save_calibration_coefficients(a, b, settings)
        
        # Load coefficients
        loaded_a, loaded_b = load_calibration_coefficients(settings)
        
        assert loaded_a == a
        assert loaded_b == b
    
    def test_load_calibration_coefficients_defaults(self):
        """Test loading calibration coefficients with defaults."""
        settings = {}
        
        a, b = load_calibration_coefficients(settings)
        
        assert a == 0.0
        assert b == 1.0
    
    def test_fit_logistic_insufficient_data(self):
        """Test logistic fitting with insufficient data."""
        x = np.array([1, 2, 3])  # Only 3 points
        y = np.array([0, 1, 0])
        
        a, b = fit_logistic(x, y)
        
        assert a == 0.0
        assert b == 1.0  # Default values
    
    def test_fit_logistic_nan_values(self):
        """Test logistic fitting with NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([0, 1, 0, 1, 0])
        
        a, b = fit_logistic(x, y)
        
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert np.isfinite(a)
        assert np.isfinite(b)


if __name__ == "__main__":
    pytest.main([__file__])
