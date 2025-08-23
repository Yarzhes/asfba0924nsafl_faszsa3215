import pytest
import pandas as pd
import numpy as np
from ultra_signals.analytics.correlation import compute_corr_groups, update_corr_state

@pytest.fixture
def mock_returns_df():
    """Provides a mock DataFrame of asset returns."""
    data = {
        "BTC/USDT": np.random.randn(100),
        "ETH/USDT": np.random.randn(100),
        "SOL/USDT": np.random.randn(100),
        "ADA/USDT": np.random.randn(100),
    }
    # Create correlation between BTC and ETH
    data["ETH/USDT"] = data["BTC/USDT"] * 0.8 + np.random.randn(100) * 0.2
    # Create correlation between SOL and ADA
    data["ADA/USDT"] = data["SOL/USDT"] * 0.9 + np.random.randn(100) * 0.1
    return pd.DataFrame(data)

def test_compute_corr_groups(mock_returns_df):
    """
    Tests that symbols are correctly grouped based on a correlation threshold.
    """
    threshold = 0.7
    groups = compute_corr_groups(mock_returns_df, threshold)
    
    assert groups["BTC/USDT"] == groups["ETH/USDT"]
    assert groups["SOL/USDT"] == groups["ADA/USDT"]
    assert groups["BTC/USDT"] != groups["SOL/USDT"]

def test_compute_corr_groups_no_correlation():
    """
    Tests that symbols are placed in their own clusters if no correlation exists.
    """
    data = {
        "A": [1, 2, 3, 4],
        "B": [4, 3, 2, 1],
        "C": [1, 2, 1, 2],
    }
    df = pd.DataFrame(data)
    groups = compute_corr_groups(df, threshold=0.95)
    
    assert groups["A"] != groups["B"]
    assert groups["A"] != groups["C"]
    assert groups["B"] != groups["C"]

def test_update_corr_state_placeholder(mock_returns_df):
    """
    Tests the placeholder functionality of update_corr_state.
    
    This test verifies that the function currently returns the new groups
    directly, as the hysteresis logic is managed externally.
    """
    prev_state = {"BTC/USDT": "cluster_0", "ETH/USDT": "cluster_0"}
    new_groups = compute_corr_groups(mock_returns_df, threshold=0.7)
    
    updated_state = update_corr_state(prev_state, new_groups, hysteresis_hits=2)
    
    assert updated_state == new_groups