import pytest
from ultra_signals.engine.position_sizer import apply_volatility_scaling

@pytest.fixture
def mock_config():
    """Provides mock volatility scaling config."""
    return {
        "atr_pct_window": 200,
        "low_vol_pct": 30,
        "high_vol_pct": 70,
        "low_vol_boost": 1.20,
        "high_vol_cut": 0.70,
    }

def test_apply_volatility_scaling_low_vol(mock_config):
    """
    Tests that risk is boosted in a low volatility environment.
    """
    base_risk = 100.0
    atr_percentile = 20  # Below low_vol_pct
    adjusted_risk = apply_volatility_scaling(base_risk, atr_percentile, mock_config)
    assert adjusted_risk == base_risk * mock_config["low_vol_boost"]

def test_apply_volatility_scaling_high_vol(mock_config):
    """
    Tests that risk is cut in a high volatility environment.
    """
    base_risk = 100.0
    atr_percentile = 80  # Above high_vol_pct
    adjusted_risk = apply_volatility_scaling(base_risk, atr_percentile, mock_config)
    assert adjusted_risk == base_risk * mock_config["high_vol_cut"]
    
def test_apply_volatility_scaling_medium_vol(mock_config):
    """
    Tests that risk is not adjusted in a medium volatility environment.
    """
    base_risk = 100.0
    atr_percentile = 50  # Between low and high thresholds
    adjusted_risk = apply_volatility_scaling(base_risk, atr_percentile, mock_config)
    assert adjusted_risk == base_risk

def test_apply_volatility_scaling_no_adjustment_if_misconfigured():
    """
    Tests that no adjustment is made if the config is missing keys.
    """
    base_risk = 100.0
    atr_percentile = 20
    # Missing 'low_vol_boost', should default to 1.0
    misconfigured = {"low_vol_pct": 30, "high_vol_pct": 70}
    adjusted_risk = apply_volatility_scaling(base_risk, atr_percentile, misconfigured)
    assert adjusted_risk == base_risk