import pytest
from ultra_signals.core.custom_types import SubSignal, EnsembleDecision
from ultra_signals.engine.ensemble import combine_subsignals

@pytest.fixture
def mock_subsignals():
    """Provides a list of mock SubSignal objects for testing."""
    return [
        SubSignal(ts=1, symbol="BTC/USDT", tf="1h", strategy_id="trend_pullback", direction="LONG", confidence_calibrated=0.9, reasons={}),
        SubSignal(ts=1, symbol="BTC/USDT", tf="1h", strategy_id="breakout_book", direction="LONG", confidence_calibrated=0.8, reasons={}),
        SubSignal(ts=1, symbol="BTC/USDT", tf="1h", strategy_id="mean_revert_vwap", direction="SHORT", confidence_calibrated=0.4, reasons={}),
        SubSignal(ts=1, symbol="BTC/USDT", tf="1h", strategy_id="sweep_reversal", direction="FLAT", confidence_calibrated=0.0, reasons={}),
    ]

@pytest.fixture
def mock_settings():
    """Provides mock ensemble settings."""
    return {
        "enabled": True,
        "strategies": ["trend_pullback", "breakout_book", "mean_revert_vwap", "sweep_reversal"],
        "weights_profiles": {
            "default": {"trend_pullback": 0.4, "breakout_book": 0.3, "mean_revert_vwap": 0.2, "sweep_reversal": 0.1},
            "trend": {"trend_pullback": 0.5, "breakout_book": 0.4, "mean_revert_vwap": 0.05, "sweep_reversal": 0.05},
        },
        "vote_threshold": 0.5,
        "veto": {
            "breakout_requires_book_flip": True,
            "mr_requires_band_pierce": True,
        },
    }

def test_combine_subsignals_long_decision(mock_subsignals, mock_settings):
    """
    Tests that a clear LONG decision is made when weights and confidences align.
    """
    decision = combine_subsignals(mock_subsignals, "default", mock_settings)
    assert decision.decision == "LONG"
    assert decision.confidence > 0
    assert decision.vote_detail["agree"] == 2
    assert decision.vote_detail["total"] == 4

def test_combine_subsignals_short_decision(mock_subsignals, mock_settings):
    """
    Tests that a clear SHORT decision can be made.
    """
    # Reverse the direction of the strong signals
    mock_subsignals[0].direction = "SHORT"
    mock_subsignals[1].direction = "SHORT"
    mock_subsignals[0].confidence_calibrated = 0.9
    mock_subsignals[1].confidence_calibrated = 0.8
    mock_subsignals[2].direction = "LONG"
    
    decision = combine_subsignals(mock_subsignals, "default", mock_settings)
    assert decision.decision == "SHORT"
    assert decision.confidence > 0

def test_combine_subsignals_flat_decision(mock_subsignals, mock_settings):
    """
    Tests that a FLAT decision is made when the weighted sum is below the threshold.
    """
    mock_settings["vote_threshold"] = 0.8  # High threshold
    decision = combine_subsignals(mock_subsignals, "default", mock_settings)
    assert decision.decision == "FLAT"

def test_combine_subsignals_with_veto(mock_subsignals, mock_settings):
    """
    Tests that a veto can override a decision (conceptual test).
    
    Note: The current `combine_subsignals` has placeholder veto logic. This
    test would need to be updated once real veto conditions are implemented.
    """
    # In a real scenario, we would modify a subsignal's 'reasons' to trigger a veto.
    # For now, this test passes as the placeholder logic does not yet veto.
    decision = combine_subsignals(mock_subsignals, "default", mock_settings)
    mock_subsignals[0].reasons = {"veto": "VETO_REASON"}
    decision = combine_subsignals(mock_subsignals, "default", mock_settings)
    assert decision.decision == "FLAT"

def test_combine_subsignals_uses_regime_weights(mock_subsignals, mock_settings):
    """
    Tests that the correct weight profile is selected based on the regime.
    """
    # Using 'trend' profile, which has different weights
    decision_trend = combine_subsignals(mock_subsignals, "trend", mock_settings)
    # The weighted sum should be different than with default weights
    decision_default = combine_subsignals(mock_subsignals, "default", mock_settings)
    
    assert decision_trend.vote_detail["weighted_sum"] != decision_default.vote_detail["weighted_sum"]
    assert decision_trend.vote_detail["profile"] == "trend"