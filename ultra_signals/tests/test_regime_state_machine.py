"""Tests for Sprint 10 Regime & Market State 2.0 implementation."""
import pytest
from ultra_signals.features.regime import RegimeStateMachine, classify_regime_full
from ultra_signals.core.custom_types import RegimeMode

@pytest.fixture
def settings():
    return {
        "regime": {
            "adx_trend_min": 25,
            "mean_revert_adx_max": 22,
            "adx_chop_max": 18,
            "atr_low_thr": 0.35,
            "atr_high_thr": 0.65,
            "hysteresis_hits": 2,
            "vol_crush_thr": 0.25,
            "vol_expansion_thr": 0.75,
        }
    }

def test_hysteresis_trend_transition(settings):
    state = RegimeStateMachine()
    # First call: high ADX & atr => candidate trend but needs 2 hits
    rf1 = classify_regime_full(30, 0.7, None, settings, state)
    assert rf1.mode in (RegimeMode.CHOP, RegimeMode.TREND)  # initial may still be chop until commit
    # Second consecutive call confirms transition
    rf2 = classify_regime_full(30, 0.7, None, settings, state)
    assert rf2.mode == RegimeMode.TREND


def test_hysteresis_prevents_flicker(settings):
    state = RegimeStateMachine()
    # Gain trend
    classify_regime_full(30, 0.7, None, settings, state)
    classify_regime_full(30, 0.7, None, settings, state)
    assert state.current == RegimeMode.TREND
    # One bar of low ADX should not instantly flip to MR
    classify_regime_full(20, 0.5, None, settings, state)
    assert state.current == RegimeMode.TREND  # still trend because need 2 hits


def test_vol_state(settings):
    state = RegimeStateMachine()
    rf_crush = classify_regime_full(15, 0.20, None, settings, state)
    assert rf_crush.vol_state.value == "crush"
    rf_exp = classify_regime_full(30, 0.80, None, settings, state)
    assert rf_exp.vol_state.value == "expansion"


def test_gates_default(settings):
    state = RegimeStateMachine()
    rf = classify_regime_full(30, 0.7, None, settings, state)
    assert "trend_following" in rf.gates
    assert isinstance(rf.gates["trend_following"], bool)

