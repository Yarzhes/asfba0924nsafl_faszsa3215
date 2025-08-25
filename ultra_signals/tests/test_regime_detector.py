import pytest
from ultra_signals.features.regime import RegimeStateMachine, classify_regime_full
from ultra_signals.core.custom_types import RegimeMode, LiquidityState, VolState

BASE_CFG = {
    "regime": {
        "hysteresis_hits": 2,
        "cooldown_bars": 4,
        "strong_override": True,
        "primary": {
            "trend": {"enter": {"adx_min": 24, "ema_sep_atr_min": 0.30}, "exit": {"adx_min": 18, "ema_sep_atr_min": 0.20}},
            "mean_revert": {"enter": {"adx_max": 16, "bb_width_pct_atr_max": 0.70}, "exit": {"adx_max": 20}},
            "chop": {"enter": {"adx_max": 20}, "exit": {"adx_max": 24}},
        },
        "vol": {"expansion_atr_pct": 0.70, "crush_atr_pct": 0.20},
        "liquidity": {"max_spread_bp": 3.0, "min_volume_z": -1.0},
    }
}

def test_trend_detection_hysteresis_and_cooldown():
    state = RegimeStateMachine()
    r1 = classify_regime_full(26, 0.55, 0.40, BASE_CFG, state)
    assert r1.mode in (RegimeMode.CHOP, RegimeMode.TREND)
    r2 = classify_regime_full(27, 0.58, 0.42, BASE_CFG, state)
    assert r2.mode == RegimeMode.TREND
    trend_flip_ts = state.last_flip_ts
    r3 = classify_regime_full(15, 0.50, 0.10, BASE_CFG, state)
    assert state.current == RegimeMode.TREND
    assert state.last_flip_ts == trend_flip_ts

def test_mean_revert_detection():
    state = RegimeStateMachine()
    classify_regime_full(15, 0.50, 0.05, BASE_CFG, state)
    r2 = classify_regime_full(14, 0.52, 0.04, BASE_CFG, state)
    assert r2.mode in (RegimeMode.MEAN_REVERT, RegimeMode.CHOP)

def test_chop_detection_low_adx():
    state = RegimeStateMachine()
    r = classify_regime_full(10, 0.30, 0.02, BASE_CFG, state)
    assert r.mode == RegimeMode.CHOP

def test_vol_states():
    state = RegimeStateMachine()
    crush = classify_regime_full(12, 0.10, 0.02, BASE_CFG, state)
    assert crush.vol_state == VolState.CRUSH
    exp = classify_regime_full(30, 0.85, 0.40, BASE_CFG, state)
    assert exp.vol_state == VolState.EXPANSION

def test_liquidity_thin():
    state = RegimeStateMachine()
    thin = classify_regime_full(25, 0.50, 0.35, BASE_CFG, state, spread_bps=5.0, volume_z=-2.0)
    assert thin.liquidity == LiquidityState.THIN

def test_confidence_range():
    state = RegimeStateMachine()
    rf = classify_regime_full(30, 0.60, 0.50, BASE_CFG, state)
    assert 0.3 <= rf.confidence <= 1.0