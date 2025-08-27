import pytest

from ultra_signals.engine.gates.mtc_gate import MTCGate, MTCGateResult
from ultra_signals.features.htf_cache import HTFFeatures


def _htf(tf, ema21, ema200, adx, macd_line, macd_prev, rsi, atrp, vwap, price, stale=False):
    macd_slope = None
    if macd_line is not None and macd_prev is not None:
        macd_slope = macd_line - macd_prev
    return HTFFeatures(
        timeframe=tf,
        ts=0,
        ema21=ema21,
        ema200=ema200,
        adx=adx,
        macd_line=macd_line,
        macd_signal=None,
        macd_hist=None,
        macd_slope=macd_slope,
        rsi=rsi,
        atr_percentile=atrp,
        vwap=vwap,
        price=price,
        price_above_vwap=None if (price is None or vwap is None) else price >= vwap,
        stale=stale,
    )


BASE_CFG = {
    "mtc": {
        "enabled": True,
        "rules": {
            "trend": {"adx_min": 18, "score_weight": 0.5},
            "momentum": {"macd_slope_min": 0.0, "rsi_band_long": [45, 70], "rsi_band_short": [30, 55], "score_weight": 0.3},
            "volatility": {"atr_pctile_max": 0.98, "score_weight": 0.1},
            "structure": {"score_weight": 0.1},
        },
        "thresholds": {"confirm_full": 0.70, "confirm_partial": 0.50},
        "actions": {"partial": {"size_mult": 0.6, "widen_stop_mult": 1.10}, "fail": {"veto": True}},
        "missing_data_policy": "SAFE",
    }
}


def build_gate():
    return MTCGate(BASE_CFG)


def test_mtc_trend_long_short():
    gate = build_gate()
    c1 = _htf("15m", ema21=105, ema200=100, adx=25, macd_line=1.2, macd_prev=1.0, rsi=55, atrp=0.5, vwap=104, price=105)
    c2 = _htf("1h", ema21=210, ema200=200, adx=30, macd_line=2.4, macd_prev=2.2, rsi=60, atrp=0.6, vwap=208, price=209)
    res_long = gate.evaluate("LONG", "BTCUSDT", "5m", 0, "trend", {"C1": c1, "C2": c2})
    assert res_long.status in ("CONFIRM", "PARTIAL")
    # Short: invert ema relation
    c1s = _htf("15m", ema21=95, ema200=100, adx=25, macd_line=-1.2, macd_prev=-1.0, rsi=40, atrp=0.5, vwap=96, price=95)
    c2s = _htf("1h", ema21=190, ema200=200, adx=30, macd_line=-2.4, macd_prev=-2.2, rsi=45, atrp=0.6, vwap=192, price=191)
    res_short = gate.evaluate("SHORT", "BTCUSDT", "5m", 0, "trend", {"C1": c1s, "C2": c2s})
    assert res_short.status in ("CONFIRM", "PARTIAL")


def test_mtc_momentum_rsi_bands():
    gate = build_gate()
    # Good momentum for LONG
    c1 = _htf("15m", 110, 100, 20, 1.3, 1.2, 60, 0.5, 109, 110)
    c2 = _htf("1h", 205, 200, 22, 2.2, 2.1, 65, 0.5, 204, 205)
    res = gate.evaluate("LONG", "BTCUSDT", "5m", 0, "trend", {"C1": c1, "C2": c2})
    assert res.scores.get("C1", 0) > 0.5
    # RSI out of band for SHORT (should reduce score)
    c1s = _htf("15m", 90, 100, 20, -1.3, -1.2, 70, 0.5, 91, 90)
    res_bad = gate.evaluate("SHORT", "BTCUSDT", "5m", 0, "trend", {"C1": c1s})
    assert res_bad.scores.get("C1", 0) < res.scores.get("C1", 1.0)


def test_mtc_vol_penalty():
    gate = build_gate()
    # Elevated ATR percentile
    c1 = _htf("15m", 110, 100, 25, 1.1, 1.0, 55, 0.995, 109, 110)
    res = gate.evaluate("LONG", "BTCUSDT", "5m", 0, "trend", {"C1": c1})
    assert res.scores["C1"] < 1.0


def test_mtc_vwap_structure():
    gate = build_gate()
    c1 = _htf("15m", 110, 100, 22, 1.0, 0.9, 55, 0.5, 109, 110)
    above = gate.evaluate("LONG", "BTCUSDT", "5m", 0, "trend", {"C1": c1})
    c1_below = _htf("15m", 110, 100, 22, 1.0, 0.9, 55, 0.5, 111, 110)  # price below vwap for LONG
    below = gate.evaluate("LONG", "BTCUSDT", "5m", 0, "trend", {"C1": c1_below})
    assert above.scores["C1"] >= below.scores["C1"]


def test_mtc_thresholds_to_actions():
    gate = build_gate()
    strong = _htf("15m", 110, 100, 30, 2.0, 1.9, 60, 0.5, 109, 110)
    weak = _htf("1h", 200, 190, 10, 0.2, 0.19, 30, 0.5, 199, 200)
    res = gate.evaluate("LONG", "BTCUSDT", "5m", 0, "trend", {"C1": strong, "C2": weak})
    assert res.action in ("DAMPEN", "ENTER", "VETO")


def test_mtc_staleness_policy():
    # C1 stale under SAFE => PARTIAL (DAMPEN)
    cfg = BASE_CFG.copy()
    cfg["mtc"] = dict(cfg["mtc"], missing_data_policy="SAFE")
    gate = MTCGate(cfg)
    stale = _htf("15m", 110, 100, 25, 1.2, 1.1, 60, 0.5, 109, 110, stale=True)
    res = gate.evaluate("LONG", "BTCUSDT", "5m", 0, "trend", {"C1": stale})
    assert res.status == "PARTIAL"
