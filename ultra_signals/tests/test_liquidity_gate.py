import time
from ultra_signals.engine.gates.liquidity_gate import evaluate_gate, LiquidityGate
from ultra_signals.market.book_health import BookHealth


def _settings():
    return {
        "micro_liquidity": {
            "enabled": True,
            "profiles": {
                "trend": {
                    "spread_cap_bps": 2.5,
                    "spread_warn_bps": 1.5,
                    "impact_cap_bps": 6.0,
                    "impact_warn_bps": 3.5,
                    "rv_cap_bps": 5.0,
                    "rv_whip_cap_bps": 8.0,
                    "dr_skew_cap": 0.65,
                    "mt_trend_min": 0.20,
                    "dampen": {"size_mult": 0.7, "widen_stop_mult": 1.15, "maker_only": True},
                }
            },
            "missing_feed_policy": "SAFE",
            "cooldown_after_veto_secs": 20,
        }
    }


def test_spread_cap_veto():
    s = _settings()
    bh = BookHealth(ts=int(time.time()), symbol="BTCUSDT", spread_bps=3.0)
    g = evaluate_gate("BTCUSDT", int(time.time()), "trend", bh, s)
    assert g.action == "VETO" and g.reason == "WIDE_SPREAD"


def test_impact_cap_veto():
    s = _settings()
    bh = BookHealth(ts=int(time.time()), symbol="BTCUSDT", spread_bps=1.0, impact_50k=7.0)
    g = evaluate_gate("BTCUSDT", int(time.time()), "trend", bh, s)
    assert g.action == "VETO" and g.reason == "THIN_BOOK"


def test_spoof_whipsaw_veto():
    s = _settings()
    bh = BookHealth(ts=int(time.time()), symbol="BTCUSDT", spread_bps=1.0, impact_50k=4.0, dr=0.8, rv_5s=6.0)
    g = evaluate_gate("BTCUSDT", int(time.time()), "trend", bh, s)
    assert g.action == "VETO" and g.reason == "SPOOFY"


def test_dampen_path():
    s = _settings()
    # spread between warn and cap triggers DAMPEN
    bh = BookHealth(ts=int(time.time()), symbol="BTCUSDT", spread_bps=1.6, impact_50k=2.0)
    g = evaluate_gate("BTCUSDT", int(time.time()), "trend", bh, s)
    assert g.action == "DAMPEN" and g.size_mult == 0.7


def test_missing_feed_policy_safe():
    s = _settings()
    g = evaluate_gate("BTCUSDT", int(time.time()), "trend", None, s)
    assert g.action == "DAMPEN" and g.reason == "MISSING_FEED"


def test_cooldown_applied():
    s = _settings()
    now = int(time.time())
    gate = LiquidityGate(s)
    bh = BookHealth(ts=now, symbol="BTCUSDT", spread_bps=3.0)  # first veto
    g1 = evaluate_gate("BTCUSDT", now, "trend", bh, s, gate)
    assert g1.action == "VETO"
    # second attempt inside cooldown window
    bh2 = BookHealth(ts=now + 5, symbol="BTCUSDT", spread_bps=1.0)  # conditions ok but cooldown should veto
    g2 = evaluate_gate("BTCUSDT", now + 5, "trend", bh2, s, gate)
    assert g2.action == "VETO" and g2.reason == "COOLDOWN"
