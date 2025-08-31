import time
from unittest.mock import MagicMock

from ultra_signals.engine.risk_filters import apply_filters
from ultra_signals.core.custom_types import Signal, SignalType
from ultra_signals.engine.sniper_counters import reset_sniper_counters


class DummyStore:
    def __init__(self):
        pass

    def get_warmup_status(self, symbol, timeframe):
        return 100

    def get_book_ticker(self, symbol):
        return (100.0, 100.1, 0.1, 10)

    def current_ts_ms(self, *args, **kwargs):
        return int(time.time() * 1000)


def make_signal():
    return Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type=SignalType.TREND_FOLLOWING,
        price=50000,
        stop_loss=49500,
        take_profit_1=51000,
        score=0.9,
        features={},
    )


def test_sniper_hourly_cap_disabled():
    """Hourly cap has been intentionally disabled; multiple rapid calls all pass.
    We keep daily cap large so it does not interfere."""
    store = DummyStore()
    sig = make_signal()
    reset_sniper_counters()
    import ultra_signals.engine.risk_filters as rf
    if hasattr(rf.apply_filters, '_sniper_history_clear'):
        rf.apply_filters._sniper_history_clear()
    settings = {"features": {"warmup_periods": 20}, "runtime": {"sniper_mode": {"enabled": True, "max_signals_per_hour": 2, "daily_signal_cap": 100, "mtf_confirm": False}}}

    r1 = apply_filters(sig, store, settings)
    assert r1.passed is True
    r2 = apply_filters(sig, store, settings)
    assert r2.passed is True
    r3 = apply_filters(sig, store, settings)
    assert r3.passed is True  # no hourly cap anymore


def test_sniper_daily_cap():
    store = DummyStore()
    sig = make_signal()
    reset_sniper_counters()
    import ultra_signals.engine.risk_filters as rf
    if hasattr(rf.apply_filters, '_sniper_history_clear'):
        rf.apply_filters._sniper_history_clear()
    settings = {"features": {"warmup_periods": 20}, "runtime": {"sniper_mode": {"enabled": True, "max_signals_per_hour": 100, "daily_signal_cap": 2, "mtf_confirm": False}}}

    r1 = apply_filters(sig, store, settings)
    assert r1.passed is True
    r2 = apply_filters(sig, store, settings)
    assert r2.passed is True
    r3 = apply_filters(sig, store, settings)
    assert r3.passed is False and r3.reason == 'SNIPER_DAILY_CAP'


def test_sniper_mtf_confirm_blocks():
    store = DummyStore()
    sig = make_signal()
    # Configure confluence to require alignment; DummyStore does not provide HTF regime so _htf_confluence_agrees returns True by default
    # To simulate disagreement, we'll monkeypatch the helper used by risk_filters
    import ultra_signals.engine.risk_filters as rf

    orig = rf._htf_confluence_agrees

    try:
        rf._htf_confluence_agrees = lambda s, st, settings: False
        settings = {"features": {"warmup_periods": 20}, "runtime": {"sniper_mode": {"enabled": True, "mtf_confirm": True, "max_signals_per_hour": 10, "daily_signal_cap": 10}}}
        r = apply_filters(sig, store, settings)
        assert r.passed is False and r.reason in ('SNIPER_MTF_REQUIRED', 'MTF_DISAGREE')
    finally:
        rf._htf_confluence_agrees = orig
