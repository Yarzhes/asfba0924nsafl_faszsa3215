import pytest
from ultra_signals.execution.fast_router import execute_fast_order

BASIC_SETTINGS = {
    'execution': {
        'mode': 'ultra_fast',
        'max_slippage_pct': 0.05,
        'use_orderbook_liquidity': True,
        'min_orderbook_depth': 5,
        'retry_attempts': 3,
        'smart_order_routing': True,
        'cancel_stale_orders': True,
        'cancel_timeout_sec': 0.5,
    }
}

def test_smart_order_routing_prefers_tighter_spread():
    quotes = {
        'binance': {'bid': 100.0, 'ask': 100.2, 'bid_size': 10, 'ask_size': 10},
        'bybit':   {'bid': 100.0, 'ask': 100.4, 'bid_size': 10, 'ask_size': 10},
    }
    res = execute_fast_order(symbol='BTCUSDT', side='LONG', size=1.0, price=100.1, settings=BASIC_SETTINGS, quotes=quotes)
    assert res.accepted
    assert res.venue == 'binance'  # tighter spread


def test_slippage_rejection():
    # artificially widen ask so (ask-mid)/mid > 0.05%
    quotes = {
        'binance': {'bid':100.0, 'ask':100.2, 'bid_size':10, 'ask_size':10}, # spread 0.2 -> mid 100.1 slip (0.1/100.1)=0.099% > 0.05%
    }
    # Adjust setting to very low 0.05% (already set). Expect rejection.
    res = execute_fast_order(symbol='BTCUSDT', side='LONG', size=1.0, price=100.0, settings=BASIC_SETTINGS, quotes=quotes)
    assert not res.accepted
    assert res.reason == 'SLIPPAGE_TOO_HIGH'


def test_liquidity_veto():
    quotes = {
        'binance': {'bid':100.0, 'ask':100.01, 'bid_size':1, 'ask_size':1},  # insufficient depth vs min 5
    }
    res = execute_fast_order(symbol='BTCUSDT', side='SHORT', size=0.5, price=100.0, settings=BASIC_SETTINGS, quotes=quotes)
    assert not res.accepted
    assert res.reason == 'DEPTH_INSUFFICIENT'


def test_fallback_when_no_quotes():
    res = execute_fast_order(symbol='BTCUSDT', side='LONG', size=1.0, price=100.0, settings=BASIC_SETTINGS, quotes={})
    assert not res.accepted
    assert res.reason == 'NO_QUOTES'


def test_retry_acquisition_success_after_errors(monkeypatch):
    class FlakyAdapter:
        def __init__(self):
            self.calls = 0
        def get_orderbook_top(self, symbol):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("network glitch")
            class OB:  # minimal orderbook top stub
                bid=100.0; ask=100.05; bid_size=10; ask_size=10
            return OB()
    flaky = FlakyAdapter()
    adapters = {'flaky': flaky}
    res = execute_fast_order(symbol='BTCUSDT', side='LONG', size=1.0, price=100.0, settings=BASIC_SETTINGS, adapters=adapters)
    assert res.accepted
    assert res.venue == 'flaky'
    # ensure at least one retry happened (attempt_errors >=1)
    assert res.retries >= 1
