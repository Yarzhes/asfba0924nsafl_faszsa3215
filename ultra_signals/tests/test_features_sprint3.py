"""
Tests for Sprint 3 features: orderbook and derivatives.
"""

import pytest

import time

from ultra_signals.core.events import BookTickerEvent, ForceOrderEvent
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.features.orderbook import OrderbookFeatures, compute_orderbook_features
from ultra_signals.features.derivatives import DerivativesFeatures, compute_derivatives_features


def test_orderbook_features_init():
    """
    Test initialization of OrderbookFeatures.
    """
    of = OrderbookFeatures()
    assert of.imbalance == 0.0
    assert of.spread == 0.0


def test_compute_orderbook_features():
    """
    Test the computation of orderbook features.
    """
    store = FeatureStore(warmup_periods=10)
    
    # Ingest a sample book ticker event
    ticker_event = BookTickerEvent(
        timestamp=1,
        symbol="BTCUSDT",
        b=100.0,      # best_bid
        B=10.0,       # best_bid_qty
        a=101.0,      # best_ask
        A=5.0,        # best_ask_qty
    )
    store._ingest_book_ticker(ticker_event)

    # Compute features
    features = compute_orderbook_features(store, "BTCUSDT")

    assert features is not None
    assert features.spread == 1.0  # 101 - 100
    assert features.bid_sum_top == 1000.0  # 100 * 10
    assert features.ask_sum_top == 505.0  # 101 * 5
    # imbalance = bid_notional / ask_notional
    assert features.imbalance == 1000.0 / 505.0
    
    mid_price = (100.0 + 101.0) / 2 # 100.5
    # slip_est = (ask - mid_price) / mid_price
    assert features.slip_est == (101.0 - 100.5) / 100.5


def test_derivatives_features_init():
    """
    Test initialization of DerivativesFeatures.
    """
    df = DerivativesFeatures()
    assert df.liq_pulse == 0


def test_compute_derivatives_features_pulse():
    """
    Test the computation of liquidation pulse with synthetic events.
    """
    store = FeatureStore(warmup_periods=10)
    now = int(time.time() * 1000)

    # More short liquidations (market buys) -> bullish pulse
    events_bullish = [
        ForceOrderEvent(timestamp=now - 2000, symbol="BTCUSDT", side="BUY", price=100, quantity=10),
        ForceOrderEvent(timestamp=now - 1000, symbol="BTCUSDT", side="SELL", price=100, quantity=2),
    ]
    for event in events_bullish:
        store._ingest_force_order(event)
    
    features_bullish = compute_derivatives_features(store, "BTCUSDT")
    assert features_bullish.liq_pulse == 1
    assert features_bullish.buy_liq_notional_5m == 1000
    assert features_bullish.sell_liq_notional_5m == 200

    # --- Reset and test bearish ----
    store = FeatureStore(warmup_periods=10)
    # More long liquidations (market sells) -> bearish pulse
    events_bearish = [
        ForceOrderEvent(timestamp=now - 2000, symbol="BTCUSDT", side="BUY", price=100, quantity=2),
        ForceOrderEvent(timestamp=now - 1000, symbol="BTCUSDT", side="SELL", price=100, quantity=10),
    ]
    for event in events_bearish:
        store._ingest_force_order(event)

    features_bearish = compute_derivatives_features(store, "BTCUSDT")
    assert features_bearish.liq_pulse == -1

    # --- Reset and test timeout ---
    store = FeatureStore(warmup_periods=10)
    # Event is too old
    event_old = ForceOrderEvent(timestamp=now - (6 * 60 * 1000), symbol="BTCUSDT", side="SELL", price=100, quantity=10)
    store._ingest_force_order(event_old)

    features_old = compute_derivatives_features(store, "BTCUSDT", timeframe_ms=5*60*1000)
    assert features_old.liq_pulse == 0
    assert features_old.sell_liq_notional_5m == 0