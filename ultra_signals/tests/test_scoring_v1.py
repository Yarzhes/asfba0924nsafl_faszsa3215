"""
Tests for the v1 scoring and signal generation logic.
"""

import pandas as pd
import pytest

from ultra_signals.engine.scoring import component_scores
from ultra_signals.engine.entries_exits import make_signal
from ultra_signals.engine.risk_filters import apply_filters
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.events import BookTickerEvent, KlineEvent
from ultra_signals.core.custom_types import FeatureVector, Signal


# --- Test Component Scoring ---

def test_trend_score_perfectly_bullish():
    """Trend score should be +1.0 for perfectly aligned bullish EMAs."""
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"ema_10": 102, "ema_20": 101, "ema_50": 100},
        orderbook={},
        derivatives={},
        funding={},
    )
    params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14},
        "volatility": {}
    }
    scores = component_scores(feature_vector, params)
    assert scores["trend"] == 1.0


def test_trend_score_perfectly_bearish():
    """Trend score should be -1.0 for perfectly aligned bearish EMAs."""
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"ema_10": 100, "ema_20": 101, "ema_50": 102},
        orderbook={},
        derivatives={},
        funding={},
    )
    params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14},
        "volatility": {}
    }
    scores = component_scores(feature_vector, params)
    assert scores["trend"] == -1.0


def test_trend_score_mixed():
    """Trend score should be between -1 and 1 for mixed EMAs."""
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"ema_10": 102, "ema_20": 100, "ema_50": 101},
        orderbook={},
        derivatives={},
        funding={},
    )
    params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14},
        "volatility": {}
    }
    scores = component_scores(feature_vector, params)
    assert -1.0 < scores["trend"] < 1.0


def test_momentum_score_neutral():
    """Momentum score should be near 0 for neutral RSI and MACD."""
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"rsi_14": 50.0, "macd_hist": 0.0},
        orderbook={},
        derivatives={},
        funding={},
    )
    params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14},
        "volatility": {}
    }
    scores = component_scores(feature_vector, params)
    assert scores["momentum"] == 0.0


def test_momentum_score_bullish():
    """Momentum score should be positive for bullish RSI and MACD."""
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"rsi_14": 65.0, "macd_hist": 0.1},
        orderbook={},
        derivatives={},
        funding={},
    )
    params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14},
        "volatility": {}
    }
    scores = component_scores(feature_vector, params)
    assert scores["momentum"] > 0


def test_momentum_score_bearish_disagreement():
    """Momentum score should be weak if RSI and MACD disagree."""
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"rsi_14": 35.0, "macd_hist": 0.1},
        orderbook={},
        derivatives={},
        funding={},
    )
    params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14},
        "volatility": {}
    }
    scores = component_scores(feature_vector, params)
    assert scores["momentum"] < 0 # RSI drives the sign
    assert abs(scores["momentum"]) < 0.5 # But the score should be weak


# --- Test Signal Generation and Filtering ---

@pytest.fixture
def dummy_ohlcv() -> pd.DataFrame:
    """A minimal OHLCV DataFrame for signal generation tests."""
    data = {'close': [100, 101, 102], 'high': [101, 102, 103], 'low': [99, 100, 101]}
    return pd.DataFrame(data)

def test_make_signal_long(settings_fixture, dummy_ohlcv):
    """Test that a high score produces a LONG signal."""
    # Strong bullish scores
    scores = {"trend": 0.9, "momentum": 0.8}
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"atr_14": 1.0},
        orderbook={},
        derivatives={},
        funding={},
    )
    
    signal = make_signal(
        symbol="BTCUSDT",
        timeframe="5m",
        component_scores=scores,
        weights=settings_fixture.engine.scoring_weights,
        thresholds=settings_fixture.engine.thresholds,
        features=feature_vector,
        ohlcv=dummy_ohlcv
    )

    assert signal.decision == "LONG"
    assert signal.score > settings_fixture.engine.thresholds.enter
    assert signal.entry_price == 102 # Last close
    assert signal.stop_loss < signal.entry_price
    assert signal.take_profit_1 > signal.entry_price


def test_make_signal_short(settings_fixture, dummy_ohlcv):
    """Test that a low score produces a SHORT signal."""
    scores = {"trend": -0.9, "momentum": -0.8}
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"atr_14": 1.0},
        orderbook={},
        derivatives={},
        funding={},
    )
    
    signal = make_signal("BTCUSDT", "5m", scores,
                         settings_fixture.engine.scoring_weights,
                         settings_fixture.engine.thresholds,
                         feature_vector, dummy_ohlcv)

    assert signal.decision == "SHORT"
    assert signal.score < -settings_fixture.engine.thresholds.enter
    assert signal.stop_loss > signal.entry_price
    assert signal.take_profit_1 < signal.entry_price


def test_make_signal_no_trade(settings_fixture, dummy_ohlcv):
    """Test that a neutral score produces NO_TRADE."""
    scores = {"trend": 0.1, "momentum": -0.2} # Weak, conflicting scores
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        ohlcv={"atr_14": 1.0},
        orderbook={},
        derivatives={},
        funding={},
    )

    signal = make_signal("BTCUSDT", "5m", scores,
                         settings_fixture.engine.scoring_weights,
                         settings_fixture.engine.thresholds,
                         feature_vector, dummy_ohlcv)
                         
    assert signal.decision == "NO_TRADE"


def test_risk_filter_warmup_pass(settings_fixture):
    """Test that the warmup filter passes if there is enough data."""
    store = FeatureStore(warmup_periods=settings_fixture.features.warmup_periods, settings=settings_fixture.model_dump())
    # Simulate having enough data by ingesting KlineEvent objects
    for i in range(settings_fixture.features.warmup_periods):
        kline = KlineEvent(
            timestamp=i * 60000,
            symbol="BTCUSDT",
            timeframe="5m",
            open=100 + i, high=101 + i, low=99 + i, close=100 + i,
            volume=10, closed=True
        )
        store.on_bar(kline.symbol, kline.timeframe, kline.model_dump())

    # Also ingest a book ticker event so spread filter doesn't fail
    store._ingest_book_ticker(
        BookTickerEvent(
            timestamp=1,
            symbol="BTCUSDT",
            b=100,      # best_bid
            B=10,       # best_bid_qty
            a=100.05,   # best_ask (0.05% spread)
            A=10,       # best_ask_qty
        )
    )

    dummy_signal = Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type="breakout",
        price=0,
        score=0,
    )
    settings_dict = settings_fixture.model_dump()

    result = apply_filters(dummy_signal, store, settings_dict)

    assert result.passed, f"Risk filter failed unexpectedly with reason: {result.reason}"


def test_risk_filter_warmup_block(settings_fixture):
    """Test that the warmup filter blocks if there is not enough data."""
    store = FeatureStore(warmup_periods=50) # Requires 50 bars
    
    # The store is empty, so it should block
    # We need a dummy signal object
    dummy_signal = Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type="breakout",
        price=0,
        score=0,
    )
    
    settings_dict = settings_fixture.model_dump()
    
    result = apply_filters(dummy_signal, store, settings_dict)

    assert not result.passed