"""
Feature Store for In-Memory Time-Series Data

This module provides the `FeatureStore`, a class responsible for managing
real-time market data (OHLCV) in memory. It serves as the single source of
truth for time-series data needed by the feature computation modules.

Design Principles:
- Centralized State: The FeatureStore holds all raw kline data, preventing
  data duplication and inconsistent state across the application.
- Efficient Storage: Uses Pandas DataFrames for efficient storage and fast
  slicing of time-series data.
- Simple API: Provides a clean, straightforward interface for ingesting new
  data (`ingest`) and retrieving historical data (`get_ohlcv`, `get_latest`).
- Decoupling: Decouples the data source (e.g., BinanceWSClient) from the
  consumers of the data (feature calculation functions).
"""

from collections import defaultdict
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from loguru import logger

from ultra_signals.core.events import (
    BookTickerEvent,
    DepthEvent,
    KlineEvent,
    MarkPriceEvent,
    MarketEvent,
    ForceOrderEvent,
    AggTradeEvent,
)
from ultra_signals.data.funding_provider import FundingProvider
from ultra_signals.core.custom_types import (
    TrendFeatures,
    MomentumFeatures,
    VolatilityFeatures,
    VolumeFlowFeatures,
)
from ultra_signals.features.cvd import CvdFeatures, CvdState, compute_cvd_features
from ultra_signals.features.momentum import compute_momentum_features
from ultra_signals.features.orderbook import (
    BookFlipState,
    OrderbookFeaturesV2,
    compute_orderbook_features_v2,
)
from ultra_signals.features.trend import compute_trend_features
from ultra_signals.features.volatility import compute_volatility_features
from ultra_signals.features.volume_flow import compute_volume_flow_features


class FeatureStore:
    """
    Manages rolling time-series data (OHLCV, order book, etc.) for multiple
    symbols and timeframes, serving as the single source of truth for calculations.
    It also computes and caches complex features.
    """

    def __init__(
        self, warmup_periods: int, funding_provider: Optional[FundingProvider] = None, settings: dict = None
    ):
        """
        Initializes the FeatureStore.

        Args:
            warmup_periods: The minimum number of data points to store per series.
            funding_provider: An optional instance of FundingProvider.
            settings: Application settings dictionary for feature configurations.
        """
        if warmup_periods <= 1:
            raise ValueError("'warmup_periods' must be greater than 1.")

        self._max_length = warmup_periods * 2
        self._funding_provider = funding_provider
        self._settings = settings or {}
        logger.info(
            f"FeatureStore initialized. Storing up to {self._max_length} OHLCV bars."
        )

        # Raw data stores
        self._ohlcv_data: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
        self._latest_book_ticker: Dict[str, BookTickerEvent] = {}
        self._latest_mark_price: Dict[str, float] = {}
        self._recent_liquidations: Dict[str, list] = defaultdict(list)
        self._latest_depth: Dict[str, DepthEvent] = {}
        self._recent_trades: Dict[str, list] = defaultdict(list)

        # Feature state and cache
        self._feature_states: Dict[str, Dict[str, object]] = defaultdict(dict)
        self._feature_cache: Dict[str, Dict[str, object]] = defaultdict(dict)

    def _get_state(self, symbol: str, state_key: str, state_class):
        """Initializes and returns a state object for a given symbol."""
        if state_key not in self._feature_states[symbol]:
            self._feature_states[symbol][state_key] = state_class()
        return self._feature_states[symbol][state_key]

    def on_bar(self, symbol: str, timeframe: str, bar: any):
        """
        Ingest a single OHLCV bar (plus timestamp) for a symbol/timeframe.
        `bar` may be dict, Series, single-row DataFrame, or 1-D array/tuple/list.
        """
        # --- Normalize `bar` into a single-row DataFrame -------------------------
        if isinstance(bar, pd.DataFrame):
            # Ensure exactly one row
            if len(bar) == 0:
                return
            new_row = bar.reset_index(drop=True).iloc[:1].copy()
        elif isinstance(bar, pd.Series):
            new_row = bar.to_frame().T
        elif isinstance(bar, dict):
            new_row = pd.DataFrame([bar])
        else:
            arr = np.asarray(bar)
            # Collapse shapes like (1, 6) → (6,)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 1:
                raise ValueError(f"on_bar expects a single row; got shape {arr.shape}")
            # If columns aren’t labeled, assume standard OHLCV ordering
            new_row = pd.DataFrame([arr], columns=["timestamp", "open", "high", "low", "close", "volume"][:arr.shape[0]])

        # Ensure all expected columns exist
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            if col not in new_row.columns:
                new_row[col] = np.nan

        # Order columns & set index
        new_row = new_row[["timestamp", "open", "high", "low", "close", "volume"]]
        new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], unit="ms", errors="coerce")
        new_row = new_row.set_index("timestamp")

        # --- existing logic to append to internal store follows ------------------
        df = self._ohlcv_data[symbol].get(timeframe)

        if df is None:
            df = new_row
        else:
            if not df.empty and df.index[-1] == new_row.index[0]:
                df.iloc[-1] = new_row.iloc[0]
            else:
                df = pd.concat([df, new_row])

            if len(df) > self._max_length:
                df = df.iloc[-self._max_length :]
        
        self._ohlcv_data[symbol][timeframe] = df
        
        # After ingesting, compute features for the new bar's timestamp
        self._compute_all_features(symbol, timeframe, new_row)


    def _compute_all_features(self, symbol: str, timeframe: str, bar: pd.DataFrame):
        """Computes and caches all component features for a given timestamp."""
        ohlcv = self.get_ohlcv(symbol, timeframe)
        if ohlcv is None or len(ohlcv) < self._settings["features"]["warmup_periods"]:
            return

        timestamp = bar.index[0]

        feature_config = self._settings["features"]
        feature_dict = {}

        # Trend
        trend_feats = compute_trend_features(ohlcv, **feature_config["trend"])
        if trend_feats:
            feature_dict["trend"] = TrendFeatures(**trend_feats)

        # Momentum
        momentum_feats = compute_momentum_features(ohlcv, **feature_config["momentum"])
        if momentum_feats:
            feature_dict["momentum"] = MomentumFeatures(**momentum_feats)

        # Volatility
        vol_feats = compute_volatility_features(ohlcv, **feature_config["volatility"])
        if vol_feats:
            feature_dict["volatility"] = VolatilityFeatures(**vol_feats)
        
        # Volume/Flow
        flow_feats = compute_volume_flow_features(ohlcv, **feature_config["volume_flow"])
        if flow_feats:
            feature_dict["volume_flow"] = VolumeFlowFeatures(**flow_feats)

        self._feature_cache[symbol][timestamp] = feature_dict
        logger.debug(f"Computed features for {symbol} at {timestamp}: {feature_dict}")
            
    def get_features(self, symbol: str, timestamp: pd.Timestamp) -> Optional[Dict]:
        """Retrieves the dictionary of all computed features for a specific timestamp."""
        return self._feature_cache.get(symbol, {}).get(timestamp)
    
    def _ingest_book_ticker(self, ticker: BookTickerEvent) -> None:
        """Updates the latest book ticker for a symbol."""
        self._latest_book_ticker[ticker.symbol] = ticker

    def _ingest_mark_price(self, mark_price: MarkPriceEvent) -> None:
        """Updates the latest mark price for a symbol."""
        self._latest_mark_price[mark_price.symbol] = mark_price.mark_price

    def _ingest_force_order(self, event: ForceOrderEvent) -> None:
        """Adds a liquidation event to the recent liquidations list."""
        notional = event.price * event.quantity
        self._recent_liquidations[event.symbol].append(
            (event.timestamp, event.side, notional)
        )

    def _ingest_depth(self, event: DepthEvent) -> None:
        """Updates the latest order book depth for a symbol."""
        self._latest_depth[event.symbol] = event

    def _ingest_agg_trade(self, event: AggTradeEvent) -> None:
        """Adds an aggregated trade to the recent trades list."""
        self._recent_trades[event.symbol].append(
            (event.timestamp, event.price, event.quantity, event.is_buyer_maker)
        )

    def get_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """
        Retrieves the entire OHLCV DataFrame for a specific symbol and timeframe.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT").
            timeframe: The kline timeframe (e.g., "5m").

        Returns:
            A Pandas DataFrame containing the OHLCV data, or None if no data
            is available for the requested series.
        """
        return self._ohlcv_data.get(symbol, {}).get(timeframe)

    def get_spread(self, symbol: str) -> tuple[float, float, float] | None:
        """
        Calculates the spread and returns bid, ask, and spread.

        Returns:
            A tuple of (best_bid, best_ask, spread), or None if data isn't available.
        """
        if ticker := self._latest_book_ticker.get(symbol):
            if ticker.best_bid > 0 and ticker.best_ask > 0:
                return ticker.best_bid, ticker.best_ask, ticker.best_ask - ticker.best_bid
        return None

    def get_book_ticker(self, symbol: str) -> tuple[float, float, float, float] | None:
        """
        Retrieves the latest full book ticker data (bid, bid_qty, ask, ask_qty).

        Returns:
            A tuple of (best_bid, best_bid_qty, best_ask, best_ask_qty), or None if not available.
        """
        if ticker := self._latest_book_ticker.get(symbol):
            return (ticker.best_bid, ticker.best_bid_qty, ticker.best_ask, ticker.best_ask_qty)
        return None

    def get_depth(self, symbol: str) -> DepthEvent | None:
        """Retrieves the most recent order book depth for a symbol."""
        return self._latest_depth.get(symbol)

    def get_recent_trades(self, symbol: str) -> list:
        """Returns the list of recent trade events for a symbol."""
        return self._recent_trades.get(symbol, [])

    def get_mark_price(self, symbol: str) -> float | None:
        """Retrieves the most recent mark price for a symbol."""
        return self._latest_mark_price.get(symbol)

    def get_latest_close(self, symbol: str, timeframe: str) -> float | None:
        """
        Retrieves the most recent close price for a symbol and timeframe.

        Args:
            symbol: The trading symbol.
            timeframe: The kline timeframe.

        Returns:
            The latest close price as a float, or None if not available.
        """
        df = self.get_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            return df["close"].iloc[-1]
        return None

    def get_warmup_status(self, symbol: str, timeframe: str) -> int:
        """
        Checks how many data points are available for a given series.

        Args:
            symbol: The trading symbol.
            timeframe: The kline timeframe.

        Returns:
            The number of available bars for the series.
        """
        df = self.get_ohlcv(symbol, timeframe)
        return len(df) if df is not None else 0

    def get_recent_liquidations(self, symbol: str) -> list:
        """Returns the list of recent liquidation events for a symbol."""
        return self._recent_liquidations.get(symbol, [])

    def prune_and_get_liquidations(self, symbol: str, cutoff_ts: int) -> list:
        """Prunes old liquidations and returns the remaining recent ones."""
        
        # Filter out old events, keeping only those more recent than the cutoff
        recent_events = [
            event for event in self._recent_liquidations.get(symbol, [])
            if event[0] >= cutoff_ts
        ]
        
        self._recent_liquidations[symbol] = recent_events
        return recent_events

    def prune_and_get_trades(self, symbol: str, cutoff_ts: int) -> list:
        """Prunes old trades and returns the remaining recent ones."""
        recent_trades = [
            trade for trade in self._recent_trades.get(symbol, [])
            if trade[0] >= cutoff_ts
        ]
        self._recent_trades[symbol] = recent_trades
        return recent_trades

    def get_funding_rate_history(self, symbol: str) -> Optional[List[Dict]]:
        """
        Retrieves the historical funding rate data from the funding provider.

        Args:
            symbol: The trading symbol.

        Returns:
            A list of historical funding rate data points, or None if not available.
        """
        if self._funding_provider:
            return self._funding_provider.get_history(symbol)
        return None


    def get_vwap_features(self, symbol: str) -> Optional[dict]:
        """Retrieves cached VWAP features for a symbol."""
        return self._feature_cache.get(symbol, {}).get('vwap')

    def get_orderbook_features_v2(self, symbol: str) -> Optional[OrderbookFeaturesV2]:
        """Retrieves cached Orderbook v2 features for a symbol."""
        return self._feature_cache.get(symbol, {}).get('orderbook_v2')

    def get_cvd_features(self, symbol: str) -> Optional[CvdFeatures]:
        """Retrieves cached CVD features for a symbol."""
        return self._feature_cache.get(symbol, {}).get('cvd')


# --- Example Usage ---
if __name__ == "__main__":
    # A simple demonstration of the FeatureStore's functionality.
    # Note: The settings dictionary would typically be loaded from a config file.
    mock_settings = {
        'features': {
            'vwap': {'vwap_window': 20, 'vwap_std_devs': (1.0, 2.0)},
            'cvd': {'cvd_lookback_seconds': 300, 'cvd_slope_period': 20},
            'orderbook_v2': {
                'depth_levels_N': 10,
                'book_flip_min_delta': 0.15,
                'book_flip_persistence_ticks': 5,
                'slippage_trade_size': 1000
            }
        }
    }
    store = FeatureStore(warmup_periods=50, settings=mock_settings)

    # 1. Create some dummy events
    kline1 = KlineEvent(event_type="kline", timestamp=1672531200000, symbol="BTCUSDT", timeframe="1m", open=20000, high=20010, low=19990, close=20005, volume=100, closed=False)
    kline1_update = KlineEvent(event_type="kline", timestamp=1672531200000, symbol="BTCUSDT", timeframe="1m", open=20000, high=20015, low=19990, close=20012, volume=150, closed=True)
    depth1 = DepthEvent(event_type="depthUpdate", timestamp=1672531200500, symbol="BTCUSDT", bids=[(20004.5, 10), (20004.0, 15)], asks=[(20005.0, 12), (20005.5, 18)])
    trade1 = AggTradeEvent(event_type="aggTrade", timestamp=1672531200600, symbol="BTCUSDT", price=20005.0, quantity=0.5, is_buyer_maker=False)

    # 2. Ingest events, which now also triggers feature computation
    store.ingest_event(kline1)
    store.ingest_event(kline1_update)
    store.ingest_event(depth1)
    store.ingest_event(trade1)
    
    logger.info("Ingested events and computed features:")

    # 3. Check feature accessors
    vwap_feats = store.get_vwap_features("BTCUSDT")
    ob_feats = store.get_orderbook_features_v2("BTCUSDT")
    cvd_feats = store.get_cvd_features("BTCUSDT")

    if vwap_feats:
        logger.info(f"\nVWAP Features for BTCUSDT: VWAP = {vwap_feats.get('vwap')}")
    if ob_feats:
        logger.info(f"Orderbook V2 Features for BTCUSDT: Imbalance = {ob_feats.imbalance_ratio:.4f}")
    if cvd_feats:
        logger.info(f"CVD Features for BTCUSDT: Slope = {cvd_feats.cvd_slope}")