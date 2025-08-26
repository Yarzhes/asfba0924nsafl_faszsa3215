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
from typing import Dict, Optional, List, Any, Tuple

import math
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
    AlphaV2Features,
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
from ultra_signals.features.alpha_v2 import compute_alpha_v2_features, build_alpha_v2_model
from ultra_signals.features.flow_metrics import compute_flow_metrics
from ultra_signals.core.custom_types import FlowMetricsFeatures


def _safe_settings(d: dict, path: Tuple[str, ...], default: Any) -> Any:
    cur = d or {}
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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
            f"FeatureStore initialized (id={id(self)}). Storing up to {self._max_length} OHLCV bars."
        )

        # Raw data stores
        # _ohlcv_data[symbol][timeframe] -> DataFrame
        self._ohlcv_data: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
        self._latest_book_ticker: Dict[str, BookTickerEvent] = {}
        self._latest_mark_price: Dict[str, float] = {}
        self._recent_liquidations: Dict[str, list] = defaultdict(list)
        self._latest_depth: Dict[str, DepthEvent] = {}
        self._recent_trades: Dict[str, list] = defaultdict(list)

        # Feature state and cache
        self._feature_states: Dict[str, Dict[str, object]] = defaultdict(dict)

        # Cache layout (fixed): _feature_cache[symbol][timeframe][pd.Timestamp] -> dict of feature groups
        self._feature_cache: Dict[str, Dict[str, Dict[pd.Timestamp, Dict[str, object]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # Sprint 10 regime detector placeholder
        self._regime_detector = None

    # -------------------------------------------------------------------------
    # Helpers: normalize timestamps and provide robust feature lookups
    # -------------------------------------------------------------------------
    @staticmethod
    def _to_timestamp(ts_like: Any) -> Optional[pd.Timestamp]:
        """
        Coerce various inputs (Timestamp, str, int epoch s/ms/ns) to tz-naive pd.Timestamp.
        """
        if ts_like is None:
            return None
        try:
            if isinstance(ts_like, (int, np.integer)):
                v = int(ts_like)
                digits = int(math.log10(abs(v))) + 1 if v != 0 else 1
                if digits <= 10:
                    ts = pd.to_datetime(v, unit="s")
                elif digits <= 13:
                    ts = pd.to_datetime(v, unit="ms")
                else:
                    ts = pd.to_datetime(v, unit="ns")
            else:
                ts = pd.to_datetime(ts_like)
        except Exception:
            return None

        # force tz-naive
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
        except Exception:
            pass
        try:
            ts = ts.tz_localize(None)
        except Exception:
            pass
        return ts

    # NOTE: Backward/forward-compatible API
    # - get_features(symbol, ts_like)                        -> search across all tfs (nearest<=)
    # - get_features(symbol, timeframe, ts_like)             -> search within timeframe (nearest<=)
    def get_features(self, symbol: str, arg2: Any, arg3: Any = None, *, nearest: bool = True) -> Optional[Dict]:
        """
        Retrieves the dictionary of all computed features.

        Compatible call styles:
          1) get_features(symbol, ts_like)
          2) get_features(symbol, timeframe, ts_like)

        If exact match is missing, falls back to nearest earlier timestamp when `nearest=True`.
        """
        # Parse args
        if arg3 is None:
            timeframe: Optional[str] = None
            ts = self._to_timestamp(arg2)
        else:
            timeframe = str(arg2) if arg2 is not None else None
            ts = self._to_timestamp(arg3)

        if ts is None:
            logger.debug("[FeatureStore id={}] get_features: invalid timestamp input", id(self))
            return None

        # Pick the cache to search
        sym_cache = self._feature_cache.get(symbol, {})
        if not sym_cache:
            logger.debug("[FeatureStore id={}] get_features: symbol={} cache empty (no timeframes)", id(self), symbol)
            return None

        if timeframe:
            series = sym_cache.get(timeframe, {})
            if not series:
                logger.debug("[FeatureStore id={}] get_features: symbol={} tf={} cache empty", id(self), symbol, timeframe)
                return None
            # Direct / nearest search within timeframe
            return self._lookup_series(series, ts, symbol, timeframe, nearest)
        else:
            # Merge search across all timeframes for this symbol
            best = None
            best_key = None
            best_tf = None
            for tf, series in sym_cache.items():
                hit = self._lookup_series(series, ts, symbol, tf, nearest, quiet=True)
                if hit is None:
                    continue
                # Choose the hit whose timestamp is closest to requested (but <=)
                # Since _lookup_series returns dict only, we need the matched key as well.
                # Re-run to get the key cheaply:
                key = self._match_key(series, ts)
                if key is None:
                    continue
                if best_key is None or key > best_key:
                    best = hit
                    best_key = key
                    best_tf = tf
            if best is not None:
                logger.debug("[FeatureStore id={}] get_features: symbol={} merged-hit tf={} ts={}", id(self), symbol, best_tf, best_key)
                return best

            logger.debug("[FeatureStore id={}] get_features: symbol={} no features across any timeframe", id(self), symbol)
            return None

    def _match_key(self, series: Dict[pd.Timestamp, Dict], ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        if not series:
            return None
        if ts in series:
            return ts
        # nearest <= ts
        keys = series.keys()
        # Protect against non-sorted keys
        try:
            candidates = [k for k in keys if k <= ts]
            return max(candidates) if candidates else None
        except Exception:
            try:
                candidates = [pd.Timestamp(k) for k in keys if pd.Timestamp(k) <= ts]
                return max(candidates) if candidates else None
            except Exception:
                return None

    def _lookup_series(
        self,
        series: Dict[pd.Timestamp, Dict],
        ts: pd.Timestamp,
        symbol: str,
        timeframe: str,
        nearest: bool,
        quiet: bool = False,
    ) -> Optional[Dict]:
        if not series:
            if not quiet:
                logger.debug("[FeatureStore id={}] get_features: symbol={} tf={} series empty", id(self), symbol, timeframe)
            return None

        key = self._match_key(series, ts) if nearest else (ts if ts in series else None)
        if key is None:
            if not quiet:
                logger.debug(
                    "[FeatureStore id={}] get_features: symbol={} tf={} miss at {} (nearest={})",
                    id(self), symbol, timeframe, ts, nearest
                )
            return None
        return series.get(key)

    # Convenience helpers
    def get_features_at_or_before(self, symbol: str, timestamp: Any, timeframe: Optional[str] = None) -> Optional[Dict]:
        return self.get_features(symbol, timeframe, timestamp, nearest=True) if timeframe else self.get_features(symbol, timestamp, nearest=True)

    def get_features_nearest(self, symbol: str, timestamp: Any, timeframe: Optional[str] = None) -> Optional[Dict]:
        return self.get_features_at_or_before(symbol, timestamp, timeframe)

    def get_latest_features(self, symbol: str, timeframe: Optional[str] = None) -> Optional[Dict]:
        """Return the most recently cached feature dict for a symbol (optionally within a timeframe)."""
        sym_cache = self._feature_cache.get(symbol, {})
        if not sym_cache:
            return None
        if timeframe:
            series = sym_cache.get(timeframe, {})
            if not series:
                return None
            try:
                latest_key = max(series.keys())
                return series.get(latest_key)
            except Exception:
                return None
        # else, pick latest across all tfs
        best = None
        best_key = None
        for tf, series in sym_cache.items():
            if not series:
                continue
            try:
                k = max(series.keys())
            except Exception:
                continue
            if best_key is None or k > best_key:
                best_key = k
                best = series.get(k)
        return best

    # -------------------------------------------------------------------------

    def _get_state(self, symbol: str, state_key: str, state_class):
        """Initializes and returns a state object for a given symbol."""
        if state_key not in self._feature_states[symbol]:
            self._feature_states[symbol][state_key] = state_class()
        return self._feature_states[symbol][state_key]

    def on_bar(self, symbol: str, timeframe: str, bar: Any):
        """
        Ingest a single OHLCV bar (plus timestamp) for a symbol/timeframe.
        `bar` may be dict, Series, single-row DataFrame, or 1-D array/tuple/list.
        """
        # --- Normalize `bar` into a single-row DataFrame -------------------------
        if isinstance(bar, pd.DataFrame):
            if len(bar) == 0:
                return
            new_row = bar.reset_index(drop=True).iloc[:1].copy()
        elif isinstance(bar, pd.Series):
            new_row = bar.to_frame().T
        elif isinstance(bar, dict):
            new_row = pd.DataFrame([bar])
        else:
            arr = np.asarray(bar)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 1:
                raise ValueError(f"on_bar expects a single row; got shape {arr.shape}")
            new_row = pd.DataFrame(
                [arr],
                columns=["timestamp", "open", "high", "low", "close", "volume"][: arr.shape[0]],
            )

        # Ensure all expected columns exist
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            if col not in new_row.columns:
                new_row[col] = np.nan

        # Order columns & set index
        new_row = new_row[["timestamp", "open", "high", "low", "close", "volume"]]
        # Kline timestamps are usually ms; tolerate already-datetime values
        if not np.issubdtype(new_row["timestamp"].dtype, np.datetime64):
            new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], unit="ms", errors="coerce")
        else:
            new_row["timestamp"] = pd.to_datetime(new_row["timestamp"])
        new_row = new_row.set_index("timestamp")

        # --- Append/replace to internal store -----------------------------------
        df = self._ohlcv_data[symbol].get(timeframe)
        if df is None or df.empty:
            df = new_row
        else:
            if df.index[-1] == new_row.index[0]:
                df.iloc[-1] = new_row.iloc[0]  # update last in-place
            else:
                df = pd.concat([df, new_row])
            if len(df) > self._max_length:
                df = df.iloc[-self._max_length :]

        self._ohlcv_data[symbol][timeframe] = df

        # After ingesting, compute features for the new bar's timestamp
        self._compute_all_features(symbol, timeframe, new_row)

    # Unified event ingestion
    def ingest_event(self, event: MarketEvent) -> None:
        """
        Ingests various market events and updates internal stores.
        """
        try:
            if isinstance(event, KlineEvent):
                self.on_bar(
                    symbol=event.symbol,
                    timeframe=event.timeframe,
                    bar={
                        "timestamp": event.timestamp,
                        "open": event.open,
                        "high": event.high,
                        "low": event.low,
                        "close": event.close,
                        "volume": event.volume,
                    },
                )
            elif isinstance(event, BookTickerEvent):
                self._ingest_book_ticker(event)
            elif isinstance(event, MarkPriceEvent):
                self._ingest_mark_price(event)
            elif isinstance(event, ForceOrderEvent):
                self._ingest_force_order(event)
            elif isinstance(event, DepthEvent):
                self._ingest_depth(event)
            elif isinstance(event, AggTradeEvent):
                self._ingest_agg_trade(event)
        except Exception as e:
            logger.exception("Error while ingesting event {}: {}", type(event).__name__, e)

    def _compute_all_features(self, symbol: str, timeframe: str, bar: pd.DataFrame):
        """Computes and caches all component features for a given timestamp."""
        ohlcv = self.get_ohlcv(symbol, timeframe)
        warmup_need = _safe_settings(self._settings, ("features", "warmup_periods"), 1)
        if ohlcv is None or len(ohlcv) < warmup_need:
            return

        timestamp = bar.index[0]

        feature_config = self._settings.get("features", {})
        feature_dict: Dict[str, object] = {}

        # Trend
        try:
            trend_feats = compute_trend_features(ohlcv, **(feature_config.get("trend") or {}))
            if trend_feats:
                feature_dict["trend"] = TrendFeatures(**trend_feats)
        except Exception as e:
            logger.exception("trend feature error: {}", e)

        # Momentum
        try:
            momentum_feats = compute_momentum_features(ohlcv, **(feature_config.get("momentum") or {}))
            if momentum_feats:
                feature_dict["momentum"] = MomentumFeatures(**momentum_feats)
        except Exception as e:
            logger.exception("momentum feature error: {}", e)

        # Volatility
        try:
            vol_feats = compute_volatility_features(ohlcv, **(feature_config.get("volatility") or {}))
            if vol_feats:
                feature_dict["volatility"] = VolatilityFeatures(**vol_feats)
        except Exception as e:
            logger.exception("volatility feature error: {}", e)

        # Volume/Flow
        try:
            flow_feats = compute_volume_flow_features(ohlcv, **(feature_config.get("volume_flow") or {}))
            if flow_feats:
                feature_dict["volume_flow"] = VolumeFlowFeatures(**flow_feats)
        except Exception as e:
            logger.exception("volume_flow feature error: {}", e)

        # Alpha V2 (Sprint 11)
        try:
            alpha_feats = compute_alpha_v2_features(ohlcv, existing_features=feature_dict)
            if alpha_feats:
                feature_dict["alpha_v2"] = AlphaV2Features(**alpha_feats)
        except Exception as e:
            logger.exception("alpha_v2 feature error: {}", e)

        # Flow Metrics (Sprint 11 advanced flow pack)
        try:
            fm_cfg = (feature_config.get("flow_metrics") or {})
            if fm_cfg.get("enabled", True):
                # Gather recent context
                recent_trades = self.prune_and_get_trades(symbol, cutoff_ts=int(timestamp.value // 1_000_000) - 10*60*1000)
                recent_liqs = self.prune_and_get_liquidations(symbol, cutoff_ts=int(timestamp.value // 1_000_000) - 10*60*1000)
                book_top = self.get_book_top(symbol)
                state = self._feature_states.setdefault(symbol, {}).setdefault("flow_metrics_state", {})
                flow_raw = compute_flow_metrics(
                    ohlcv=ohlcv,
                    trades=recent_trades,
                    liquidations=recent_liqs,
                    book_top=book_top,
                    settings=self._settings,
                    state=state,
                )
                if flow_raw:
                    feature_dict["flow_metrics"] = FlowMetricsFeatures(**flow_raw)
                    try:
                        logger.debug("[FLOW_METRICS] {} {} ts={} -> {}", symbol, timeframe, int(timestamp.value // 1_000_000), flow_raw)
                    except Exception:
                        pass
        except Exception as e:
            logger.exception("flow_metrics feature error: {}", e)

        self._feature_cache[symbol][timeframe][timestamp] = feature_dict
        logger.debug(
            "Computed features (store id={}) for {} {} at {}: {}",
            id(self), symbol, timeframe, timestamp, feature_dict
        )
        # --- Sprint 10 Regime state (per-bar) ---
        try:
            if (self._settings.get("regime", {}) or {}).get("enabled", True):
                from ultra_signals.engine.regime import RegimeDetector
                if self._regime_detector is None:
                    self._regime_detector = RegimeDetector(self._settings)
                from ultra_signals.core.custom_types import FeatureVector
                fv = FeatureVector(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv={},
                    trend=feature_dict.get("trend"),
                    momentum=feature_dict.get("momentum"),
                    volatility=feature_dict.get("volatility"),
                    volume_flow=feature_dict.get("volume_flow"),
                )
                spread_bps = self.get_spread_bps(symbol)
                volume_z = None
                if feature_dict.get("volume_flow"):
                    volume_z = getattr(feature_dict["volume_flow"], "volume_z_score", None)
                rf = self._regime_detector.detect(fv, spread_bps=spread_bps, volume_z=volume_z)
                feature_dict["regime"] = rf
                cooldown = getattr(getattr(self._regime_detector, "_STATE", None), "cooldown_left", None)
                logger.debug(
                    "[REGIME] ts={} prim={}({:.2f}) vol={} liq={} cooldown={}",
                    int(timestamp.value // 1_000_000),
                    rf.profile.value,
                    rf.confidence,
                    rf.vol_state.value,
                    rf.liquidity.value,
                    cooldown,
                )
        except Exception as e:
            logger.exception("Regime compute error: {}", e)

    # ------------------- Latest market snapshots -------------------

    def _ingest_book_ticker(self, ticker: BookTickerEvent) -> None:
        self._latest_book_ticker[ticker.symbol] = ticker

    def _ingest_mark_price(self, mark_price: MarkPriceEvent) -> None:
        self._latest_mark_price[mark_price.symbol] = mark_price.mark_price

    def _ingest_force_order(self, event: ForceOrderEvent) -> None:
        notional = event.price * event.quantity
        self._recent_liquidations[event.symbol].append((event.timestamp, event.side, notional))

    def _ingest_depth(self, event: DepthEvent) -> None:
        self._latest_depth[event.symbol] = event

    def _ingest_agg_trade(self, event: AggTradeEvent) -> None:
        self._recent_trades[event.symbol].append((event.timestamp, event.price, event.quantity, event.is_buyer_maker))

    # ------------------- Public getters -------------------

    def get_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        return self._ohlcv_data.get(symbol, {}).get(timeframe)

    def get_spread(self, symbol: str) -> Optional[Tuple[float, float, float]]:
        if ticker := self._latest_book_ticker.get(symbol):
            if ticker.best_bid > 0 and ticker.best_ask > 0:
                return ticker.best_bid, ticker.best_ask, ticker.best_ask - ticker.best_bid
        return None

    def get_book_ticker(self, symbol: str) -> Optional[Tuple[float, float, float, float]]:
        if ticker := self._latest_book_ticker.get(symbol):
            return (ticker.best_bid, ticker.best_bid_qty, ticker.best_ask, ticker.best_ask_qty)
        return None

    def get_depth(self, symbol: str) -> Optional[DepthEvent]:
        return self._latest_depth.get(symbol)

    def get_recent_trades(self, symbol: str) -> list:
        return self._recent_trades.get(symbol, [])

    def get_mark_price(self, symbol: str) -> Optional[float]:
        return self._latest_mark_price.get(symbol)

    def get_latest_close(self, symbol: str, timeframe: str) -> Optional[float]:
        df = self.get_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    def get_warmup_status(self, symbol: str, timeframe: str) -> int:
        df = self.get_ohlcv(symbol, timeframe)
        return len(df) if df is not None else 0

    def get_recent_liquidations(self, symbol: str) -> list:
        return self._recent_liquidations.get(symbol, [])

    def prune_and_get_liquidations(self, symbol: str, cutoff_ts: int) -> list:
        recent_events = [event for event in self._recent_liquidations.get(symbol, []) if event[0] >= cutoff_ts]
        self._recent_liquidations[symbol] = recent_events
        return recent_events

    def prune_and_get_trades(self, symbol: str, cutoff_ts: int) -> list:
        recent_trades = [trade for trade in self._recent_trades.get(symbol, []) if trade[0] >= cutoff_ts]
        self._recent_trades[symbol] = recent_trades
        return recent_trades

    def get_funding_rate_history(self, symbol: str) -> Optional[List[Dict]]:
        if self._funding_provider:
            return self._funding_provider.get_history(symbol)
        return None

    # Legacy convenience wrappers to pull a sub-feature from the latest cached bar
    def get_vwap_features(self, symbol: str, timeframe: Optional[str] = None) -> Optional[dict]:
        feats = self.get_latest_features(symbol, timeframe)
        if feats and "volume_flow" in feats:
            vf = feats["volume_flow"]
            return {"vwap": getattr(vf, "vwap", None), "volume_z_score": getattr(vf, "volume_z_score", None)}
        return None

    def get_orderbook_features_v2(self, symbol: str, timeframe: Optional[str] = None) -> Optional[OrderbookFeaturesV2]:
        feats = self.get_latest_features(symbol, timeframe)
        return feats.get("orderbook_v2") if feats else None

    def get_cvd_features(self, symbol: str, timeframe: Optional[str] = None) -> Optional[CvdFeatures]:
        feats = self.get_latest_features(symbol, timeframe)
        return feats.get("cvd") if feats else None

    # =======================  SPRINT 9 ADDITIONS  =======================
    # (No removals from your original code. Everything below is additive.)

    # --- Simple scratchpad for auxiliary values (optional use by filters)
    def set_aux(self, symbol: str, key: str, value: Any) -> None:
        self._aux = getattr(self, "_aux", {})
        self._aux.setdefault(symbol, {})[key] = value

    def get_aux(self, symbol: str, key: str, default: Any = None) -> Any:
        return getattr(self, "_aux", {}).get(symbol, {}).get(key, default)

    # --- Latest timestamp (ms) for a symbol/timeframe (used by news/funding veto)
    def current_ts_ms(self, symbol: str, timeframe: str) -> Optional[int]:
        df = self.get_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            return None
        try:
            return int(pd.Timestamp(df.index[-1]).value // 1_000_000)  # ns → ms
        except Exception:
            return None

    # --- Return recent OHLCV rows as list of dicts (for CVD proxy, etc.)
    def get_recent_ohlcv(self, symbol: str, timeframe: str, bars: int = 200) -> List[dict]:
        out: List[dict] = []
        df = self.get_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            return out
        tail = df.tail(bars)
        for idx, row in tail.iterrows():
            out.append({
                "ts": int(pd.Timestamp(idx).value // 1_000_000),
                "open": float(row.get("open", np.nan)),
                "high": float(row.get("high", np.nan)),
                "low": float(row.get("low", np.nan)),
                "close": float(row.get("close", np.nan)),
                "volume": float(row.get("volume", np.nan)),
            })
        return out

    # --- Book top in a dict form the depth/thinness check expects
    def get_book_top(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Returns a dict with keys matching Sprint 9 helpers:
          bid, ask, B (bid_qty), A (ask_qty)
        """
        t = self._latest_book_ticker.get(symbol)
        if not t:
            return None
        return {
            "bid": float(t.best_bid),
            "ask": float(t.best_ask),
            "B": float(getattr(t, "best_bid_qty", 0.0)),
            "A": float(getattr(t, "best_ask_qty", 0.0)),
        }

    # --- Spread in basis points (bps) convenience
    def get_spread_bps(self, symbol: str) -> Optional[float]:
        sp = self.get_spread(symbol)
        if not sp:
            return None
        bid, ask, _ = sp
        mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
        if mid <= 0:
            return None
        return float((ask - bid) / mid * 10_000.0)

    # --- Safe getters from cached features (if present)
    def get_atr_percentile(self, symbol: str, timeframe: str) -> Optional[float]:
        feats = self.get_latest_features(symbol, timeframe)
        if not feats:
            return None
        vol = feats.get("volatility")
        # Try common attribute names
        for name in ("atr_percentile", "atr_pct", "atrp"):
            v = getattr(vol, name, None) if vol is not None else None
            if v is not None:
                return float(v)
        return None

    def get_adx(self, symbol: str, timeframe: str) -> Optional[float]:
        feats = self.get_latest_features(symbol, timeframe)
        if not feats:
            return None
        tr = feats.get("trend")
        v = getattr(tr, "adx", None) if tr is not None else None
        return float(v) if v is not None else None

    # --- Simple TR compression metric: mean((high-low)/close) over a small window
    def get_tr_compression(self, symbol: str, timeframe: str, window: Optional[int] = None) -> Optional[float]:
        df = self.get_ohlcv(symbol, timeframe)
        if df is None or len(df) < 5:
            return None
        if window is None:
            window = int(_safe_settings(self._settings, ("filters", "tr_compression_window"), 20))
        tail = df.tail(max(5, window)).copy()
        try:
            rng = (tail["high"] - tail["low"]).astype(float)
            comp = (rng / tail["close"].replace(0, np.nan).astype(float)).mean()
            return float(comp) if pd.notna(comp) else None
        except Exception:
            return None

    # --- Coarse regime helper for HTF confluence
    def get_regime(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Tries to read a trend regime from cached trend features.
        Falls back to a simple ADX + price slope heuristic.
        Returns one of: 'trend_up', 'trend_down', 'mixed', or None.
        """
        feats = self.get_latest_features(symbol, timeframe)
        if not feats:
            return None
        # Prefer new Sprint 10 regime if present
        reg_obj = feats.get("regime")
        if reg_obj is not None:
            try:
                prim = getattr(reg_obj, "profile", None)
                if prim is not None:
                    return prim.value
            except Exception:
                pass
        tr = feats.get("trend")
        # Preferred: explicit regime if your TrendFeatures defines it
        for name in ("regime", "trend_regime"):
            reg = getattr(tr, name, None) if tr is not None else None
            if isinstance(reg, str):
                return reg

        # Fallback heuristic
        adx = getattr(tr, "adx", None) if tr is not None else None
        df = self.get_ohlcv(symbol, timeframe)
        if adx is None or df is None or len(df) < 10:
            return None
        closes = df["close"].astype(float).tail(20).values
        if len(closes) < 3:
            return "mixed"
        # sign of simple slope
        x = np.arange(len(closes))
        denom = ( (x - x.mean())**2 ).sum() or 1.0
        slope = float(((x - x.mean()) * (closes - closes.mean())).sum() / denom)
        if adx >= 20 and slope > 0:
            return "trend_up"
        if adx >= 20 and slope < 0:
            return "trend_down"
        return "mixed"

    # --- Funding provider bridge used by filters (minutes to next funding)
    def get_minutes_to_next_funding(self, symbol: str, now_ms: Optional[int]) -> Optional[int]:
        if not self._funding_provider or now_ms is None:
            return None
        try:
            return int(self._funding_provider.minutes_to_next(symbol, now_ms))
        except Exception:
            # Keep FeatureStore resilient even if provider glitches
            return None

    def get_regime_state(self, symbol: str, timeframe: str):
        feats = self.get_latest_features(symbol, timeframe)
        if feats and "regime" in feats:
            return feats["regime"]
        return None


# --- Example Usage ---
if __name__ == "__main__":
    # A simple demonstration of the FeatureStore's functionality.
    mock_settings = {
        'features': {
            'warmup_periods': 50,
            'trend': {'ema_short': 12, 'ema_medium': 26, 'ema_long': 50, 'adx_period': 14},
            'momentum': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
            'volatility': {'atr_period': 14, 'bbands_period': 20, 'bbands_std': 2.0},
            'volume_flow': {'vwap_window': 20, 'volume_z_window': 50},
        }
    }
    store = FeatureStore(warmup_periods=50, settings=mock_settings)

    # 1. Create some dummy events
    kline1 = KlineEvent(event_type="kline", timestamp=1672531200000, symbol="BTCUSDT", timeframe="1m",
                        open=20000, high=20010, low=19990, close=20005, volume=100, closed=False)
    kline1_update = KlineEvent(event_type="kline", timestamp=1672531200000, symbol="BTCUSDT", timeframe="1m",
                               open=20000, high=20015, low=19990, close=20012, volume=150, closed=True)

    # 2. Ingest events, which also triggers feature computation
    store.ingest_event(kline1)
    store.ingest_event(kline1_update)

    ts_exact = pd.to_datetime(1672531200000, unit="ms")
    # both calls work:
    _a = store.get_features("BTCUSDT", "1m", ts_exact)
    _b = store.get_features("BTCUSDT", ts_exact)
    logger.info("Feature hit with timeframe: {}", _a is not None)
    logger.info("Feature hit legacy form:   {}", _b is not None)
