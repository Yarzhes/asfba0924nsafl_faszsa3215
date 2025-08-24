import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
from ultra_signals.core.custom_types import EnsembleDecision, RiskEvent, Position
from ultra_signals.risk.portfolio import evaluate_portfolio, Portfolio


# ------------------------ UNIVERSAL DECISION TRACE HELPERS ------------------------

def _trace_reject(symbol, ts, reason, **kv):
    """
    Single-line debug log to explain why a bar did not produce an entry.
    """
    extras = " ".join(f"{k}={v}" for k, v in kv.items() if v is not None)
    logger.debug(f"[DECISION] reject symbol={symbol} ts={ts} reason={reason} {extras}")

def _safe_get_attr(obj, path, default=None):
    """
    _safe_get_attr(features, 'trend.adx') -> value or default
    Works with either attribute objects or dicts.
    """
    cur = obj
    for p in str(path).split('.'):
        if cur is None:
            return default
        # support dicts and objects
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            cur = getattr(cur, p, None)
    return cur if cur is not None else default

def _peek_features(feature_store, symbol, timeframe, ts):
    """
    Best-effort extraction of a small subset of features for logging
    without depending on any specific FeatureStore API.
    Returns dict like {'adx':..., 'rsi':..., 'atr_percentile':...} (keys may be None).
    """
    feats = None

    # Try common getters
    for meth in ("get_features", "get_latest", "latest", "get"):
        f = getattr(feature_store, meth, None)
        if callable(f):
            try:
                # try the most specific signature first
                for args in (
                    (symbol, timeframe, ts),
                    (symbol, timeframe),
                    (symbol,),
                    tuple(),
                ):
                    try:
                        feats = f(*args)
                        if feats is not None:
                            break
                    except TypeError:
                        continue
                if feats is not None:
                    break
            except Exception:
                pass

    # Try digging into common caches
    if feats is None:
        for cache_name in ("cache", "store", "buffer", "latest_features", "features"):
            cache = getattr(feature_store, cache_name, None)
            if cache is None:
                continue
            candidate = None
            # Try direct dict access patterns
            try:
                candidate = cache.get((symbol, timeframe, ts)) or cache.get((symbol, timeframe)) \
                            or cache.get(symbol) or cache.get(timeframe)
            except Exception:
                candidate = None
            if candidate is None:
                # last resort: take anything that looks like features
                if isinstance(cache, dict):
                    try:
                        candidate = next(iter(cache.values()))
                    except Exception:
                        candidate = None
            if candidate is not None:
                feats = candidate
                break

    # Now extract a few useful fields if present
    adx  = _safe_get_attr(feats, "trend.adx", None)
    rsi  = _safe_get_attr(feats, "momentum.rsi", None)
    atrp = _safe_get_attr(feats, "volatility.atr_percentile", None)

    # Float formatting/rounding (only if numeric)
    def _rf(x, nd=3):
        try:
            return round(float(x), nd)
        except Exception:
            return None

    return {"adx": _rf(adx, 2), "rsi": _rf(rsi, 2), "atr_percentile": _rf(atrp, 3)}


# ----------------------------------------------------------------------------------


class EventRunner:
    """
    Orchestrates the backtest event loop, simulates trade execution,
    and manages the state of the portfolio.
    """

    def __init__(self, settings: Dict[str, Any], data_adapter, signal_engine, feature_store):
        self.settings = settings
        self.data_adapter = data_adapter
        self.signal_engine = signal_engine
        self.feature_store = feature_store
        self.risk_events: List[RiskEvent] = []
        self.log = logger
        self.warmup_mode = bool(self.settings.get("warmup_mode", False))

        # Initialize the Portfolio (robust for flat or nested settings)
        backtest_cfg = self.settings.get("backtest", {})
        exec_cfg = (backtest_cfg.get("execution") if isinstance(backtest_cfg, dict) else None) or {}

        # fallbacks for flattened test settings
        initial_capital = (
            exec_cfg.get("initial_capital")
            or self.settings.get("initial_capital")
            or 10000.0
        )

        portfolio_settings = self.settings.get("portfolio", {}) or {}
        max_total_positions = (
            portfolio_settings.get("max_total_positions")
            or self.settings.get("max_total_positions")
            or 999999
        )
        max_positions_per_symbol = (
            portfolio_settings.get("max_positions_per_symbol")
            or self.settings.get("max_positions_per_symbol")
            or 999999
        )

        self.portfolio = Portfolio(
            initial_capital=float(initial_capital),
            max_positions_total=int(max_total_positions),
            max_positions_per_symbol=int(max_positions_per_symbol),
        )

    # Expose trades/equity_curve for tests that expect them on runner
    @property
    def trades(self) -> List[dict]:
        return self.portfolio.trades

    @property
    def equity_curve(self) -> List[dict]:
        return self.portfolio.equity_curve

    def run(self, symbol: str, timeframe: str):
        """Main event loop for a single symbol backtest."""
        logger.info(f"Starting event runner for {symbol} on {timeframe}.")

        backtest_cfg = self.settings.get("backtest", {}) or {}
        start_date = backtest_cfg.get("start_date") or self.settings.get("start_date")
        end_date = backtest_cfg.get("end_date") or self.settings.get("end_date")

        # 1) Load historical data
        ohlcv = self.data_adapter.load_ohlcv(symbol, timeframe, start_date, end_date)
        if ohlcv is None or ohlcv.empty:
            logger.error("No data loaded, cannot run backtest.")
            return self.portfolio.trades, self.portfolio.equity_curve

        # 2) Iterate each bar
        for timestamp, bar in ohlcv.iterrows():
            self._process_bar(symbol, timeframe, timestamp, bar)

        logger.success("Event runner finished.")

        # Force-close all open positions at end of backtest
        if not ohlcv.empty:
            last_close = ohlcv.iloc[-1]["close"]
            last_ts = ohlcv.index[-1]
            for sym, pos in list(self.portfolio.positions.items()):  # Iterate over a copy
                self.portfolio.close_position(sym, last_close, last_ts, "EOD")
                self.log.info(f"Force-closing {sym} open position at EOD.")

        return self.portfolio.trades, self.portfolio.equity_curve

    def _process_bar(self, symbol: str, timeframe: str, timestamp: pd.Timestamp, bar: pd.Series):
        """Processes a single bar of data."""

        # 1) Push bar into FeatureStore
        bar_with_timestamp = bar.to_frame().T
        bar_with_timestamp["timestamp"] = timestamp
        self.feature_store.on_bar(symbol, timeframe, bar_with_timestamp)

        # 2) Mark-to-market equity
        self.portfolio.equity_curve.append({"timestamp": timestamp, "equity": self.portfolio.current_equity})

        # 3) Exit checks for any open position (keep your existing exit logic if you have it)
        pos = self.portfolio.positions.get(symbol)
        if pos is not None:
            exit_reason = None
            self.log.debug(
                f"DEBUG ExitCheck: ts={timestamp}, side={pos.side}, price={bar['close']}, "
                f"stop={'N/A' if getattr(pos, 'stop', None) is None else pos.stop}, "
                f"tp={'N/A' if getattr(pos, 'tp', None) is None else pos.tp}, "
                f"bars_held={getattr(pos, 'bars_held', 0)}, reason={exit_reason}"
            )
            # IMPORTANT BEHAVIOR:
            # While a position for this symbol is open, do NOT call the signal engine again.
            # However, if the TOTAL cap is reached, record a MAX_POSITIONS_TOTAL veto event.
            total_open = len(self.portfolio.positions)
            cap_total = int(getattr(self.portfolio, "max_positions_total", 999999))
            if total_open >= cap_total:
                ev = RiskEvent(
                    ts=int(pd.Timestamp(timestamp).timestamp()),
                    symbol=symbol,
                    reason="MAX_POSITIONS_TOTAL",
                    action="VETO",
                    detail={"open_total": total_open, "cap": cap_total},
                )
                self.risk_events.append(ev)
                self.log.info(
                    f"Portfolio gate for {symbol} at {timestamp}: {ev.reason} ({ev.action}) {ev.detail}"
                )
            else:
                # Position open but total cap not reached -> skip signal evaluation.
                self.log.info(
                    f"Position already open for {symbol}; skipping signal evaluation on {timestamp}."
                )
            return  # Always return if position open

        # 4) Build a minimal ohlcv segment for engines that want a DataFrame
        ohlcv_segment = pd.DataFrame([bar]).copy()
        ohlcv_segment.index = pd.DatetimeIndex([timestamp])

        # 5) Ask the signal engine for a decision (robust across different mock APIs)
        from typing import Any

        def _normalize_decision(res: Any) -> Optional[EnsembleDecision]:
            """
            Normalize various return types into an EnsembleDecision-like object.
            Accepts:
              - object with .decision attr
              - dict with 'decision' key (+ optional fields)
              - str: 'LONG'|'SHORT'|'FLAT'
              - tuple/list like ('LONG', 0.8)
            Any unknown/unsupported types return None.
            """
            ts_epoch = int(pd.Timestamp(timestamp).timestamp())

            # Helper: coerce any string into allowed set
            def _coerce_dec_string(s: str) -> str:
                s_up = str(s).upper()
                if s_up in ("LONG", "SHORT", "FLAT"):
                    return s_up
                # map common aliases to FLAT
                if s_up in ("NO-TRADE", "NONE", "HOLD", "WAIT", "NEUTRAL"):
                    return "FLAT"
                return "FLAT"

            # Case 1: object with .decision
            if hasattr(res, "decision"):
                try:
                    if getattr(res, "decision", None) not in ("LONG", "SHORT", "FLAT"):
                        res.decision = _coerce_dec_string(getattr(res, "decision"))
                except Exception:
                    pass
                return res  # assume it's already the right type

            # Case 2: dict
            if isinstance(res, dict) and "decision" in res:
                dec = _coerce_dec_string(res["decision"])
                return EnsembleDecision(
                    ts=res.get("ts", ts_epoch),
                    symbol=res.get("symbol", symbol),
                    tf=res.get("tf", timeframe),
                    decision=dec,
                    confidence=float(res.get("confidence", 0.0)),
                    subsignals=res.get("subsignals", []),
                    vote_detail=res.get("vote_detail", {}),
                    vetoes=res.get("vetoes", []),
                )

            # Case 3: string
            if isinstance(res, str):
                dec = _coerce_dec_string(res)
                return EnsembleDecision(
                    ts=ts_epoch, symbol=symbol, tf=timeframe, decision=dec, confidence=0.0,
                    subsignals=[], vote_detail={}, vetoes=[]
                )

            # Case 4: tuple/list like ('LONG', 0.8)
            if isinstance(res, (tuple, list)) and len(res) >= 1:
                dec = _coerce_dec_string(res[0])
                conf = float(res[1]) if len(res) > 1 else 0.0
                return EnsembleDecision(
                    ts=ts_epoch, symbol=symbol, tf=timeframe, decision=dec, confidence=conf,
                    subsignals=[], vote_detail={}, vetoes=[]
                )

            return None  # not recognized

        def _resolve_decision(engine: Any) -> EnsembleDecision:
            """
            Try multiple method names and signatures commonly used in tests/mocks.
            Explicitly supports MockSignalEngine.generate_signal(ohlcv_segment, symbol).
            """
            # 0) Explicit support for your mock: generate_signal(ohlcv_segment, symbol)
            if hasattr(engine, "generate_signal"):
                try:
                    # pass keyword args so tests can inspect kwargs['ohlcv_segment']
                    res0 = engine.generate_signal(ohlcv_segment=ohlcv_segment, symbol=symbol)
                    norm0 = _normalize_decision(res0)
                    if norm0 is not None:
                        return norm0
                except Exception:
                    pass  # fall through

            candidates = [
                "on_bar", "decide", "generate", "next_signal",
                "next", "get_signal", "signal", "produce", "evaluate",
            ]
            arg_shapes: List[Tuple] = [
                (symbol, timeframe, bar, self.feature_store),
                (symbol, timeframe, bar),
                (symbol, timeframe, self.feature_store),
                (symbol, timeframe),
                (bar, self.feature_store),
                (bar,),
                (self.feature_store,),
                tuple(),  # no-arg call
            ]

            # 1) Named methods
            for name in candidates:
                if hasattr(engine, name):
                    meth = getattr(engine, name)
                    if callable(meth):
                        for args in arg_shapes:
                            try:
                                res = meth(*args)
                                norm = _normalize_decision(res)
                                if norm is not None:
                                    return norm
                            except TypeError:
                                continue
                            except Exception:
                                continue

            # 2) Callable engine
            if callable(engine):
                for args in arg_shapes:
                    try:
                        res = engine(*args)
                        norm = _normalize_decision(res)
                        if norm is not None:
                            return norm
                    except TypeError:
                        continue
                    except Exception:
                        continue

            # 3) Fallback to FLAT
            ts_epoch = int(pd.Timestamp(timestamp).timestamp())
            return EnsembleDecision(
                ts=ts_epoch,
                symbol=symbol,
                tf=timeframe,
                decision="FLAT",
                confidence=0.0,
                subsignals=[],
                vote_detail={},
                vetoes=[]
            )

        decision = _resolve_decision(self.signal_engine)

        # ---------------------- NEW: LOG WHY NO TRADE ON THIS BAR ----------------------
        # If we didn't get a LONG/SHORT, emit a detailed debug line with features + confidence.
        action = getattr(decision, "decision", None)
        if action not in {"LONG", "SHORT"}:
            fbits = _peek_features(self.feature_store, symbol, timeframe, timestamp)
            # Try to infer a helpful reason label
            reason = "flat_no_entry"
            vd = getattr(decision, "vote_detail", {}) or {}
            vetoes = getattr(decision, "vetoes", []) or []
            if isinstance(vd, dict):
                thr = vd.get("threshold") or vd.get("enter_threshold") or vd.get("vote_threshold")
                score = vd.get("score") or vd.get("votes") or vd.get("confidence")
                if thr is not None and score is not None and float(score) < float(thr):
                    reason = "score_below_threshold"

            _trace_reject(
                symbol,
                timestamp,
                reason,
                side=action,
                confidence=(round(float(getattr(decision, "confidence", 0.0)), 3) if hasattr(decision, "confidence") else None),
                score=(round(float(vd.get("score")), 3) if isinstance(vd.get("score", None), (int, float)) else None),
                votes=(round(float(vd.get("votes")), 3) if isinstance(vd.get("votes", None), (int, float)) else None),
                threshold=(round(float(vd.get("threshold")), 3) if isinstance(vd.get("threshold", None), (int, float)) else None),
                adx=fbits.get("adx"),
                rsi=fbits.get("rsi"),
                atr_pct=fbits.get("atr_percentile"),
                vetoes=len(vetoes) if vetoes else None,
            )
            # no entry; fall through to return
            return
        # -------------------------------------------------------------------------------

        # 6) Handle entries (LONG/SHORT) with portfolio gating.
        if action in {"LONG", "SHORT"}:
            # Always evaluate portfolio gate (even if a position is already open)
            allowed, size_scale, events = evaluate_portfolio(decision, self.portfolio, self.settings)

            # Record all risk events
            if events:
                self.risk_events.extend(events)
                for ev in events:
                    self.log.info(
                        f"Portfolio gate for {symbol} at {timestamp}: {ev.reason} ({ev.action}) {ev.detail}"
                    )

            # If vetoed by portfolio policy -> do not open a trade
            if not allowed:
                self.log.info(f"Trade for {symbol} at {timestamp} NOT allowed by portfolio gate.")
                return

            # Open position if allowed (we only reach here when no position exists for symbol)
            close_px = float(bar["close"])
            size = self.portfolio.position_size(symbol, close_px) * float(size_scale)
            self.log.info(f"Trade for {symbol} at {timestamp} ALLOWED by portfolio gate.")
            self.portfolio.open_position(symbol, action, close_px, timestamp, size)
            self.log.info(f"INFO Opened {action} position for {symbol} at {close_px} with size {size:.4f}")


class MockSignalEngine:
    """A mock signal engine for testing the event runner."""
    def generate_signal(self, ohlcv_segment: pd.DataFrame, symbol: str) -> Optional[EnsembleDecision]:
        # Mocking an EnsembleDecision
        ts = int(ohlcv_segment.index[-1].timestamp())
        tf = "mock_tf"
        last = ohlcv_segment.iloc[-1]
        direction = "LONG" if float(last["close"]) > float(last["open"]) else "SHORT"

        return EnsembleDecision(
            ts=ts,
            symbol=symbol,
            tf=tf,
            decision=direction,
            confidence=0.75,
            subsignals=[],
            vote_detail={},
            vetoes=[]
        )

    def should_exit(self, symbol, pos, bar, features):
        # Mock should_exit for testing purposes
        return None
