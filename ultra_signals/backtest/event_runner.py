import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
from ultra_signals.core.custom_types import EnsembleDecision, RiskEvent, Position
from ultra_signals.risk.portfolio import evaluate_portfolio, Portfolio


# ------------------------ UNIVERSAL DECISION TRACE HELPERS ------------------------

def _trace_reject(symbol, ts, reason, **kv):
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
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            cur = getattr(cur, p, None)
    return cur if cur is not None else default

def _peek_features(feature_store, symbol, timeframe, ts):
    """
    Best-effort extraction of a few features for logging without a hard API dependency.
    """
    feats = None
    for meth in ("get_features", "get_latest", "latest", "get"):
        f = getattr(feature_store, meth, None)
        if callable(f):
            try:
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

    if feats is None:
        for cache_name in ("cache", "store", "buffer", "latest_features", "features"):
            cache = getattr(feature_store, cache_name, None)
            if cache is None:
                continue
            candidate = None
            try:
                candidate = cache.get((symbol, timeframe, ts)) or cache.get((symbol, timeframe)) \
                            or cache.get(symbol) or cache.get(timeframe)
            except Exception:
                candidate = None
            if candidate is None and isinstance(cache, dict):
                try:
                    candidate = next(iter(cache.values()))
                except Exception:
                    candidate = None
            if candidate is not None:
                feats = candidate
                break

    adx  = _safe_get_attr(feats, "trend.adx", None)
    rsi  = _safe_get_attr(feats, "momentum.rsi", None)
    atrp = _safe_get_attr(feats, "volatility.atr_percentile", None)

    def _rf(x, nd=3):
        try:
            return round(float(x), nd)
        except Exception:
            return None

    return {"adx": _rf(adx, 2), "rsi": _rf(rsi, 2), "atr_percentile": _rf(atrp, 3)}


# ------------------------ SMALL HELPERS ------------------------

def _read_exec_cfg(settings: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Read backtest.execution and return possibly-None floats
    so we can distinguish 'missing' from '0.0'.
    """
    bt = settings.get("backtest", {}) or {}
    ex = (bt.get("execution") if isinstance(bt, dict) else None) or {}

    def _maybe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    return {
        "sl_pct": _maybe_float(ex.get("sl_pct", ex.get("stop_loss_pct"))),
        "tp_pct": _maybe_float(ex.get("tp_pct", ex.get("take_profit_pct"))),
        "atr_sl_mult": _maybe_float(ex.get("atr_sl_mult")) or 2.0,
        "atr_tp_mult": _maybe_float(ex.get("atr_tp_mult")) or 3.0,
    }

def _pos_entry_price(pos, fallback: float) -> float:
    for name in ("entry_price", "entry", "avg_entry_price", "avg_price", "price", "open_price", "fill_price"):
        try:
            if hasattr(pos, name):
                v = getattr(pos, name)
                if v is not None:
                    return float(v)
        except Exception:
            pass
        if isinstance(pos, dict) and name in pos and pos[name] is not None:
            try:
                return float(pos[name])
            except Exception:
                pass
    return float(fallback)

def _get_features_for_ts(feature_store, symbol, timeframe, ts):
    feats = None
    for meth in ("get_features", "get_latest", "latest", "get"):
        f = getattr(feature_store, meth, None)
        if callable(f):
            try:
                for args in (
                    (symbol, timeframe, ts),
                    (symbol, timeframe),
                    (symbol,),
                    tuple(),
                ):
                    try:
                        feats = f(*args)
                        if feats is not None:
                            return feats
                    except TypeError:
                        continue
            except Exception:
                pass
    return feats

def _compute_initial_risk_levels(feature_store, symbol, timeframe, ts, side_up: str, entry_px: float, settings: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Decide stop/tp using (1) % config if provided and >0,
    else (2) ATR fallback if available,
    else (3) default fallback 5%/5% so we NEVER leave them None.
    """
    cfg = _read_exec_cfg(settings)
    sl_pct = cfg["sl_pct"]
    tp_pct = cfg["tp_pct"]
    stop_val = tp_val = None
    source = None

    # 1) % based
    if sl_pct is not None and sl_pct > 0:
        stop_val = entry_px * (1.0 - sl_pct) if side_up == "LONG" else entry_px * (1.0 + sl_pct)
        source = "percent"
    if tp_pct is not None and tp_pct > 0:
        tp_val = entry_px * (1.0 + tp_pct) if side_up == "LONG" else entry_px * (1.0 - tp_pct)
        source = source or "percent"

    # 2) ATR fallback (only if any side still missing)
    if stop_val is None or tp_val is None:
        feats = _get_features_for_ts(feature_store, symbol, timeframe, ts)
        atr = _safe_get_attr(feats, "volatility.atr", None)
        try:
            atr = float(atr) if atr is not None else None
        except Exception:
            atr = None
        if atr is not None:
            if stop_val is None:
                m = cfg["atr_sl_mult"]
                stop_val = entry_px - m * atr if side_up == "LONG" else entry_px + m * atr
            if tp_val is None:
                m = cfg["atr_tp_mult"]
                tp_val = entry_px + m * atr if side_up == "LONG" else entry_px - m * atr
            source = "atr"

    # 3) Last resort defaults (if still None because no % and ATR not ready yet)
    if stop_val is None or tp_val is None:
        default_sl = 0.05
        default_tp = 0.05
        if stop_val is None:
            stop_val = entry_px * (1.0 - default_sl) if side_up == "LONG" else entry_px * (1.0 + default_sl)
        if tp_val is None:
            tp_val = entry_px * (1.0 + default_tp) if side_up == "LONG" else entry_px * (1.0 - default_tp)
        source = (source or "fallback_percent")

    return {"stop": float(stop_val), "tp": float(tp_val), "source": source}


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

        # Per-symbol risk mirrors so exits work even if Position disallows custom attrs
        self._risk_levels: Dict[str, Dict[str, Optional[float]]] = {}

        # Initialize the Portfolio (robust for flat or nested settings)
        backtest_cfg = self.settings.get("backtest", {})
        exec_cfg = (backtest_cfg.get("execution") if isinstance(backtest_cfg, dict) else None) or {}

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

        # ---------------- NEW (additive): optional global entry budget toggle ---------------
        # If >0, we’ll enforce an overall entries cap; if 0/None, we DO NOT use a global cap.
        self._global_entry_budget: int = int(portfolio_settings.get("global_entry_budget", 0) or 0)
        # ------------------------------------------------------------------------------------

        self.portfolio = Portfolio(
            initial_capital=float(initial_capital),
            max_positions_total=int(max_total_positions),
            max_positions_per_symbol=int(max_positions_per_symbol),
        )

        # ---------------- NEW (additive): honor default_size_pct from settings --------------
        try:
            dsp = exec_cfg.get("default_size_pct", None)
            if dsp is not None:
                dsp = float(dsp)
                self.portfolio.default_size_pct = (dsp / 100.0) if dsp > 1.0 else dsp
        except Exception:
            pass
        # ------------------------------------------------------------------------------------

        # NEW: per-run entry counter to enforce "total entries" budget (tests expect this)
        self.entries_count_total: int = 0

        # Optional (additive) time stop bars per trade; 0/None keeps current behavior
        self._max_bars_in_trade: int = int(exec_cfg.get("max_bars_in_trade", 0) or 0)

        # ---------------- NEW: trace FeatureStore id and self-heal if mismatched ----------------
        self._store_id = id(self.feature_store)
        self.log.debug(f"[EventRunner] using FeatureStore id={self._store_id}")

        if hasattr(self.signal_engine, "feature_store"):
            eng_store = getattr(self.signal_engine, "feature_store")
            if eng_store is not self.feature_store:
                self.log.warning(
                    "EventRunner detected a different FeatureStore on the SignalEngine; "
                    "rebinding engine.feature_store -> runner.feature_store "
                    f"(runner_store_id={id(self.feature_store)}, engine_store_id={id(eng_store)})"
                )
                try:
                    setattr(self.signal_engine, "feature_store", self.feature_store)
                except Exception:
                    setter = getattr(self.signal_engine, "set_feature_store", None)
                    if callable(setter):
                        try:
                            setter(self.feature_store)
                        except Exception:
                            pass
        # ---------------------------------------------------------------------------------------

    @property
    def trades(self) -> List[dict]:
        return self.portfolio.trades

    @property
    def equity_curve(self) -> List[dict]:
        return self.portfolio.equity_curve

    def run(self, symbol: str, timeframe: str):
        """Main event loop for a single symbol backtest."""
        logger.info(f"Starting event runner for {symbol} on {timeframe}.")

        # reset per-run counters / buckets
        self.entries_count_total = 0
        self.risk_events = []

        backtest_cfg = self.settings.get("backtest", {}) or {}
        start_date = backtest_cfg.get("start_date") or self.settings.get("start_date")
        end_date = backtest_cfg.get("end_date") or self.settings.get("end_date")

        ohlcv = self.data_adapter.load_ohlcv(symbol, timeframe, start_date, end_date)
        if ohlcv is None or ohlcv.empty:
            logger.error("No data loaded, cannot run backtest.")
            return self.portfolio.trades, self.portfolio.equity_curve

        for timestamp, bar in ohlcv.iterrows():
            self._process_bar(symbol, timeframe, timestamp, bar)

        logger.success("Event runner finished.")

        # Force-close all open positions at end of backtest
        if not ohlcv.empty:
            last_close = ohlcv.iloc[-1]["close"]
            last_ts = ohlcv.index[-1]
            for sym, pos in list(self.portfolio.positions.items()):
                self.portfolio.close_position(sym, last_close, last_ts, "EOD")
                self._risk_levels.pop(sym, None)
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

        # 3) Exit checks for any open position
        pos = self.portfolio.positions.get(symbol)
        if pos is not None:
            pos.bars_held = getattr(pos, "bars_held", 0) + 1

            try:
                bar_high = float(bar["high"]); bar_low = float(bar["low"]); bar_close = float(bar["close"])
            except Exception:
                bar_high = bar_low = bar_close = float(bar.get("close", 0.0))

            side_up = str(getattr(pos, "side", "LONG")).upper()

            # Ensure risk levels exist; if missing, (re)compute
            rl = self._risk_levels.get(symbol)
            if rl is None or (rl.get("stop") is None and rl.get("tp") is None):
                entry_px = _pos_entry_price(pos, bar_close)
                rl_calc = _compute_initial_risk_levels(self.feature_store, symbol, timeframe, timestamp, side_up, entry_px, self.settings)
                rl = {"stop": rl_calc["stop"], "tp": rl_calc["tp"]}
                self._risk_levels[symbol] = rl
                self.log.info(
                    f"Risk levels set for {symbol} on {timestamp}: stop={rl['stop']}, tp={rl['tp']} (source={rl_calc['source']})"
                )

            stop_val = rl.get("stop")
            tp_val   = rl.get("tp")

            # --- NEW (Sprint12 trailing stop using risk_model metadata) ---
            try:
                rl_meta = rl.get('_meta') if isinstance(rl, dict) else None
                if rl_meta and rl_meta.get('source') == 'risk_model':
                    atr = float(rl_meta.get('atr') or 0.0)
                    atr_mult_stop = float(rl_meta.get('atr_mult_stop') or 0.0)
                    entry_px_meta = float(rl_meta.get('entry_price') or 0.0)
                    side_meta = rl_meta.get('side') or side_up
                    # Apply simple trailing rule: once move > 1*ATR, raise (or lower for SHORT) stop by 0.5*ATR*mult
                    if atr > 0 and atr_mult_stop > 0 and entry_px_meta > 0:
                        if side_meta == 'LONG' and (bar_close - entry_px_meta) > atr:
                            new_stop = max(stop_val, bar_close - atr * atr_mult_stop * 0.5)
                            if new_stop is not None and new_stop > (stop_val or 0):
                                rl['stop'] = new_stop
                                stop_val = new_stop
                        elif side_meta == 'SHORT' and (entry_px_meta - bar_close) > atr:
                            new_stop = min(stop_val, bar_close + atr * atr_mult_stop * 0.5) if stop_val is not None else bar_close + atr * atr_mult_stop * 0.5
                            if new_stop is not None and (stop_val is None or new_stop < stop_val):
                                rl['stop'] = new_stop
                                stop_val = new_stop
            except Exception:
                pass
            # --------------------------------------------------------------

            self.log.debug(
                f"DEBUG ExitCheck: ts={timestamp}, side={side_up}, price={bar_close}, "
                f"stop={'N/A' if stop_val is None else stop_val}, "
                f"tp={'N/A' if tp_val is None else tp_val}, "
                f"bars_held={getattr(pos, 'bars_held', 0)}, reason=None"
            )

            # ===== Correct intrabar exit logic: only close if actually touched =====
            exit_reason = None
            exit_price: Optional[float] = None

            if side_up == "LONG":
                if stop_val is not None and bar_low <= stop_val:
                    exit_reason, exit_price = "SL", float(stop_val)
                elif tp_val is not None and bar_high >= tp_val:
                    exit_reason, exit_price = "TP", float(tp_val)
            else:  # SHORT
                if stop_val is not None and bar_high >= stop_val:
                    exit_reason, exit_price = "SL", float(stop_val)
                elif tp_val is not None and bar_low <= tp_val:
                    exit_reason, exit_price = "TP", float(tp_val)
            # =====================================================================

            if exit_reason:
                self.portfolio.close_position(symbol, exit_price, timestamp, reason=exit_reason)
                self._risk_levels.pop(symbol, None)
                self.log.info(f"Closed {symbol} {side_up} at {exit_price} due to {exit_reason}.")
                return  # do not re-enter on the same bar

            # Strategy/engine exits (optional)
            if hasattr(self.signal_engine, "should_exit"):
                if hasattr(self.signal_engine, "feature_store"):
                    eng_store = getattr(self.signal_engine, "feature_store")
                    if eng_store is not self.feature_store:
                        self.log.warning(
                            "EventRunner detected a different FeatureStore on the SignalEngine (exit path); "
                            "rebinding engine.feature_store -> runner.feature_store "
                            f"(runner_store_id={id(self.feature_store)}, engine_store_id={id(eng_store)})"
                        )
                        try:
                            setattr(self.signal_engine, "feature_store", self.feature_store)
                        except Exception:
                            setter = getattr(self.signal_engine, "set_feature_store", None)
                            if callable(setter):
                                try:
                                    setter(self.feature_store)
                                except Exception:
                                    pass

                feats_for_engine = _get_features_for_ts(self.feature_store, symbol, timeframe, timestamp) or {}
                try:
                    engine_reason = self.signal_engine.should_exit(symbol, pos, bar, feats_for_engine)
                except TypeError:
                    try:
                        engine_reason = self.signal_engine.should_exit(symbol, pos, bar)
                    except Exception:
                        engine_reason = None
                except Exception:
                    engine_reason = None

                if engine_reason:
                    self.portfolio.close_position(symbol, bar_close, timestamp, reason=str(engine_reason))
                    self._risk_levels.pop(symbol, None)
                    self.log.info(f"Closed {symbol} {side_up} at {bar_close} due to {engine_reason}.")
                    return

            # ---------------- NEW (additive): optional bar-based time stop -------------------
            if self._max_bars_in_trade and getattr(pos, "bars_held", 0) >= self._max_bars_in_trade:
                self.portfolio.close_position(symbol, bar_close, timestamp, reason=f"TIME_STOP_{self._max_bars_in_trade}")
                self._risk_levels.pop(symbol, None)
                self.log.info(f"Closed {symbol} {side_up} at {bar_close} due to TIME_STOP_{self._max_bars_in_trade}.")
                return
            # --------------------------------------------------------------------------------

            # Still open: just log and return (do not try to re-enter)
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
                self.log.info(f"Position already open for {symbol}; skipping signal evaluation on {timestamp}.")
            return

        # 4) Build a minimal ohlcv segment for engines that want a DataFrame
        ohlcv_segment = pd.DataFrame([bar]).copy()
        ohlcv_segment.index = pd.DatetimeIndex([timestamp])

        # 5) Ask the signal engine for a decision (robust across different mock APIs)
        from typing import Any

        def _normalize_decision(res: Any) -> Optional[EnsembleDecision]:
            ts_epoch = int(pd.Timestamp(timestamp).timestamp())

            def _coerce_dec_string(s: str) -> str:
                s_up = str(s).upper()
                if s_up in ("LONG", "SHORT", "FLAT"):
                    return s_up
                if s_up in ("NO-TRADE", "NONE", "HOLD", "WAIT", "NEUTRAL"):
                    return "FLAT"
                return "FLAT"

            if hasattr(res, "decision"):
                try:
                    if getattr(res, "decision", None) not in ("LONG", "SHORT", "FLAT"):
                        res.decision = _coerce_dec_string(getattr(res, "decision"))
                except Exception:
                    pass
                return res

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

            if isinstance(res, str):
                dec = _coerce_dec_string(res)
                return EnsembleDecision(
                    ts=ts_epoch, symbol=symbol, tf=timeframe, decision=dec, confidence=0.0,
                    subsignals=[], vote_detail={}, vetoes=[]
                )

            if isinstance(res, (tuple, list)) and len(res) >= 1:
                dec = _coerce_dec_string(res[0])
                conf = float(res[1]) if len(res) > 1 else 0.0
                return EnsembleDecision(
                    ts=ts_epoch, symbol=symbol, tf=timeframe, decision=dec, confidence=conf,
                    subsignals=[], vote_detail={}, vetoes=[]
                )

            return None

        def _resolve_decision(engine: Any) -> EnsembleDecision:
            if hasattr(engine, "feature_store"):
                eng_store = getattr(engine, "feature_store")
                if eng_store is not self.feature_store:
                    self.log.warning(
                        "EventRunner detected a different FeatureStore on the SignalEngine (entry path); "
                        "rebinding engine.feature_store -> runner.feature_store "
                        f"(runner_store_id={id(self.feature_store)}, engine_store_id={id(eng_store)})"
                    )
                    try:
                        setattr(engine, "feature_store", self.feature_store)
                    except Exception:
                        setter = getattr(engine, "set_feature_store", None)
                        if callable(setter):
                            try:
                                setter(self.feature_store)
                            except Exception:
                                pass

            if hasattr(engine, "generate_signal"):
                try:
                    res0 = engine.generate_signal(ohlcv_segment=ohlcv_segment, symbol=symbol)
                    norm0 = _normalize_decision(res0)
                    if norm0 is not None:
                        return norm0
                except Exception:
                    pass

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
                tuple(),
            ]

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

        # ---------------------- WHY NO TRADE ON THIS BAR ----------------------
        action = getattr(decision, "decision", None)
        if action not in {"LONG", "SHORT"}:
            fbits = _peek_features(self.feature_store, symbol, timeframe, timestamp)
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
            return
        # ----------------------------------------------------------------------

        # 6) Handle entries (LONG/SHORT) with portfolio gating.
        if action in {"LONG", "SHORT"}:
            # ----------------- MODIFIED (additive guard): global entries budget -----------------
            # Your old HARD GATE is now only active when portfolio.global_entry_budget > 0.
            if self._global_entry_budget:
                max_total = int(self._global_entry_budget)
                if self.entries_count_total >= max_total:
                    ev = RiskEvent(
                        ts=int(pd.Timestamp(timestamp).timestamp()),
                        symbol=symbol,
                        reason="GLOBAL_ENTRY_BUDGET",   # clearer reason when toggle is on
                        action="VETO",
                        detail={"entries_count_total": self.entries_count_total, "cap": max_total},
                    )
                    self.risk_events.append(ev)
                    self.log.info(f"Trade for {symbol} at {timestamp} VETOED by global budget gate: {ev.reason}")
                    return
            # -----------------------------------------------------------------------------------

            allowed, size_scale, events = evaluate_portfolio(decision, self.portfolio, self.settings)

            # Record all risk events
            if events:
                self.risk_events.extend(events)
                for ev in events:
                    self.log.info(
                        f"Portfolio gate for {symbol} at {timestamp}: {ev.reason} ({ev.action}) {ev.detail}"
                    )

            if not allowed:
                self.log.info(f"Trade for {symbol} at {timestamp} NOT allowed by portfolio gate.")
                return

            close_px = float(bar["close"])
            size = self.portfolio.position_size(symbol, close_px) * float(size_scale)
            # --- NEW (Sprint12): override with adaptive risk_model sizing & stops if present ---
            try:
                rm = None
                vd = getattr(decision, 'vote_detail', None)
                if isinstance(vd, dict):
                    rm = vd.get('risk_model') if isinstance(vd.get('risk_model'), dict) else None
                if rm:
                    # Preferred: playbook-adjusted size if present
                    size_quote = rm.get('position_size_playbook') or rm.get('position_size')
                    if size_quote and close_px > 0:
                        size_adaptive = float(size_quote) / float(close_px)
                        if size_adaptive > 0:
                            size = size_adaptive
                    # Inject risk model SL/TP directly if available
                    sl = rm.get('stop_loss')
                    tp = rm.get('take_profit')
                    if sl is not None or tp is not None:
                        self._risk_levels[symbol] = {
                            'stop': float(sl) if sl is not None else None,
                            'tp': float(tp) if tp is not None else None,
                            '_meta': {
                                'source': 'risk_model',
                                'atr': rm.get('atr'),
                                'atr_mult_stop': rm.get('atr_mult_stop') or self.settings.get('risk_model', {}).get('atr_multiplier_stop'),
                                'atr_mult_tp': rm.get('atr_mult_tp') or self.settings.get('risk_model', {}).get('atr_multiplier_tp'),
                                'entry_price': close_px,
                                'side': action
                            }
                        }
            except Exception:
                pass
            # -------------------------------------------------------------------------------
            self.log.info(f"Trade for {symbol} at {timestamp} ALLOWED by portfolio gate.")
            self.portfolio.open_position(symbol, action, close_px, timestamp, size)
            self.entries_count_total += 1  # count successful entries
            self.log.info(f"INFO Opened {action} position for {symbol} at {close_px} with size {size:.4f}")

            # If risk_model did not set SL/TP, compute legacy levels
            if symbol not in self._risk_levels or (self._risk_levels.get(symbol, {}).get('stop') is None and self._risk_levels.get(symbol, {}).get('tp') is None):
                side_up = str(action).upper()
                rl_calc = _compute_initial_risk_levels(self.feature_store, symbol, timeframe, timestamp, side_up, close_px, self.settings)
                self._risk_levels[symbol] = {"stop": rl_calc["stop"], "tp": rl_calc["tp"]}
                self.log.info(
                    f"Risk levels set for {symbol} on entry: stop={rl_calc['stop']}, tp={rl_calc['tp']} (source={rl_calc['source']})"
                )


class MockSignalEngine:
    """A mock signal engine for testing the event runner."""
    def generate_signal(self, ohlcv_segment: pd.DataFrame, symbol: str) -> Optional[EnsembleDecision]:
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
        return None
