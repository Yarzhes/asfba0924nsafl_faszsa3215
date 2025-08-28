import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
from ultra_signals.core.custom_types import EnsembleDecision, RiskEvent, Position
from ultra_signals.risk.portfolio import evaluate_portfolio, Portfolio
try:  # Sprint 24 execution attribution (optional)
    from ultra_signals.execution.metrics import ExecAttribution
except Exception:  # pragma: no cover
    ExecAttribution = None  # type: ignore


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
        self.risk_events = []
        self.log = logger
        self.warmup_mode = bool(self.settings.get("warmup_mode", False))
        # Execution attribution per symbol (filled once position opens)
        self._exec_attr = {}
        # Microstructure pending entry orders (symbol -> dict)
        self._pending_entries: Dict[str, Dict[str, Any]] = {}
        # Cache microstructure config (prefer backtest.execution.microstructure)
        try:
            self._micro_cfg = ((self.settings.get("backtest", {}) or {}).get("execution", {}) or {}).get("microstructure", {}) or {}
        except Exception:
            self._micro_cfg = {}

        # Per-symbol risk mirrors
        self._risk_levels = {}

        backtest_cfg = self.settings.get("backtest", {}) or {}
        exec_cfg = (backtest_cfg.get("execution") if isinstance(backtest_cfg, dict) else None) or {}
        initial_capital = exec_cfg.get("initial_capital") or self.settings.get("initial_capital") or 10000.0
        portfolio_settings = self.settings.get("portfolio", {}) or {}
        max_total_positions = portfolio_settings.get("max_total_positions") or self.settings.get("max_total_positions") or 999999
        max_positions_per_symbol = portfolio_settings.get("max_positions_per_symbol") or self.settings.get("max_positions_per_symbol") or 999999
        self._global_entry_budget = int(portfolio_settings.get("global_entry_budget", 0) or 0)

        self.portfolio = Portfolio(
            initial_capital=float(initial_capital),
            max_positions_total=int(max_total_positions),
            max_positions_per_symbol=int(max_positions_per_symbol),
        )
        # Sprint 30: track last bar MTC status to tag trades
        self._last_mtc_gate = None

        # Hedge tracking
        self._equity_unhedged = []
        self._equity_hedged = []
        self._hedge_avg_price = 0.0
        self._hedge_realized_pnl = 0.0
        self._hedge_unrealized_pnl = 0.0
        self._last_leader_price = None
        self._hedge_funding_pnl = 0.0
        # AdvancedSizer equity peak for drawdown tracking
        self._equity_peak = float(initial_capital)

        # Default size pct
        try:
            dsp = exec_cfg.get("default_size_pct")
            if dsp is not None:
                dsp = float(dsp)
                self.portfolio.default_size_pct = (dsp / 100.0) if dsp > 1.0 else dsp
        except Exception:
            pass

        self.entries_count_total = 0
        self._max_bars_in_trade = int(exec_cfg.get("max_bars_in_trade", 0) or 0)

        self._store_id = id(self.feature_store)
        self.log.debug(f"[EventRunner] using FeatureStore id={self._store_id}")
        if hasattr(self.signal_engine, "feature_store"):
            eng_store = getattr(self.signal_engine, "feature_store")
            if eng_store is not self.feature_store:
                self.log.warning(
                    "EventRunner detected a different FeatureStore on the SignalEngine; rebinding.")
                try:
                    setattr(self.signal_engine, "feature_store", self.feature_store)
                except Exception:
                    setter = getattr(self.signal_engine, "set_feature_store", None)
                    if callable(setter):
                        try:
                            setter(self.feature_store)
                        except Exception:
                            pass
        # Sprint 36: broker simulator adapter (lazy init)
        self._broker_adapter = None
        try:
            bcfg = (self.settings.get('broker_sim') or {}) if isinstance(self.settings, dict) else {}
            if bcfg.get('enabled'):
                from ultra_signals.sim.router_adapter import BrokerRouterAdapter  # local import
                self._broker_adapter = BrokerRouterAdapter(self.settings)
                self.log.debug('[BrokerSim] adapter initialized')
        except Exception as e:  # pragma: no cover
            self.log.warning(f"[BrokerSim] init failed: {e}")
        # Sprint 36: fill ledger for detailed CSV export
        self._sim_fills: list[dict] = []

    @property
    def trades(self) -> List[dict]:
        return self.portfolio.trades

    @property
    def equity_curve(self) -> List[dict]:
        return self.portfolio.equity_curve

    @property
    def event_metrics(self) -> Dict[str, Any]:
        return getattr(self, '_event_metrics', {})

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

        # ---------------------------------------------------------------------------------
        # Optional: Funding / OI replay (load exported CSV and inject into FeatureStore
        # and FundingProvider cache so compute_derivatives_posture can operate in backtests)
        try:
            fr_cfg = (self.settings.get('backtest') or {}).get('funding_replay', {}) or {}
            enabled = bool(fr_cfg.get('enabled', True))
            if enabled:
                import os, pandas as _pd
                # priority: explicit path in backtest.funding_replay.path -> backtest.output_dir -> default file
                path = fr_cfg.get('path') or ((self.settings.get('backtest') or {}).get('output_dir')) or 'funding_export.csv'
                if os.path.isfile(path):
                    try:
                        df = _pd.read_csv(path)
                        if not df.empty:
                            # normalize timestamp column to ms epoch
                            if 'ts' in df.columns:
                                df['funding_time'] = _pd.to_datetime(df['ts'], unit='ms', errors='coerce')
                            elif 'funding_time' in df.columns:
                                try:
                                    df['funding_time'] = _pd.to_datetime(df['funding_time'], unit='ms', errors='raise')
                                except Exception:
                                    df['funding_time'] = _pd.to_datetime(df['funding_time'], errors='coerce')
                            else:
                                # fall back to any timestamp-like column or index
                                ts_candidates = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
                                if ts_candidates:
                                    df['funding_time'] = _pd.to_datetime(df[ts_candidates[0]], errors='coerce')
                                else:
                                    try:
                                        df = df.reset_index()
                                        df['funding_time'] = _pd.to_datetime(df['index'], errors='coerce')
                                    except Exception:
                                        df['funding_time'] = _pd.NaT

                            # derive ms integer and build per-symbol lists
                            df['funding_time_ms'] = (df['funding_time'].astype('int64') // 1_000_000).astype('Int64')
                            grouped = {}
                            for _, r in df.iterrows():
                                sym = str(r.get('symbol') or r.get('sym') or '').strip()
                                if not sym:
                                    continue
                                fr = float(r.get('funding_rate') or r.get('fund') or 0.0)
                                oi = None
                                try:
                                    oi = float(r.get('oi_notional') if r.get('oi_notional') is not None else r.get('oi') if r.get('oi') is not None else 0.0)
                                except Exception:
                                    oi = None
                                venue = r.get('venue') or r.get('exchange') or 'unknown'
                                ts_ms = int(r.get('funding_time_ms') or 0)
                                item = {'funding_rate': fr, 'funding_time': ts_ms, 'venue': venue}
                                if oi is not None:
                                    item['oi_notional'] = oi
                                grouped.setdefault(sym, []).append(item)
                                # buffer into store's macro export buffer for traceability
                                try:
                                    self.feature_store._buffer_funding_row({'symbol': sym, 'ts': ts_ms, 'funding_rate': fr, 'oi_notional': oi, 'venue': venue})
                                except Exception:
                                    pass

                            # inject into FundingProvider cache if present (override existing cache for replay)
                            fp = getattr(self.feature_store, '_funding_provider', None)
                            if fp is not None:
                                try:
                                    for s, lst in grouped.items():
                                        lst_sorted = sorted(lst, key=lambda x: int(x.get('funding_time', 0)))
                                        # use provider API if available
                                        lr = getattr(fp, 'load_replay', None)
                                        if callable(lr):
                                            lr(s, lst_sorted)
                                        else:
                                            fp._cache[s] = lst_sorted
                                except Exception:
                                    pass
                    except Exception:
                        logger.debug('Funding replay load failed or file malformed; skipping funding replay')
        except Exception:
            pass
        # ---------------------------------------------------------------------------------

        # Event metrics buckets
        self._event_metrics = {
            'bars': 0,
            'veto_bars': 0,
            'dampen_bars': 0,
            'force_closes': 0,
            'cooldown_blocks': 0,
            'dampen_trades': 0,
            'counterfactual_pnl_gain': 0.0,
            # Sprint 30 MTC metrics buckets
            'mtc_bars': 0,
            'mtc_confirm_bars': 0,
            'mtc_partial_bars': 0,
            'mtc_fail_bars': 0,
            # Sprint 30: MTC score samples for histogram
            'mtc_score_samples_c1': [],
            'mtc_score_samples_c2': [],
            # Sprint 34 adaptive exits metrics
            'adaptive_trailing_adjustments': 0,
            'adaptive_breakeven_moves': 0,
            'adaptive_partial_fills': 0,
            'adaptive_exit_counts': {},  # reason -> count
            'adaptive_rr_samples': [],   # achieved R multiples at close
        }
        from typing import Optional as _Opt
        last_veto_ts: _Opt[int] = None
        cooldown_min = int(((self.settings.get('event_risk') or {}).get('cooldown_minutes_after_veto', 0)) or 0)
        cooldown_sec = cooldown_min * 60

        for timestamp, bar in ohlcv.iterrows():
            self._event_metrics['bars'] += 1
            # Sprint 33: portfolio risk estimator update + periodic rebalance (once per bar per symbol loop)
            try:
                pr_cfg = (self.settings.get('portfolio_risk') or {}) if isinstance(self.settings, dict) else {}
                if pr_cfg.get('enabled') and hasattr(self.signal_engine, '_pr_est') and getattr(self.signal_engine, '_pr_est'):
                    # Update estimator with OHLC for this symbol
                    try:
                        self.signal_engine._pr_est.update(symbol,
                            high=float(bar.get('high', bar.get('close', 0.0))),
                            low=float(bar.get('low', bar.get('close', 0.0))),
                            close=float(bar.get('close', 0.0))
                        )
                    except Exception:
                        pass
                    # Provide fresh open positions snapshot to engine so entry scaling uses real portfolio
                    open_snapshot = []
                    for sym, pos in self.portfolio.positions.items():
                        try:
                            qty = float(getattr(pos,'qty', getattr(pos,'size',0.0)) or 0.0)
                            if qty == 0: continue
                            entry_px = float(getattr(pos,'entry_price', getattr(pos,'price', None)) or 0.0)
                            side = str(getattr(pos,'side','LONG')).upper()
                            stop_dist = float(getattr(pos,'adv_stop_distance', 0.0) or 0.0)
                            risk_amt = float(getattr(pos,'risk_amount_at_entry', 0.0) or 0.0)
                            open_snapshot.append({'symbol': sym,'qty': qty,'entry_price': entry_px,'side': side,'risk_amount': risk_amt,'stop_distance': stop_dist})
                        except Exception:
                            continue
                    try:
                        self.signal_engine.set_open_positions(open_snapshot)
                    except Exception:
                        pass
                    # Periodic rebalance check (only trigger on leader symbol loop OR all symbols? single-symbol run so ok)
                    try:
                        interval = int(pr_cfg.get('rebalance_interval_bars', 0) or 0)
                    except Exception:
                        interval = 0
                    if interval and (self._event_metrics['bars'] % interval == 0) and open_snapshot:
                        # Build open positions as candidate list and run allocator with no new candidate
                        try:
                            alloc = getattr(self.signal_engine, '_pr_alloc', None)
                            est = getattr(self.signal_engine, '_pr_est', None)
                            if alloc and est and est.ready():
                                # Evaluate current portfolio (candidate None) by wrapping None symbol
                                # Reuse allocator.evaluate with a dummy candidate that results in no new position actions
                                adjustments, metrics = alloc.evaluate(open_snapshot, None, int(pd.Timestamp(timestamp).timestamp()))
                                # Apply only scale adjustments (ignore rejects for existing positions) partially per rebalance_strength already inside allocator
                                for adj in adjustments or []:
                                    if adj.get('action') == 'scale':
                                        sym = adj.get('symbol')
                                        target_mult = float(adj.get('size_mult',1.0))
                                        pos_obj = self.portfolio.positions.get(sym)
                                        if pos_obj and target_mult > 0:
                                            try:
                                                cur_qty = float(getattr(pos_obj,'qty', getattr(pos_obj,'size',0.0)) or 0.0)
                                                new_qty = cur_qty * target_mult
                                                if new_qty <= 0: continue
                                                # Simple proportional adjust: treat as closing or adding difference at current close
                                                px = float(bar.get('close', 0.0))
                                                delta = new_qty - cur_qty
                                                if abs(delta) / (cur_qty+1e-9) > 0.01:  # threshold to avoid noise
                                                    if delta > 0:
                                                        # add to position
                                                        self.portfolio.add_to_position(sym, px, timestamp, delta)
                                                        self.log.info(f"[PR-REB] Upsized {sym} qty {cur_qty:.4f}->{new_qty:.4f} mult={target_mult:.3f}")
                                                    else:
                                                        # partial reduce
                                                        self.portfolio.reduce_position(sym, px, timestamp, -delta, reason='PR_REBAL')
                                                        self.log.info(f"[PR-REB] Downsized {sym} qty {cur_qty:.4f}->{new_qty:.4f} mult={target_mult:.3f}")
                                            except Exception:
                                                continue
                                # Store metrics timeseries (one row tagged rebalance)
                                try:
                                    if metrics:
                                        row = dict(metrics); row['ts'] = int(pd.Timestamp(timestamp).timestamp()); row['rebalance'] = True
                                        self.signal_engine._pr_metrics_ts.append(row)
                                except Exception:
                                    pass
                        except Exception as _e:
                            self.log.debug(f"[PR-REB] error { _e }")
            except Exception:
                pass
            # Microstructure simulation hook (placeholder for future partial fill model)
            # No-op currently; real implementation will adjust position entries with partial maker fills.
            # Pre bar: enforce force_close or cooldown blocks using gate evaluation
            try:
                from ultra_signals import events as _events
                gate = _events.evaluate_gate(symbol, int(pd.Timestamp(timestamp).timestamp())*1000, None, None, self.settings)
                if gate.action == 'VETO':
                    self._event_metrics['veto_bars'] += 1
                    now_sec = int(pd.Timestamp(timestamp).timestamp())
                    last_veto_ts = now_sec
                    # Force close open position if flagged
                    if gate.force_close and symbol in self.portfolio.positions:
                        pos = self.portfolio.positions.get(symbol)
                        try:
                            exit_px = float(bar.get('close'))
                        except Exception:
                            exit_px = float(bar.get('close', 0.0))
                        # capture pre-close size for stats
                        self.portfolio.close_position(symbol, exit_px, timestamp, reason='EVENT_FORCE_CLOSE')
                        self._risk_levels.pop(symbol, None)
                        self._event_metrics['force_closes'] += 1
                        self.log.info(f"[EVENT] Force-closed {symbol} due to event veto @ {exit_px}")
                        # Do not skip rest; still record bar metrics, but prevent re-entry on same bar
                        # Continue to next bar (skip entry evaluation this bar)
                        continue
                elif gate.action == 'DAMPEN':
                    self._event_metrics['dampen_bars'] += 1
            except Exception:
                gate = None  # silent

            # Cooldown new entries (but still allow exit management) if within cooldown window after last veto
            cooldown_block = False
            if last_veto_ts and (int(pd.Timestamp(timestamp).timestamp()) - last_veto_ts) < cooldown_sec:
                cooldown_block = True

            self._process_bar(symbol, timeframe, timestamp, bar, cooldown_block=cooldown_block)
            # Sprint 30: collect per-bar MTC status if decision object exposed it (decision stored in last call?)
            try:
                # We infer last decision via portfolio entries or risk events? Simplest: feature_store may keep last decision, but not guaranteed.
                # Instead, reuse gate outcome stored in last decision through instrumented attribute on runner if present.
                mtc_meta = getattr(self, '_last_mtc_gate', None)
                if mtc_meta:
                    self._event_metrics['mtc_bars'] += 1
                    status = mtc_meta.get('status')
                    if status == 'CONFIRM':
                        self._event_metrics['mtc_confirm_bars'] += 1
                    elif status == 'PARTIAL':
                        self._event_metrics['mtc_partial_bars'] += 1
                    elif status == 'FAIL':
                        self._event_metrics['mtc_fail_bars'] += 1
                    # collect score samples
                    try:
                        sc = mtc_meta.get('scores') or {}
                        c1 = sc.get('C1'); c2 = sc.get('C2')
                        if isinstance(c1,(int,float)):
                            self._event_metrics['mtc_score_samples_c1'].append(float(c1))
                        if isinstance(c2,(int,float)):
                            self._event_metrics['mtc_score_samples_c2'].append(float(c2))
                    except Exception:
                        pass
            except Exception:
                pass
        logger.success("Event runner finished.")

        # Sprint 29: finalize liquidity gate derived metrics
        try:
            bars = max(1, self._event_metrics.get('bars', 0))
            lv = float(self._event_metrics.get('liquidity_veto_bars', 0))
            ld = float(self._event_metrics.get('liquidity_dampen_bars', 0))
            self._event_metrics['liquidity_veto_rate_pct'] = round(lv / bars * 100.0, 3)
            self._event_metrics['liquidity_dampen_rate_pct'] = round(ld / bars * 100.0, 3)
        except Exception:
            pass

        # Flush any still-pending microstructure entries as taker at last close before EOD close logic
        try:
            if self._pending_entries:
                last_close = float(ohlcv.iloc[-1]["close"])
                last_ts = ohlcv.index[-1]
                for sym, order in list(self._pending_entries.items()):
                    if order.get("remaining", 0) > 0:
                        rem = float(order["remaining"])
                        # record final taker fill
                        self._record_pending_fill(sym, last_ts, last_close, rem, maker=False)
                        order["remaining"] = 0.0
                        # convert to position
                        fills = order["fills"]
                        total_qty = sum(f[2] for f in fills)
                        vwap = sum(f[1]*f[2] for f in fills)/total_qty if total_qty else last_close
                        self.portfolio.open_position(sym, order["side"], vwap, last_ts, total_qty)
                        self.entries_count_total += 1
                        self.log.info(f"[MICRO] EOD convert pending -> position {sym} {order['side']} VWAP={vwap:.4f} size={total_qty:.4f}")
                # clear after flush (positions will be force-closed below)
                self._pending_entries.clear()
        except Exception:
            pass

        # Sprint22: After run, if hedging enabled, compute comparative performance snapshot
        try:
            if getattr(self.signal_engine, "_ph_enabled", False):
                tf = str((self.settings.get("runtime", {}) or {}).get("primary_timeframe", "5m"))
                comp = self.signal_engine._ph_report.compare(self._equity_hedged, self._equity_unhedged, timeframe=tf)
                self.log.info(f"[HEDGE_REPORT] summary={self.signal_engine._ph_report.summary()} deltas={ {k: round(v['delta'],4) for k,v in comp.get('delta',{}).items()} }")
                # Persist (best-effort) if output dir configured
                out_dir = (self.settings.get("backtest", {}) or {}).get("output_dir") or self.settings.get("output_dir")
                if out_dir:
                    import os, json
                    os.makedirs(out_dir, exist_ok=True)
                    try:
                        self.signal_engine._ph_report.to_json(os.path.join(out_dir, f"hedge_report_{symbol}_{tf}.json"))
                    except Exception:
                        pass
                    try:
                        self.signal_engine._ph_report.to_csv(os.path.join(out_dir, f"hedge_snapshots_{symbol}_{tf}.csv"))
                    except Exception:
                        pass
        except Exception:
            pass

        # Force-close all open positions at end of backtest
        if not ohlcv.empty:
            last_close = ohlcv.iloc[-1]["close"]
            last_ts = ohlcv.index[-1]
            for sym, pos in list(self.portfolio.positions.items()):
                try:
                    attr = self._exec_attr.get(sym)
                    if attr:
                        attr.paper_exit_px = float(last_close)
                        attr.real_exit_px = float(last_close)
                except Exception:
                    pass
                self.portfolio.close_position(sym, last_close, last_ts, "EOD")
                try:
                    if sym in self._exec_attr and self.portfolio.trades:
                        t = self.portfolio.trades[-1]
                        attr = self._exec_attr.pop(sym, None)
                        if attr:
                            t['paper_entry_px'] = attr.paper_entry_px
                            t['real_entry_px'] = attr.real_entry_px
                            t['paper_exit_px'] = attr.paper_exit_px
                            t['real_exit_px'] = attr.real_exit_px
                            t['slippage_bps'] = attr.slippage_bps()
                            t['maker_fraction'] = attr.maker_fraction()
                            t['exec_alpha'] = attr.exec_alpha()
                except Exception:
                    pass
                self._risk_levels.pop(sym, None)
                self.log.info(f"Force-closing {sym} open position at EOD.")

        # Sprint 36: write detailed simulated fills CSV & charts
        try:
            if self._sim_fills:
                import os, csv, statistics
                out_dir = (self.settings.get('backtest', {}) or {}).get('output_dir') or self.settings.get('output_dir') or 'reports'
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, 'fills_detailed.csv')
                cols = ['trade_id','symbol','side','venue','order_type','submit_ts','first_fill_ts','last_fill_ts','time_to_first_ms','time_to_full_ms','avg_fill_price','mark_at_submit','slippage_bps','fee_bps','liquidity','partials','post_only_reject','cancel_reason']
                with open(path,'w',newline='',encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in self._sim_fills]
                # basic slippage chart
                try:
                    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
                    slips = [r['slippage_bps'] for r in self._sim_fills if r.get('slippage_bps') is not None]
                    if slips:
                        plt.figure(figsize=(5,3)); plt.hist(slips, bins=30, color='teal', edgecolor='black'); plt.title('Slippage (bps)'); plt.xlabel('bps'); plt.ylabel('freq'); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'slippage_hist.png')); plt.close()
                except Exception:
                    pass
        except Exception as e:  # pragma: no cover
            self.log.warning(f"[BrokerSim] export error: {e}")

        # Sprint 30: aggregate and finalize MTC metrics
        try:
            mb = max(1, self._event_metrics.get('mtc_bars', 0))
            self._event_metrics['mtc_confirm_rate_pct'] = round(self._event_metrics.get('mtc_confirm_bars',0)/mb*100.0,2)
            self._event_metrics['mtc_partial_rate_pct'] = round(self._event_metrics.get('mtc_partial_bars',0)/mb*100.0,2)
            self._event_metrics['mtc_fail_rate_pct'] = round(self._event_metrics.get('mtc_fail_bars',0)/mb*100.0,2)
            # Build histograms (10 bins 0.0-1.0)
            import math
            for key in ['mtc_score_samples_c1','mtc_score_samples_c2']:
                samples = self._event_metrics.get(key) or []
                if samples:
                    bins = [0]*10
                    for v in samples:
                        if v is None: continue
                        try:
                            idx = int(min(9, max(0, math.floor(float(v)*10))))
                            bins[idx]+=1
                        except Exception:
                            continue
                    self._event_metrics[key.replace('samples','hist')] = bins
            # Sprint 34: RR distribution plot if adaptive samples exist
            try:
                rr_samples = self._event_metrics.get('adaptive_rr_samples') or []
                if rr_samples:
                    out_dir = (self.settings.get('backtest', {}) or {}).get('output_dir') or self.settings.get('output_dir')
                    if out_dir:
                        import os
                        os.makedirs(out_dir, exist_ok=True)
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(6,4))
                        plt.hist(rr_samples, bins=30, alpha=0.8, color='steelblue', edgecolor='black')
                        plt.title('Adaptive Exits RR Distribution')
                        plt.xlabel('R Multiple')
                        plt.ylabel('Frequency')
                        plt.axvline(0, color='red', linestyle='--', linewidth=1)
                        try:
                            import numpy as np
                            mean_rr = float(np.mean(rr_samples)) if rr_samples else 0.0
                            plt.axvline(mean_rr, color='green', linestyle='--', linewidth=1, label=f"Mean {mean_rr:.2f}R")
                            plt.legend()
                        except Exception:
                            pass
                        fname = os.path.join(out_dir, f"rr_distribution_{symbol}_{timeframe}.png")
                        try:
                            plt.tight_layout(); plt.savefig(fname)
                            self.log.info(f"[ADAPTIVE] Saved RR distribution plot -> {fname}")
                        except Exception as _e:
                            self.log.warning(f"[ADAPTIVE] Failed to save RR distribution plot: {_e}")
                        finally:
                            try: plt.close()
                            except Exception: pass
            except Exception:
                pass
        except Exception:
            pass

        # Sprint 35: slippage & latency visualizations
        try:
            out_dir = (self.settings.get('backtest', {}) or {}).get('output_dir') or self.settings.get('output_dir')
            if out_dir:
                import os
                os.makedirs(out_dir, exist_ok=True)
                slips = []
                makers = []
                lats = []
                # latency placeholder: since backtest lacks real timing, approximate decision -> entry as 0
                for t in self.portfolio.trades:
                    sbps = t.get('slippage_bps')
                    if isinstance(sbps,(int,float)):
                        slips.append(float(sbps))
                        makers.append(float(t.get('maker_fraction',0.0) or 0.0))
                if slips:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import numpy as np
                    # Heatmap: slippage (x) vs maker_fraction (y) density
                    x = np.array(slips)
                    y = np.array(makers)
                    try:
                        plt.figure(figsize=(5,4))
                        plt.hist2d(x, y, bins=30, cmap='viridis')
                        plt.colorbar(label='Count')
                        plt.xlabel('Slippage (bps)')
                        plt.ylabel('Maker Fraction')
                        plt.title('Slippage vs Maker Fraction')
                        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"slippage_heatmap_{symbol}_{timeframe}.png"))
                        plt.close()
                        self.log.info(f"[S35] Saved slippage heatmap -> {out_dir}")
                    except Exception as _e:
                        self.log.warning(f"[S35] Heatmap save failed: {_e}")
                # Latency distribution (placeholder zeros -> skip if not implemented later)
                if lats:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(5,4))
                        plt.hist(lats, bins=25, color='slateblue', edgecolor='black')
                        plt.xlabel('Latency ms')
                        plt.ylabel('Freq')
                        plt.title('Execution Latency Distribution')
                        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"latency_distribution_{symbol}_{timeframe}.png"))
                        plt.close()
                    except Exception:
                        pass
        except Exception:
            pass

        return self.portfolio.trades, self.portfolio.equity_curve
    # (Unreachable due to earlier return; block retained for reference)

    def _process_bar(self, symbol: str, timeframe: str, timestamp: pd.Timestamp, bar: pd.Series, cooldown_block: bool=False):
        """Processes a single bar of data."""

        # 1) Push bar into FeatureStore
        bar_with_timestamp = bar.to_frame().T
        bar_with_timestamp["timestamp"] = timestamp
        self.feature_store.on_bar(symbol, timeframe, bar_with_timestamp)
        # Ensure DC integration hook registered (idempotent)
        try:
            from ultra_signals.dc.integration import register_with_store
            register_with_store(self.feature_store)
        except Exception:
            pass
        # Sprint 29: if no real BookHealth snapshot present, compute proxy from latest features
        try:
            get_bh = getattr(self.feature_store, 'get_latest_book_health', None)
            bh_now = get_bh(symbol) if callable(get_bh) else None
            if bh_now is None:  # only build proxy if absent
                feats = self.feature_store.get_latest_features(symbol, timeframe) or {}
                if feats:
                    from ultra_signals.market.book_health_proxy import compute_proxies  # local import to avoid backtest cold start cost
                    from ultra_signals.market.book_health import BookHealth
                    proxies = compute_proxies({**feats, 'ohlcv': bar.to_dict()})
                    ts_epoch = int(pd.Timestamp(timestamp).timestamp())
                    proxy_bh = BookHealth(ts=ts_epoch, symbol=symbol,
                                           spread_bps=proxies.get('spread_bps'),
                                           dr=proxies.get('dr'),
                                           impact_50k=proxies.get('impact_50k'),
                                           rv_5s=proxies.get('rv_5s'),
                                           source='proxy')
                    set_bh = getattr(self.feature_store, 'set_latest_book_health', None)
                    if callable(set_bh):
                        set_bh(symbol, proxy_bh)
        except Exception:
            pass

        # 2) Mark-to-market equity
        self.portfolio.equity_curve.append({"timestamp": timestamp, "equity": self.portfolio.current_equity})
        # Update drawdown peak
        try:
            if self.portfolio.current_equity > self._equity_peak:
                self._equity_peak = self.portfolio.current_equity
        except Exception:
            pass
        # Baseline (unhedged) equity path mirrors existing curve
        try:
            self._equity_unhedged.append({"timestamp": timestamp, "equity": self.portfolio.current_equity})
        except Exception:
            pass
        # Hedge mark-to-market & funding accrual
        try:
            if getattr(self.signal_engine, "_ph_enabled", False):
                leader = getattr(self.signal_engine._ph_hedger, "leader", None)
                if leader and symbol == leader:
                    close_px = float(bar.get("close", 0.0))
                    if close_px > 0 and self.signal_engine._ph_hedger.current_hedge_notional != 0:
                        # Update unrealized PnL
                        self._hedge_unrealized_pnl = self.signal_engine._ph_hedger.unrealized_pnl(close_px, self._hedge_avg_price or close_px)
                        # Simple funding: proportional to notional * funding_rate_per_bar
                        ph_cfg = (self.signal_engine.settings.get("portfolio_hedge") or {}) if isinstance(self.signal_engine.settings, dict) else {}
                        costs_cfg = (ph_cfg.get("costs") or {})
                        daily_rate = float(costs_cfg.get("funding_daily", 0.0) or 0.0)
                        if daily_rate:
                            # approximate bars per day from timeframe (fallback 288 for 5m) -> adapt later
                            tf = str((self.signal_engine.settings.get("runtime", {}) or {}).get("primary_timeframe", "5m"))
                            bars_per_day = 288.0
                            try:
                                if tf.endswith('m'):
                                    mins = int(tf[:-1])
                                    bars_per_day = 1440 / mins
                                elif tf.endswith('h'):
                                    hrs = int(tf[:-1])
                                    bars_per_day = 24 / hrs
                            except Exception:
                                bars_per_day = 288.0
                            funding_per_bar = daily_rate / bars_per_day
                            self._hedge_funding_pnl -= abs(self.signal_engine._ph_hedger.current_hedge_notional) * funding_per_bar
                # Record hedged equity curve (baseline + hedge PnL components)
                hedged_equity = self.portfolio.current_equity + self._hedge_unrealized_pnl + self._hedge_realized_pnl + self._hedge_funding_pnl
                self._equity_hedged.append({"timestamp": timestamp, "equity": hedged_equity})
        except Exception:
            pass
        # Sprint22: hedge mark-to-market update (simple)
        try:
            if getattr(self.signal_engine, "_ph_enabled", False):
                leader = getattr(self.signal_engine._ph_hedger, "leader", None)
                if leader and symbol == leader:
                    close_px = float(bar.get("close", 0.0))
                    if close_px > 0:
                        # approximate hedge PnL vs last snapshot price stored in hedger report history (not stored yet, so simple diff not implemented fully)
                        # Placeholder: no separate avg price tracking here, could extend with fills
                        pass
        except Exception:
            pass

        # 3) Exit checks for any open position
        # 3a) Process any pending microstructure entries BEFORE exit logic so fills can finalize this bar
        try:
            if self._pending_entries:
                self._process_pending_entries(timestamp, bar)
        except Exception:
            pass

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

            # Sprint 34 Adaptive exits runtime mechanics (breakeven, trailing, partials)
            try:
                # Adaptive metadata stored in position? replicate from decision vote_detail on entry via _risk_levels meta
                ae = rl.get('adaptive') if isinstance(rl, dict) else None
                # If absent and risk_model stored adaptive_exits in decision, copy on first bar after entry
                if ae is None and isinstance(rl, dict) and rl.get('_meta') and rl['_meta'].get('source') == 'risk_model':
                    # nothing to do here; adaptive stored at entry section below when setting _risk_levels
                    pass
                # Pull position entry price
                entry_px = _pos_entry_price(pos, bar_close)
                risk_per_unit = abs(entry_px - (rl.get('stop') or entry_px)) or 1e-9
                unrealized_R = 0.0
                if risk_per_unit > 0:
                    if side_up == 'LONG':
                        unrealized_R = (bar_close - entry_px) / risk_per_unit
                    else:
                        unrealized_R = (entry_px - bar_close) / risk_per_unit
                # Breakeven move
                if rl.get('breakeven') and rl['breakeven'].get('enabled') and rl.get('stop'):
                    be_trig = float(rl['breakeven'].get('trigger_rr', 1.2))
                    if unrealized_R >= be_trig:
                        # Move stop to entry (lock zero)
                        if side_up == 'LONG' and rl['stop'] < entry_px:
                            rl['stop'] = entry_px
                            try:
                                self._event_metrics['adaptive_breakeven_moves'] += 1
                            except Exception:
                                pass
                        elif side_up == 'SHORT' and rl['stop'] > entry_px:
                            rl['stop'] = entry_px
                            try:
                                self._event_metrics['adaptive_breakeven_moves'] += 1
                            except Exception:
                                pass
                # Trailing stop by ATR step
                if rl.get('trail_config') and rl['trail_config'].get('enabled') and rl.get('meta_atr'):
                    atr_val = float(rl.get('meta_atr') or 0.0)
                    step = float(rl['trail_config'].get('step') or atr_val)
                    if atr_val > 0 and step > 0:
                        if side_up == 'LONG' and bar_close - (rl.get('trail_last_px') or entry_px) >= step:
                            # raise stop preserving distance = initial stop distance? simple ratchet near current - step
                            candidate = bar_close - step
                            if candidate > rl.get('stop', candidate):
                                rl['stop'] = candidate
                                try:
                                    self._event_metrics['adaptive_trailing_adjustments'] += 1
                                except Exception:
                                    pass
                            rl['trail_last_px'] = bar_close
                        elif side_up == 'SHORT' and (rl.get('trail_last_px') or entry_px) - bar_close >= step:
                            candidate = bar_close + step
                            if candidate < rl.get('stop', candidate):
                                rl['stop'] = candidate
                                try:
                                    self._event_metrics['adaptive_trailing_adjustments'] += 1
                                except Exception:
                                    pass
                            rl['trail_last_px'] = bar_close
                # Partial take profits: check levels sequentially
                if rl.get('partials') and isinstance(rl['partials'], list) and rl['partials']:
                    filled_partials = rl.setdefault('_filled_partials', [])
                    for lvl in rl['partials']:
                        if lvl.get('id') in filled_partials:
                            continue
                        lvl_price = float(lvl.get('price'))
                        hit = (side_up == 'LONG' and bar_high >= lvl_price) or (side_up == 'SHORT' and bar_low <= lvl_price)
                        if hit:
                            # Close fraction of position
                            pct = float(lvl.get('pct', 0.0))
                            if pct > 0 and pct < 1.0 and pos.size > 0:
                                close_qty = pos.size * pct
                                # Close portion at level price
                                fill_px = lvl_price
                                if side_up == 'LONG':
                                    pnl = (fill_px - pos.entry_price) * close_qty
                                else:
                                    pnl = (pos.entry_price - fill_px) * close_qty
                                self.current_equity = getattr(self.portfolio, 'current_equity', self.portfolio.current_equity)
                                self.portfolio.current_equity += pnl
                                # Reduce position size
                                pos.size -= close_qty
                                filled_partials.append(lvl.get('id', lvl_price))
                                try:
                                    self._event_metrics['adaptive_partial_fills'] += 1
                                except Exception:
                                    pass
                                self.log.info(f"[ADAPTIVE_PARTIAL] {symbol} hit RR {lvl.get('rr')} pct={pct} new_size={pos.size}")
                                # If closed fully by rounding, mark exit
                                if pos.size <= 1e-9:
                                    self.portfolio.close_position(symbol, fill_px, timestamp, f"PARTIAL_FINAL")
                                    try:
                                        self._event_metrics['adaptive_exit_counts']['PARTIAL_FINAL'] = self._event_metrics['adaptive_exit_counts'].get('PARTIAL_FINAL',0)+1
                                        # Record R multiple for final exit
                                        init_stop = rl.get('initial_stop') or rl.get('stop')
                                        ep = entry_px
                                        if init_stop and ep:
                                            risk = abs(ep - init_stop)
                                            if risk>0:
                                                r_mult = ((fill_px - ep)/risk) if side_up=='LONG' else ((ep - fill_px)/risk)
                                                self._event_metrics['adaptive_rr_samples'].append(float(r_mult))
                                    except Exception:
                                        pass
                                    self._risk_levels.pop(symbol, None)
                                    return
                stop_val = rl.get('stop')
            except Exception:
                pass

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
                # Attach attribution exit metrics if tracking
                try:
                    attr = self._exec_attr.get(symbol)
                    if attr:
                        attr.paper_exit_px = exit_price
                        attr.real_exit_px = exit_price
                except Exception:
                    pass
                # Adaptive exit classification & RR logging
                try:
                    self._event_metrics['adaptive_exit_counts'][exit_reason] = self._event_metrics['adaptive_exit_counts'].get(exit_reason,0)+1
                    rl_snapshot = self._risk_levels.get(symbol)
                    if rl_snapshot:
                        ep = _pos_entry_price(pos, exit_price)
                        init_stop = rl_snapshot.get('initial_stop') or rl_snapshot.get('orig_stop') or rl_snapshot.get('stop')
                        if init_stop and ep:
                            risk = abs(ep - init_stop)
                            if risk>0:
                                r_mult = ((exit_price - ep)/risk) if side_up=='LONG' else ((ep - exit_price)/risk)
                                self._event_metrics['adaptive_rr_samples'].append(float(r_mult))
                except Exception:
                    pass
                self.portfolio.close_position(symbol, exit_price, timestamp, reason=exit_reason)
                # Enrich trade dict just appended
                try:
                    if symbol in self._exec_attr and self.portfolio.trades:
                        t = self.portfolio.trades[-1]
                        attr = self._exec_attr.pop(symbol, None)
                        if attr:
                            t['paper_entry_px'] = attr.paper_entry_px
                            t['real_entry_px'] = attr.real_entry_px
                            t['paper_exit_px'] = attr.paper_exit_px
                            t['real_exit_px'] = attr.real_exit_px
                            t['slippage_bps'] = attr.slippage_bps()
                            t['maker_fraction'] = attr.maker_fraction()
                            t['exec_alpha'] = attr.exec_alpha()
                except Exception:
                    pass
                self._risk_levels.pop(symbol, None)
                # Sprint22: remove exposure for hedging subsystem
                try:
                    if getattr(self.signal_engine, "_ph_enabled", False):
                        self.signal_engine._ph_exposure.remove_symbol(symbol)
                except Exception:
                    pass
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
                    try:
                        if getattr(self.signal_engine, "_ph_enabled", False):
                            self.signal_engine._ph_exposure.remove_symbol(symbol)
                    except Exception:
                        pass
                    self.log.info(f"Closed {symbol} {side_up} at {bar_close} due to {engine_reason}.")
                    return

            # ---------------- NEW (additive): optional bar-based time stop -------------------
            if self._max_bars_in_trade and getattr(pos, "bars_held", 0) >= self._max_bars_in_trade:
                self.portfolio.close_position(symbol, bar_close, timestamp, reason=f"TIME_STOP_{self._max_bars_in_trade}")
                self._risk_levels.pop(symbol, None)
                try:
                    if getattr(self.signal_engine, "_ph_enabled", False):
                        self.signal_engine._ph_exposure.remove_symbol(symbol)
                except Exception:
                    pass
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
        # Sprint 30: capture mtc_gate for metrics (if present)
        try:
            if isinstance(getattr(decision,'vote_detail',{}), dict) and decision.vote_detail.get('mtc_gate'):
                self._last_mtc_gate = decision.vote_detail.get('mtc_gate')
        except Exception:
            pass

        # Sprint 29: collect liquidity gate metrics from decision (if present)
        try:
            lq = getattr(decision, 'vote_detail', {}).get('liquidity_gate') if isinstance(getattr(decision,'vote_detail',{}), dict) else None
            if lq:
                if lq.get('action') == 'VETO':
                    self._event_metrics.setdefault('liquidity_veto_bars', 0)
                    self._event_metrics['liquidity_veto_bars'] += 1
                elif lq.get('action') == 'DAMPEN':
                    self._event_metrics.setdefault('liquidity_dampen_bars', 0)
                    self._event_metrics['liquidity_dampen_bars'] += 1
        except Exception:
            pass

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
            # Sprint 35 pre-trade microstructure vetoes (spoofing / cascade heuristic placeholders)
            try:
                exec_cfg = (self.settings.get('execution') or {}) if isinstance(self.settings, dict) else {}
                if exec_cfg.get('mode') == 'ultra_fast':
                    # Simple spoofing heuristic: if latest BookHealth spread_bps suddenly > 2x 20-bar median -> veto
                    try:
                        get_bh = getattr(self.feature_store, 'get_latest_book_health', None)
                        bh_now = get_bh(symbol) if callable(get_bh) else None
                        if bh_now and getattr(bh_now,'spread_bps', None):
                            hist_spreads = getattr(self, '_bh_spreads', [])
                            hist_spreads.append(float(getattr(bh_now,'spread_bps') or 0.0))
                            if len(hist_spreads) > 40:
                                hist_spreads = hist_spreads[-40:]
                            self._bh_spreads = hist_spreads
                            if len(hist_spreads) >= 20:
                                import numpy as _np
                                med = float(_np.median(hist_spreads[:-1])) if len(hist_spreads)>1 else 0.0
                                if med > 0 and hist_spreads[-1] > med * 2.2:
                                    self.log.info(f"[S35] Spoofing spread spike veto {symbol} spread_bps={hist_spreads[-1]:.2f} med={med:.2f}")
                                    return
                    except Exception:
                        pass
                    # Simple liquidation cascade heuristic: if last 3 bars ATR percentile rising & down move > 1.2*ATR (long entry) skip
                    try:
                        feats = _get_features_for_ts(self.feature_store, symbol, timeframe, timestamp) or {}
                        vol = feats.get('volatility') if isinstance(feats, dict) else None
                        atr = float(getattr(vol,'atr',0.0) or 0.0)
                        atrp = float(getattr(vol,'atr_percentile',0.0) or 0.0)
                        self._atrp_hist = getattr(self,'_atrp_hist', []) + [atrp]
                        if len(self._atrp_hist) > 10:
                            self._atrp_hist = self._atrp_hist[-10:]
                        bar_low = float(bar.get('low', bar.get('close',0.0)))
                        bar_high = float(bar.get('high', bar.get('close',0.0)))
                        bar_close = float(bar.get('close',0.0))
                        bar_open = float(bar.get('open', bar_close))
                        downward_impulse = (bar_open - bar_low)
                        upward_impulse = (bar_high - bar_open)
                        if action == 'LONG' and atr>0 and downward_impulse > 1.2*atr and sum(x>0.9 for x in self._atrp_hist[-3:])>=2:
                            self.log.info(f"[S35] Cascade veto LONG {symbol} downward_impulse={downward_impulse:.4f} atr={atr:.4f}")
                            return
                        if action == 'SHORT' and atr>0 and upward_impulse > 1.2*atr and sum(x>0.9 for x in self._atrp_hist[-3:])>=2:
                            self.log.info(f"[S35] Cascade veto SHORT {symbol} upward_impulse={upward_impulse:.4f} atr={atr:.4f}")
                            return
                    except Exception:
                        pass
            except Exception:
                pass
            # Event gate metadata instrumentation: counterfactual if veto converted decision earlier (engine recorded pre_decision)
            try:
                eg = getattr(decision, 'vote_detail', {}).get('event_gate') if isinstance(getattr(decision,'vote_detail',{}), dict) else None
                if eg and eg.get('action') == 'DAMPEN':
                    self._event_metrics['dampen_trades'] += 1
            except Exception:
                pass
            if cooldown_block:
                self.log.info(f"[EVENT] Cooldown blocking new entry for {symbol} at {timestamp}")
                return
            # Microstructure mode: create pending maker entry instead of immediate position
            if self._micro_cfg.get("enabled"):
                if symbol not in self._pending_entries and symbol not in self.portfolio.positions:
                    close_px = float(bar["close"])
                    size_est = self.portfolio.position_size(symbol, close_px)
                    self._pending_entries[symbol] = {
                        "side": action,
                        "limit_px": close_px,  # maker attempt at current close
                        "posted_ts": timestamp,
                        "posted_bar": int(pd.Timestamp(timestamp).timestamp()),
                        "remaining": size_est,
                        "fills": [],  # (ts, px, qty, maker)
                        "bars_waited": 0,
                    }
                    # Initialize attribution skeleton (paper entry at decision close)
                    if ExecAttribution:
                        self._exec_attr[symbol] = ExecAttribution(paper_entry_px=close_px)
                    self.log.info(f"[MICRO] Posted pending {action} entry for {symbol} @ {close_px}")
                else:
                    self.log.debug(f"[MICRO] Pending entry already exists for {symbol}, skipping new signal")
                return  # do NOT proceed to portfolio.open_position yet
            # ----------------- MODIFIED (additive guard): global entries budget -----------------
            if self._global_entry_budget:
                max_total = int(self._global_entry_budget)
                if self.entries_count_total >= max_total:
                    ev = RiskEvent(
                        ts=int(pd.Timestamp(timestamp).timestamp()),
                        symbol=symbol,
                        reason="GLOBAL_ENTRY_BUDGET",
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
                    size_quote = rm.get('position_size_playbook') or rm.get('position_size')
                    if size_quote and close_px > 0:
                        size_adaptive = float(size_quote) / float(close_px)
                        if size_adaptive > 0:
                            size = size_adaptive
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
            # Sprint 32: AdvancedSizer integration (portfolio-aware)
            try:
                from ultra_signals.engine.sizing.advanced_sizer import AdvancedSizer
                if (self.settings.get('sizer') or {}).get('enabled', False):
                    if not hasattr(self, '_adv_sizer'):
                        try:
                            self._adv_sizer = AdvancedSizer(self.settings)
                        except Exception:
                            self._adv_sizer = None
                    adv = getattr(self, '_adv_sizer', None)
                    if adv and getattr(adv, 'enabled', False):
                        # Build open_positions risk list (approx by using size * stop distance once we have risk_model; else assume risk_pct ~ base_risk_pct)
                        open_positions = []
                        for sym,pos in self.portfolio.positions.items():
                            try:
                                # Use stored risk amount if available else fallback to 0.5% equity
                                ra = getattr(pos,'risk_amount_at_entry', None)
                                if ra is None:
                                    ra = self.portfolio.current_equity*0.005
                                open_positions.append({'symbol': sym,'risk_amount': float(ra),'side': getattr(pos,'side','LONG')})
                            except Exception:
                                continue
                        p_meta = None; mtc_status=None; lq_action=None; atr_val=None
                        try:
                            vd_local = getattr(decision,'vote_detail',{}) or {}
                            mg = vd_local.get('meta_gate'); mt = vd_local.get('mtc_gate'); lq = vd_local.get('liquidity_gate')
                            if mg: p_meta = mg.get('p')
                            if mt: mtc_status = mt.get('status')
                            if lq: lq_action = lq.get('action')
                        except Exception:
                            pass
                        # ATR from volatility features if present in risk_model meta else compute via stop
                        if symbol in self._risk_levels:
                            try:
                                atr_val = self._risk_levels[symbol]['_meta'].get('atr')
                            except Exception:
                                pass
                        # Current drawdown fraction
                        dd_frac = 0.0
                        try:
                            dd_frac = max(0.0, (self._equity_peak - self.portfolio.current_equity) / self._equity_peak)
                        except Exception:
                            dd_frac = 0.0
                        adv_res = adv.compute(symbol, action, close_px, float(self.portfolio.current_equity), {
                            'p_meta': p_meta,
                            'mtc_status': mtc_status,
                            'liquidity_gate_action': lq_action,
                            'atr': atr_val,
                            'stop_distance': None,
                            'drawdown': dd_frac,
                            'open_positions': open_positions,
                        })
                        if adv_res and adv_res.qty > 0:
                            size = adv_res.qty
                            # store breakdown into decision.vote_detail for trades export
                            try:
                                decision.vote_detail.setdefault('advanced_sizer', adv_res.breakdown)
                            except Exception:
                                pass
                        else:
                            # zero size => block entry
                            if adv_res and adv_res.qty == 0:
                                self.log.info(f"[SIZER_BLOCK] {symbol} {action} reasons={adv_res.breakdown.get('reasons')}")
                                return
            except Exception as _e:
                self.log.debug(f"AdvancedSizer integration error: {_e}")

            self.log.info(f"Trade for {symbol} at {timestamp} ALLOWED by portfolio gate.")
            self.portfolio.open_position(symbol, action, close_px, timestamp, size)
            # Sprint 36: if broker_sim enabled, replace entry price with simulated fills
            try:
                if self._broker_adapter:
                    sim_res = self._broker_adapter.execute_fast_order(symbol=symbol, side=action, size=size, price=close_px, settings=self.settings)
                    if sim_res.accepted and sim_res.order and sim_res.order.get('price'):
                        fill_px = float(sim_res.order['price'])
                        # adjust position entry (simple overwrite) and track slippage
                        pos_obj = self.portfolio.positions.get(symbol)
                        if pos_obj:
                            setattr(pos_obj,'entry_price', fill_px)
                        # mark reference mid = close_px
                        slip_bps = (fill_px - close_px)/close_px*10_000 if close_px else 0.0
                        self._sim_fills.append({
                            'trade_id': f"{symbol}-{int(pd.Timestamp(timestamp).timestamp())}",
                            'symbol': symbol,
                            'side': action,
                            'venue': sim_res.venue,
                            'order_type': 'MARKET',
                            'submit_ts': int(pd.Timestamp(timestamp).timestamp()*1000),
                            'first_fill_ts': int(pd.Timestamp(timestamp).timestamp()*1000),
                            'last_fill_ts': int(pd.Timestamp(timestamp).timestamp()*1000),
                            'time_to_first_ms': 0,
                            'time_to_full_ms': 0,
                            'avg_fill_price': fill_px,
                            'mark_at_submit': close_px,
                            'slippage_bps': round(slip_bps,3),
                            'fee_bps': 4.0 if action in ('LONG','SHORT') else 0.0,
                            'liquidity': 'TAKER',
                            'partials': 1,
                            'post_only_reject': False,
                            'cancel_reason': ''
                        })
            except Exception as e:
                self.log.debug(f"[BrokerSim] entry integration error: {e}")
            # Sprint 33: capture portfolio risk allocator metrics for this entry if present
            try:
                pr_meta = getattr(decision,'vote_detail',{}).get('portfolio_risk') if isinstance(getattr(decision,'vote_detail',{}), dict) else None
                if pr_meta and isinstance(pr_meta, dict):
                    metrics = pr_meta.get('metrics') or {}
                    if metrics:
                        row = dict(metrics); row['ts'] = int(pd.Timestamp(timestamp).timestamp()); row['symbol'] = symbol; row['entry'] = True
                        self.signal_engine._pr_metrics_ts.append(row)
            except Exception:
                pass
            # Attach risk amount & stop distance to position for later R calc / ledger
            try:
                pos_obj = self.portfolio.positions.get(symbol)
                if pos_obj:
                    # Stop distance: use risk model stop if available else adv stop distance from breakdown
                    stop_px = None
                    try:
                        rm = decision.vote_detail.get('risk_model') if isinstance(decision.vote_detail, dict) else None
                        stop_px = rm.get('stop_loss') if rm else None
                    except Exception:
                        stop_px = None
                    if stop_px and stop_px > 0:
                        stop_dist = abs(close_px - float(stop_px))
                    else:
                        stop_dist = decision.vote_detail.get('advanced_sizer',{}).get('stop_distance') or 0.0
                    risk_amount = 0.0
                    if stop_dist and stop_dist > 0:
                        risk_amount = stop_dist * size
                    setattr(pos_obj,'risk_amount_at_entry', risk_amount)
                    setattr(pos_obj,'adv_stop_distance', stop_dist)
            except Exception:
                pass
            self.entries_count_total += 1
            self.log.info(f"INFO Opened {action} position for {symbol} at {close_px} with size {size:.4f}")
            # Sprint 30: tag position with latest MTC metadata for PnL stratification
            try:
                pos = self.portfolio.positions.get(symbol)
                mtc_meta = getattr(self, '_last_mtc_gate', None)
                if pos and mtc_meta:
                    setattr(pos,'mtc_status', mtc_meta.get('status'))
                    setattr(pos,'mtc_action', mtc_meta.get('action'))
                    setattr(pos,'mtc_scores', mtc_meta.get('scores'))
                    setattr(pos,'mtc_observe_only', mtc_meta.get('observe_only'))
            except Exception:
                pass

            # Apply stop widen if dampen event provided widen_stop_mult
            try:
                eg = getattr(decision, 'vote_detail', {}).get('event_gate') if isinstance(getattr(decision,'vote_detail',{}), dict) else None
                if eg and eg.get('widen_stop_mult') and symbol in self._risk_levels:
                    rl = self._risk_levels.get(symbol)
                    mult = float(eg['widen_stop_mult'])
                    if rl and rl.get('stop'):
                        rl['stop'] = rl['stop'] * mult
                        rl.setdefault('_meta', {})['event_gate_stop_widened'] = mult
            except Exception:
                pass

            if symbol not in self._risk_levels or (self._risk_levels.get(symbol, {}).get('stop') is None and self._risk_levels.get(symbol, {}).get('tp') is None):
                side_up = str(action).upper()
                rl_calc = _compute_initial_risk_levels(self.feature_store, symbol, timeframe, timestamp, side_up, close_px, self.settings)
                # Attempt to attach adaptive exits metadata from decision
                rl_adaptive = {"stop": rl_calc["stop"], "tp": rl_calc["tp"]}
                try:
                    ae = getattr(decision,'vote_detail',{}).get('adaptive_exits') if isinstance(getattr(decision,'vote_detail',{}), dict) else None
                    if ae:
                        rl_adaptive['stop'] = ae.get('stop_price', rl_adaptive['stop'])
                        rl_adaptive['tp'] = ae.get('target_price', rl_adaptive['tp'])
                        rl_adaptive['breakeven'] = ae.get('breakeven')
                        rl_adaptive['trail_config'] = ae.get('trail_config')
                        rl_adaptive['meta_atr'] = ae.get('meta',{}).get('atr')
                        # Tag partials with ids for tracking
                        parts = []
                        for idx, p in enumerate(ae.get('partial_tp') or []):
                            parts.append({'id': idx, **p})
                        rl_adaptive['partials'] = parts
                        rl_adaptive.setdefault('_meta', {})['adaptive'] = True
                except Exception:
                    pass
                self._risk_levels[symbol] = rl_adaptive
                try:
                    if rl_adaptive.get('stop') is not None:
                        rl_adaptive['initial_stop'] = rl_adaptive['stop']
                    if rl_adaptive.get('tp') is not None:
                        rl_adaptive['initial_tp'] = rl_adaptive['tp']
                except Exception:
                    pass
                self.log.info(
                    f"Risk levels set for {symbol} on entry: stop={rl_calc['stop']}, tp={rl_calc['tp']} (source={rl_calc['source']})"
                )

            try:
                if getattr(self.signal_engine, "_ph_enabled", False):
                    side_sign = 1.0 if action == "LONG" else -1.0
                    notional = close_px * size * side_sign
                    self.signal_engine._ph_exposure.update_position(symbol, notional)
                    beta_p = self.signal_engine._ph_corr.portfolio_beta(
                        self.signal_engine._ph_exposure.symbol_notionals,
                        self.signal_engine._ph_equity,
                    )
                    leader = getattr(self.signal_engine._ph_hedger, "leader", None)
                    leader_price = close_px if symbol == leader else None
                    if leader and leader_price is None:
                        try:
                            feats_leader = _get_features_for_ts(self.feature_store, leader, timeframe, timestamp) or {}
                            lp = _safe_get_attr(feats_leader, "close", None) or _safe_get_attr(feats_leader, "price", None)
                            leader_price = float(lp) if lp else None
                        except Exception:
                            leader_price = None
                    if leader_price is None or leader_price <= 0:
                        leader_price = close_px
                    plan = self.signal_engine._ph_hedger.compute_plan(
                        bar_index=int(pd.Timestamp(timestamp).timestamp()),
                        portfolio_beta=beta_p,
                        equity=self.signal_engine._ph_equity,
                        beta_target=0.0,
                    )
                    if plan.action in ("OPEN", "ADJUST", "CLOSE") and plan.delta_notional:
                        try:
                            if plan.action == "CLOSE" and self._hedge_unrealized_pnl:
                                self._hedge_realized_pnl += self._hedge_unrealized_pnl
                                self._hedge_unrealized_pnl = 0.0
                        except Exception:
                            pass
                        try:
                            if plan.action in ("OPEN", "ADJUST"):
                                prev_notional = self.signal_engine._ph_hedger.current_hedge_notional
                                new_notional_total = plan.target_notional
                                if abs(new_notional_total) < 1e-9:
                                    self._hedge_avg_price = leader_price
                                else:
                                    if prev_notional == 0:
                                        self._hedge_avg_price = leader_price
                                    else:
                                        w_prev = prev_notional
                                        w_new = plan.delta_notional
                                        if (w_prev + w_new) != 0:
                                            self._hedge_avg_price = (
                                                self._hedge_avg_price * w_prev + leader_price * w_new
                                            ) / (w_prev + w_new)
                        except Exception:
                            pass
                        self.signal_engine._ph_hedger.apply_plan(plan, bar_index=int(pd.Timestamp(timestamp).timestamp()))
                        self.log.info(
                            f"[HEDGE] {plan.action} delta_notional={plan.delta_notional:.2f} target={plan.target_notional:.2f} beta_p={beta_p:.4f}"
                        )
            except Exception:
                pass

    # ------------------ Microstructure Helpers ------------------
    def _process_pending_entries(self, timestamp: pd.Timestamp, bar: pd.Series):
        if not self._pending_entries:
            return
        # Process only entries for the current symbol context (run is per symbol)
        for sym, order in list(self._pending_entries.items()):
            side = order["side"]
            limit_px = float(order["limit_px"])
            remaining = float(order["remaining"])
            if remaining <= 0:
                continue
            high = float(bar.get("high", bar.get("close", limit_px)))
            low = float(bar.get("low", bar.get("close", limit_px)))
            close_px = float(bar.get("close", limit_px))
            order["bars_waited"] += 1
            bars_waited = order["bars_waited"]
            deadline = int(self._micro_cfg.get("taker_deadline_bars", 3))
            touched = (side == "LONG" and low <= limit_px) or (side == "SHORT" and high >= limit_px)
            p_min = float(self._micro_cfg.get("fill_prob_min", 0.05))
            p_max = float(self._micro_cfg.get("fill_prob_max", 0.95))
            base_p = 0.15
            if touched:
                base_p = 0.55
            if side == "LONG" and close_px < limit_px:
                base_p += 0.1
            if side == "SHORT" and close_px > limit_px:
                base_p += 0.1
            prob = max(p_min, min(p_max, base_p))

            if bars_waited >= deadline:
                # Fill remainder as taker at close
                fill_qty = remaining
                self._record_pending_fill(sym, timestamp, close_px, fill_qty, maker=False)
                order["remaining"] = 0.0
            else:
                if touched:
                    fill_qty = min(remaining, remaining * prob)
                    fill_qty = min(fill_qty, remaining * 0.6)
                    if fill_qty > 0:
                        self._record_pending_fill(sym, timestamp, limit_px, fill_qty, maker=True)
                        order["remaining"] = remaining - fill_qty

            if order["remaining"] <= 1e-9:
                # convert to real position at VWAP of fills
                fills = order["fills"]
                total_qty = sum(f[2] for f in fills)
                if total_qty <= 0:
                    continue
                vwap = sum(f[1]*f[2] for f in fills)/total_qty
                self.portfolio.open_position(sym, side, vwap, timestamp, total_qty)
                self.entries_count_total += 1
                self.log.info(f"[MICRO] Converted pending -> position {sym} {side} VWAP={vwap:.4f} size={total_qty:.4f}")
                # remove from pending
                self._pending_entries.pop(sym, None)

    def _record_pending_fill(self, symbol: str, ts: pd.Timestamp, px: float, qty: float, maker: bool):
        order = self._pending_entries.get(symbol)
        if not order:
            return
        order.setdefault("fills", []).append((ts, px, qty, maker))
        # attribution
        try:
            attr = self._exec_attr.get(symbol)
            if attr:
                attr.add_fill(px=px, qty=qty, maker=maker, ts_ms=int(pd.Timestamp(ts).timestamp()*1000))
        except Exception:
            pass


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
