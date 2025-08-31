"""Engine worker consuming market events and producing execution plans.

This is a highly simplified version for initial Sprint 21 scaffolding. It:
  * Consumes Kline closed events only (placeholder for full feature pipeline)
  * Emits a NOOP plan structure for demonstration
  * Respects a per-event latency budget; if exceeded emits abstain
"""
from __future__ import annotations
import asyncio
import time
import os
import sys
import traceback
from typing import Dict, Any, Optional
from loguru import logger
from ultra_signals.core.events import KlineEvent, MarketEvent, BookTickerEvent
from ultra_signals.core.events import AggTradeEvent
from ultra_signals.features.vpin import VPINEngine, fuse_toxicity
try:
    from ultra_signals.guards.live_guards import pre_bar_guard
except Exception:  # pragma: no cover
    pre_bar_guard = None  # type: ignore
try:  # Sprint 35 ultra-fast execution integration
    from ultra_signals.execution.fast_router import execute_fast_order
except Exception:  # pragma: no cover
    execute_fast_order = None  # type: ignore
try:
    from ultra_signals.execution.pricing import build_exec_plan
    from ultra_signals.execution.guards import pre_trade_guards
except Exception:  # pragma: no cover
    build_exec_plan = None  # type: ignore
    pre_trade_guards = None  # type: ignore
try:
    from ultra_signals.dq.venue_merge import composite_mid as _dq_composite_mid
except Exception:  # pragma: no cover
    _dq_composite_mid = None  # type: ignore
try:  # Telegram transport (optional)
    from ultra_signals.core.custom_types import EnsembleDecision, SubSignal  # type: ignore
    from ultra_signals.transport.telegram import send_decision  # type: ignore
except Exception:  # pragma: no cover
    EnsembleDecision = None  # type: ignore
    SubSignal = None  # type: ignore
    send_decision = None  # type: ignore
try:  # Minimal strategy components (lightweight placeholder)
    from ultra_signals.core.feature_store import FeatureStore  # type: ignore
    from ultra_signals.engine.risk_filters import apply_filters  # type: ignore
    from ultra_signals.core.custom_types import Signal as LegacySignal, SignalType, SubSignal, EnsembleDecision  # type: ignore
except Exception:  # pragma: no cover
    FeatureStore = None  # type: ignore
    apply_filters = None  # type: ignore
    LegacySignal = None  # type: ignore
    SignalType = None  # type: ignore


class EngineWorker:
    def __init__(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue, latency_budget_ms: int = 150, metrics=None, safety=None, extra_delay_ms: int = 0, feature_writer=None):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.latency_budget_ms = latency_budget_ms
        self._running = False
        # injected helpers / test hooks
        self.metrics = metrics
        self.safety = safety
        self.extra_delay_ms = extra_delay_ms  # for tests to force deadline breach
        # optional feature writer for persisting per-slice telemetry
        self.feature_writer = feature_writer
        # optional references set by runner
        self.feed_ref = None
        self._settings = None
        self._exec_cfg = None
        # per-symbol VPIN engines (live incremental)
        self._vpin_engines = {}
        # vpin per-symbol transient state (normal, toxic, cooldown)
        self._vpin_states = {}

    async def run(self):
        self._running = True
        # Sprint 35 microstructure state
        self._book_snaps = []  # list[(mono_ts, bid, ask, bid_sz, ask_sz)]
        self._spoof_flag_until = 0.0
        self._flip_flag_until = 0.0
        self._cascade_flag_until = 0.0
        self._last_direction_changes = []  # list[(mono_ts, direction)] where direction in {+1,-1}
        # Main event loop
        while self._running:
            try:
                evt: MarketEvent = await self.in_queue.get()
            except asyncio.CancelledError:
                break
            # Fast forced-signal path on BookTicker when canary override enabled (allows rapid wiring test without waiting for bar close)
            if isinstance(evt, BookTickerEvent):
                force_signals_bt = (
                    os.getenv('CANARY_FORCE_SIGNALS') == '1' or
                    ((self._settings or {}).get('runtime', {}) or {}).get('canary_force_signals') is True
                )
                if force_signals_bt:
                    now_m = time.perf_counter()
                    if not hasattr(self, '_last_forced_emit'):
                        self._last_forced_emit = {}
                    last_emit = self._last_forced_emit.get(evt.symbol, 0)
                    if (now_m - last_emit) >= 2.0:  # emit at most every 2s per symbol
                        side = 'LONG' if ((getattr(self, '_canary_flip_state', 0) % 2) == 0) else 'SHORT'
                        self._canary_flip_state = getattr(self, '_canary_flip_state', 0) + 1
                        plan = {
                            'ts': int(time.time()),
                            'symbol': evt.symbol,
                            'timeframe': 'TICK',
                            'side': side,
                            'price': evt.best_ask or evt.best_bid,
                            'size': 1.0,
                            'qty': 1.0,
                            'version': 1,
                            '_decision_monotonic': time.perf_counter(),
                            'exec_plan': {'order_type': 'MARKET', 'post_only': False},
                        }
                        try:
                            self.out_queue.put_nowait(plan)
                            logger.debug(f"[EngineWorker][FORCE-BT] emitted tick plan side={side} sym={evt.symbol}")
                        except asyncio.QueueFull:
                            logger.error('[EngineWorker] order queue full; dropping forced tick plan')
                        self._last_forced_emit[evt.symbol] = now_m
                        # Emit synthetic EnsembleDecision for Telegram (canary wiring)
                        try:
                            if side in ('LONG','SHORT') and EnsembleDecision and SubSignal and send_decision:
                                # Resolve telegram settings (dict with 'telegram' key expected by transport)
                                tcfg = None
                                sref = getattr(self, '_settings', None)
                                if isinstance(sref, dict) and 'transport' in sref:
                                    # full settings dict already attached
                                    tcfg = (sref.get('transport') or {})
                                elif self.feed_ref and hasattr(self.feed_ref, '_settings'):
                                    frs = getattr(self.feed_ref, '_settings')
                                    try:
                                        if hasattr(frs, 'transport') and hasattr(frs.transport, 'telegram'):
                                            tcfg = {'telegram': frs.transport.telegram.model_dump() if hasattr(frs.transport.telegram,'model_dump') else dict(getattr(frs.transport,'telegram',{}))}
                                    except Exception:
                                        pass
                                if tcfg and 'telegram' in tcfg:
                                    # Build minimal decision object
                                    sub = SubSignal(ts=plan['ts'], symbol=plan['symbol'], tf=plan['timeframe'], strategy_id='canary_forced', direction=side, confidence_calibrated=0.60, reasons={})
                                    # Derive simplistic pseudo entry/SL/TP for display (spread around current price)
                                    px = float(plan.get('price') or 0.0) or 1.0
                                    rr_tp_mult = 1.004 if side=='LONG' else 0.996
                                    rr_sl_mult = 0.996 if side=='LONG' else 1.004
                                    entry = px
                                    tp = px * rr_tp_mult
                                    sl = px * rr_sl_mult
                                    rr = abs((tp-entry)/(entry-sl)) if (entry-sl)!=0 else 1.0
                                    dec = EnsembleDecision(
                                        ts=plan['ts'], symbol=plan['symbol'], tf=plan['timeframe'], decision=side, confidence=0.60,
                                        subsignals=[sub],
                                        vote_detail={'agree':1,'total':1,'profile':'canary','weighted_sum':1.0,
                                                     'pre_trade':{'p_win':0.55,'regime':'canary','veto_count':0,'lat_ms':{'p50':10.0,'p90':15.0}},
                                                     'entry':entry,'sl':sl,'tp':tp,'lev':1.0,'risk_pct':0.02,'rr':rr},
                                        vetoes=[])
                                    # fire and forget
                                    asyncio.create_task(send_decision(dec, {'telegram': tcfg.get('telegram')}))
                        except Exception:
                            pass
            if isinstance(evt, BookTickerEvent):
                # cache best bid/ask plus sizes (if provided)
                setattr(self, "_last_bidask", (evt.symbol, evt.best_bid, evt.best_ask))
                try:
                    bid_sz = getattr(evt, 'best_bid_size', 0.0) or 0.0
                    ask_sz = getattr(evt, 'best_ask_size', 0.0) or 0.0
                except Exception:
                    bid_sz = ask_sz = 0.0
                now_m = time.perf_counter()
                try:
                    self._book_snaps.append((now_m, float(evt.best_bid or 0), float(evt.best_ask or 0), float(bid_sz), float(ask_sz)))
                    # keep last 1.5s of snapshots
                    cutoff = now_m - 1.5
                    self._book_snaps = [s for s in self._book_snaps if s[0] >= cutoff]
                except Exception:
                    pass
                # Spoofing / liquidity vanish: >=70% total top size drop within 200ms
                try:
                    if len(self._book_snaps) >= 2:
                        cur = self._book_snaps[-1]
                        prev = self._book_snaps[-2]
                        dt = cur[0] - prev[0]
                        if dt <= 0.2:
                            prev_total = (prev[3] + prev[4]) or 1.0
                            cur_total = cur[3] + cur[4]
                            if prev_total > 0 and cur_total / prev_total <= 0.30:  # >=70% vanish
                                self._spoof_flag_until = now_m + 0.6  # 600ms cooloff
                except Exception:
                    pass
                # Flip count: >3 alternating micro-direction changes (bid price up/down) in 300ms window
                try:
                    if len(self._book_snaps) >= 2:
                        p2 = self._book_snaps[-2][1]
                        p1 = self._book_snaps[-1][1]
                        if p1 != p2:
                            direction = 1 if p1 > p2 else -1
                            self._last_direction_changes.append((now_m, direction))
                            # retain 0.35s
                            cutoff_dir = now_m - 0.35
                            self._last_direction_changes = [d for d in self._last_direction_changes if d[0] >= cutoff_dir]
                            if len(self._last_direction_changes) >= 4:
                                # check alternating pattern
                                dirs = [d[1] for d in self._last_direction_changes[-4:]]
                                if dirs == [1,-1,1,-1] or dirs == [-1,1,-1,1]:
                                    self._flip_flag_until = now_m + 0.5
                except Exception:
                    pass
                # Cascade: rapid price move >0.3% within 300ms + spread widening >2x rolling median spread
                try:
                    if len(self._book_snaps) >= 3:
                        # price move based on mid
                        mids = [(s[1]+s[2])/2.0 for s in self._book_snaps]
                        span = self._book_snaps[-1][0] - self._book_snaps[0][0]
                        if span <= 0.35 and self._book_snaps[0][1] and self._book_snaps[-1][1]:
                            m0 = mids[0]; m1 = mids[-1]
                            if m0 > 0 and abs(m1-m0)/m0 >= 0.003:  # 0.3%
                                # compute rolling median spread (last up to 10)
                                spreads = [ (s[2]-s[1]) for s in self._book_snaps[-10:] if s[1]>0 and s[2]>0 ]
                                if len(spreads) >= 3:
                                    spreads_sorted = sorted(spreads)
                                    med = spreads_sorted[len(spreads_sorted)//2]
                                    cur_spread = spreads[-1]
                                    if med>0 and cur_spread/med >= 2.0:
                                        self._cascade_flag_until = time.perf_counter() + 0.8
                except Exception:
                    pass
                continue
            # ingest aggTrade events into local VPIN engines for realtime toxicity estimation
            if isinstance(evt, AggTradeEvent):
                try:
                    sym = evt.symbol
                    eng = self._vpin_engines.get(sym)
                    if eng is None:
                        # read default bucket from feed settings if available
                        try:
                            cfg = getattr(self.feed_ref, 'settings', {})
                            vcfg = ((cfg or {}).get('features', {}) or {}).get('vpin', {})
                            V_bucket = float(vcfg.get('V_bucket', 250000))
                            K_buckets = int(vcfg.get('K_buckets', 50))
                        except Exception:
                            V_bucket = 250000.0; K_buckets = 50
                        eng = VPINEngine(V_bucket=V_bucket, K_buckets=K_buckets)
                        self._vpin_engines[sym] = eng
                    # convert to tuple (ts, price, qty, is_buyer_maker)
                    eng.ingest_trade((evt.timestamp, float(evt.price), float(evt.quantity), bool(getattr(evt, 'is_buyer_maker', False))), book_top=(getattr(self,'_last_bidask', None) and {'bid': self._last_bidask[1], 'ask': self._last_bidask[2], 'B': 0.0, 'A': 0.0}))
                except Exception:
                    pass
                continue
            if isinstance(evt, KlineEvent) and evt.closed:
                # DQ bar-level guard
                try:
                    if pre_bar_guard:
                        # timeframe ms heuristic
                        tf = evt.timeframe
                        tf_ms = 60_000
                        if tf.endswith('m'): tf_ms = int(tf[:-1]) * 60_000
                        elif tf.endswith('h'): tf_ms = int(tf[:-1]) * 3_600_000
                        pre_bar_guard(evt.symbol, tf_ms, getattr(self,'_settings',{}) or {})
                except Exception as e:
                    logger.error(f"engine.pre_bar_guard_block symbol={evt.symbol} err={e}")
                    continue
                try:
                    store_ref_dbg = getattr(self.feed_ref, 'feature_store', None) or getattr(self, 'feature_store', None)
                    warm_ct = None
                    if store_ref_dbg is not None and hasattr(store_ref_dbg, 'get_warmup_status'):
                        try:
                            warm_ct = store_ref_dbg.get_warmup_status(evt.symbol, evt.timeframe)
                        except Exception:
                            warm_ct = None
                    logger.info(f"[EngineWorker][BAR] closed kline sym={evt.symbol} tf={evt.timeframe} warmup_ct={warm_ct} store_attached={store_ref_dbg is not None}")
                except Exception:
                    pass
                started = time.perf_counter()
                # Force artificial delay if requested (test hook)
                if self.extra_delay_ms:
                    await asyncio.sleep(self.extra_delay_ms / 1000.0)
                # --- BASIC STRATEGY & SIGNAL PIPELINE (minimal viable) ---
                await asyncio.sleep(0)  # cooperative yield
                # --- Minimal strategy placeholder -> build naive signals ---
                strategy_signals = []
                try:
                    if LegacySignal and SignalType:
                        side_hint = 'LONG' if evt.close >= evt.open else 'SHORT'
                        sig = LegacySignal(symbol=evt.symbol, timeframe=evt.timeframe, decision=side_hint, signal_type=SignalType.TREND_FOLLOWING, price=evt.close, confidence=0.55)
                        strategy_signals.append(sig)
                except Exception:
                    pass
                # Apply risk filters
                filtered_signals = []
                for sig in strategy_signals:
                    if self.metrics and hasattr(self.metrics, 'record_candidate'):
                        try:
                            self.metrics.record_candidate()
                        except Exception:
                            pass
                    store_ref = None
                    try:
                        store_ref = getattr(self.feed_ref, 'feature_store', None) or getattr(self.feed_ref, 'store', None) or getattr(self, 'feature_store', None)
                    except Exception:
                        store_ref = None
                    if apply_filters and store_ref is not None:
                        try:
                            fr = apply_filters(sig, store_ref, self._settings or {}, self.metrics, record_candidate=False)  # type: ignore
                            if not fr.passed:
                                if self.metrics and hasattr(self.metrics, 'record_block'):
                                    try:
                                        self.metrics.record_block(fr.reason or 'BLOCK')
                                    except Exception:
                                        pass
                                continue
                        except Exception as e:
                            # treat as block if filters explode
                            logger.error(f"[EngineWorker] FILTER_ERR: Exception during apply_filters for {sig.symbol} {sig.timeframe}: {e}")
                            traceback.print_exc(file=sys.stderr) # Print full traceback to stderr
                            if self.metrics and hasattr(self.metrics, 'record_block'):
                                try:
                                    self.metrics.record_block('FILTER_ERR')
                                except Exception:
                                    pass
                            continue
                    elif apply_filters and store_ref is None:
                        if self.metrics and hasattr(self.metrics, 'record_block'):
                            try:
                                self.metrics.record_block('NO_STORE')
                            except Exception:
                                pass
                        continue
                    if self.metrics and hasattr(self.metrics, 'record_allowed'):
                        try:
                            self.metrics.record_allowed()
                        except Exception:
                            pass
                    filtered_signals.append(sig)
                # Use first allowed signal to drive side
                if filtered_signals:
                    side = filtered_signals[0].decision
                ingest = getattr(evt, "_ingest_monotonic", None)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if ingest is not None and self.metrics:
                    self.metrics.latency_tick_to_decision.observe((time.perf_counter() - ingest)*1000.0)
                if elapsed_ms > self.latency_budget_ms:
                    logger.warning(f"[EngineWorker] Deadline exceeded {elapsed_ms:.1f}ms >= {self.latency_budget_ms}ms – abstain")
                    continue
                if self.safety and self.safety.state.paused:
                    logger.warning("[EngineWorker] Safety paused – skipping plan emission")
                    continue
                # Spread guardrail & plan construction
                bidask = getattr(self, "_last_bidask", None)
                if bidask and bidask[0] == evt.symbol:
                    _, bid, ask = bidask
                    if bid and ask and ask > 0:
                        spread_pct = (ask - bid) / ask * 100
                        max_spread_pct = getattr(self.safety, 'max_spread_pct', 0.05) if self.safety else 0.05
                        if spread_pct > max_spread_pct:
                            logger.warning(f"[EngineWorker] Spread {spread_pct:.3f}% > {max_spread_pct}% – abstain")
                            continue
                # Placeholder: in future incorporate real decision engine.
                # Canary override: allow forcing alternating LONG/SHORT decisions to test end-to-end wiring.
                side = "FLAT" if not filtered_signals else filtered_signals[0].decision  # default fallback
                force_signals = (
                    os.getenv('CANARY_FORCE_SIGNALS') == '1' or
                    ((self._settings or {}).get('runtime', {}) or {}).get('canary_force_signals') is True
                )
                if force_signals and not filtered_signals:
                    # simple deterministic flip-flop so tests are reproducible
                    flip = getattr(self, '_canary_flip_state', 0)
                    side = 'LONG' if flip % 2 == 0 else 'SHORT'
                    self._canary_flip_state = flip + 1
                plan = {
                    "ts": evt.timestamp,
                    "symbol": evt.symbol,
                    "timeframe": evt.timeframe,
                    "side": side,
                    "price": evt.close,
                    # Provide a deterministic tiny notional/size so executor can simulate fills
                    "size": 1.0,
                    "version": 1,
                    "_decision_monotonic": time.perf_counter(),
                }
                if getattr(self.safety, 'size_downscale', None):
                    plan['size_mult'] = getattr(self.safety, 'size_downscale')
                # If decision were actionable (LONG/SHORT) attempt to build execution plan
                if side in ("LONG","SHORT"):
                    # First run legacy guard + plan path if modules available
                    if build_exec_plan and pre_trade_guards:
                        bidask = getattr(self, "_last_bidask", None)
                        if bidask and bidask[0]==evt.symbol:
                            _, bid, ask = bidask
                            book = {"bid":bid, "ask":ask}
                            guards = pre_trade_guards(evt.symbol, side, book=book, decision_latency_ms=int(elapsed_ms), ex_cfg=(getattr(self,'_exec_cfg',{}) or {}))
                            if guards.get('blocked'):
                                logger.debug(f"[EngineWorker] guards blocked side={side} reason={guards.get('reason')}")
                            else:
                                ep = build_exec_plan(evt.symbol, side, {"bid":bid, "ask":ask}, tick_size=0.1, atr=None, atr_pct=None, regime=None, settings=getattr(self,'_settings',{}), now_ms=int(evt.timestamp*1000))
                                if ep:
                                    plan['exec_plan'] = ep.to_order()
                    # Sprint 35 fast_router integration (taker-first smart routing) if enabled
                    try:
                        ex_cfg = getattr(self,'_settings',{}).get('execution',{}) if hasattr(self,'_settings') else {}
                        if execute_fast_order and ex_cfg.get('mode') in ('ultra_fast','fast'):
                            # Acquire simple quotes snapshot from cached bidask only for now
                            bidask = getattr(self,'_last_bidask', None)
                            quotes = None
                            if bidask and bidask[0]==evt.symbol:
                                _, bid, ask = bidask
                                quotes = { 'SIM': {'bid': bid, 'ask': ask, 'bid_size': 1e6, 'ask_size': 1e6} }
                            fr = execute_fast_order(symbol=evt.symbol, side=side, size=plan.get('size',1.0), price=evt.close, settings=getattr(self,'_settings',{}), quotes=quotes)
                            plan.setdefault('fast_exec', {})
                            plan['fast_exec'].update({'accepted': fr.accepted, 'reason': fr.reason, 'venue': fr.venue, 'spread_bps': fr.spread_bps, 'depth_ok': fr.depth_ok, 'retries': fr.retries})
                            if fr.accepted and fr.order:
                                plan.setdefault('exec_plan', {}).update(fr.order)
                    except Exception:
                        pass
                    # Microstructure veto flags (skip emission if unstable) unless disabled for canary
                    disable_micro_veto = (
                        os.getenv('CANARY_DISABLE_MICRO_VETO') == '1' or
                        ((self._settings or {}).get('runtime', {}) or {}).get('disable_microstructure_veto') is True
                    )
                    if not disable_micro_veto:
                        active_flags = []
                        now_m = time.perf_counter()
                        if now_m < self._spoof_flag_until: active_flags.append('SPOOF_LIQ_VANISH')
                        if now_m < self._flip_flag_until: active_flags.append('BOOK_FLIP_UNSTABLE')
                        if now_m < self._cascade_flag_until: active_flags.append('CASCADE_UNSTABLE')
                        if active_flags:
                            logger.warning(f"[EngineWorker] Microstructure veto {active_flags} – suppress order plan")
                            continue
                        if active_flags:
                            plan.setdefault('microstructure', {}).update({'flags': active_flags})
                    # Ensure a minimal exec_plan so downstream order path triggers in canary mode
                    if 'exec_plan' not in plan:
                        plan['exec_plan'] = {'order_type': 'MARKET', 'post_only': False}
                    # Ensure qty alias for any downstream paths expecting qty instead of size
                    if 'qty' not in plan:
                        plan['qty'] = plan.get('size', 1.0)
                # Composite mid (if multi-venue snapshots available on feed_ref) placed after plan creation
                try:
                    feed_ref = getattr(self, 'feed_ref', None)
                    if feed_ref and hasattr(feed_ref, '_venue_books') and len(getattr(feed_ref, '_venue_books')) >= 2 and _dq_composite_mid:  # type: ignore
                        venues_data = {}
                        for vid, snap in feed_ref._venue_books.items():
                            try:
                                venues_data[vid] = __import__('pandas').DataFrame([{'ts': snap.get('ts'), 'bid': snap.get('bid'), 'ask': snap.get('ask')}])
                            except Exception:
                                pass
                        if venues_data:
                            comp_df, flags = _dq_composite_mid(venues_data, getattr(self,'_settings',{}) or {})
                            if hasattr(comp_df, 'empty') and not comp_df.empty:
                                plan.setdefault('composite', {})['mid'] = float(comp_df['mid'].iloc[-1])
                                try:
                                    plan['composite']['spread_bps'] = float(comp_df['venue_spread_bps'].iloc[-1])
                                except Exception:
                                    pass
                                plan['composite']['flags'] = flags
                except Exception:
                    pass
                # Latency enrich: quote freshness & book spread
                try:
                    bidask = getattr(self,'_last_bidask', None)
                    if bidask and self._book_snaps:
                        last_book_ts = self._book_snaps[-1][0]
                        plan['_lat_quote_recency_ms'] = (time.perf_counter() - last_book_ts)*1000.0
                        bid, ask = self._book_snaps[-1][1], self._book_snaps[-1][2]
                        if bid>0 and ask>0:
                            plan['_lat_spread_bps'] = (ask-bid)/((ask+bid)/2.0)*10000
                        if self.metrics and hasattr(self.metrics, 'latency_quote_to_decision'):
                            self.metrics.latency_quote_to_decision.observe(plan['_lat_quote_recency_ms'])
                except Exception:
                    pass
                try:
                    # Apply VPIN toxicity policy (veto/resize) if present
                    try:
                        sym = plan.get('symbol')
                        eng = self._vpin_engines.get(sym)
                        if eng is not None:
                            v = eng.get_latest_vpin()
                            pctl = float(v.get('vpin_pctl', 0.0))
                            # read policy from feed settings if available
                            try:
                                cfg = getattr(self.feed_ref, 'settings', {})
                                vcfg = ((cfg or {}).get('features', {}) or {}).get('vpin', {})
                                pol = (vcfg.get('policy', {}) or {}).get('mode', vcfg.get('mode', 'veto'))
                                size_mult_cfg = float((vcfg.get('policy', {}) or {}).get('size_mult', 0.5))
                                hi = float((vcfg.get('policy', {}) or {}).get('hi_th', vcfg.get('hi_th', 0.85)))
                                lo = float((vcfg.get('policy', {}) or {}).get('lo_th', vcfg.get('lo_th', 0.7)))
                            except Exception:
                                pol = 'veto'; size_mult_cfg = 0.5; hi = 0.85; lo = 0.7
                            prev = self._vpin_states.get(sym, 'normal')
                            # hysteresis transitions
                            if prev != 'toxic' and pctl >= hi:
                                self._vpin_states[sym] = 'toxic'
                                prev = 'toxic'
                            elif prev == 'toxic' and pctl <= lo:
                                self._vpin_states[sym] = 'cooldown'
                                prev = 'cooldown'
                            # enforce policy
                            if prev == 'toxic':
                                if pol.lower() == 'veto':
                                    logger.warning(f"[EngineWorker][VPIN] vetoing plan for {sym} pctl={pctl:.2f}")
                                    # increment metric if available
                                    if self.metrics and hasattr(self.metrics, 'inc'):
                                        try:
                                            self.metrics.inc('vpin_veto')
                                        except Exception:
                                            pass
                                    continue
                                elif pol.lower() == 'resize':
                                    plan['size_mult'] = float(plan.get('size_mult', 1.0)) * size_mult_cfg
                    except Exception:
                        pass
                    # If plan requests VWAP child_algo, perform slicing and enqueue child orders
                    try:
                        if plan.get('child_algo') == 'VWAP':
                            # lazy import to avoid top-level dependency
                            from ultra_signals.routing.vwap_adapter import VWAPExecutor
                            from ultra_signals.routing.types import VenueInfo
                            symbol = plan.get('symbol')
                            child_notional = float((plan.get('vwap_cfg') or {}).get('child_notional') or plan.get('child_notional') or plan.get('size') or 0.0)
                            if child_notional <= 0:
                                # nothing to do
                                self.out_queue.put_nowait(plan)
                            else:
                                # build venues mapping from venue_router if available
                                venues_map = {}
                                try:
                                    vr = getattr(self.feed_ref, 'venue_router', None)
                                    if vr and hasattr(vr, 'venues'):
                                        for vid, adapter in vr.venues.items():
                                            # best-effort fee extraction
                                            fee = 0.0
                                            try:
                                                fee = float(getattr(adapter, 'taker_fee', 0.0) or 0.0)
                                            except Exception:
                                                fee = 0.0
                                            venues_map[vid] = VenueInfo(vid, maker_bps=0.0, taker_bps=fee, min_notional=1.0, lot_size=0.0001)
                                except Exception:
                                    venues_map = {}
                                vcfg = plan.get('vwap_cfg') or {}
                                vexec = VWAPExecutor(venues_map, volume_curve=vcfg.get('volume_curve'), pr_cap=float(vcfg.get('pr_cap', 0.1)), jitter_frac=float(vcfg.get('jitter_frac', 0.05)), max_slice_notional=vcfg.get('max_slice_notional'), feature_writer=self.feature_writer)
                                # optional feature provider: attempt to read lightweight features from feed_ref if present
                                def feat_provider(sym):
                                    try:
                                        # feed_ref may expose a small feature view cache
                                        ff = getattr(self.feed_ref, 'feature_view', None)
                                        if callable(ff):
                                            return ff().get(sym) or {}
                                    except Exception:
                                        pass
                                    return {}
                                vexec.feature_provider = feat_provider
                                # execute slices
                                slices = vexec.execute(self.feed_ref, plan.get('side','LONG'), child_notional, symbol)
                                # enqueue each slice as one or more venue child plans based on allocation
                                for sl in slices:
                                    alloc = sl.get('router_allocation') or {}
                                    slice_id = sl.get('slice_id')
                                    expected_cost = sl.get('expected_cost_bps')
                                    # expected_price may be present in sl['router_decision'] if router provided; else None
                                    expected_price = None
                                    rd = sl.get('router_decision')
                                    if rd and hasattr(rd, 'expected_price'):
                                        expected_price = getattr(rd, 'expected_price')
                                    # if allocation only a single venue, create one child
                                    if not alloc:
                                        child = {
                                            'ts': int(time.time()),
                                            'symbol': symbol,
                                            'side': plan.get('side'),
                                            'size': sl.get('slice_notional'),
                                            'price': expected_price,
                                            'parent_id': plan.get('ts'),
                                            'slice_id': slice_id,
                                            'exec_plan': {'order_type': 'MARKET' if sl.get('style') == 'MARKET' else 'LIMIT', 'post_only': True if sl.get('style') == 'LIMIT' else False, 'expected_cost_bps': expected_cost, 'expected_price': expected_price},
                                        }
                                        try:
                                            self.out_queue.put_nowait(child)
                                        except asyncio.QueueFull:
                                            logger.error('[EngineWorker] order queue full when enqueueing VWAP child')
                                    else:
                                        for vid, pct in alloc.items():
                                            child = {
                                                'ts': int(time.time()),
                                                'symbol': symbol,
                                                'side': plan.get('side'),
                                                'size': sl.get('slice_notional') * float(pct),
                                                'price': expected_price,
                                                'parent_id': plan.get('ts'),
                                                'slice_id': slice_id,
                                                'venue': vid,
                                                'exec_plan': {'order_type': 'MARKET' if sl.get('style') == 'MARKET' else 'LIMIT', 'post_only': True if sl.get('style') == 'LIMIT' else False, 'expected_cost_bps': expected_cost, 'expected_price': expected_price},
                                            }
                                            try:
                                                self.out_queue.put_nowait(child)
                                            except asyncio.QueueFull:
                                                logger.error('[EngineWorker] order queue full when enqueueing VWAP child')
                                # also persist the parent plan for bookkeeping
                                self.out_queue.put_nowait(plan)
                            continue
                    except Exception:
                        # if anything goes wrong, fall back to emitting the original plan
                        pass
                    try:
                        self.out_queue.put_nowait(plan)
                        if force_signals:
                            logger.debug(f"[EngineWorker][FORCE-KLINE] emitted bar plan side={side} sym={evt.symbol}")
                    except asyncio.QueueFull:
                        logger.error("[EngineWorker] order queue full; dropping plan")
                    # Emit synthetic EnsembleDecision for Telegram (canary wiring) on bar-close forced signals
                    try:
                        if force_signals and side in ('LONG','SHORT') and EnsembleDecision and SubSignal and send_decision:
                            tcfg = None
                            sref = getattr(self, '_settings', None)
                            if isinstance(sref, dict) and 'transport' in sref:
                                tcfg = (sref.get('transport') or {})
                            elif self.feed_ref and hasattr(self.feed_ref, '_settings'):
                                frs = getattr(self.feed_ref, '_settings')
                                try:
                                    if hasattr(frs, 'transport') and hasattr(frs.transport, 'telegram'):
                                        tcfg = {'telegram': frs.transport.telegram.model_dump() if hasattr(frs.transport.telegram,'model_dump') else dict(getattr(frs.transport,'telegram',{}))}
                                except Exception:
                                    pass
                            if tcfg and 'telegram' in tcfg:
                                sub = SubSignal(ts=plan['ts'], symbol=plan['symbol'], tf=plan['timeframe'], strategy_id='canary_forced', direction=side, confidence_calibrated=0.60, reasons={})
                                px = float(plan.get('price') or 0.0) or 1.0
                                rr_tp_mult = 1.004 if side=='LONG' else 0.996
                                rr_sl_mult = 0.996 if side=='LONG' else 1.004
                                entry = px; tp = px * rr_tp_mult; sl = px * rr_sl_mult
                                rr = abs((tp-entry)/(entry-sl)) if (entry-sl)!=0 else 1.0
                                dec = EnsembleDecision(
                                    ts=plan['ts'], symbol=plan['symbol'], tf=plan['timeframe'], decision=side, confidence=0.60,
                                    subsignals=[sub],
                                    vote_detail={'agree':1,'total':1,'profile':'canary','weighted_sum':1.0,
                                                 'pre_trade':{'p_win':0.55,'regime':'canary','veto_count':0,'lat_ms':{'p50':10.0,'p90':15.0}},
                                                 'entry':entry,'sl':sl,'tp':tp,'lev':1.0,'risk_pct':0.02,'rr':rr},
                                    vetoes=[])
                                asyncio.create_task(send_decision(dec, {'telegram': tcfg.get('telegram')}))
                    except Exception:
                        pass
                except asyncio.QueueFull:
                    logger.error("[EngineWorker] order queue full; dropping plan")

    def stop(self):  # pragma: no cover
        self._running = False

__all__ = ["EngineWorker"]
