"""Order executor with idempotency & retry/backoff (simplified)."""
from __future__ import annotations
import asyncio
import time
import math
import hashlib
import random
import inspect
from typing import Dict, Any, Callable, Optional
from loguru import logger
try:
    from ultra_signals.execution.brackets import build_brackets
except Exception:  # pragma: no cover
    build_brackets = None  # type: ignore
from .state_store import StateStore
from ultra_signals.persist.db import upsert_order_pending, update_order_after_ack


def make_client_order_id(plan: Dict[str, Any]) -> str:
    key = f"{plan.get('ts')}|{plan.get('symbol')}|{plan.get('side')}|{plan.get('price')}|{plan.get('version',1)}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


class OrderExecutor:
    def __init__(
        self,
        queue: asyncio.Queue,
        store: StateStore,
        rate_limits: Dict[str, int],
        retry_cfg: Dict[str, Any],
        dry_run: bool = True,
        safety=None,
        metrics=None,
        order_sender: Optional[Callable[[Dict[str, Any], str], None]] = None,
        feature_writer: Optional[object] = None,
        simulator_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.queue = queue
        self.store = store
        self.rate_limits = rate_limits or {"orders_per_sec": 8, "cancels_per_sec": 8}
        self.retry_cfg = retry_cfg or {"max_attempts": 3, "base_delay_ms": 100}
        self.dry_run = dry_run
        self._recent_order_timestamps = []  # rate limit window
        self._running = False
        self.safety = safety
        self.metrics = metrics
        self._order_sender = order_sender  # injectable for tests
        self.feature_writer = feature_writer
        self.simulator_cfg = simulator_cfg or {}
        # optionally pluggable tick-based adapter for dry-run realism
        self._tick_adapter = None
        try:
            if self.simulator_cfg.get('tick_adapter'):
                from ultra_signals.sim.execution_adapter import ExecutionAdapter
                # allow passing an adapter instance directly
                if isinstance(self.simulator_cfg.get('tick_adapter'), ExecutionAdapter):
                    self._tick_adapter = self.simulator_cfg.get('tick_adapter')
                else:
                    self._tick_adapter = ExecutionAdapter()
        except Exception:
            self._tick_adapter = None

    async def run(self):
        self._running = True
        while self._running:
            try:
                plan = await self.queue.get()
            except asyncio.CancelledError:
                break
            if plan is None:
                continue
            await self._process_plan(plan)

    async def _process_plan(self, plan: Dict[str, Any]):
        client_id = make_client_order_id(plan)
        # Journal-first insert (exactly-once semantics) using persist layer
        try:
            upsert_order_pending({
                'client_order_id': client_id,
                'venue': plan.get('venue') or (plan.get('exec_plan') or {}).get('venue'),
                'symbol': plan.get('symbol'),
                'side': plan.get('side'),
                'type': (plan.get('exec_plan') or {}).get('order_type','LIMIT'),
                'qty': plan.get('size') or plan.get('qty') or 0.0,
                'price': plan.get('price'),
                'reduce_only': (plan.get('exec_plan') or {}).get('reduce_only', False),
                'parent_id': plan.get('parent_id'),
                'profile_id': plan.get('profile_id'),
                'cfg_hash': plan.get('cfg_hash'),
            })
        except Exception:
            pass
        if not self.store.ensure_order(client_id):
            logger.info(f"[OrderExec] Duplicate plan ignored (idempotent) {client_id}")
            return
        if self.safety and self.safety.state.paused:
            logger.warning("[OrderExec] Safety paused – suppressing order send")
            return
        attempts = 0
        max_attempts = int(self.retry_cfg.get("max_attempts", 3))
        base_delay = int(self.retry_cfg.get("base_delay_ms", 100)) / 1000.0
        while attempts < max_attempts:
            attempts += 1
            # rate limit
            now = time.time()
            self._recent_order_timestamps = [t for t in self._recent_order_timestamps if now - t < 1.0]
            if len(self._recent_order_timestamps) >= self.rate_limits.get("orders_per_sec", 8):
                await asyncio.sleep(0.05)
                continue
            self._recent_order_timestamps.append(now)
            try:
                # Injectable sender for tests / live
                if self._order_sender:
                    result = self._order_sender(plan, client_id)
                    if inspect.isawaitable(result):
                        result = await result
                    # If router-style result provided, update store based on ack
                    # normalize ack shapes: support dict-with-ack, dict-flat, or object ack
                    ack = None
                    venue_id = None
                    if isinstance(result, dict):
                        if result.get('ack'):
                            ack = result.get('ack')
                            venue_id = result.get('venue') or result.get('venue_id')
                        else:
                            # flattened dict may contain status/avg_px/venue
                            ack = type('AckObj', (), {})()
                            setattr(ack, 'status', result.get('status', None))
                            setattr(ack, 'avg_px', result.get('avg_px', result.get('exec_price', None)))
                            setattr(ack, 'venue_order_id', result.get('venue_order_id', None) or result.get('order_id', None))
                            venue_id = result.get('venue') or result.get('venue_id')
                    else:
                        # support object-like ack
                        ack = result
                        try:
                            venue_id = getattr(result, 'venue', None) or getattr(result, 'venue_id', None)
                        except Exception:
                            venue_id = None

                    # helper to extract fields from either dict-like or attr-like ack
                    def _g(o, name, default=None):
                        try:
                            if isinstance(o, dict):
                                return o.get(name, default)
                            return getattr(o, name, default)
                        except Exception:
                            return default

                    if venue_id:
                        # persist ack update (db layer) + state store venue_id
                        try:
                            update_order_after_ack(client_id, status=_g(ack, 'status', 'ACKED'), venue_order_id=_g(ack, 'venue_order_id', None))
                        except Exception:
                            pass
                        try:
                            self.store.update_order(client_id, venue_id=venue_id)
                        except Exception:
                            pass

                    status = _g(ack, 'status', 'FILLED')
                    ex_id = _g(ack, 'venue_order_id', None)
                    filled = _g(ack, 'filled_qty', None) or _g(ack, 'filled', None)
                    avg_px = _g(ack, 'avg_px', None) or _g(ack, 'exec_price', None) or _g(ack, 'avg_price', None)
                    self.store.update_order(client_id, status=status, exchange_order_id=ex_id, filled_qty=filled, exec_price=avg_px)
                    # update FeatureView with realized cost if slice-linked and avg_px available
                    try:
                        slice_id = plan.get('slice_id') or (plan.get('exec_plan') or {}).get('slice_id')
                        if self.feature_writer and slice_id and avg_px:
                            exp_px = (plan.get('exec_plan') or {}).get('expected_price')
                            if exp_px and isinstance(exp_px, (int,float)) and exp_px != 0:
                                side = (plan.get('side') or 'LONG')
                                realized = (avg_px - exp_px) / exp_px * 10000.0
                                if side.upper() in ('SHORT','SELL'):
                                    realized = -realized
                                try:
                                    self.feature_writer.update_by_slice_id(slice_id, {'realized_cost_bps': realized})
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if status not in ("REJECTED", "ERROR") and self.metrics:
                        self.metrics.inc("orders_sent")
                    return
                elif self.dry_run:
                    # If tick adapter present, use it to simulate fills deterministically
                    if self._tick_adapter:
                        try:
                            plan_for_exec = { 'side': plan.get('side'), 'size': plan.get('size') or plan.get('qty'), 'price': plan.get('price') }
                            res = self._tick_adapter.place_order(plan_for_exec)
                            # normalize fill into store
                            filled = float(res.get('filled_qty', 0.0) or res.get('filled', 0.0) or 0.0)
                            exec_px = res.get('fills') and res['fills'][-1].get('px') if res.get('fills') else res.get('price')
                            self.store.update_order(client_id, status="FILLED" if filled >= (plan.get('size') or plan.get('qty') or 0) else "PARTIAL", exchange_order_id=f"TICKSIM-{client_id[:8]}", exec_price=exec_px, filled_qty=filled)
                            if self.metrics:
                                self.metrics.inc('tick_fill_events')
                                # record observed fill ratio and latency if present
                                req = float(plan.get('size') or plan.get('qty') or 0)
                                if req:
                                    self.metrics.fill_ratio.observe((filled/req) * 1.0)
                                if res.get('latency_ms') and isinstance(self.metrics.queue_wait_ms, object):
                                    self.metrics.queue_wait_ms.observe(res.get('latency_ms'))
                            return
                        except Exception:
                            # fallback to legacy dry-run if adapter fails
                            pass
                
                    # legacy dry-run continues below
                    # Simulator probabilities
                    reject_prob = float(self.simulator_cfg.get("reject_prob", 0.02))
                    part_prob = float(self.simulator_cfg.get("partial_fill_prob", 0.2))
                    if random.random() < reject_prob:
                        # For maker-post-only path we want fallback opportunity; mark PENDING instead of hard REJECT
                        ep = plan.get('exec_plan') or {}
                        if ep.get('post_only') and ep.get('taker_fallback_after_ms'):
                            self.store.update_order(client_id, status="PENDING", last_error="SIM_REJECT_POST_ONLY")
                            logger.warning(f"[OrderExec] DRY-RUN simulated post-only reject (will fallback) {client_id}")
                        else:
                            self.store.update_order(client_id, status="REJECTED", last_error="SIM_REJECT")
                            if self.metrics:
                                self.metrics.inc("orders_errors")
                            logger.warning(f"[OrderExec] DRY-RUN simulated rejection {client_id}")
                            # record reject in TCA engine if available
                            try:
                                te = getattr(self, 'tca_engine', None)
                                if te is not None:
                                    ven = plan.get('venue') or (plan.get('exec_plan') or {}).get('venue') or 'UNKNOWN'
                                    te.record_reject(str(ven))
                                    # per-symbol reject
                                    if plan.get('symbol'):
                                        te.record_reject_for_symbol(str(ven), str(plan.get('symbol')))
                            except Exception:
                                pass
                            return
                    if random.random() < part_prob:
                        part = round(random.uniform(0.3, 0.7), 4)
                        logger.info(f"[OrderExec] DRY-RUN partial fill {part*100:.1f}% {client_id}")
                        self.store.update_order(client_id, status="PARTIAL", exchange_order_id=f"SIM-{client_id[:8]}", last_error=None)
                        # schedule completion
                        asyncio.create_task(self._finalize_partial(client_id))
                        if self.metrics:
                            self.metrics.inc("orders_sent")
                        return
                    # Slippage application
                    slip_min = float(self.simulator_cfg.get("slippage_bps_min", -1.0))
                    slip_max = float(self.simulator_cfg.get("slippage_bps_max", 1.5))
                    slip_bps = random.uniform(slip_min, slip_max)
                    px = plan.get("price")
                    exec_price = px * (1 + slip_bps/10000.0) if isinstance(px, (int,float)) else px
                    logger.info(f"[OrderExec] DRY-RUN fill {client_id} slip_bps={slip_bps:.3f} exec_price={exec_price}")
                    self.store.update_order(client_id, status="FILLED", exchange_order_id=f"SIM-{client_id[:8]}", exec_price=exec_price)
                    # If slice-linked, update FeatureView realized cost
                    try:
                        slice_id = plan.get('slice_id') or (plan.get('exec_plan') or {}).get('slice_id')
                        if self.feature_writer and slice_id:
                            exp_px = None
                            try:
                                exp_px = (plan.get('exec_plan') or {}).get('expected_price')
                            except Exception:
                                exp_px = None
                            if exp_px and isinstance(exec_price, (int,float)) and isinstance(exp_px, (int,float)) and exp_px != 0:
                                side = (plan.get('side') or 'LONG')
                                # realized cost in bps (positive = adverse)
                                realized = (exec_price - exp_px) / exp_px * 10000.0
                                # for SHORT invert sign
                                if side.upper() in ('SHORT','SELL'):
                                    realized = -realized
                                try:
                                    self.feature_writer.update_by_slice_id(slice_id, {'realized_cost_bps': realized})
                                except Exception:
                                    pass
                    except Exception:
                        pass
                        # record fill into TCA engine if present
                        try:
                            te = getattr(self, 'tca_engine', None)
                            if te is not None:
                                ven = plan.get('venue') or (plan.get('exec_plan') or {}).get('venue') or 'SIM'
                                ev = {'venue': ven, 'symbol': plan.get('symbol'), 'arrival_px': (plan.get('exec_plan') or {}).get('expected_price') or plan.get('price'), 'fill_px': exec_price, 'filled_qty': plan.get('size') or plan.get('qty') or 0.0, 'requested_qty': plan.get('size') or plan.get('qty') or 0.0, 'arrival_ts_ms': int(time.time()*1000), 'completion_ts_ms': int(time.time()*1000), 'order_type': (plan.get('exec_plan') or {}).get('order_type')}
                                try:
                                    te.record_fill(ev)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    # After fill place brackets if spec present
                    try:
                        if build_brackets and plan.get('exec_plan') and plan['exec_plan'].get('post_only') is not None:
                            # minimal bracket creation using close price as entry
                            brackets = build_brackets(entry_px=exec_price, side=plan.get('side','LONG'), atr=plan.get('atr'), size=plan.get('size',1.0), execution_cfg=(plan.get('settings') or {}))
                            if brackets:
                                # Insert bracket legs into store as child orders (reduce-only simulation)
                                for ix, tp in enumerate(brackets.tps):
                                    cid_child = f"{client_id[:18]}TP{ix}"
                                    self.store.insert_child_order(cid_child, parent_id=client_id, status='OPEN')
                                cid_sl = f"{client_id[:18]}SL"
                                self.store.insert_child_order(cid_sl, parent_id=client_id, status='OPEN')
                    except Exception:
                        pass
                else:
                    logger.info(f"[OrderExec] LIVE place order {client_id}: {plan}")
                    self.store.update_order(client_id, status="FILLED", exchange_order_id=f"EX-{client_id[:8]}")
                if self.metrics:
                    dm = plan.get("_decision_monotonic")
                    if dm:
                        self.metrics.latency_decision_to_order.observe((time.perf_counter() - dm)*1000.0)
                    self.metrics.inc("orders_sent")
                # Schedule taker fallback amend if maker-first
                try:
                    ep = plan.get('exec_plan') or {}
                    if ep.get('post_only') and ep.get('taker_fallback_after_ms'):
                        delay = ep['taker_fallback_after_ms']/1000.0
                        asyncio.create_task(self._maker_fallback(client_id, ep, delay))
                except Exception:
                    pass
                return
            except Exception as e:
                delay = base_delay * (2 ** (attempts - 1))
                self.store.update_order(client_id, status="ERROR", last_error=str(e), retries=attempts)
                if self.metrics:
                    self.metrics.inc("orders_errors")
                if self.safety:
                    try:
                        self.safety.record_order_error()
                    except Exception:
                        pass
                logger.exception(f"[OrderExec] order error attempt={attempts} id={client_id} {e}")
                await asyncio.sleep(delay + (0.01 * random.random()))
        logger.error(f"[OrderExec] Exhausted retries for {client_id}")

    def stop(self):  # pragma: no cover
        self._running = False

    async def _finalize_partial(self, client_id: str):  # pragma: no cover (timing nondeterministic)
        await asyncio.sleep(random.uniform(0.05, 0.2))
        try:
            # mark filled and attempt FeatureView update using stored order info
            row = self.store.get_order(client_id)
            self.store.update_order(client_id, status="FILLED")
            try:
                if self.feature_writer and row:
                    slice_id = row.get('slice_id') or (row.get('exec_plan') or {}).get('slice_id')
                    exec_price = row.get('exec_price')
                    exp_px = None
                    try:
                        exp_px = (row.get('exec_plan') or {}).get('expected_price')
                    except Exception:
                        exp_px = None
                    if slice_id and exec_price and exp_px and isinstance(exec_price, (int,float)) and isinstance(exp_px, (int,float)) and exp_px != 0:
                        side = (row.get('side') or 'LONG')
                        realized = (exec_price - exp_px) / exp_px * 10000.0
                        if side.upper() in ('SHORT','SELL'):
                            realized = -realized
                        try:
                            self.feature_writer.update_by_slice_id(slice_id, {'realized_cost_bps': realized})
                        except Exception:
                            pass
            except Exception:
                pass
            logger.info(f"[OrderExec] DRY-RUN partial fill completed {client_id}")
        except Exception:
            pass

    async def _maker_fallback(self, client_id: str, exec_plan: Dict[str, Any], delay: float):  # pragma: no cover (timing)
        await asyncio.sleep(delay)
        try:
            row = self.store.get_order(client_id)
            if not row or row.get('status') != 'PENDING':
                return
            taker_px = exec_plan.get('taker_price')
            if taker_px:
                self.store.update_order(client_id, status="FILLED", exec_price=taker_px)
                # update FeatureView realized cost for this slice if present
                try:
                    if self.feature_writer and row:
                        slice_id = row.get('slice_id') or (row.get('exec_plan') or {}).get('slice_id')
                        exp_px = None
                        try:
                            exp_px = (row.get('exec_plan') or {}).get('expected_price')
                        except Exception:
                            exp_px = None
                        if slice_id and exp_px and isinstance(taker_px, (int,float)) and isinstance(exp_px, (int,float)) and exp_px != 0:
                            side = (row.get('side') or 'LONG')
                            realized = (taker_px - exp_px) / exp_px * 10000.0
                            if side.upper() in ('SHORT','SELL'):
                                realized = -realized
                            try:
                                self.feature_writer.update_by_slice_id(slice_id, {'realized_cost_bps': realized})
                            except Exception:
                                pass
                except Exception:
                    pass
                logger.info(f"[OrderExec] maker->taker fallback fill {client_id} px={taker_px}")
        except Exception:
            pass

__all__ = ["OrderExecutor", "make_client_order_id"]
