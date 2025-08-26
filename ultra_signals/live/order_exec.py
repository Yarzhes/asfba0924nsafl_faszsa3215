"""Order executor with idempotency & retry/backoff (simplified)."""
from __future__ import annotations
import asyncio
import time
import math
import hashlib
import random
from typing import Dict, Any, Callable, Optional
from loguru import logger
from .state_store import StateStore


def make_client_order_id(plan: Dict[str, Any]) -> str:
    key = f"{plan.get('ts')}|{plan.get('symbol')}|{plan.get('side')}|{plan.get('price')}|{plan.get('version',1)}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


class OrderExecutor:
    def __init__(self, queue: asyncio.Queue, store: StateStore, rate_limits: Dict[str, int], retry_cfg: Dict[str, Any], dry_run: bool = True, safety=None, metrics=None, order_sender: Optional[Callable[[Dict[str, Any], str], None]] = None, simulator_cfg: Optional[Dict[str, Any]] = None):
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
        self.simulator_cfg = simulator_cfg or {}

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
                    self._order_sender(plan, client_id)
                elif self.dry_run:
                    # Simulator probabilities
                    reject_prob = float(self.simulator_cfg.get("reject_prob", 0.02))
                    part_prob = float(self.simulator_cfg.get("partial_fill_prob", 0.2))
                    if random.random() < reject_prob:
                        self.store.update_order(client_id, status="REJECTED", last_error="SIM_REJECT")
                        if self.metrics:
                            self.metrics.inc("orders_errors")
                        logger.warning(f"[OrderExec] DRY-RUN simulated rejection {client_id}")
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
                else:
                    logger.info(f"[OrderExec] LIVE place order {client_id}: {plan}")
                    self.store.update_order(client_id, status="FILLED", exchange_order_id=f"EX-{client_id[:8]}")
                if self.metrics:
                    dm = plan.get("_decision_monotonic")
                    if dm:
                        self.metrics.latency_decision_to_order.observe((time.perf_counter() - dm)*1000.0)
                    self.metrics.inc("orders_sent")
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
            self.store.update_order(client_id, status="FILLED")
            logger.info(f"[OrderExec] DRY-RUN partial fill completed {client_id}")
        except Exception:
            pass

__all__ = ["OrderExecutor", "make_client_order_id"]
