"""Engine worker consuming market events and producing execution plans.

This is a highly simplified version for initial Sprint 21 scaffolding. It:
  * Consumes Kline closed events only (placeholder for full feature pipeline)
  * Emits a NOOP plan structure for demonstration
  * Respects a per-event latency budget; if exceeded emits abstain
"""
from __future__ import annotations
import asyncio
import time
from typing import Dict, Any, Optional
from loguru import logger
from ultra_signals.core.events import KlineEvent, MarketEvent


class EngineWorker:
    def __init__(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue, latency_budget_ms: int = 150, metrics=None, safety=None, extra_delay_ms: int = 0):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.latency_budget_ms = latency_budget_ms
        self._running = False
        self.metrics = metrics
        self.safety = safety
        self.extra_delay_ms = extra_delay_ms  # for tests to force deadline breach

    async def run(self):
        self._running = True
        while self._running:
            try:
                evt: MarketEvent = await self.in_queue.get()
            except asyncio.CancelledError:
                break
            if isinstance(evt, KlineEvent) and evt.closed:
                started = time.perf_counter()
                # Force artificial delay if requested (test hook)
                if self.extra_delay_ms:
                    await asyncio.sleep(self.extra_delay_ms / 1000.0)
                # placeholder compute – future integration with real engine
                await asyncio.sleep(0)  # yield to loop
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
                plan = {
                    "ts": evt.timestamp,
                    "symbol": evt.symbol,
                    "timeframe": evt.timeframe,
                    "side": "FLAT",
                    "price": evt.close,
                    "version": 1,
                    "_decision_monotonic": time.perf_counter(),  # for decision->order latency metric
                }
                try:
                    self.out_queue.put_nowait(plan)
                except asyncio.QueueFull:
                    logger.error("[EngineWorker] order queue full; dropping plan")

    def stop(self):  # pragma: no cover
        self._running = False

__all__ = ["EngineWorker"]
