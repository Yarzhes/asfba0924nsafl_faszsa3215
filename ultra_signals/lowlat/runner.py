"""Low-latency runner skeleton with synthetic tick generator and pipeline hooks.

This module demonstrates a single-writer ring buffer feeding a consumer
that performs parse -> feature -> gate -> route -> encode -> send steps.
All blocking IO is avoided; the "send" step is a stub that simulates
an async send by calling a callback.
"""
from __future__ import annotations
import asyncio
import time
from typing import Callable

from ultra_signals.lowlat.ringbuffer import RingBuffer
from ultra_signals.live.metrics import Metrics
from ultra_signals.lowlat.router import make_default_router


class Runner:
    def __init__(self, capacity: int = 4096):
        self.ring = RingBuffer(capacity)
        self.metrics = Metrics()
        self.router = make_default_router()
        self._stop = False

    def produce_tick(self, raw: bytes) -> bool:
        """Called by ingest path (single-writer) to push raw tick bytes."""
        ts = time.perf_counter_ns()
        # store tuple (ts_ns, raw)
        return self.ring.push((ts, raw))

    async def consumer_loop(self, send_fn: Callable[[bytes], asyncio.Future], batch_ms: int = 1):
        """Consume ticks and run a minimal pipeline; send_fn must be async.

        send_fn(payload: bytes) -> awaitable
        """
        while not self._stop:
            item = self.ring.pop()
            if item is None:
                await asyncio.sleep(batch_ms / 1000.0)
                continue

            ts_ns, raw = item
            # parse (hot path): here raw is already bytes -> minimal parsing
            parse_ts = time.perf_counter_ns()
            symbol = raw.decode(errors='ignore').split()[1] if b' ' in raw else 'BTC-USD'

            # feature extraction (micro): stub
            feat_ts = time.perf_counter_ns()
            features = {"dummy": 1}

            # gate: simple pass
            gate_ts = time.perf_counter_ns()

            # route
            route = self.router.select(symbol, lowlat=True)

            # encode: use prebuilt payload when possible
            payload = route.prebuilt_payload

            # measure latencies
            self.metrics.observe_tick_to_decision(ts_ns, parse_ts)
            self.metrics.observe_decision_to_order(parse_ts, time.perf_counter_ns())

            # send (simulate async send)
            send_start = time.perf_counter_ns()
            try:
                await send_fn(payload)
                self.metrics.inc("orders_sent", 1)
            except Exception:
                self.metrics.inc("orders_errors", 1)
            finally:
                self.metrics.observe_wire_to_ack(send_start, time.perf_counter_ns())

    def stop(self) -> None:
        self._stop = True


async def fake_send(payload: bytes) -> None:
    # simulate tiny network delay
    await asyncio.sleep(0.0005)


async def synthetic_run(peek_n: int = 1000) -> Metrics:
    r = Runner(capacity=peek_n * 2)

    async def producer():
        for i in range(peek_n):
            ok = r.produce_tick(b"TICK BTC-USD")
            if not ok:
                r.metrics.inc("tick_fill_events", 1)
            await asyncio.sleep(0)  # yield
        # allow backlog to drain

    prod = asyncio.create_task(producer())
    cons = asyncio.create_task(r.consumer_loop(fake_send))
    await prod
    # wait for queue to empty
    while len(r.ring) > 0:
        await asyncio.sleep(0.001)
    r.stop()
    await cons
    return r.metrics


__all__ = ["Runner", "synthetic_run"]
