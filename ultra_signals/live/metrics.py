"""Lightweight in-process metrics collectors for live trading.

Provides simple histograms (stored as ring buffer of recent samples) and
point-in-time gauges for queue depths, latency and order rates. Kept
dependency-free (no Prometheus client) but structure allows easy future
export.
"""
from __future__ import annotations
import time
import statistics
from collections import deque
from typing import Dict, Any


class Histogram:
    def __init__(self, max_samples: int = 5000):
        self.buf = deque(maxlen=max_samples)

    def observe(self, value: float):
        self.buf.append(float(value))

    def snapshot(self) -> Dict[str, float]:  # percentile snapshot
        if not self.buf:
            return {"count": 0}
        data = list(self.buf)
        data.sort()
        def pct(p: float):
            if not data:
                return 0.0
            k = int(p * (len(data) - 1))
            return data[k]
        return {
            "count": len(data),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": data[-1],
            "min": data[0],
            "mean": statistics.fmean(data) if len(data) > 1 else data[0],
        }


class Metrics:
    def __init__(self):
        self.started_monotonic = time.perf_counter()
        self.latency_tick_to_decision = Histogram()
        self.latency_decision_to_order = Histogram()
        self.queue_depths: Dict[str, int] = {}
        self.counters: Dict[str, int] = {"orders_sent": 0, "orders_errors": 0}

    def inc(self, key: str, value: int = 1):
        self.counters[key] = self.counters.get(key, 0) + value

    def set_queue_depth(self, name: str, depth: int):
        self.queue_depths[name] = depth

    def snapshot(self) -> Dict[str, Any]:
        up = time.perf_counter() - self.started_monotonic
        return {
            "uptime_sec": round(up, 1),
            "latency_tick_to_decision": self.latency_tick_to_decision.snapshot(),
            "latency_decision_to_order": self.latency_decision_to_order.snapshot(),
            "queues": dict(self.queue_depths),
            "counters": dict(self.counters),
        }

__all__ = ["Metrics"]
