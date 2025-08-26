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
        self._last_export_headers = False

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

    # --- Prometheus style exposition (minimal) ---
    def to_prometheus(self) -> str:
        snap = self.snapshot()
        lines = []
        c = snap["counters"]
        lines.append(f"orders_sent_total {c.get('orders_sent',0)}")
        lines.append(f"orders_errors_total {c.get('orders_errors',0)}")
        lt = snap["latency_tick_to_decision"]
        if lt.get("count"):
            lines.append(f"latency_tick_to_decision_ms_p50 {lt.get('p50',0):.3f}")
            lines.append(f"latency_tick_to_decision_ms_p99 {lt.get('p99',0):.3f}")
        ld = snap["latency_decision_to_order"]
        if ld.get("count"):
            lines.append(f"latency_decision_to_order_ms_p50 {ld.get('p50',0):.3f}")
            lines.append(f"latency_decision_to_order_ms_p99 {ld.get('p99',0):.3f}")
        for qn, depth in snap["queues"].items():
            lines.append(f"queue_depth{{queue=\"{qn}\"}} {depth}")
        lines.append(f"process_uptime_seconds {snap['uptime_sec']}")
        return "\n".join(lines) + "\n"

    def export_csv(self, path: str):  # pragma: no cover (simple IO)
        snap = self.snapshot()
        import csv, os
        flat = {
            "uptime_sec": snap["uptime_sec"],
            "q_feed": snap["queues"].get("feed", 0),
            "q_orders": snap["queues"].get("orders", 0),
            "orders_sent": snap["counters"].get("orders_sent", 0),
            "orders_errors": snap["counters"].get("orders_errors", 0),
            "lat_tick_p50": snap["latency_tick_to_decision"].get("p50", 0),
            "lat_tick_p99": snap["latency_tick_to_decision"].get("p99", 0),
            "lat_dec_p50": snap["latency_decision_to_order"].get("p50", 0),
            "lat_dec_p99": snap["latency_decision_to_order"].get("p99", 0),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not self._last_export_headers or not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                w.writeheader()
                self._last_export_headers = True
            w.writerow(flat)

__all__ = ["Metrics"]
