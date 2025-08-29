"""Minimal, clean metrics for the low-latency runner.

This file intentionally small: a Histogram and Metrics container.
"""
from __future__ import annotations
import time
import statistics
from collections import deque
from typing import Dict, Any, Optional


class Histogram:
    def __init__(self, max_samples: int = 4096):
        self._buf = deque(maxlen=max_samples)

    def observe(self, value: float) -> None:
        # keep hot path cheap: cast and append
        self._buf.append(float(value))

    def snapshot(self) -> Dict[str, float]:
        if not self._buf:
            return {"count": 0}
        data = sorted(self._buf)
        n = len(data)

        def pct(p: float) -> float:
            if n == 0:
                return 0.0
            idx = int(p * (n - 1))
            return float(data[idx])

        return {
            "count": n,
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
            "min": float(data[0]),
            "max": float(data[-1]),
            "mean": statistics.fmean(data) if n else 0.0,
        }


class Metrics:
    """Runtime, low-latency metrics container.

    Kept deliberately small and import-safe (no heavy deps). This class
    collects a few histograms and a simple pre-trade summary which may be
    rendered by transport formatters.
    """

    def __init__(self) -> None:
        self.started_monotonic = time.perf_counter()

        # core latency histograms (milliseconds)
        self.latency_tick_to_decision = Histogram()
        self.latency_decision_to_order = Histogram()
        self.latency_wire_to_ack = Histogram()

        # misc histograms
        self.fill_ratio = Histogram()
        self.fill_slip_bps = Histogram()

        # counters and gauges
        self.queue_depths = {}
        # Core counters. Additional dynamic counters (per-block reason) created on demand.
        self.counters = {
            "orders_sent": 0,
            "orders_errors": 0,
            "sniper_hourly_cap": 0,
            "sniper_daily_cap": 0,
            "sniper_mtf_required": 0,
            # Signal lifecycle counters
            "signals_candidates": 0,
            "signals_allowed": 0,
            "signals_blocked": 0,
        }

        self.needs_resync_counter = 0

        # latest pre-trade summary (for transport/telemetry)
        # example: {"p_win": 0.62, "regime": "risk_on", "veto_count": 0, "lat_ms": {"p50": ..}}
        self.last_pre_trade = None

    # basic mutators -------------------------------------------------
    def inc(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + int(value)

    def set_queue_depth(self, name: str, depth: int) -> None:
        self.queue_depths[name] = int(depth)

    # observation helpers -------------------------------------------
    def observe_tick_to_decision(self, start_ns: int, end_ns: Optional[int] = None) -> None:
        end = end_ns if end_ns is not None else time.perf_counter_ns()
        self.latency_tick_to_decision.observe((end - int(start_ns)) / 1_000_000.0)

    def observe_decision_to_order(self, start_ns: int, end_ns: Optional[int] = None) -> None:
        end = end_ns if end_ns is not None else time.perf_counter_ns()
        self.latency_decision_to_order.observe((end - int(start_ns)) / 1_000_000.0)

    def observe_wire_to_ack(self, start_ns: int, end_ns: Optional[int] = None) -> None:
        end = end_ns if end_ns is not None else time.perf_counter_ns()
        self.latency_wire_to_ack.observe((end - int(start_ns)) / 1_000_000.0)

    # snapshot / export ------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        up = time.perf_counter() - self.started_monotonic
        return {
            "uptime_sec": round(up, 1),
            "latency_tick_to_decision": self.latency_tick_to_decision.snapshot(),
            "latency_decision_to_order": self.latency_decision_to_order.snapshot(),
            "latency_wire_to_ack": self.latency_wire_to_ack.snapshot(),
            "queues": dict(self.queue_depths),
            "counters": dict(self.counters),
            "fill_ratio": self.fill_ratio.snapshot(),
            "fill_slip_bps": self.fill_slip_bps.snapshot(),
            "needs_resync": int(self.needs_resync_counter),
            "last_pre_trade": dict(self.last_pre_trade) if self.last_pre_trade is not None else None,
        }

    def export_csv(self, filepath: str) -> None:
        import csv
        snap = self.snapshot()

        rows = []
        for key, value in snap.items():
            if not isinstance(value, dict):
                rows.append((key, value))

        for key, value in snap.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    rows.append((f"{key}_{subkey}", subval))

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    def to_prometheus(self) -> str:
        snap = self.snapshot()
        lines = []

        for key, value in snap.get("counters", {}).items():
            lines.append(f"# TYPE {key}_total counter")
            lines.append(f"{key}_total {value}")

        lines.append(f"# TYPE uptime_sec gauge")
        lines.append(f"uptime_sec {snap.get('uptime_sec', 0)}")

        for hist_name in ["latency_tick_to_decision", "latency_decision_to_order", "latency_wire_to_ack", "fill_ratio", "fill_slip_bps"]:
            hist_data = snap.get(hist_name, {})
            if hist_data.get("count", 0) > 0:
                lines.append(f"# TYPE {hist_name} histogram")
                lines.append(f"{hist_name}_count {hist_data.get('count', 0)}")
                lines.append(f"{hist_name}_sum {hist_data.get('mean', 0) * hist_data.get('count', 0)}")
                for pct in ["p50", "p90", "p99"]:
                    if pct in hist_data:
                        quantile = {"p50": "0.5", "p90": "0.9", "p99": "0.99"}[pct]
                        lines.append(f"{hist_name}{{quantile=\"{quantile}\"}} {hist_data[pct]}")

        return "\n".join(lines) + "\n"

    # ----- Pre-trade summary helpers -----------------------------------
    def set_pre_trade_summary(self, summary: Dict[str, Any]) -> None:
        try:
            self.last_pre_trade = dict(summary)
        except Exception:
            self.last_pre_trade = None

    def clear_pre_trade_summary(self) -> None:
        self.last_pre_trade = None

    # ----- Sniper metrics helpers -----------------------------------
    def inc_sniper_rejection(self, reason: str) -> None:
        """Increment sniper rejection counter by reason."""
        reason_lower = reason.lower()
        if 'hourly' in reason_lower:
            self.inc('sniper_hourly_cap')
        elif 'daily' in reason_lower:
            self.inc('sniper_daily_cap')
        elif 'mtf' in reason_lower:
            self.inc('sniper_mtf_required')

    # ----- Signal lifecycle helpers -----------------------------------
    def record_candidate(self) -> None:
        self.inc('signals_candidates')

    def record_allowed(self) -> None:
        self.inc('signals_allowed')

    def record_block(self, reason: str) -> None:
        self.inc('signals_blocked')
        if not reason:
            return
        # Support multi-reason strings separated by ;
        for r in str(reason).split(';'):
            r = r.strip()
            if not r:
                continue
            key = f"block_{r.lower()}"
            self.inc(key)


__all__ = ["Metrics", "Histogram"]
