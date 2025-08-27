"""Venue health metrics & circuit breaker logic (clean rewrite)."""
from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Dict, List


@dataclass
class RollingMetric:
    values: List[float] = field(default_factory=list)
    maxlen: int = 50

    def add(self, v: float):
        self.values.append(v)
        if len(self.values) > self.maxlen:
            self.values.pop(0)

    def avg(self) -> float:
        return sum(self.values)/len(self.values) if self.values else 0.0


@dataclass
class VenueHealthState:
    venue_id: str
    rest_latency_ms: RollingMetric = field(default_factory=RollingMetric)
    rest_errors: int = 0
    auth_errors: int = 0
    ws_staleness_ms: float = 0.0
    ws_reconnects: int = 0
    order_submit_latency_ms: RollingMetric = field(default_factory=RollingMetric)
    order_rejects: int = 0
    last_update: float = field(default_factory=time.time)
    tripped_until: float = 0.0
    consecutive_green: int = 0

    def score(self, cfg: Dict[str, float]) -> float:
        staleness_penalty = min(1.0, self.ws_staleness_ms / max(1.0, cfg.get("staleness_ms_max", 2500)))
        rest_penalty = min(1.0, self.rest_latency_ms.avg() / 1000.0)
        order_penalty = min(1.0, self.order_submit_latency_ms.avg() / 800.0)
        error_penalty = min(1.0, (self.rest_errors + self.order_rejects + self.auth_errors * 2) / 25.0)
        raw = 1.0 - (0.35 * staleness_penalty + 0.25 * rest_penalty + 0.25 * order_penalty + 0.15 * error_penalty)
        return max(0.0, min(1.0, raw))

    def color(self, cfg: Dict[str, float]) -> str:
        s = self.score(cfg)
        if s < cfg.get("red_threshold", 0.35):
            return "red"
        if s < cfg.get("yellow_threshold", 0.65):
            return "yellow"
        return "green"

    def update_ws_tick(self, age_ms: float):
        self.ws_staleness_ms = age_ms
        self.last_update = time.time()

    def circuit_tripped(self) -> bool:
        return time.time() < self.tripped_until

    def maybe_trip(self, cfg: Dict[str, float]):
        if self.color(cfg) == "red" and not self.circuit_tripped():
            self.tripped_until = time.time() + float(cfg.get("cooloff_sec", 30))
            self.consecutive_green = 0
        elif self.color(cfg) == "green" and not self.circuit_tripped():
            self.consecutive_green += 1
        if self.circuit_tripped() and self.color(cfg) == "green":
            return
        if not self.circuit_tripped() and self.consecutive_green >= 3:
            self.consecutive_green = 3


class HealthRegistry:
    def __init__(self, cfg: Dict[str, float]):
        self.cfg = cfg
        self._venues: Dict[str, VenueHealthState] = {}

    def ensure(self, venue_id: str) -> VenueHealthState:
        if venue_id not in self._venues:
            self._venues[venue_id] = VenueHealthState(venue_id)
        return self._venues[venue_id]

    def record_rest_latency(self, venue_id: str, ms: float):
        self.ensure(venue_id).rest_latency_ms.add(ms)
        self.ensure(venue_id).maybe_trip(self.cfg)

    def record_order_latency(self, venue_id: str, ms: float):
        self.ensure(venue_id).order_submit_latency_ms.add(ms)
        self.ensure(venue_id).maybe_trip(self.cfg)

    def record_ws_staleness(self, venue_id: str, ms: float):
        self.ensure(venue_id).update_ws_tick(ms)
        self.ensure(venue_id).maybe_trip(self.cfg)

    def record_error(self, venue_id: str):
        self.ensure(venue_id).rest_errors += 1
        self.ensure(venue_id).maybe_trip(self.cfg)

    def record_order_reject(self, venue_id: str):
        self.ensure(venue_id).order_rejects += 1
        self.ensure(venue_id).maybe_trip(self.cfg)

    def record_auth_error(self, venue_id: str):  # pragma: no cover
        self.ensure(venue_id).auth_errors += 1
        self.ensure(venue_id).maybe_trip(self.cfg)

    def snapshot(self):
        out = {}
        for vid, st in self._venues.items():
            out[vid] = {
                "score": st.score(self.cfg),
                "color": st.color(self.cfg),
                "tripped": st.circuit_tripped(),
                "staleness_ms": st.ws_staleness_ms,
            }
        return out

__all__ = ["HealthRegistry", "VenueHealthState"]
