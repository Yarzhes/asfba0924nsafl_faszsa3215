"""High-level Economic Event Service (Sprint 46 scaffold).

Responsibilities:
    * Maintain in-memory list of normalized EconomicEvent objects.
    * Periodically invoke registered collectors to refresh raw events.
    * Dedupe + update statuses (scheduled->live->done) based on wall clock.
    * Expose a method to build EconFeatures for current bar.

This is an intentionally lightweight first scaffold so downstream integration
tests can wire the feature pack quickly; collectors will be fleshed out with
real parsing logic incrementally.
"""
from __future__ import annotations

from typing import List, Dict, Callable, Optional, Iterable
import time
import hashlib

from loguru import logger

from ultra_signals.core.custom_types import (
    EconomicEvent, EconEventClass, EconEventStatus, EconSeverity, EconFeatures, EconWindowSide
)


CollectorFn = Callable[[int], Iterable[EconomicEvent]]  # receives now_ms -> yields events


def _stable_event_id(source: str, raw_id: str | None, ts_start: int, title: str) -> str:
    h = hashlib.sha1(f"{source}|{raw_id or ''}|{ts_start}|{title}".encode()).hexdigest()[:16]
    return h


class EconEventService:
    def __init__(self, config: Dict):
        self.config = config
        self.collectors: Dict[str, CollectorFn] = {}
        self.events: Dict[str, EconomicEvent] = {}
        self.last_refresh_ms: int = 0
        self.template_windows = config.get("risk_windows", {  # minutes
            "cpi": (-30, 45),
            "fomc": (-60, 90),
            "nfp": (-30, 60),
            "gdp": (-30, 45),
            "exchange_maint": (0, 30),
            "holiday": (0, 0),
            "earnings_coin": (-30, 60),
            "earnings_mstr": (-30, 60),
        })
        self.severity_policy = config.get("severity_policy", {  # size multipliers / veto
            "high": {"size_mult": 0.0, "veto": True},
            "med": {"size_mult": 0.5, "veto": False},
            "low": {"size_mult": 0.8, "veto": False},
        })

    # ------------------------------------------------------------------
    # Collector registration
    # ------------------------------------------------------------------
    def register_collector(self, name: str, fn: CollectorFn):
        self.collectors[name] = fn
        logger.info(f"Registered econ collector {name}")

    # ------------------------------------------------------------------
    def refresh(self, now_ms: Optional[int] = None):
        now_ms = now_ms or int(time.time() * 1000)
        ttl_ms = int(self.config.get("refresh_min", 5) * 60_000)
        if self.last_refresh_ms and now_ms - self.last_refresh_ms < ttl_ms:
            return  # within cadence
        for name, fn in self.collectors.items():
            try:
                for ev in fn(now_ms):
                    if not ev.id:
                        ev.id = _stable_event_id(ev.source, ev.raw_id, ev.ts_start, ev.title)
                    existing = self.events.get(ev.id)
                    if existing:
                        # merge basic mutable fields
                        existing.status = ev.status
                        existing.actual = ev.actual or existing.actual
                        existing.surprise_score = ev.surprise_score or existing.surprise_score
                        existing.updated_ts = now_ms
                    else:
                        self.events[ev.id] = ev
            except Exception as e:
                logger.error(f"Collector {name} failed: {e}")
        self.last_refresh_ms = now_ms
        # Update SCHEDULED->LIVE->DONE transitions
        self._advance_status(now_ms)

    # ------------------------------------------------------------------
    def _advance_status(self, now_ms: int):
        for ev in self.events.values():
            if ev.status == EconEventStatus.SCHEDULED and now_ms >= ev.ts_start:
                ev.status = EconEventStatus.LIVE
            if ev.status == EconEventStatus.LIVE:
                end = ev.ts_end or ev.ts_start + 5 * 60_000  # default 5m duration
                if now_ms > end:
                    ev.status = EconEventStatus.DONE

    # ------------------------------------------------------------------
    def _dominant_event(self, now_ms: int) -> Optional[EconomicEvent]:
        # choose highest severity active or soon upcoming (next 6h) event
        active: List[EconomicEvent] = []
        upcoming: List[EconomicEvent] = []
        horizon_ms = now_ms + 6 * 60 * 60 * 1000
        for ev in self.events.values():
            pre, post = self.template_windows.get(ev.cls.value, (-15, 30))
            if ev.risk_pre_min is not None or ev.risk_post_min is not None:
                pre = ev.risk_pre_min or pre
                post = ev.risk_post_min or post
            if ev.in_risk_window(now_ms, abs(pre), post):
                active.append(ev)
            elif ev.ts_start > now_ms and ev.ts_start < horizon_ms:
                upcoming.append(ev)
        def sev_rank(ev: EconomicEvent):
            return {"high": 3, "med": 2, "low": 1}.get(ev.severity.value, 0)
        if active:
            return sorted(active, key=lambda e: (-sev_rank(e), e.ts_start))[0]
        if upcoming:
            return sorted(upcoming, key=lambda e: (e.ts_start, -sev_rank(e)))[0]
        return None

    # ------------------------------------------------------------------
    def build_features(self, now_ms: Optional[int] = None) -> EconFeatures:
        now_ms = now_ms or int(time.time() * 1000)
        dom = self._dominant_event(now_ms)
        feat = EconFeatures()
        if not dom:
            return feat
        pre_t, post_t = self.template_windows.get(dom.cls.value, (-15, 30))
        if dom.risk_pre_min is not None:
            pre_t = dom.risk_pre_min
        if dom.risk_post_min is not None:
            post_t = dom.risk_post_min
        # Determine window side
        start_pre = dom.ts_start - pre_t * 60_000
        end_post = (dom.ts_end or dom.ts_start) + post_t * 60_000
        if now_ms < dom.ts_start:
            side = EconWindowSide.PRE.value if now_ms >= start_pre else EconWindowSide.OUT.value
        elif dom.status == EconEventStatus.LIVE and now_ms <= (dom.ts_end or dom.ts_start + 5 * 60_000):
            side = EconWindowSide.LIVE.value
        elif now_ms <= end_post:
            side = EconWindowSide.POST.value
        else:
            side = EconWindowSide.OUT.value
        feat.econ_window_side = side
        feat.econ_risk_class = dom.cls.value
        feat.econ_severity = dom.severity.value
        feat.econ_countdown_min = max(0.0, (dom.ts_start - now_ms) / 60000.0) if now_ms < dom.ts_start else 0.0
        feat.econ_surprise_score = dom.surprise_score
        feat.econ_risk_active = 1 if side in ("pre", "live", "post") else 0
        # size policy
        pol = self.severity_policy.get(dom.severity.value, {"size_mult": 1.0})
        feat.allowed_size_mult_econ = pol.get("size_mult", 1.0)
        feat.flags = [f"{dom.cls.value.upper()}_{side.upper()}"]
        return feat


# ---------------------------------------------------------------------------
# Minimal local collector example (reads static config events)
# ---------------------------------------------------------------------------
def static_config_collector_factory(static_events: List[Dict]) -> CollectorFn:
    def _collector(now_ms: int):
        for raw in static_events:
            try:
                cls = EconEventClass(raw.get("cls", "other")) if raw.get("cls") in EconEventClass._value2member_map_ else EconEventClass.OTHER
                sev = EconSeverity(raw.get("severity", "med")) if raw.get("severity") in EconSeverity._value2member_map_ else EconSeverity.MED
                ev = EconomicEvent(
                    id=raw.get("id") or _stable_event_id("static", raw.get("raw_id"), raw["ts_start"], raw.get("title", "")),
                    source="static",
                    raw_id=raw.get("raw_id"),
                    cls=cls,
                    title=raw.get("title", cls.value),
                    region=raw.get("region"),
                    severity=sev,
                    symbols=raw.get("symbols"),
                    ts_start=raw["ts_start"],
                    ts_end=raw.get("ts_end"),
                    status=EconEventStatus.SCHEDULED,
                    risk_pre_min=raw.get("risk_pre_min"),
                    risk_post_min=raw.get("risk_post_min"),
                    notes=raw.get("notes"),
                )
                yield ev
            except Exception as e:
                logger.error(f"static_config_collector bad event {raw}: {e}")
    return _collector
