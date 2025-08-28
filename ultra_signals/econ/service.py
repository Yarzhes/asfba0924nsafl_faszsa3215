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
import json
import os
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
        self.last_refresh_ms = 0
        self.persist_path = config.get('persist_path', 'econ_events.json')
        self.health: Dict[str, Dict[str, int]] = {}
        self.template_windows = config.get("risk_windows", {
            "cpi": (-30, 45),
            "fomc": (-60, 90),
            "nfp": (-30, 60),
            "gdp": (-30, 45),
            "exchange_maint": (0, 30),
            "holiday": (0, 0),
            "earnings_coin": (-30, 60),
            "earnings_mstr": (-30, 60),
        })
        self.severity_policy = config.get("severity_policy", {
            "high": {"size_mult": 0.0, "veto": True},
            "med": {"size_mult": 0.5, "veto": False},
            "low": {"size_mult": 0.8, "veto": False},
        })
        # Optional per-class size overrides: { 'cpi': 0.2, 'fomc': 0.0 }
        self.class_size_overrides: Dict[str, float] = config.get('class_size_overrides', {}) or {}
        # Track which pre-alert thresholds already fired per event id. NOTE: must be defined BEFORE _load_persisted
        # so that persisted alert_state isn't immediately overwritten (bug fix for duplicate alerts after restart).
        self._alert_state: Dict[str, set] = {}
        self._load_persisted()
        self._alert_minutes: List[int] = sorted(set(config.get('alert_minutes', [30, 10])))
        # Severity-specific alert minutes (override global for that severity if present)
        self._alert_minutes_sev: Dict[str, List[int]] = {
            k: sorted(set(v)) for k, v in (config.get('alert_minutes_severity', {}) or {}).items() if isinstance(v, (list, tuple))
        }
        # Auto-register configured collector sources
        try:
            if config.get('ics_sources'):
                from .collectors_ics import ics_collector_factory
                self.register_collector('ics', ics_collector_factory(config.get('ics_sources')))
            if config.get('exchange_maint'):
                from .collectors_exchange import exchange_static_collector_factory
                self.register_collector('exchange', exchange_static_collector_factory(config.get('exchange_maint')))
            if config.get('earnings'):
                from .collectors_earnings import earnings_static_collector_factory
                self.register_collector('earnings', earnings_static_collector_factory(config.get('earnings')))
            if config.get('exchange_http_sources'):
                from .collectors_exchange import exchange_http_collector_factory
                self.register_collector('exchange_http', exchange_http_collector_factory(config.get('exchange_http_sources')))
            if config.get('earnings_rss_sources'):
                from .collectors_earnings import earnings_rss_collector_factory
                self.register_collector('earnings_rss', earnings_rss_collector_factory(config.get('earnings_rss_sources')))
        except Exception as e:  # pragma: no cover
            logger.debug(f"Econ auto-register error: {e}")
        # Telegram callback placeholder: user supplies callable taking (event, phase)
        self._telegram_cb = config.get('telegram_callback')  # Optional callable

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
                        existing.status = ev.status
                        existing.actual = ev.actual or existing.actual
                        existing.surprise_score = ev.surprise_score or existing.surprise_score
                        existing.updated_ts = now_ms
                        # If an event moved to LIVE and we have expected vs actual -> compute surprise
                        try:
                            if existing.status == EconEventStatus.LIVE and existing.actual and existing.expected and existing.surprise_score is None:
                                existing.surprise_score = self._compute_surprise(existing.expected, existing.actual)
                        except Exception:
                            pass
                    else:
                        self.events[ev.id] = ev
                        # New event, maybe schedule telegram pre alert at registration (handled externally)
                h = self.health.setdefault(name, {"error_count": 0, "last_success_ms": 0})
                h["last_success_ms"] = now_ms
            except Exception as e:
                logger.error(f"Collector {name} failed: {e}")
                h = self.health.setdefault(name, {"error_count": 0, "last_success_ms": 0})
                h["error_count"] += 1
        self.last_refresh_ms = now_ms
        # Update SCHEDULED->LIVE->DONE transitions
        self._advance_status(now_ms)
        # Recompute missing surprise scores (post status transitions)
        self._compute_missing_surprises()
        # Emit pre-alerts after fresh state
        self._emit_pre_alerts(now_ms)
        # Persist occasionally
        if int(time.time()) % 60 == 0:  # crude throttle once per minute
            self._persist()

    # ------------------------------------------------------------------
    def _advance_status(self, now_ms: int):
        for ev in self.events.values():
            if ev.status == EconEventStatus.SCHEDULED and now_ms >= ev.ts_start:
                ev.status = EconEventStatus.LIVE
                self._maybe_telegram(ev, 'live')
            if ev.status == EconEventStatus.LIVE:
                end = ev.ts_end or ev.ts_start + 5 * 60_000  # default 5m duration
                if now_ms > end:
                    ev.status = EconEventStatus.DONE
                    self._maybe_telegram(ev, 'done')

    # ------------------------------------------------------------------
    def _persist(self):  # best-effort; ignore errors
        try:
            serial_events = [e.model_dump() for e in self.events.values()]
            serial_alerts = {eid: list(ths) for eid, ths in self._alert_state.items() if ths}
            payload = {"events": serial_events, "alert_state": serial_alerts}
            tmp = self.persist_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, separators=(',', ':'), ensure_ascii=False)
            os.replace(tmp, self.persist_path)
        except Exception:
            pass

    def _load_persisted(self):
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):  # backward compatibility
                arr = data
                for raw in arr:
                    try:
                        ev = EconomicEvent(**raw)
                        self.events[ev.id] = ev
                    except Exception:
                        continue
            elif isinstance(data, dict):
                for raw in data.get('events', []):
                    try:
                        ev = EconomicEvent(**raw)
                        self.events[ev.id] = ev
                    except Exception:
                        continue
                try:
                    alerts = data.get('alert_state', {}) or {}
                    self._alert_state = {k: set(v) for k, v in alerts.items() if isinstance(v, list)}
                except Exception:
                    pass
            logger.info(f"Loaded {len(self.events)} persisted economic events from {self.persist_path}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _active_and_upcoming(self, now_ms: int):
        active: List[EconomicEvent] = []
        upcoming: List[EconomicEvent] = []
        horizon_ms = now_ms + 6 * 60 * 60 * 1000
        for ev in self.events.values():
            pre, post = self.template_windows.get(ev.cls.value, (-15, 30))
            if ev.risk_pre_min is not None:
                pre = ev.risk_pre_min
            if ev.risk_post_min is not None:
                post = ev.risk_post_min
            if ev.in_risk_window(now_ms, abs(pre), post):
                active.append(ev)
            elif now_ms < ev.ts_start < horizon_ms:
                upcoming.append(ev)
        return active, upcoming

    def _choose_dominant(self, active: List[EconomicEvent], upcoming: List[EconomicEvent]) -> Optional[EconomicEvent]:
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
        active, upcoming = self._active_and_upcoming(now_ms)
        dom = self._choose_dominant(active, upcoming)
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
        # Multi-event aggregation
        feat.econ_risk_active = 1 if any(e for e in active) and side in ("pre", "live", "post") else 0
        # Compute min size multiplier across active events by severity
        if active:
            size_mults = []
            flags = []
            for ev in active:
                pol_ev = self.severity_policy.get(ev.severity.value, {"size_mult": 1.0})
                base_mult = pol_ev.get("size_mult", 1.0)
                # Apply class override if configured (min takes precedence for safety)
                cls_override = self.class_size_overrides.get(ev.cls.value)
                if cls_override is not None:
                    base_mult = min(base_mult, float(cls_override))
                size_mults.append(base_mult)
                # Determine window side per event for flagging
                pre_a, post_a = self.template_windows.get(ev.cls.value, (-15, 30))
                if ev.risk_pre_min is not None:
                    pre_a = ev.risk_pre_min
                if ev.risk_post_min is not None:
                    post_a = ev.risk_post_min
                start_pre_ev = ev.ts_start - pre_a * 60_000
                end_post_ev = (ev.ts_end or ev.ts_start) + post_a * 60_000
                if now_ms < ev.ts_start and now_ms >= start_pre_ev:
                    side_ev = "PRE"
                elif ev.status == EconEventStatus.LIVE and now_ms <= (ev.ts_end or ev.ts_start + 5 * 60_000):
                    side_ev = "LIVE"
                elif now_ms <= end_post_ev:
                    side_ev = "POST"
                else:
                    side_ev = "OUT"
                flags.append(f"{ev.cls.value.upper()}_{side_ev}")
            feat.allowed_size_mult_econ = min(size_mults) if size_mults else 1.0
            feat.flags = flags
        else:
            pol = self.severity_policy.get(dom.severity.value, {"size_mult": 1.0})
            base_mult = pol.get("size_mult", 1.0)
            cls_override = self.class_size_overrides.get(dom.cls.value)
            if cls_override is not None:
                base_mult = min(base_mult, float(cls_override))
            feat.allowed_size_mult_econ = base_mult
            feat.flags = [f"{dom.cls.value.upper()}_{side.upper()}"]
        # Meta health snapshot
        try:
            stale_ct = 0
            now_ms_i = now_ms
            for h in self.health.values():
                if now_ms_i - h.get("last_success_ms", 0) > 60*60*1000:  # >1h stale
                    stale_ct += 1
            feat.meta = {"sources": float(len(self.health)), "stale": float(stale_ct), "active_events": float(len(active))}
        except Exception:
            pass
        return feat

    # ------------------------------------------------------------------
    def _compute_surprise(self, expected: str, actual: str) -> Optional[float]:
        """Compute a naive surprise score: (actual - expected)/abs(expected).
        Falls back to None if parse fails. Only numeric extraction of first float."""
        try:
            import re
            def _first_float(s: str):
                m = re.search(r"-?\d+(?:\.\d+)?", s)
                return float(m.group(0)) if m else None
            e = _first_float(str(expected))
            a = _first_float(str(actual))
            if e is None or a is None or e == 0:
                return None
            return (a - e)/abs(e)
        except Exception:
            return None

    def _maybe_telegram(self, ev: EconomicEvent, phase: str):  # phase: 'live'|'done'
        cb = self._telegram_cb
        if not cb:
            return
        try:
            cb(ev, phase)
        except Exception:
            pass

    def _compute_missing_surprises(self):
        for ev in self.events.values():
            if ev.surprise_score is None and ev.actual and ev.expected:
                try:
                    ev.surprise_score = self._compute_surprise(ev.expected, ev.actual)
                except Exception:
                    continue

    def _emit_pre_alerts(self, now_ms: int):
        if not self._telegram_cb or not self._alert_minutes:
            return
        for ev in self.events.values():
            if ev.status != EconEventStatus.SCHEDULED:
                continue
            mins = (ev.ts_start - now_ms) / 60000.0
            if mins < 0:
                continue
            fired = self._alert_state.setdefault(ev.id, set())
            # Determine threshold list: severity-specific override if configured
            sev_key = getattr(ev.severity, 'value', None)
            thresholds = self._alert_minutes_sev.get(sev_key, self._alert_minutes)
            for threshold in thresholds:
                if mins <= threshold and threshold not in fired:
                    # fire alert
                    try:
                        self._telegram_cb(ev, 'pre', threshold)
                        fired.add(threshold)
                    except Exception:
                        continue


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
