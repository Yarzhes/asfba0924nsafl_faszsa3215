from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List


@dataclass
class VenueStats:
    fills: int = 0
    total_filled_qty: float = 0.0
    total_requested_qty: float = 0.0
    slip_sum_bps: float = 0.0
    slip_sq_sum: float = 0.0
    latency_sum_ms: float = 0.0
    rejects: int = 0
    # EWMA fields (useful for per-symbol fast adjustments)
    ewma_slip_bps: Optional[float] = None
    ewma_alpha: float = 0.2

    def record_fill(self, slip_bps: float, filled_qty: float, requested_qty: float, latency_ms: float):
        self.fills += 1
        self.total_filled_qty += float(filled_qty)
        self.total_requested_qty += float(requested_qty)
        self.slip_sum_bps += float(slip_bps)
        self.slip_sq_sum += float(slip_bps) * float(slip_bps)
        self.latency_sum_ms += float(latency_ms)
        # update EWMA
        try:
            if self.ewma_slip_bps is None:
                self.ewma_slip_bps = float(slip_bps)
            else:
                a = float(self.ewma_alpha or 0.2)
                self.ewma_slip_bps = a * float(slip_bps) + (1.0 - a) * float(self.ewma_slip_bps)
        except Exception:
            pass

    def record_reject(self):
        self.rejects += 1

    @property
    def avg_slip_bps(self) -> Optional[float]:
        return (self.slip_sum_bps / self.fills) if self.fills else None

    @property
    def fill_ratio(self) -> Optional[float]:
        if self.total_requested_qty <= 0:
            return None
        return self.total_filled_qty / self.total_requested_qty

    @property
    def avg_latency_ms(self) -> Optional[float]:
        return (self.latency_sum_ms / self.fills) if self.fills else None

    @property
    def slip_variance(self) -> Optional[float]:
        if self.fills <= 1:
            return None
        mean = self.avg_slip_bps
        try:
            return max(0.0, (self.slip_sq_sum / self.fills) - (mean * mean))
        except Exception:
            return None




class TCAEngine:
    """Lightweight TCA engine collecting per-fill metrics and exposing
    simple venue-level scores for routing feedback.

    Notes:
    - Keeps in-memory aggregates and appends a compact JSON line to a log file.
    - Exposes get_effective_cost_bps(...) that adjusts a base cost using
      observed average slip and a latency penalty (lambda tunable).
    """

    def __init__(self, logfile: Optional[str] = None, latency_lambda: float = 0.001):
        # logfile path where per-fill events are appended as json lines
        self.logfile = logfile or os.path.join(os.getcwd(), 'tca_events.jsonl')
        self.latency_lambda = float(latency_lambda)
        self._venues: Dict[str, VenueStats] = defaultdict(VenueStats)
        # per-symbol per-venue aggregates (EWMA & quick lookup)
        # structure: symbol -> venue -> VenueStats
        self._symbol_venues = defaultdict(lambda: defaultdict(VenueStats))
        self._lock = threading.Lock()

        # alerting thresholds (sigma multiplier) default
        self.alert_sigma = 2.0
        # optional callable to publish alerts (callable signature like publish_alert(name, msg, severity='WARN', meta=None))
        self._publish_alert = None

    def set_alert_publisher(self, publisher_callable):
        """Set an optional alert publisher function used by check_alerts.

        This allows tests to inject a stub to capture published alerts.
        """
        try:
            if publisher_callable is None:
                self._publish_alert = None
            elif callable(publisher_callable):
                self._publish_alert = publisher_callable
        except Exception:
            self._publish_alert = None

    # --- Test / helper APIs -------------------------------------------------
    def synthesize_venue_stats(self, venue: str, avg_slip_bps: float, variance: float = 0.0, fills: int = 1, ewma: Optional[float] = None, latency_ms: float = 0.0, rejects: int = 0):
        """Create or update a VenueStats for testing or backfill.

        This avoids tests poking internal dicts directly.
        """
        with self._lock:
            vs = self._venues[str(venue)]
            vs.fills = int(max(1, fills))
            vs.slip_sum_bps = float(avg_slip_bps) * float(vs.fills)
            vs.slip_sq_sum = (float(variance) + float(avg_slip_bps) ** 2) * float(vs.fills)
            vs.latency_sum_ms = float(latency_ms) * float(vs.fills)
            vs.rejects = int(rejects)
            if ewma is not None:
                vs.ewma_slip_bps = float(ewma)

    def synthesize_symbol_venue_stats(self, symbol: str, venue: str, avg_slip_bps: float, variance: float = 0.0, fills: int = 1, ewma: Optional[float] = None, latency_ms: float = 0.0, rejects: int = 0):
        """Create or update a symbol-specific VenueStats."""
        with self._lock:
            sv = self._symbol_venues[symbol][venue]
            sv.fills = int(max(1, fills))
            sv.slip_sum_bps = float(avg_slip_bps) * float(sv.fills)
            sv.slip_sq_sum = (float(variance) + float(avg_slip_bps) ** 2) * float(sv.fills)
            sv.latency_sum_ms = float(latency_ms) * float(sv.fills)
            sv.rejects = int(rejects)
            if ewma is not None:
                sv.ewma_slip_bps = float(ewma)

    # per-symbol last-checked timestamp (ms) to allow independent scheduling
    def _mark_symbol_checked(self, symbol: str):
        try:
            import time as _time
            with self._lock:
                if not hasattr(self, '_symbol_last_checked'):
                    self._symbol_last_checked = {}
                self._symbol_last_checked[str(symbol)] = int(_time.time() * 1000)
        except Exception:
            pass

    def _get_symbol_last_checked(self, symbol: str) -> Optional[int]:
        with self._lock:
            if not hasattr(self, '_symbol_last_checked'):
                return None
            return self._symbol_last_checked.get(str(symbol))

    def record_fill(self, event: Dict[str, Any]):
        """Record a fill event.

        Expected event keys (best-effort):
        - venue, symbol, fill_px, arrival_px, filled_qty, requested_qty, arrival_ts_ms, completion_ts_ms
        """
        try:
            venue = str(event.get('venue') or 'UNKNOWN')
            filled_qty = float(event.get('filled_qty') or event.get('qty') or 0.0)
            requested_qty = float(event.get('requested_qty') or event.get('requested') or filled_qty or 0.0)
            arrival_px = event.get('arrival_px')
            fill_px = event.get('fill_px') or event.get('exec_price') or event.get('avg_px')
            arrival_ts = event.get('arrival_ts_ms') or event.get('arrival_ts')
            completion_ts = event.get('completion_ts_ms') or event.get('completion_ts') or event.get('ts')

            slip_bps = None
            if arrival_px and fill_px and arrival_px != 0:
                try:
                    slip_bps = (float(fill_px) - float(arrival_px)) / float(arrival_px) * 10000.0
                except Exception:
                    slip_bps = None

            latency_ms = None
            if arrival_ts and completion_ts:
                try:
                    latency_ms = float(completion_ts) - float(arrival_ts)
                except Exception:
                    latency_ms = None

            # fallback defaults
            slip_bps = 0.0 if slip_bps is None else slip_bps
            latency_ms = 0.0 if latency_ms is None else latency_ms

            with self._lock:
                vs = self._venues[venue]
                vs.record_fill(slip_bps, filled_qty, requested_qty, latency_ms)
                # per-symbol per-venue
                try:
                    if event.get('symbol'):
                        sym = str(event.get('symbol'))
                        sv = self._symbol_venues[sym][venue]
                        sv.record_fill(slip_bps, filled_qty, requested_qty, latency_ms)
                except Exception:
                    pass

            # append compact event to log for offline analysis
            rec = {
                'ts': int(completion_ts) if completion_ts is not None else None,
                'venue': venue,
                'symbol': event.get('symbol'),
                'slip_bps': slip_bps,
                'filled_qty': filled_qty,
                'requested_qty': requested_qty,
                'latency_ms': latency_ms,
                'order_type': event.get('order_type'),
            }
            try:
                with open(self.logfile, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                # logging failures must not break the engine
                pass
        except Exception:
            # swallow errors to stay non-invasive
            return

    def record_reject(self, venue: str):
        with self._lock:
            vs = self._venues[str(venue)]
            vs.record_reject()

    def record_reject_for_symbol(self, venue: str, symbol: str):
        with self._lock:
            sv = self._symbol_venues.get(symbol)
            if sv is None:
                sv = defaultdict(VenueStats)
                self._symbol_venues[symbol] = sv
            vs = sv[venue]
            vs.record_reject()

    def get_venue_stats(self, venue: str) -> Optional[VenueStats]:
        return self._venues.get(str(venue))

    def get_effective_cost_bps(self, venue: str, base_cost_bps: float, rtt_ms: Optional[float] = None) -> float:
        """Return an adjusted cost (bps) for a venue using observed TCA metrics.

        effective_cost = base_cost + avg_slip_bps + latency_lambda * avg_latency_ms
        If no stats exist for venue, returns base_cost_bps unchanged.
        """
        vs = self.get_venue_stats(venue)
        if not vs:
            return float(base_cost_bps)

        adj = float(base_cost_bps)
        if vs.avg_slip_bps is not None:
            adj += vs.avg_slip_bps
        lat = vs.avg_latency_ms if vs.avg_latency_ms is not None else (float(rtt_ms) if rtt_ms is not None else 0.0)
        adj += self.latency_lambda * float(lat)
        # penalize venues with high reject rates mildly
        if vs.rejects > 0:
            adj += min(10.0, float(vs.rejects) * 0.5)
        return adj

    def get_top_venues(self, symbol: str, k: int = 3, base_costs: Optional[Dict[str, float]] = None) -> List[str]:
        """Return top-k venues for a given symbol sorted by effective cost."""
        out = []
        with self._lock:
            sv = self._symbol_venues.get(symbol) or {}
            for vid, vs in sv.items():
                base = 0.0
                if base_costs and vid in base_costs:
                    base = float(base_costs[vid])
                eff = self.get_effective_cost_bps(vid, base, None)
                out.append((vid, eff))
        out.sort(key=lambda x: x[1])
        return [v for v, _ in out[:k]]

    def check_alerts(self, symbol: Optional[str] = None, sigma: Optional[float] = None):
        """Check slip deviations and publish alerts when a venue slips > mean + sigma*stddev.

        If symbol is provided, checks per-symbol per-venue EWMA vs the venue global stats.
        """
        s = float(sigma or self.alert_sigma)
        try:
            # optional alert publisher
            from ultra_signals.core.alerts import publish_alert
        except Exception:
            publish_alert = None

        alerts = []
        with self._lock:
            if symbol:
                sv = self._symbol_venues.get(symbol, {})
                for vid, vs in sv.items():
                    mean = vs.avg_slip_bps
                    var = vs.slip_variance
                    std = (var ** 0.5) if var else None
                    ewma = vs.ewma_slip_bps
                    if ewma is not None and mean is not None and std is not None and std > 0 and ewma >= mean + s * std:
                        alerts.append((symbol, vid, ewma, mean, std))
            else:
                for vid, vs in self._venues.items():
                    mean = vs.avg_slip_bps
                    var = vs.slip_variance
                    std = (var ** 0.5) if var else None
                    ewma = vs.ewma_slip_bps
                    if ewma is not None and mean is not None and std is not None and std > 0 and ewma >= mean + s * std:
                        alerts.append((None, vid, ewma, mean, std))

        for a in alerts:
            sym, vid, ewma, mean, std = a
            msg = f"TCA_ALERT venue={vid} sym={sym or '*'} ewma={ewma:.2f} mean={mean:.2f} std={std:.2f}"
            try:
                if publish_alert:
                    publish_alert('TCA_SLIP_ALERT', msg, severity='WARN', meta={'venue': vid, 'symbol': sym, 'ewma': ewma, 'mean': mean, 'std': std})
            except Exception:
                pass
        return alerts


__all__ = ['TCAEngine', 'VenueStats']
