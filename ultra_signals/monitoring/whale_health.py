"""Whale Pipeline Health Snapshot.

Produces per-source health metrics for ops dashboards:
  - last_event_age_sec
  - event_count_1h
  - active (bool)
  - error counters (future extension)

Usage: call WhaleHealthMonitor.update(source, ts_ms) from collectors, then
snapshot() to retrieve summary.
"""
from __future__ import annotations
import time
from typing import Dict, Any

class WhaleHealthMonitor:
    def __init__(self):
        self._sources: Dict[str, Dict[str, Any]] = {}

    def update(self, source: str, ts_ms: int | None = None):
        now = int(time.time()*1000)
        rec = self._sources.setdefault(source, {'events': []})
        rec['events'].append(ts_ms or now)
        if len(rec['events']) > 2000:
            del rec['events'][:1000]

    def snapshot(self) -> Dict[str, Any]:
        now = int(time.time()*1000)
        out: Dict[str, Any] = {}
        for src, rec in self._sources.items():
            events = rec.get('events') or []
            last_ts = events[-1] if events else None
            events_1h = [e for e in events if e >= now - 3600_000]
            out[src] = {
                'last_event_age_sec': None if last_ts is None else max(0,(now-last_ts)//1000),
                'event_count_1h': len(events_1h),
                'active': bool(last_ts and (now-last_ts) < 5*60*1000),
            }
        return out
