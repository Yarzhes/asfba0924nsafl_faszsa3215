from __future__ import annotations
"""
Sprint 16: News & Event Veto System
-----------------------------------
Lightweight loader for scheduled macro / exchange events.
Supports a local YAML file of events with fields: time, title, impact.
Provides is_event_now(ts_ms) -> (blocked: bool, reason: str)
Embargo window: N minutes before & after event time blocks trading.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import yaml
from pathlib import Path

@dataclass
class NewsEvent:
    time: pd.Timestamp
    title: str
    impact: str  # LOW/MEDIUM/HIGH

class NewsVeto:
    def __init__(self, settings: dict):
        self.settings = settings or {}
        cfg = (self.settings.get('news_veto') or {})
        self.enabled = bool(cfg.get('enabled', True))
        self.embargo_minutes = int(cfg.get('embargo_minutes', 15))
        self.high_impact_only = bool(cfg.get('high_impact_only', False))
        self._events: List[NewsEvent] = []
        self._load_events(cfg.get('sources', []))

    def _load_events(self, sources: List[Any]):
        if not sources:
            return
        for src in sources:
            if isinstance(src, dict) and 'local_file' in src:
                path = Path(src['local_file'])
                if path.exists():
                    try:
                        data = yaml.safe_load(path.read_text()) or {}
                        for ev in data.get('events', []) or []:
                            try:
                                t = pd.Timestamp(ev.get('time')).tz_localize(None)
                                self._events.append(NewsEvent(time=t, title=str(ev.get('title','')), impact=str(ev.get('impact','')).upper()))
                            except Exception:
                                continue
                    except Exception:
                        continue
        # sort events
        self._events.sort(key=lambda e: e.time)

    def is_event_now(self, ts_ms: int) -> Tuple[bool, Optional[str]]:
        if not self.enabled or not self._events or ts_ms is None:
            return False, None
        try:
            now = pd.Timestamp(ts_ms, unit='ms')
        except Exception:
            return False, None
        for ev in self._events:
            if self.high_impact_only and ev.impact != 'HIGH':
                continue
            # embargo window
            start = ev.time - pd.Timedelta(minutes=self.embargo_minutes)
            end = ev.time + pd.Timedelta(minutes=self.embargo_minutes)
            if start <= now <= end:
                return True, f"{ev.title} ({ev.impact})"
        return False, None
