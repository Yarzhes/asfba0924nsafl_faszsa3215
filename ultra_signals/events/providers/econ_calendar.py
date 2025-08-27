"""Economic calendar provider (stub).

Intended future implementation ideas:
 - Pull CSV/JSON from a macro calendar API (e.g. ForexFactory, Investing.com,
   EconDB, or custom paid feed) – ensure licensing compliance.
 - Convert local time zone stamps to UTC epoch ms.

For now returns an empty list so the system can run in observe-only mode.
"""
from __future__ import annotations
from typing import List
from .base import EventProvider, RawEvent
from loguru import logger
import os, yaml, hashlib, datetime as dt


class EconCalendarProvider(EventProvider):
    provider_name = "econ_calendar"

    def __init__(self, config: dict | None = None):  # config placeholder
        self.config = config or {}

    def fetch_upcoming(self, from_ts: int, to_ts: int) -> List[RawEvent]:  # pragma: no cover - trivial
        try:
            # TODO: Implement real fetch logic.
            out: List[RawEvent] = []
            path = self.config.get('local_file') or 'news_events.yaml'
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f) or {}
                    for ev in data.get('events', []):
                        t = ev.get('time') or ev.get('datetime')
                        if not t:
                            continue
                        try:
                            ts = int(dt.datetime.fromisoformat(t.replace('Z','+00:00')).timestamp()*1000)
                        except Exception:
                            continue
                        if ts > to_ts or ts < from_ts:
                            continue
                        title = str(ev.get('title') or ev.get('name') or 'Event')
                        imp = ev.get('impact') or ev.get('importance') or 'MED'
                        # build ID stable by provider+title+ts
                        hid = hashlib.sha256(f"econ|{title}|{ts}".encode()).hexdigest()[:16]
                        out.append(RawEvent(
                            provider=self.provider_name,
                            id=hid,
                            name=title,
                            start_ts=ts,
                            end_ts=ts,
                            importance={'LOW':1,'MED':2,'HIGH':3}.get(str(imp).upper(),2),
                            country=ev.get('country'),
                            payload=ev
                        ))
                except Exception as e:
                    logger.warning("[events] econ_calendar local parse error: {}", e)
            return out
        except Exception as e:  # safety
            logger.warning("[events] econ_calendar fetch error: {}", e)
            return []


__all__ = ["EconCalendarProvider"]
