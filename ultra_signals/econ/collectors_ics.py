"""ICS / iCal based calendar collector (holidays, central bank meetings).

Lightweight manual parser sufficient for VEVENT blocks with DTSTART / DTEND lines.
Avoid external dependency to keep footprint small. Supports config:
  econ: {
     ics_sources: [ { 'url': 'path_or_url', 'class': 'holiday', 'severity':'low' } ]
  }
If url starts with 'http', we attempt requests via urllib; otherwise treat as local file.
Cache handled by outer EconEventService refresh cadence.
"""
from __future__ import annotations
import re, time, os, urllib.request
from typing import Iterable, Dict, List
from loguru import logger
from ultra_signals.core.custom_types import EconomicEvent, EconEventClass, EconSeverity, EconEventStatus
from .service import _stable_event_id

DT_RE = re.compile(r"^DTSTART(?::|;VALUE=DATE:)([0-9TzZ]+)")
DTEND_RE = re.compile(r"^DTEND(?::|;VALUE=DATE:)([0-9TzZ]+)")
SUMMARY_RE = re.compile(r"^SUMMARY:(.+)")
UID_RE = re.compile(r"^UID:(.+)")

# Simple datetime parse supporting YYYYMMDD or YYYYMMDDT HH formats
from datetime import datetime

def _parse_dt(val: str) -> int:
    val = val.strip()
    # date only
    if len(val) == 8 and val.isdigit():
        dt = datetime.strptime(val, "%Y%m%d")
        return int(dt.timestamp()*1000)
    # basic form yyyymmddThhmmssZ
    try:
        if 'T' in val:
            # trim Z
            if val.endswith('Z'):
                val = val[:-1]
            # take first 14 digits at least
            core = val[:15]
            fmt = "%Y%m%dT%H%M%S" if len(core)==15 else "%Y%m%dT%H%M"
            dt = datetime.strptime(core, fmt)
            return int(dt.timestamp()*1000)
    except Exception:
        pass
    # fallback now
    return int(time.time()*1000)

def ics_collector_factory(sources: List[Dict]):
    def _collector(now_ms: int) -> Iterable[EconomicEvent]:
        for src in sources:
            url = src.get('url')
            cls_raw = src.get('class','holiday')
            cls = EconEventClass(cls_raw) if cls_raw in EconEventClass._value2member_map_ else EconEventClass.HOLIDAY
            sev_raw = src.get('severity','low')
            sev = EconSeverity(sev_raw) if sev_raw in EconSeverity._value2member_map_ else EconSeverity.LOW
            try:
                if not url:
                    continue
                if url.startswith('http'):
                    with urllib.request.urlopen(url, timeout=10) as resp:
                        data = resp.read().decode('utf-8','ignore')
                else:
                    if not os.path.exists(url):
                        continue
                    data = open(url,'r',encoding='utf-8').read()
                blocks = data.split('BEGIN:VEVENT')
                for b in blocks[1:]:
                    lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
                    dtstart = None; dtend=None; summary=None; uid=None
                    for ln in lines:
                        m = DT_RE.match(ln)
                        if m: dtstart=_parse_dt(m.group(1)); continue
                        m = DTEND_RE.match(ln)
                        if m: dtend=_parse_dt(m.group(1)); continue
                        m = SUMMARY_RE.match(ln)
                        if m: summary=m.group(1); continue
                        m = UID_RE.match(ln)
                        if m: uid=m.group(1); continue
                    if not dtstart or not summary:
                        continue
                    ev = EconomicEvent(
                        id=_stable_event_id('ics', uid, dtstart, summary),
                        source='ics',
                        raw_id=uid,
                        cls=cls,
                        title=summary,
                        severity=sev,
                        ts_start=dtstart,
                        ts_end=dtend,
                        status=EconEventStatus.SCHEDULED,
                        risk_pre_min=src.get('risk_pre_min'),
                        risk_post_min=src.get('risk_post_min'),
                        notes=src.get('notes')
                    )
                    yield ev
            except Exception as e:
                logger.debug(f"ICS collector source {url} error: {e}")
    return _collector
