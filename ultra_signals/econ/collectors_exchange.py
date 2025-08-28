"""Exchange status / maintenance collector.
Fetches simple JSON endpoints or static config mapping upcoming maint windows to EconomicEvent objects.
Config example:
 econ: {
   exchange_maint: [ { 'exchange':'binance','title':'Binance Spot Maint','ts_start':..,'ts_end':..,'severity':'high'} ]
 }
Future: implement HTTP fetch to status API; current version consumes static list for determinism.
"""
from __future__ import annotations
from typing import Iterable, Dict, List
import json, urllib.request, time, os
_LM_CACHE = {}
from loguru import logger
from ultra_signals.core.custom_types import EconomicEvent, EconEventClass, EconSeverity, EconEventStatus
from .service import _stable_event_id

def exchange_static_collector_factory(events: List[Dict]):
    def _collector(now_ms: int) -> Iterable[EconomicEvent]:
        for raw in events:
            try:
                sev_raw = raw.get('severity','high')
                sev = EconSeverity(sev_raw) if sev_raw in EconSeverity._value2member_map_ else EconSeverity.HIGH
                title = raw.get('title', 'Exchange Maintenance')
                ts_start = int(raw['ts_start'])
                ts_end = int(raw.get('ts_end') or ts_start + 30*60*1000)
                ev = EconomicEvent(
                    id=raw.get('id') or _stable_event_id('exchange', raw.get('raw_id'), ts_start, title),
                    source='exchange',
                    raw_id=raw.get('raw_id'),
                    cls=EconEventClass.EXCHANGE_MAINT,
                    title=title,
                    severity=sev,
                    ts_start=ts_start,
                    ts_end=ts_end,
                    status=EconEventStatus.SCHEDULED,
                    risk_pre_min=raw.get('risk_pre_min', 0),
                    risk_post_min=raw.get('risk_post_min', 30),
                    notes=raw.get('notes'),
                    symbols=raw.get('symbols')
                )
                yield ev
            except Exception:
                continue
    return _collector


def exchange_http_collector_factory(sources: List[Dict]):
    """Fetch maintenance windows from simple JSON endpoints.
    Expected JSON schema per endpoint (flexible):
      {
        "maintenances": [ { "title": "..", "start": 169.. (ms) , "end":169..(ms), "severity":"high" } ]
      }
    Config item example: { 'url': 'https://example/api/status', 'exchange':'binance'}
    """
    # Optional shared disk cache path
    cache_path = None
    for s in sources:
        if s.get('cache_path'):
            cache_path = s['cache_path']
            break
    http_cache = None
    if cache_path:
        try:
            from .cache import HTTPCache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            http_cache = HTTPCache(cache_path)
        except Exception:
            http_cache = None

    def _collector(now_ms: int) -> Iterable[EconomicEvent]:
        for src in sources:
            url = src.get('url')
            if not url:
                continue
            try:
                headers = {}
                if http_cache:
                    headers.update(http_cache.conditional_headers(url))
                req = urllib.request.Request(url, headers=headers)
                lm = _LM_CACHE.get(url)
                if lm and 'If-Modified-Since' not in req.headers:
                    req.add_header('If-Modified-Since', lm)
                try:
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        if resp.status == 304:
                            if http_cache:
                                entry = http_cache.get(url)
                                if entry and entry.get('body'):
                                    raw = entry['body']
                                else:
                                    continue
                            else:
                                continue
                        else:
                            raw = resp.read().decode('utf-8','ignore')
                            lm_new = resp.headers.get('Last-Modified')
                            etag = resp.headers.get('ETag')
                            if lm_new:
                                _LM_CACHE[url] = lm_new
                            if http_cache:
                                http_cache.put(url, {
                                    'fetched_at_ms': int(time.time()*1000),
                                    'last_modified': lm_new,
                                    'etag': etag,
                                    'status': resp.status,
                                    'body': raw,
                                })
                except urllib.error.HTTPError as he:
                    if he.code == 304 and http_cache:
                        entry = http_cache.get(url)
                        if not entry:
                            continue
                        raw = entry.get('body','')
                    else:
                        raise
                data = json.loads(raw)
                maints = data.get('maintenances') or data.get('maintenance') or []
                for m in maints:
                    try:
                        ts_start = int(m.get('start') or m.get('ts_start'))
                        ts_end = int(m.get('end') or m.get('ts_end') or (ts_start + 30*60*1000))
                        title = m.get('title') or f"{src.get('exchange','exchange')} Maintenance"
                        sev_raw = m.get('severity','high')
                        sev = EconSeverity(sev_raw) if sev_raw in EconSeverity._value2member_map_ else EconSeverity.HIGH
                        ev = EconomicEvent(
                            id=_stable_event_id('exchange_http', src.get('exchange'), ts_start, title),
                            source='exchange_http',
                            raw_id=str(m.get('id') or ts_start),
                            cls=EconEventClass.EXCHANGE_MAINT,
                            title=title,
                            severity=sev,
                            ts_start=ts_start,
                            ts_end=ts_end,
                            status=EconEventStatus.SCHEDULED,
                            risk_pre_min=src.get('risk_pre_min', 0),
                            risk_post_min=src.get('risk_post_min', 30),
                            notes=src.get('exchange')
                        )
                        yield ev
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"exchange_http source {url} error: {e}")
    return _collector
