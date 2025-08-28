"""Simplified earnings collector.
Accepts static config of earnings dates for COIN / MSTR (placeholder until web scraping/RSS added).
Config example:
 econ: {
    earnings: [ { 'symbol':'COIN','ts_start': 1699999999000, 'severity':'med'} ]
 }
Maps to EconEventClass.EARNINGS_COIN or EARNINGS_MSTR.
"""
from __future__ import annotations
from typing import Iterable, Dict, List
import urllib.request, time, re, os
_LM_CACHE_RSS = {}
from loguru import logger
from ultra_signals.core.custom_types import EconomicEvent, EconEventClass, EconSeverity, EconEventStatus
from .service import _stable_event_id

MAP = {
    'COIN': EconEventClass.EARNINGS_COIN,
    'MSTR': EconEventClass.EARNINGS_MSTR,
}

def earnings_static_collector_factory(items: List[Dict]):
    def _collector(now_ms: int) -> Iterable[EconomicEvent]:
        for it in items:
            try:
                sym = str(it.get('symbol','COIN')).upper()
                cls = MAP.get(sym, EconEventClass.EARNINGS_OTHER)
                sev_raw = it.get('severity','med')
                sev = EconSeverity(sev_raw) if sev_raw in EconSeverity._value2member_map_ else EconSeverity.MED
                ts_start = int(it['ts_start'])
                title = it.get('title') or f"{sym} Earnings"
                ev = EconomicEvent(
                    id=it.get('id') or _stable_event_id('earnings', sym, ts_start, title),
                    source='earnings',
                    raw_id=sym,
                    cls=cls,
                    title=title,
                    severity=sev,
                    ts_start=ts_start,
                    ts_end=it.get('ts_end'),
                    status=EconEventStatus.SCHEDULED,
                    risk_pre_min=it.get('risk_pre_min',30),
                    risk_post_min=it.get('risk_post_min',60),
                    notes=it.get('notes')
                )
                yield ev
            except Exception:
                continue
    return _collector


def earnings_rss_collector_factory(sources: List[Dict]):
    """Parse simple RSS feeds for earnings announcements (title/date mapping heuristic).
    Heuristic: look for lines containing a symbol (e.g., COIN) and a future datetime.
    Config item: { 'url': 'https://example/rss', 'symbol':'COIN', 'regex_datetime': '(\\d{4}-\\d{2}-\\d{2})' }
    """
    dt_regex_default = re.compile(r"(\d{4}-\d{2}-\d{2})")
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
            sym = str(src.get('symbol','COIN')).upper()
            cls = MAP.get(sym, EconEventClass.EARNINGS_OTHER)
            if not url:
                continue
            try:
                headers = {}
                if http_cache:
                    headers.update(http_cache.conditional_headers(url))
                req = urllib.request.Request(url, headers=headers)
                lm = _LM_CACHE_RSS.get(url)
                if lm and 'If-Modified-Since' not in req.headers:
                    req.add_header('If-Modified-Since', lm)
                try:
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        if resp.status == 304:
                            if http_cache:
                                entry = http_cache.get(url)
                                if entry and entry.get('body'):
                                    txt = entry['body']
                                else:
                                    continue
                            else:
                                continue
                        else:
                            txt = resp.read().decode('utf-8','ignore')
                            lm_new = resp.headers.get('Last-Modified')
                            etag = resp.headers.get('ETag')
                            if lm_new:
                                _LM_CACHE_RSS[url] = lm_new
                            if http_cache:
                                http_cache.put(url, {
                                    'fetched_at_ms': int(time.time()*1000),
                                    'last_modified': lm_new,
                                    'etag': etag,
                                    'status': resp.status,
                                    'body': txt,
                                })
                except urllib.error.HTTPError as he:
                    if he.code == 304 and http_cache:
                        entry = http_cache.get(url)
                        if not entry:
                            continue
                        txt = entry.get('body','')
                    else:
                        raise
                # naive parse: split items
                items = txt.split('<item')
                r_dt = re.compile(src.get('regex_datetime')) if src.get('regex_datetime') else dt_regex_default
                for it in items[1:]:
                    if sym not in it:
                        continue
                    m = r_dt.search(it)
                    if not m:
                        continue
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(m.group(1))
                        ts_start = int(dt.timestamp()*1000)
                        if ts_start < now_ms:
                            continue
                        title_match = re.search(r"<title>(.*?)</title>", it, re.IGNORECASE|re.DOTALL)
                        title = title_match.group(1).strip() if title_match else f"{sym} Earnings"
                        sev = EconSeverity.MED
                        ev = EconomicEvent(
                            id=_stable_event_id('earnings_rss', sym, ts_start, title),
                            source='earnings_rss',
                            raw_id=sym,
                            cls=cls,
                            title=title,
                            severity=sev,
                            ts_start=ts_start,
                            ts_end=ts_start + 60*60*1000,
                            status=EconEventStatus.SCHEDULED,
                            risk_pre_min=src.get('risk_pre_min',30),
                            risk_post_min=src.get('risk_post_min',60)
                        )
                        yield ev
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"earnings_rss source {url} error: {e}")
    return _collector
