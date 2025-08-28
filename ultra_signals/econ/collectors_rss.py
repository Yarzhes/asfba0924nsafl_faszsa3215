"""Generic RSS collectors for macro / regulatory / earnings sources.

Uses httpx (already a dependency) for async fetch with simple cache support.
We intentionally parse minimally to extract title + published date.
"""
from __future__ import annotations

import re
import time
from typing import Iterable, Dict, Optional, List
import httpx
from datetime import datetime, timezone
from loguru import logger
from xml.etree import ElementTree as ET

from ultra_signals.core.custom_types import EconomicEvent, EconEventClass, EconEventStatus, EconSeverity
from .cache import HTTPCache


RSS_SOURCE_MAP = {
    # url: (event_class, severity, region)
    "https://www.federalreserve.gov/feeds/press_all.xml": (EconEventClass.FOMC, EconSeverity.HIGH, "US"),
    "https://www.bls.gov/feed/bls_latest.rss": (EconEventClass.CPI, EconSeverity.HIGH, "US"),
    "https://www.ecb.europa.eu/press/rss/press.html": (EconEventClass.FOMC, EconSeverity.MED, "EU"),  # treat ECB press like medium severity for now
    "https://www.sec.gov/news/pressreleases.rss": (EconEventClass.SEC_ETF, EconSeverity.MED, "US"),
}

DATE_TAGS = ["pubDate", "published", "updated"]


def _parse_rss_datetime(node_text: str) -> Optional[int]:
    try:
        # RFC822 style e.g. Tue, 27 Aug 2025 18:00:00 GMT
        dt = email.utils.parsedate_to_datetime(node_text)  # type: ignore
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        # fallback naive parse
        try:
            dt = datetime.fromisoformat(node_text.replace('Z','+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp()*1000)
        except Exception:
            return None


async def fetch_rss(url: str, cache: HTTPCache, timeout: float = 10.0) -> Optional[str]:
    now_ms = int(time.time()*1000)
    headers = cache.conditional_headers(url, now_ms)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=timeout, headers=headers)
            if resp.status_code == 304:  # not modified
                entry = cache.get(url, now_ms)
                if entry:
                    return entry.get('body')
                return None
            resp.raise_for_status()
            body = resp.text
            cache.put(url, {
                'fetched_at_ms': now_ms,
                'etag': resp.headers.get('ETag'),
                'last_modified': resp.headers.get('Last-Modified'),
                'ttl_ms': 15*60*1000,
                'body': body,
                'status': resp.status_code,
            })
            return body
        except Exception as e:
            logger.error(f"RSS fetch failed {url}: {e}")
            return None


def parse_rss_events(url: str, xml_text: str) -> List[Dict]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.findall('.//item'):
        title_el = item.find('title')
        title = (title_el.text or '').strip() if title_el is not None else 'Untitled'
        # Find date
        ts_ms = None
        for tag in DATE_TAGS:
            d_el = item.find(tag)
            if d_el is not None and d_el.text:
                ts_ms = _parse_rss_datetime(d_el.text.strip())
                if ts_ms:
                    break
        if ts_ms is None:
            ts_ms = int(time.time()*1000)
        link_el = item.find('link')
        link = link_el.text.strip() if link_el is not None and link_el.text else None
        items.append({
            'title': title,
            'ts_start': ts_ms,
            'url': link,
        })
    return items


async def rss_collector(now_ms: int, cache: HTTPCache, enabled_urls: Optional[List[str]] = None):  # async generator style
    for url, meta in RSS_SOURCE_MAP.items():
        if enabled_urls and url not in enabled_urls:
            continue
        body = await fetch_rss(url, cache)
        if not body:
            continue
        try:
            raw_items = parse_rss_events(url, body)
            cls, sev, region = meta
            for r in raw_items[:10]:  # limit to latest 10 per feed to avoid flooding
                ev = EconomicEvent(
                    id="",  # will be set by service
                    source=f"rss:{url.split('/')[2]}",
                    raw_id=r['title'][:120],
                    cls=cls,
                    title=r['title'],
                    region=region,
                    severity=sev,
                    ts_start=r['ts_start'],
                    ts_end=None,
                    status=EconEventStatus.SCHEDULED,
                    notes=None,
                    url=r.get('url'),
                )
                yield ev
        except Exception as e:
            logger.error(f"rss parse error {url}: {e}")

__all__ = ["rss_collector", "parse_rss_events", "fetch_rss", "RSS_SOURCE_MAP"]
