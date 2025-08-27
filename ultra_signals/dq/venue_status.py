"""Venue status polling & caching (Sprint 39)."""
from __future__ import annotations
import time
from typing import Callable, Dict, Optional
from loguru import logger

_STATUS_FN: Optional[Callable[[str], dict]] = None
_CACHE: Dict[str, dict] = {}
_LAST: Dict[str, float] = {}
_LOCK_TTL = 30


def configure(get_status_snapshot: Callable[[str], dict]):
    global _STATUS_FN
    _STATUS_FN = get_status_snapshot


def get_venue_status(venue: str, force: bool = False) -> str:
    if _STATUS_FN is None:
        return "operational"  # default optimistic
    now = time.time()
    ttl = 30
    if force or venue not in _CACHE or now - _LAST.get(venue, 0) > ttl:
        try:
            snap = _STATUS_FN(venue) or {}
            _CACHE[venue] = snap
            _LAST[venue] = now
        except Exception as e:  # pragma: no cover
            logger.warning(f"venue_status.poll_failed venue={venue} err={e}")
            return _CACHE.get(venue, {}).get('status', 'operational')
    return _CACHE.get(venue, {}).get('status', 'operational')

__all__ = ['configure','get_venue_status']
