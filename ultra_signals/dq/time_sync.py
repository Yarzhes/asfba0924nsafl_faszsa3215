"""Time synchronization utilities (Sprint 39).

Provides clock skew estimation vs venue server time.
The actual network call for server time must be supplied by caller via
an injected "get_server_time_ms(venue)" function (dependency inversion).
"""
from __future__ import annotations
import time
import threading
from typing import Callable, Dict, Optional
from loguru import logger

# Simple EMA helper
class _EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.v: Optional[float] = None
    def update(self, x: float) -> float:
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v
    def value(self) -> Optional[float]:
        return self.v

# Global registry (module-level – lightweight)
_LOCK = threading.RLock()
_SKEW_EMA: Dict[str, _EMA] = {}
_LAST_POLL: Dict[str, float] = {}
_SERVER_TIME_FN: Optional[Callable[[str], int]] = None

class CircuitBreak(Exception):
    pass

def configure(get_server_time_ms: Callable[[str], int]):
    """Inject the server time retrieval implementation."""
    global _SERVER_TIME_FN
    _SERVER_TIME_FN = get_server_time_ms

_DEF_ALPHA = 0.2


def _monotonic_ms() -> int:
    return int(time.monotonic() * 1000)

def _wall_clock_ms() -> int:
    return int(time.time() * 1000)

def poll_venue(venue: str, settings: dict):  # pragma: no cover (network timing nondeterministic)
    if _SERVER_TIME_FN is None:
        raise RuntimeError("time_sync not configured with get_server_time_ms")
    try:
        server_ms = int(_SERVER_TIME_FN(venue))
    except Exception as e:  # noqa
        logger.warning(f"time_sync.poll_failed venue={venue} err={e}")
        return
    local_wall = _wall_clock_ms()
    # skew = server - local
    skew = server_ms - local_wall
    with _LOCK:
        ema = _SKEW_EMA.get(venue)
        if ema is None:
            ema = _EMA(_DEF_ALPHA)
            _SKEW_EMA[venue] = ema
        val = ema.update(skew)
        _LAST_POLL[venue] = time.time()
    logger.debug(f"time_sync.poll venue={venue} server={server_ms} local={local_wall} skew={skew} ema={val}")


def get_skew_ms(venue: str) -> float:
    with _LOCK:
        ema = _SKEW_EMA.get(venue)
        return float(ema.value()) if ema and ema.value() is not None else 0.0


def assert_within_skew(settings: dict, venues: Optional[list] = None):
    dq = (settings or {}).get("data_quality", {})
    if not dq.get("enabled", False):
        return
    threshold = float(dq.get("max_clock_skew_ms", 250))
    targets = venues or dq.get("multi_venue", {}).get("primary", []) or []
    for v in targets:
        skew = abs(get_skew_ms(v))
        if skew > threshold:
            logger.error(f"dq.skew_exceeded venue={v} skew_ms={skew} threshold={threshold}")
            raise CircuitBreak(f"Clock skew {skew}ms exceeds {threshold}ms for {v}")


def now_utc_synced(prefer_venue: Optional[str] = None) -> int:
    base = _wall_clock_ms()
    if prefer_venue:
        return int(base + get_skew_ms(prefer_venue))
    return base

__all__ = [
    "configure",
    "poll_venue",
    "get_skew_ms",
    "assert_within_skew",
    "now_utc_synced",
    "CircuitBreak",
]
