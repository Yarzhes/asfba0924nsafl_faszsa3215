"""Provider base interfaces for events subsystem.

Providers fetch upcoming raw events in a (from_ts, to_ts) epoch ms range and
return lightweight dictionaries. Classification into internal categories &
severity is deferred to `classifier`.
"""
from __future__ import annotations
from typing import Protocol, List, TypedDict, Optional


class RawEvent(TypedDict, total=False):
    provider: str
    id: str            # provider-unique identifier
    name: str
    start_ts: int      # epoch ms UTC
    end_ts: int        # epoch ms UTC (may equal start if point-in-time)
    importance: Optional[int]
    country: Optional[str]
    symbol: Optional[str]
    payload: dict      # original provider payload for audit


class EventProvider(Protocol):
    provider_name: str

    def fetch_upcoming(self, from_ts: int, to_ts: int) -> List[RawEvent]:
        """Fetch upcoming events in [from_ts, to_ts].

        Must return events with UTC epoch ms timestamps. Implementations should
        be resilient: swallow network errors and return an empty list so the
        gating layer can apply missing_feed_policy.
        """
        ...  # pragma: no cover – interface

