"""Crypto incidents provider (stub).

Future scope:
 - Exchange maint / upgrade windows
 - Symbol listings / delistings
 - Protocol forks / airdrops / unlocks
 - ETF flows / large on-chain events

Current version returns empty list so gating infra can be validated independently.
"""
from __future__ import annotations
from typing import List
from .base import EventProvider, RawEvent
from loguru import logger


class CryptoIncidentsProvider(EventProvider):
    provider_name = "crypto_incidents"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def fetch_upcoming(self, from_ts: int, to_ts: int) -> List[RawEvent]:  # pragma: no cover - trivial
        try:
            return []
        except Exception as e:
            logger.warning("[events] crypto_incidents fetch error: {}", e)
            return []


__all__ = ["CryptoIncidentsProvider"]
