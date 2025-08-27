"""Symbol mapping utilities.

Config supplies per-venue symbols; we expose a small helper to unify access &
support round-trip tests.
"""
from __future__ import annotations
from typing import Dict


class SymbolMapper:
    def __init__(self, mapping: Dict[str, Dict[str, str]] | None = None):
        self._map = mapping or {}

    def to_venue(self, internal: str, venue_id: str) -> str:
        return (self._map.get(internal, {}) or {}).get(venue_id, internal)

    def from_venue(self, venue_symbol: str, venue_id: str) -> str:
        # naive reverse search (small map) – optimize if needed
        for internal, venues in self._map.items():
            if venues.get(venue_id) == venue_symbol:
                return internal
        return venue_symbol

    def round_trip_ok(self, symbol: str, venue_id: str) -> bool:
        return self.from_venue(self.to_venue(symbol, venue_id), venue_id) == symbol

__all__ = ["SymbolMapper"]
