"""Multi-venue abstraction layer (Sprint 23).

Modules:
  base.py     : ExchangeVenue protocol & common dataclasses.
  symbols.py  : Internal <-> venue symbol mapping utilities.
  health.py   : Health metrics collection & circuit breaker state.
  binance_usdm.py / bybit_perp.py : Thin async venue adapters (minimal stub impl for now).
  router.py   : VenueRouter selecting venues for data / orders with failover + stickiness.

Design goals:
  * Small surface area (only the methods engine actually needs).
  * Idempotent order flow via client_order_id (hash from plan) preserved end-to-end.
  * Fail closed: if all venues unhealthy, caller can stay FLAT / pause orders.

Initial implementation intentionally lightweight (no real HTTP calls yet) so tests can
exercise routing semantics. Real REST/WS integration can be added incrementally without
breaking the contract established here.
"""

from .base import ExchangeVenue, OrderAck, CancelAck, BookTop, Position, AccountInfo  # noqa: F401
from .router import VenueRouter  # noqa: F401
from .symbols import SymbolMapper  # noqa: F401

__all__ = [
    "ExchangeVenue",
    "OrderAck",
    "CancelAck",
    "BookTop",
    "Position",
    "AccountInfo",
    "VenueRouter",
    "SymbolMapper",
]
