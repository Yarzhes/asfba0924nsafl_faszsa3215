"""Sprint 46 Economic / Event Engine package.

Modules:
    collectors: Individual source collectors (RSS/ICS/HTML/status JSON/local drop).
    normalize: Canonical normalization & dedupe helpers.
    risk: Risk window computation & policy (pre/post windows, size/veto logic).
    service: High-level orchestrator to refresh sources on cadence & expose
             current event snapshots + feature extraction helper.

Design Goals:
    * Offline tolerant (local ICS/CSV drop folder optional)
    * Polite rate limiting & basic caching (TTL aware, pass in now_ms)
    * Minimal mandatory deps (reuse stdlib + existing httpx/pyyaml)
"""

from .service import EconEventService  # convenience import

__all__ = ["EconEventService"]
