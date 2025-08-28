import time
from typing import Dict


class HealthMonitor:
    def __init__(self, stale_threshold_s: float = 2.0):
        self._last_ts: Dict[str, float] = {}
        self.stale_threshold_s = stale_threshold_s

    def heartbeat(self, venue: str):
        self._last_ts[venue] = time.time()

    def is_healthy(self, venue: str) -> bool:
        ts = self._last_ts.get(venue)
        if ts is None:
            return False
        return (time.time() - ts) <= self.stale_threshold_s

    def unhealthy_venues(self):
        now = time.time()
        return [v for v, t in self._last_ts.items() if (now - t) > self.stale_threshold_s]
