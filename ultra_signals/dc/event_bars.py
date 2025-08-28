from typing import List, Dict, Any


class EventBarBuilder:
    """Aggregate last N events into an event-bar for feature computation."""

    def __init__(self, n_events: int = 5):
        self.n = n_events
        self.buffer: List[Dict[str, Any]] = []

    def add_event(self, ev: Dict[str, Any]):
        self.buffer.append(ev)
        if len(self.buffer) > self.n:
            self.buffer.pop(0)

    def get_bar(self) -> Dict[str, Any]:
        # simple aggregation — counts, last directions, os stats
        if not self.buffer:
            return {}
        types = [e.get("type") for e in self.buffer]
        count = len(self.buffer)
        os_ranges = [e.get("os_range") for e in self.buffer if e.get("os_range") is not None]
        return {
            "count": count,
            "types": types,
            "os_mean": (sum(os_ranges) / len(os_ranges)) if os_ranges else None,
            "os_max": max(os_ranges) if os_ranges else None,
        }
