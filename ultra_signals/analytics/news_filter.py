from __future__ import annotations
import json, os
from typing import List, Dict, Optional

class NewsFilter:
    def __init__(self, cfg: dict):
        n = cfg.get("news_filter", {})
        self.enabled = bool(n.get("enabled", True))
        self.path = n.get("file", "data/calendar/events.json")
        self.window = int(n.get("window_minutes", 30))
        self.min_sev = int(n.get("min_severity", 2))
        self._cache: Optional[List[Dict]] = None

    def _load(self) -> List[Dict]:
        if self._cache is not None: return self._cache
        if not os.path.exists(self.path): 
            self._cache = []
            return self._cache
        with open(self.path, "r") as f:
            self._cache = json.load(f)
        return self._cache

    def is_blocked(self, now_ms: int) -> bool:
        if not self.enabled: return False
        events = self._load()
        win_ms = self.window * 60 * 1000
        for ev in events:
            sev = int(ev.get("severity", 1))
            ts  = int(ev.get("ts"))  # unix ms
            if sev >= self.min_sev and abs(now_ms - ts) <= win_ms:
                return True
        return False
