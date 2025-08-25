from __future__ import annotations
import csv, os, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

@dataclass
class FundingPoint:
    ts: int     # unix ms
    rate: float # e.g. 0.0001 = 0.01%

class FundingProvider:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = cfg.get("alpha", {}).get("funding", {}).get("dir", "data/funding")
        self.fallback = float(cfg.get("alpha", {}).get("funding", {}).get("fallback_rate", 0.0001))
        self.source = cfg.get("alpha", {}).get("funding", {}).get("source", "csv")

    def load_trail(self, symbol: str, start_ms: int, end_ms: int) -> List[FundingPoint]:
        if self.source == "csv":
            path = os.path.join(self.dir, f"{symbol}_funding.csv")
            if not os.path.exists(path):
                return []
            out: List[FundingPoint] = []
            with open(path, "r", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    ts = int(row["ts"])
                    if start_ms <= ts <= end_ms:
                        out.append(FundingPoint(ts=ts, rate=float(row["rate"])))
            return sorted(out, key=lambda x: x.ts)
        elif self.source == "synthetic":
            # every 8h, constant rate
            out = []
            step = 8 * 60 * 60 * 1000
            t = start_ms - (start_ms % step) + step
            while t <= end_ms:
                out.append(FundingPoint(ts=t, rate=self.fallback))
                t += step
            return out
        else:
            # TODO: exchange REST (later)
            return []

    def minutes_to_next(self, symbol: str, now_ms: int) -> Optional[int]:
        # naive: find next funding point >= now
        trail = self.load_trail(symbol, now_ms, now_ms + 24*60*60*1000)
        if not trail:
            return None
        nxt = min(trail, key=lambda p: p.ts)
        return int((nxt.ts - now_ms) / 60000)
