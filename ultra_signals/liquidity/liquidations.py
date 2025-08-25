from __future__ import annotations
import csv, os
import numpy as np
from typing import List, Dict

class LiqPulse:
    def __init__(self, cfg: dict):
        a = cfg.get("alpha", {}).get("liquidations", {})
        self.dir = a.get("dir", "data/liquidations")
        self.win = int(a.get("spike_window", 50))
        self.z   = float(a.get("spike_zscore", 3.0))
        self.source = a.get("source", "csv")

    def load_series(self, symbol: str) -> List[Dict]:
        path = os.path.join(self.dir, f"{symbol}_liqs.csv")
        if not os.path.exists(path): return []
        out = []
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                out.append({"ts": int(row["ts"]), "amt": float(row["amt"])})
        return out

    def latest_spike_flag(self, symbol: str, now_ms: int) -> Dict[str, float]:
        series = self.load_series(symbol)
        if not series: return {"liq_spike": 0.0, "liq_z": 0.0}
        # Keep only last window around now
        vals = np.array([s["amt"] for s in series[-self.win:]], dtype=float)
        if len(vals) < 5:
            return {"liq_spike": 0.0, "liq_z": 0.0}
        mu, sd = float(vals.mean()), float(vals.std() or 1.0)
        z_last = (vals[-1] - mu) / sd
        return {"liq_spike": 1.0 if z_last >= self.z else 0.0, "liq_z": z_last}
