from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Any, Deque

from loguru import logger

class SentimentAggregator:
    """Rolls up scored items into per-symbol windows & z-scores.

    Maintains short (minutes) and medium (hours) horizons plus burst metrics.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        win_cfg = self.cfg.get("windows", {})
        self.short_minutes = int(win_cfg.get("short_minutes", 60))
        self.medium_hours = int(win_cfg.get("medium_hours", 12))
        self.z_lookback = int(win_cfg.get("z_lookback", 240))
        self.symbol_buffers: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.z_lookback*4))
        self.latest_per_symbol: Dict[str, Dict[str, Any]] = {}

    def ingest(self, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        now = int(time.time())
        by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for it in items:
            syms = it.get("symbols") or []
            for s in syms:
                clone = dict(it)
                clone["symbol"] = s
                by_symbol[s].append(clone)
                self.symbol_buffers[s].append(clone)
        out: Dict[str, Dict[str, Any]] = {}
        for sym, rows in by_symbol.items():
            snap = self._compute(sym, now)
            if snap:
                out[sym] = snap
                self.latest_per_symbol[sym] = snap
        return out

    def _compute(self, symbol: str, now: int) -> Dict[str, Any]:
        buf = list(self.symbol_buffers.get(symbol, []))
        if not buf:
            return {}
        short_cutoff = now - self.short_minutes * 60
        medium_cutoff = now - self.medium_hours * 3600
        short_vals = [b.get("polarity", 0.0) for b in buf if b.get("ts", 0) >= short_cutoff]
        med_vals = [b.get("polarity", 0.0) for b in buf if b.get("ts", 0) >= medium_cutoff]
        def _avg(v):
            return sum(v)/len(v) if v else 0.0
        sent_s = _avg(short_vals)
        sent_m = _avg(med_vals)
        # rolling z based on entire buffer
        vals = [b.get("polarity", 0.0) for b in buf[-self.z_lookback:]]
        z = 0.0
        if len(vals) >= 10:
            import statistics as stats
            mu = stats.mean(vals)
            try:
                sd = stats.pstdev(vals)
            except Exception:
                sd = 0.0
            if sd > 1e-9:
                z = (sent_s - mu)/sd
        # burst: posts per minute last N minutes
        short_posts = len(short_vals)
        burst = short_posts / max(1, self.short_minutes)
        # extremes
        cfg = self.cfg.get("extremes", {})
        zthr = float(cfg.get("z_threshold", 2.0))
        extreme_bull = int(z >= zthr and sent_s > 0)
        extreme_bear = int(z <= -zthr and sent_s < 0)
        # Derive supplemental metrics from latest meta (funding, oi z, etc.)
        latest_meta = buf[-1].get("meta", {}) if buf else {}
        funding_z = latest_meta.get("funding_z")
        oi_z = latest_meta.get("oi_z")
        basis_z = latest_meta.get("basis_z")
        fg_index = latest_meta.get("fg_index")
        out = {
            "ts": now,
            "sent_score_s": float(sent_s),
            "sent_score_m": float(sent_m),
            "sent_z_s": float(z),
            "sent_burst_s": float(burst),
            "extreme_flag_bull": extreme_bull,
            "extreme_flag_bear": extreme_bear,
        }
        if funding_z is not None:
            out["funding_z"] = float(funding_z)
        if oi_z is not None:
            out["oi_z"] = float(oi_z)
        if basis_z is not None:
            out["basis_z"] = float(basis_z)
        if fg_index is not None:
            out["fg_index"] = float(fg_index)
        return out
