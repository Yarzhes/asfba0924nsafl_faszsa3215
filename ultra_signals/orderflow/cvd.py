from __future__ import annotations
import numpy as np
from typing import Dict, List

class CVDComputer:
    def __init__(self, lookback: int = 200, slope_window: int = 20):
        self.lb = lookback
        self.win = slope_window

    def compute_proxy(self, ohlcv_rows: List[dict]) -> Dict[str, float]:
        """
        ohlcv_rows: list of dicts with keys: open, high, low, close, volume
        Proxy rule: uptick volume when close>prev_close else downtick.
        Returns rolling CVD and slope estimate.
        """
        if len(ohlcv_rows) < 2:
            return {"cvd": 0.0, "cvd_slope": 0.0}

        closes = np.array([r["close"] for r in ohlcv_rows[-self.lb:]], dtype=float)
        vols   = np.array([r["volume"] for r in ohlcv_rows[-self.lb:]], dtype=float)
        signs  = np.sign(np.diff(closes, prepend=closes[0]))
        # Treat zero move as neutral; no volume counted
        upvol  = np.where(signs > 0, vols, 0.0)
        downv  = np.where(signs < 0, vols, 0.0)
        cvd = np.cumsum(upvol - downv)
        # slope via simple linear reg on last window
        w = min(self.win, len(cvd))
        x = np.arange(w)
        y = cvd[-w:]
        denom = (w * (w-1) / 2.0) or 1.0
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / ( (x - x.mean())**2 ).sum()) if w > 2 else 0.0
        return {"cvd": float(cvd[-1]), "cvd_slope": slope}
