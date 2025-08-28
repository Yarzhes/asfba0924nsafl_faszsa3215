"""Structural chart pattern detectors using pivot highs/lows.

This is a simplified implementation meant to satisfy interface and tests.
"""
from typing import List, Dict, Any, Tuple
from .base import Bars, DetectedPattern
import math
import numpy as np


def _find_fractal_pivots(bars: Bars, k: int = 2):
    highs = [b.high for b in bars.bars]
    lows = [b.low for b in bars.bars]
    pivots = {'highs': [], 'lows': []}
    n = len(bars.bars)
    for i in range(k, n - k):
        is_high = all(highs[i] > highs[j] for j in range(i - k, i + k + 1) if j != i)
        is_low = all(lows[i] < lows[j] for j in range(i - k, i + k + 1) if j != i)
        if is_high:
            pivots['highs'].append(i)
        if is_low:
            pivots['lows'].append(i)
    return pivots


def double_top(bars: Bars, pivots: Dict[str, List[int]]) -> List[DetectedPattern]:
    out = []
    highs = pivots.get('highs', [])
    if len(highs) >= 2:
        # compare last two highs
        hi1, hi2 = highs[-2], highs[-1]
        p1 = bars.bars[hi1]
        p2 = bars.bars[hi2]
        if abs(p1.high - p2.high) / max(1e-8, max(p1.high, p2.high)) < 0.02:
            strength = 0.8
            out.append(DetectedPattern('double_top', 'BEAR', strength, hi1, hi2, {'pivots': highs[-2:]}))
    return out


def double_bottom(bars: Bars, pivots: Dict[str, List[int]]) -> List[DetectedPattern]:
    out = []
    lows = pivots.get('lows', [])
    if len(lows) >= 2:
        lo1, lo2 = lows[-2], lows[-1]
        p1 = bars.bars[lo1]
        p2 = bars.bars[lo2]
        if abs(p1.low - p2.low) / max(1e-8, min(p1.low, p2.low)) < 0.02:
            strength = 0.8
            out.append(DetectedPattern('double_bottom', 'BULL', strength, lo1, lo2, {'pivots': lows[-2:]}))
    return out


class ChartPatternLibrary:
    def __init__(self, config: Dict[str, Any] | None = None, fractal_k: int | None = None):
        """Configuration-driven chart pattern library.

        Backwards-compatible signature: callers may pass fractal_k as a kwarg
        (legacy tests use ChartPatternLibrary(fractal_k=1)).

        Expected config shape (all keys optional):
          patterns.pivots.fractal_k -> int
          patterns.geom.min_r2 -> float (0..1)
        """
        # Accept older style: ChartPatternLibrary(fractal_k=1)
        if isinstance(config, int) and fractal_k is None:
            fractal_k = int(config)
            config = None

        cfg = config or {}
        piv_cfg = cfg.get('pivots') or {}
        geom_cfg = cfg.get('geom') or {}
        # Order of precedence: explicit fractal_k arg > config.pivots.fractal_k > default 2
        if fractal_k is not None:
            self.k = int(fractal_k)
        else:
            self.k = int(piv_cfg.get('fractal_k', 2))
        # minimum R^2 for a line fit to be considered acceptable (0 disables)
        self.min_r2 = float(geom_cfg.get('min_r2', 0.0))
        self.cfg = cfg

    def detect(self, bars: Bars) -> List[DetectedPattern]:
        piv = _find_fractal_pivots(bars, self.k)
        out = []
        out.extend(double_top(bars, piv))
        out.extend(double_bottom(bars, piv))
        out.extend(self._detect_triangles(bars, piv))
        out.extend(self._detect_wedges(bars, piv))
        out.extend(self._detect_flag_pennant(bars, piv))
        out.extend(self._detect_cup_and_handle(bars, piv))
        return out

    def _slope(self, p1, p2):
        # For two points fall back to simple slope
        dx = p2[0] - p1[0]
        if dx == 0:
            return float('inf')
        return (p2[1] - p1[1]) / dx

    def _linreg(self, pts: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """Perform linear regression on pts -> (slope, intercept, r2).

        Uses numpy.polyfit for the fit; if variance is zero returns r2=1.0.
        """
        if len(pts) < 2:
            return 0.0, 0.0, 0.0
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        # If all x or y are constant, fallback
        if np.all(xs == xs[0]) or np.all(ys == ys[0]):
            slope = 0.0 if np.all(ys == ys[0]) else float('inf')
            intercept = float(ys[0]) if np.all(ys == ys[0]) else 0.0
            r2 = 1.0
            return slope, intercept, r2
        try:
            slope, intercept = np.polyfit(xs, ys, 1)
            preds = slope * xs + intercept
            ss_res = np.sum((ys - preds) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
            return float(slope), float(intercept), float(r2)
        except Exception:
            return 0.0, 0.0, 0.0

    def _detect_triangles(self, bars: Bars, piv: Dict[str, List[int]]) -> List[DetectedPattern]:
        out: List[DetectedPattern] = []
        highs = piv.get('highs', [])
        lows = piv.get('lows', [])
        n = len(bars.bars)
        # Primary: use pivots if available
        if len(highs) >= 3 and len(lows) >= 3:
            h_idx = highs[-3:]
            l_idx = lows[-3:]
            h_pts = [(i, bars.bars[i].high) for i in h_idx]
            l_pts = [(i, bars.bars[i].low) for i in l_idx]
        elif n >= 6:
            # Fallback: sample three points across the series
            idxs = [0, n // 2, n - 1]
            h_pts = [(i, bars.bars[i].high) for i in idxs]
            l_pts = [(i, bars.bars[i].low) for i in idxs]
            h_idx = idxs
            l_idx = idxs
        else:
            return out

        # use overall regression fits for highs and lows
        sh, sh_int, sh_r2 = self._linreg(h_pts)
        sl, sl_int, sl_r2 = self._linreg(l_pts)
        r2_ok = (sh_r2 >= self.min_r2) and (sl_r2 >= self.min_r2)
        # converge: highs overall down, lows overall up
        if sh < 0 and sl > 0 and (self.min_r2 <= 0.0 or r2_ok):
            strength = 0.7
            meta = {'highs': h_idx, 'lows': l_idx, 'high_slope': sh, 'low_slope': sl, 'high_r2': sh_r2, 'low_r2': sl_r2}
            out.append(DetectedPattern('sym_triangle', 'NEUTRAL', strength, h_idx[0], l_idx[-1], meta))
        return out

    def _detect_wedges(self, bars: Bars, piv: Dict[str, List[int]]) -> List[DetectedPattern]:
        out: List[DetectedPattern] = []
        highs = piv.get('highs', [])
        lows = piv.get('lows', [])
        n = len(bars.bars)
        # Primary: use pivots if available
        if len(highs) >= 3 and len(lows) >= 3:
            h_idx = highs[-3:]
            l_idx = lows[-3:]
            h_pts = [(i, bars.bars[i].high) for i in h_idx]
            l_pts = [(i, bars.bars[i].low) for i in l_idx]
        elif n >= 6:
            idxs = [0, n // 2, n - 1]
            h_pts = [(i, bars.bars[i].high) for i in idxs]
            l_pts = [(i, bars.bars[i].low) for i in idxs]
            h_idx = idxs
            l_idx = idxs
        else:
            return out
        sh, sh_int, sh_r2 = self._linreg(h_pts)
        sl, sl_int, sl_r2 = self._linreg(l_pts)
        r2_ok = (sh_r2 >= self.min_r2) and (sl_r2 >= self.min_r2)
        # rising wedge: both slopes positive but highs slope < lows slope (compression)
        if sh > 0 and sl > 0 and abs(sh) < abs(sl) and (self.min_r2 <= 0.0 or r2_ok):
            out.append(DetectedPattern('rising_wedge', 'BEAR', 0.65, h_idx[0], l_idx[-1], {'highs': h_idx, 'lows': l_idx, 'high_r2': sh_r2, 'low_r2': sl_r2}))
        # falling wedge
        if sh < 0 and sl < 0 and abs(sh) < abs(sl) and (self.min_r2 <= 0.0 or r2_ok):
            out.append(DetectedPattern('falling_wedge', 'BULL', 0.65, h_idx[0], l_idx[-1], {'highs': h_idx, 'lows': l_idx, 'high_r2': sh_r2, 'low_r2': sl_r2}))
        return out

    def _detect_flag_pennant(self, bars: Bars, piv: Dict[str, List[int]]) -> List[DetectedPattern]:
        out: List[DetectedPattern] = []
        highs = piv.get('highs', [])
        lows = piv.get('lows', [])
        # simple flag/pennant: short consolidation after a strong move
        n = len(bars.bars)
        if n < 8:
            return out
        # measure recent run: compare average of first half vs last half
        mid = n // 2
        first = [b.close for b in bars.bars[:mid]]
        second = [b.close for b in bars.bars[mid:]]
        rise = (second[-1] - first[0]) if first else 0
        run_pct = abs(rise) / max(1e-8, sum(first)/len(first))
        # consolidation small range
        tail = bars.bars[-6:]
        highs_tail = max(b.high for b in tail)
        lows_tail = min(b.low for b in tail)
        band = (highs_tail - lows_tail) / max(1e-8, (highs_tail + lows_tail)/2)
        if run_pct > 0.03 and band < 0.02:
            # direction based on run sign
            direction = 'BULL' if rise > 0 else 'BEAR'
            out.append(DetectedPattern('flag_pennant', direction, 0.6, n-6, n-1, {'run_pct': run_pct, 'band': band}))
        return out

    def _detect_cup_and_handle(self, bars: Bars, piv: Dict[str, List[int]]) -> List[DetectedPattern]:
        out: List[DetectedPattern] = []
        n = len(bars.bars)
        if n < 9:
            return out
        closes = [b.close for b in bars.bars]
        # find global center (min close)
        center_idx = int(min(range(len(closes)), key=lambda i: closes[i]))
        center_min = closes[center_idx]
        left_max = max(closes[:center_idx]) if center_idx > 0 else 0
        right_max = max(closes[center_idx + 1 :]) if center_idx < n - 1 else 0
        # require both sides have higher rims and center not extremely deep
        if left_max > center_min and right_max > center_min and (center_min / max(left_max, right_max)) > 0.75:
            # handle: last ~20% should be a small pullback vs right rim
            handle_region = closes[int(n * 0.75) :]
            if handle_region:
                hmax = max(handle_region)
                # allow handle to reach up to rim (<=) but require it not exceed and be near rim (small pullback)
                if hmax <= right_max and hmax >= 0.9 * right_max:
                    out.append(DetectedPattern('cup_and_handle', 'BULL', 0.7, 0, n - 1, {'center_idx': center_idx, 'cup_center': center_min, 'handle_max': hmax}))
        return out
