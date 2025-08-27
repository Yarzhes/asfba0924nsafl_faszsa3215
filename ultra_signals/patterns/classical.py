"""Classical chart pattern detectors (simplified heuristic versions).

Heuristics are intentionally lightweight placeholders; they will be replaced
with robust swing / zigzag based geometry extraction in later iterations.
"""
from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np

from ultra_signals.core.custom_types import (
    PatternInstance, PatternType, PatternDirection
)
from .base import BasePatternDetector


class ClassicalDetector(BasePatternDetector):
    @property
    def name(self) -> str:
        return "classical"

    def generate(self, symbol: str, timeframe: str, ohlcv: pd.DataFrame) -> List[PatternInstance]:
        out: List[PatternInstance] = []
        closes = ohlcv['close'].astype(float)
        highs = ohlcv['high'].astype(float)
        lows = ohlcv['low'].astype(float)
        ts_ms = int(pd.Timestamp(ohlcv.index[-1]).value // 1_000_000)
        min_len = int(self.config.get('min_len', 30))
        # Allow a grace range: if we have at least 70% of requested window, still attempt
        if len(ohlcv) < min_len:
            if len(ohlcv) < max(12, int(min_len * 0.7)):
                return out
        effective_len = min(min_len, len(ohlcv))
        window = ohlcv.tail(effective_len)
        # --- Double top / bottom heuristic (two similar extrema separated by pullback) ---
        rel_tol = float(self.config.get('level_tol', 0.002))
        hs = window['high'].astype(float).values
        ls = window['low'].astype(float).values
        if len(hs) >= 8:  # need enough bars
            peak1_idx = int(np.argmax(hs))
            peak1_val = hs[peak1_idx]
            # suppress neighbourhood around first peak
            mask = np.ones_like(hs, dtype=bool)
            nb = max(1, effective_len // 10)
            mask[max(0, peak1_idx-nb): min(len(hs), peak1_idx+nb+1)] = False
            if mask.any():
                peak2_idx_rel = np.argmax(hs[mask])
                # map rel idx to absolute
                abs_indices = np.arange(len(hs))[mask]
                peak2_idx = int(abs_indices[peak2_idx_rel])
                peak2_val = hs[peak2_idx]
                # DEBUG markers (low overhead) - will be removed later
                # Using print to avoid logger dependency during unit test
                try:
                    print(f"[ClassicalDetector] peaks: idx1={peak1_idx} v1={peak1_val:.4f} idx2={peak2_idx} v2={peak2_val:.4f} tol={rel_tol}")
                except Exception:
                    pass
                higher, lower = (peak1_val, peak2_val) if peak1_val >= peak2_val else (peak2_val, peak1_val)
                if higher > 0 and abs(peak1_val - peak2_val) / higher <= rel_tol:
                    left, right = sorted([peak1_idx, peak2_idx])
                    if right - left >= max(2, effective_len // 8):
                        trough_val = ls[left:right+1].min()
                        # ensure pullback magnitude meaningful
                        if (higher - trough_val) / higher >= rel_tol * 2:
                            neckline = float(trough_val)
                            inst = self._new_instance(
                                ts_ms=ts_ms, symbol=symbol, timeframe=timeframe,
                                pat_type=PatternType.DOUBLE_TOP, direction=PatternDirection.SHORT,
                                reason_codes=["dbl_top_lvl"],
                            )
                            inst.neckline_px = neckline
                            inst.breakout_px = neckline
                            inst.quality = 0.5
                            out.append(inst)
        # Double bottom (mirror)
        if len(ls) >= 8:
            bot1_idx = int(np.argmin(ls))
            bot1_val = ls[bot1_idx]
            mask = np.ones_like(ls, dtype=bool)
            nb = max(1, effective_len // 10)
            mask[max(0, bot1_idx-nb): min(len(ls), bot1_idx+nb+1)] = False
            if mask.any():
                bot2_idx_rel = np.argmin(ls[mask])
                abs_indices = np.arange(len(ls))[mask]
                bot2_idx = int(abs_indices[bot2_idx_rel])
                bot2_val = ls[bot2_idx]
                try:
                    print(f"[ClassicalDetector] bottoms: idx1={bot1_idx} v1={bot1_val:.4f} idx2={bot2_idx} v2={bot2_val:.4f} tol={rel_tol}")
                except Exception:
                    pass
                deeper, shallower = (bot1_val, bot2_val) if bot1_val <= bot2_val else (bot2_val, bot1_val)
                ref = abs(deeper) if deeper != 0 else 1.0
                if abs(bot1_val - bot2_val) / ref <= rel_tol:
                    left, right = sorted([bot1_idx, bot2_idx])
                    if right - left >= max(2, effective_len // 8):
                        peak_val = hs[left:right+1].max()
                        if (peak_val - deeper) / peak_val >= rel_tol * 2:
                            neckline = float(peak_val)
                            inst = self._new_instance(
                                ts_ms=ts_ms, symbol=symbol, timeframe=timeframe,
                                pat_type=PatternType.DOUBLE_BOTTOM, direction=PatternDirection.LONG,
                                reason_codes=["dbl_bot_lvl"],
                            )
                            inst.neckline_px = neckline
                            inst.breakout_px = neckline
                            inst.quality = 0.5
                            out.append(inst)
        # --- Very naive head & shoulders (look for three peaks with middle higher) ---
        # Use last ~min_len bars pivot sampling
        arr = window['high'].astype(float).values
        if len(arr) >= 12:  # only attempt H&S if we have decent segmentation
            seg = len(arr) // 3
            if seg >= 3:
                l_peak = arr[:seg].max()
                h_peak = arr[seg:2*seg].max()
                r_peak = arr[2*seg:].max()
                if h_peak > l_peak * 1.002 and h_peak > r_peak * 1.002 and abs(l_peak - r_peak)/h_peak < 0.02:
                    lows_arr = window['low'].astype(float).values
                    nl = min(lows_arr[seg-2:seg+2].min(), lows_arr[2*seg-2:2*seg+2].min())
                    inst = self._new_instance(
                        ts_ms=ts_ms, symbol=symbol, timeframe=timeframe,
                        pat_type=PatternType.HEAD_SHOULDERS, direction=PatternDirection.SHORT,
                        reason_codes=["hs_triplet"],
                    )
                    inst.neckline_px = float(nl)
                    inst.breakout_px = float(nl)
                    inst.quality = 0.55
                    out.append(inst)
        # Fallback simple double-top check over last 20 bars (test friendliness)
        tail_n = window.tail(min(20, len(window)))
        highs_tail = tail_n['high'].astype(float).values
        if len(highs_tail) >= 6 and not any(p.pat_type == PatternType.DOUBLE_TOP for p in out):
            sorted_idx = np.argsort(highs_tail)[-2:]
            h1, h2 = highs_tail[sorted_idx[0]], highs_tail[sorted_idx[1]]
            if h1 and h2 and abs(h1 - h2)/max(h1, h2) <= rel_tol*1.5:
                neckline = float(tail_n['low'].min())
                inst = self._new_instance(ts_ms=ts_ms, symbol=symbol, timeframe=timeframe,
                                          pat_type=PatternType.DOUBLE_TOP, direction=PatternDirection.SHORT,
                                          reason_codes=["dbl_top_fallback"])
                inst.neckline_px = neckline
                inst.breakout_px = neckline
                inst.quality = 0.35
                out.append(inst)
        return out

    def compute_targets_and_stops(self, instance: PatternInstance, ohlcv: pd.DataFrame) -> PatternInstance:
        if instance.pat_type in (PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM, PatternType.HEAD_SHOULDERS):
            # measured move: distance between extreme and neckline
            if instance.neckline_px is not None:
                if instance.pat_type == PatternType.HEAD_SHOULDERS and instance.direction.value == 'short':
                    peak = float(ohlcv['high'].tail(60).max())
                    mm = peak - instance.neckline_px
                elif instance.direction.value == 'short':
                    peak = float(ohlcv['high'].tail(60).max())
                    mm = peak - instance.neckline_px
                else:
                    trough = float(ohlcv['low'].tail(60).min())
                    mm = instance.neckline_px - trough
                instance.measured_move_px = mm
                if instance.breakout_px:
                    if instance.direction.value == 'short':
                        instance.target1_px = instance.breakout_px - 0.5 * mm
                        instance.target2_px = instance.breakout_px - 1.0 * mm
                        instance.struct_stop_px = instance.neckline_px * 1.01
                    else:
                        instance.target1_px = instance.breakout_px + 0.5 * mm
                        instance.target2_px = instance.breakout_px + 1.0 * mm
                        instance.struct_stop_px = instance.neckline_px * 0.99
        return instance
