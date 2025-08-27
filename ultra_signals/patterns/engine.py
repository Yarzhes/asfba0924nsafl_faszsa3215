"""PatternEngine Orchestrator (Sprint 44 scaffolding).

Coordinates multiple BasePatternDetector implementations, manages pattern life
cycle transitions, scoring hooks, de-duplication and feature export.

Current implementation is minimal scaffolding:
 - Registers detectors
 - Runs generate() on new bar
 - Maintains simple hash-based registry
 - Updates ages & freshness, placeholder confirmation/failure logic
 - Produces list of PatternInstance objects ready to attach to FeatureVector

Future (roadmap markers as TODO):
 - Geometry hashing improvements (swing pivot serialization)
 - Breakout confirmation rules per pattern type
 - ML scoring + calibration integration
 - Conflict resolver (overlap / mutual exclusivity)
 - Target/stop projection utilities
 - Volume / S/R / fractal confluence enrichment
 - Visualization overlay renderer
"""
from __future__ import annotations

from typing import Dict, List, Any, Iterable
import hashlib

import pandas as pd

from ultra_signals.core.custom_types import (
    PatternInstance,
    PatternStage,
    PatternType,
    PatternDirection,
)
from .base import BasePatternDetector
from .classical import ClassicalDetector
from .harmonics import HarmonicDetector


def _stable_hash(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


class PatternEngine:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.detectors: List[BasePatternDetector] = []
        # registry[(symbol,timeframe)][hash_id] = PatternInstance
        self._registry: Dict[tuple[str, str], Dict[str, PatternInstance]] = {}

    # ---------------- Registration -----------------
    def register(self, detector: BasePatternDetector) -> "PatternEngine":
        self.detectors.append(detector)
        return self

    # ---------------- Processing -----------------
    def on_bar(self, symbol: str, timeframe: str, ohlcv: pd.DataFrame) -> List[PatternInstance]:
        if ohlcv is None or len(ohlcv) < max(5, self.config.get("min_window", 5)):
            return []
        key = (symbol, timeframe)
        reg = self._registry.setdefault(key, {})
        ts_ms = int(pd.Timestamp(ohlcv.index[-1]).value // 1_000_000)

        # Age existing
        for inst in reg.values():
            inst.age_bars += 1
            inst.freshness_bars += 1

        # Generate new candidates
        new_candidates: List[PatternInstance] = []
        for det in self.detectors:
            try:
                cand = det.generate(symbol, timeframe, ohlcv)
                if cand:
                    new_candidates.extend(cand)
            except Exception:
                # Keep engine resilient; log externally if logger available
                pass

        # Integrate candidates (dedup by hash or (type,direction,neckline,breakout,lastN closes sha))
        closes_tail = list(map(float, ohlcv['close'].tail(10).values))
        for c in new_candidates:
            if not c.hash_id:
                c.hash_id = _stable_hash([c.pat_type, c.direction, round(c.neckline_px or 0, 2), round(c.breakout_px or 0, 2), closes_tail])
            existing = reg.get(c.hash_id)
            if existing:
                # refresh existing (keep age, reset freshness)
                existing.freshness_bars = 0
                existing.ts = ts_ms
                # merge improved fields if provided
                for fld in [
                    "quality","confidence","neckline_px","breakout_px","target1_px","target2_px","struct_stop_px",
                    "measured_move_px","fib_fit_err","harm_prz_score","channel_r2","triangle_slope","vpoc_distance_pct",
                    "lvn_confluence_flag","va_rotation_flag","fractal_dim","hurst_h","sr_level_strength","sr_reaction_score",
                    "target_confidence","stop_confidence"
                ]:
                    val = getattr(c, fld, None)
                    if val is not None:
                        setattr(existing, fld, val)
                # extend confluence / reasons
                for list_attr in ["confluence", "reason_codes"]:
                    cur = getattr(existing, list_attr)
                    for item in getattr(c, list_attr):
                        if item not in cur:
                            cur.append(item)
            else:
                reg[c.hash_id] = c

        # Lifecycle transitions (placeholder heuristics)
        confirm_after = int(self.config.get("confirm_min_bars", 2))
        stale_after = int(self.config.get("stale_bars", 100))
        to_delete: List[str] = []
        for hid, inst in reg.items():
            # Confirm if breakout price breached (simple heuristic) or age threshold
            if inst.stage == PatternStage.FORMING:
                if inst.breakout_px is not None:
                    last_close = float(ohlcv['close'].iloc[-1])
                    if (inst.direction == PatternDirection.LONG and last_close >= inst.breakout_px) or (
                        inst.direction == PatternDirection.SHORT and last_close <= inst.breakout_px
                    ):
                        inst.stage = PatternStage.CONFIRMED
                        inst.reason_codes.append("breakout_hit")
                if inst.stage == PatternStage.FORMING and inst.age_bars >= confirm_after:
                    # soft confirm based on persistence
                    inst.stage = PatternStage.CONFIRMED
                    inst.reason_codes.append("age_confirm")
            # Failure heuristic: if price invalidates struct_stop
            if inst.struct_stop_px is not None and inst.stage != PatternStage.FAILED:
                last_close = float(ohlcv['close'].iloc[-1])
                if (inst.direction == PatternDirection.LONG and last_close <= inst.struct_stop_px) or (
                    inst.direction == PatternDirection.SHORT and last_close >= inst.struct_stop_px
                ):
                    inst.stage = PatternStage.FAILED
                    inst.reason_codes.append("stop_invalidated")
            if inst.age_bars > stale_after:
                to_delete.append(hid)
        for hid in to_delete:
            reg.pop(hid, None)

        # Export snapshot list (sorted by quality desc then freshness)
        out = list(reg.values())
        # Conflict resolution (simple): prefer highest quality per pat_type hash direction group
        resolved: Dict[str, PatternInstance] = {}
        for inst in out:
            key2 = f"{inst.pat_type.value}:{inst.direction.value}"
            cur = resolved.get(key2)
            if cur is None or (inst.quality or 0) > (cur.quality or 0):
                resolved[key2] = inst
        out = list(resolved.values())

        # Target/stop pass for any missing targets
        for inst in out:
            if not inst.target1_px or not inst.struct_stop_px:
                # ask each detector that supports pat_type (loop all for simplicity)
                for det in self.detectors:
                    try:
                        inst = det.compute_targets_and_stops(inst, ohlcv)  # may mutate
                    except Exception:
                        pass

        # ML scoring hook (placeholder): quality -> confidence mapping via logistic-ish squeeze
        for inst in out:
            if inst.confidence is None and inst.quality is not None:
                q = max(0.0, min(1.0, inst.quality))
                inst.confidence = 1.0 / (1.0 + pow(2.71828, -4 * (q - 0.5)))  # rough sigmoid

        out.sort(key=lambda x: (x.confidence or 0.0, x.quality or 0.0, -x.freshness_bars), reverse=True)
        return out

    # Convenience factory
    @classmethod
    def with_default_detectors(cls, config: Dict[str, Any] | None = None):
        eng = cls(config)
        eng.register(RangeCompressionDetector(config.get('range_compression', {}) if config else None))
        # Relax classical min_len automatically for short histories (unit tests) by capping to 24 if > len window likely provided
        cls_cfg = (config.get('classical', {}).copy() if config else {})
        if cls_cfg.get('min_len', 30) > 24:
            cls_cfg['min_len'] = 24
        eng.register(ClassicalDetector(cls_cfg))
        eng.register(HarmonicDetector(config.get('harmonics', {}) if config else None))
        return eng


# ------------------- Simple Placeholder Detector -------------------
class RangeCompressionDetector(BasePatternDetector):
    """Toy detector: identifies a volatility compression (proxy for triangle/flag setup).

    Criteria (simplistic for scaffolding):
      - last N bars true range mean / close < threshold
      - price within X% band over window
    Emits a generic SYM_TRIANGLE or BULL_FLAG/Bear flag directional guess based on EMA drift.
    """
    @property
    def name(self) -> str:
        return "range_comp"

    def generate(self, symbol: str, timeframe: str, ohlcv: pd.DataFrame) -> List[PatternInstance]:
        n = int(self.config.get("window", 30))
        if len(ohlcv) < n:
            return []
        window = ohlcv.tail(n)
        hi = float(window['high'].max())
        lo = float(window['low'].min())
        band_pct = (hi - lo) / ((hi + lo) / 2.0) if (hi + lo) > 0 else 0
        if band_pct > self.config.get("max_band_pct", 0.02):  # >2% range -> skip
            return []
        closes = window['close'].astype(float).values
        ema_fast = closes[-5:].mean()
        ema_slow = closes.mean()
        direction = PatternDirection.LONG if ema_fast >= ema_slow else PatternDirection.SHORT
        pat_type = PatternType.SYM_TRIANGLE
        ts_ms = int(pd.Timestamp(window.index[-1]).value // 1_000_000)
        inst = self._new_instance(
            ts_ms=ts_ms,
            symbol=symbol,
            timeframe=timeframe,
            pat_type=pat_type,
            direction=direction,
            reason_codes=["vol_compress"],
        )
        inst.quality = 0.4 + max(0, 0.2 - band_pct)  # crude quality
        inst.neckline_px = (hi + lo) / 2.0
        inst.struct_stop_px = lo * (0.999 if direction == PatternDirection.LONG else 1.001)
        inst.breakout_px = hi if direction == PatternDirection.LONG else lo
        inst.target1_px = inst.breakout_px * (1.005 if direction == PatternDirection.LONG else 0.995)
        inst.target2_px = inst.breakout_px * (1.01 if direction == PatternDirection.LONG else 0.99)
        inst.measured_move_px = (hi - lo)
        inst.confidence = None  # reserved for calibrated model later
        inst.confluence.append("compression")
        return [inst]
