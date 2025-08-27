"""Multi-Timeframe Confirmation Gate (Sprint 30)

Calculates a weighted confirmation score for up to two higher timeframes.
Returns action ENTER | DAMPEN | VETO mapped from internal status CONFIRM | PARTIAL | FAIL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

try:  # type hints
    from ultra_signals.features.htf_cache import HTFFeatures
except Exception:  # pragma: no cover
    HTFFeatures = object  # type: ignore


@dataclass(slots=True)
class MTCGateResult:
    action: str
    status: str
    scores: Dict[str, float]
    reasons: List[str]
    meta: Dict | None = None


class MTCGate:
    def __init__(self, settings: Dict):
        self.settings = settings or {}
        self.cfg = (self.settings.get("mtc") or {}) if isinstance(self.settings, dict) else {}

    def evaluate(self, side: str, symbol: str, ttf: str, ts_epoch: int, profile: str, htf_feats: Dict[str, Any]):
        if not self.cfg.get("enabled", True):
            return MTCGateResult(action="ENTER", status="CONFIRM", scores={}, reasons=["MTC_DISABLED"])
        side_u = side.upper()
        if side_u not in ("LONG", "SHORT"):
            return MTCGateResult(action="ENTER", status="CONFIRM", scores={}, reasons=["NON_DIRECTIONAL"])
        rules = self.cfg.get("rules", {})
        thresholds = self.cfg.get("thresholds", {})
        confirm_full = float(thresholds.get("confirm_full", 0.7))
        confirm_partial = float(thresholds.get("confirm_partial", 0.5))
        missing_policy = str(self.cfg.get("missing_data_policy", "SAFE")).upper()

        scores: Dict[str, float] = {}
        reasons: List[str] = []

        for tag in ("C1", "C2"):
            hf = htf_feats.get(tag)  # type: ignore
            if hf is None:
                if missing_policy in ("OPEN", "OFF"):
                    reasons.append(f"{tag}_MISSING_IGNORED")
                    continue
                reasons.append(f"{tag}_MISSING_SAFE")
                continue
            if hf.stale:
                reasons.append(f"{tag}_STALE")
            score, comp_reasons = self._score_htf(side_u, hf, rules)
            scores[tag] = score
            reasons.extend([f"{tag}_{r}" for r in comp_reasons])

        # SAFE policy -> any missing/stale becomes PARTIAL
        if missing_policy == "SAFE" and (any(r.startswith("C1_MISSING") for r in reasons) or any(r.startswith("C2_MISSING") for r in reasons) or any(r.endswith("_STALE") for r in reasons)):
            return MTCGateResult(action="DAMPEN", status="PARTIAL", scores=scores, reasons=reasons)

        c1 = scores.get("C1", 0.0)
        c2 = scores.get("C2", 0.0)
        if c1 >= confirm_full and c2 >= confirm_full:
            return MTCGateResult(action="ENTER", status="CONFIRM", scores=scores, reasons=reasons)
        if c1 >= confirm_full and c2 >= confirm_partial:
            return MTCGateResult(action="DAMPEN", status="PARTIAL", scores=scores, reasons=reasons)
        # fail path
        fail_cfg = (self.cfg.get("actions", {}) or {}).get("fail", {})
        veto = bool(fail_cfg.get("veto", True))
        return MTCGateResult(action=("VETO" if veto else "DAMPEN"), status="FAIL", scores=scores, reasons=reasons)

    def _score_htf(self, side: str, hf, rules: Dict) -> Tuple[float, List[str]]:
        trend_cfg = rules.get("trend", {})
        mom_cfg = rules.get("momentum", {})
        vol_cfg = rules.get("volatility", {})
        struct_cfg = rules.get("structure", {})
        w_trend = float(trend_cfg.get("score_weight", 0.5))
        w_mom = float(mom_cfg.get("score_weight", 0.3))
        w_vol = float(vol_cfg.get("score_weight", 0.1))
        w_struct = float(struct_cfg.get("score_weight", 0.1))
        total_w = w_trend + w_mom + w_vol + w_struct or 1.0
        comp: List[str] = []

        # Trend
        trend_score = 0.0
        if hf.ema21 is not None and hf.ema200 is not None:
            adx_min = float(trend_cfg.get("adx_min", 18))
            adx_ok = (hf.adx or 0) >= adx_min if hf.adx is not None else False
            if side == "LONG":
                if hf.ema21 > hf.ema200 and adx_ok:
                    trend_score = 1.0
                    comp.append("TREND_OK")
                else:
                    ema_dist = (hf.ema21 - hf.ema200) / abs(hf.ema200 or 1.0)
                    if ema_dist > 0:
                        trend_score = min(1.0, 0.5 + ema_dist)
                    if adx_ok:
                        trend_score = max(trend_score, 0.5)
                    comp.append("TREND_WEAK")
            else:
                if hf.ema21 < hf.ema200 and adx_ok:
                    trend_score = 1.0
                    comp.append("TREND_OK")
                else:
                    ema_dist = (hf.ema200 - hf.ema21) / abs(hf.ema200 or 1.0)
                    if ema_dist > 0:
                        trend_score = min(1.0, 0.5 + ema_dist)
                    if adx_ok:
                        trend_score = max(trend_score, 0.5)
                    comp.append("TREND_WEAK")
        else:
            comp.append("TREND_MISSING")

        # Momentum
        mom_score = 0.0
        if hf.rsi is not None or hf.macd_slope is not None:
            rsi_band_long = mom_cfg.get("rsi_band_long", [45, 70])
            rsi_band_short = mom_cfg.get("rsi_band_short", [30, 55])
            rsi_ok = False
            if hf.rsi is not None:
                if side == "LONG":
                    rsi_ok = rsi_band_long[0] <= hf.rsi <= rsi_band_long[1]
                else:
                    rsi_ok = rsi_band_short[0] <= hf.rsi <= rsi_band_short[1]
            slope = hf.macd_slope or 0.0
            slope_ok = slope > (mom_cfg.get("macd_slope_min", 0.0) or 0.0) if side == "LONG" else slope < -(mom_cfg.get("macd_slope_min", 0.0) or 0.0)
            if slope_ok and rsi_ok:
                mom_score = 1.0
                comp.append("MOM_OK")
            elif slope_ok or rsi_ok:
                mom_score = 0.6
                comp.append("MOM_PARTIAL")
            else:
                comp.append("MOM_WEAK")
        else:
            comp.append("MOM_MISSING")

        # Volatility (penalty)
        vol_score = 1.0
        if hf.atr_percentile is not None:
            cap = float(vol_cfg.get("atr_pctile_max", 0.98))
            atrp = hf.atr_percentile
            try:
                if atrp > 1.2:  # treat >1.2 as possibly 0-100 input
                    if atrp > 100:
                        atrp = atrp / 100.0
            except Exception:
                pass
            if atrp > cap:
                vol_score = max(0.0, 1.0 - (atrp - cap) * 3)
                comp.append("VOL_HIGH")
            else:
                comp.append("VOL_OK")
        else:
            comp.append("VOL_MISSING")

        struct_score = 0.0
        if hf.price is not None and hf.vwap is not None:
            if side == "LONG":
                if hf.price >= hf.vwap:
                    struct_score = 1.0
                    comp.append("STRUCT_OK")
                else:
                    comp.append("STRUCT_BELOW_VWAP")
            else:
                if hf.price <= hf.vwap:
                    struct_score = 1.0
                    comp.append("STRUCT_OK")
                else:
                    comp.append("STRUCT_ABOVE_VWAP")
        else:
            comp.append("STRUCT_MISSING")

        weighted = (trend_score * w_trend + mom_score * w_mom + vol_score * w_vol + struct_score * w_struct) / total_w
        return max(0.0, min(1.0, weighted)), comp


def evaluate_gate(side: str, symbol: str, ttf: str, ts_epoch: int, profile: str, settings: Dict, htf_map: Dict[str, Any]):
    gate = MTCGate(settings)
    try:
        return gate.evaluate(side, symbol, ttf, ts_epoch, profile, htf_map)
    except Exception as e:  # pragma: no cover
        logger.debug(f"MTC gate error: {e}")
        return MTCGateResult(action="ENTER", status="CONFIRM", scores={}, reasons=["ERR"])

__all__ = ["MTCGate", "MTCGateResult", "evaluate_gate"]
