"""Regime Router (Sprint 13)

Determines simplified trading regime and selects appropriate alpha profile list.
Uses existing computed features (trend, momentum, volatility, flow_metrics, regime) plus
lightweight heuristics. Non-intrusive: if anything fails returns 'mixed' and default profile.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from loguru import logger
import math

DEFAULT_REGIME = "mixed"

class RegimeRouter:
    @staticmethod
    def detect_regime(features: Dict[str, object], settings: Dict) -> str:
        try:
            # Prefer Sprint 10 regime classifier if present
            reg_obj = features.get("regime")
            if reg_obj is not None:
                prof = getattr(reg_obj, "profile", None)
                if prof:
                    return prof if isinstance(prof, str) else getattr(prof, 'value', DEFAULT_REGIME)
        except Exception:
            pass

        cfg = (settings.get("regime_detection") or {})
        adx_thr = float(cfg.get("adx_threshold", 22))
        chop_vol_pct = float(cfg.get("chop_volatility", 15))  # treat as atr_percentile threshold
        mr_rsi_thr = float(cfg.get("mean_revert_rsi", 70))

        trend_f = features.get("trend")
        vol_f = features.get("volatility")
        mom_f = features.get("momentum")

        adx = getattr(trend_f, "adx", None) if trend_f else None
        atr_pct = getattr(vol_f, "atr_percentile", None) if vol_f else None
        rsi = getattr(mom_f, "rsi", None) if mom_f else None

        # Basic fallback heuristics
        try:
            adx_val = float(adx) if adx is not None and math.isfinite(adx) else None
        except Exception: adx_val = None
        try:
            atrp_val = float(atr_pct) if atr_pct is not None and math.isfinite(atr_pct) else None
        except Exception: atrp_val = None
        try:
            rsi_val = float(rsi) if rsi is not None and math.isfinite(rsi) else None
        except Exception: rsi_val = None

        # Decide regime
        # Trend: ADX high plus optional price displacement (ema alignment already in scoring)
        if adx_val is not None and adx_val >= adx_thr:
            return "trend"
        # Chop: low volatility percentile
        if atrp_val is not None and atrp_val <= chop_vol_pct:
            return "chop"
        # Mean revert: RSI extremes
        if rsi_val is not None and (rsi_val >= mr_rsi_thr or rsi_val <= (100 - mr_rsi_thr)):
            return "mean_revert"
        return DEFAULT_REGIME

    @staticmethod
    def pick_alphas(regime: str, settings: Dict) -> Tuple[List[str], Dict]:
        profiles = (settings.get("alpha_profiles") or {})
        prof_cfg = profiles.get(regime, profiles.get("trend", {}))
        alphas = prof_cfg.get("alphas", []) or []
        return alphas, prof_cfg

    @staticmethod
    def route(features: Dict[str, object], settings: Dict) -> Dict:
        reg = RegimeRouter.detect_regime(features, settings)
        alphas, prof_cfg = RegimeRouter.pick_alphas(reg, settings)
        out = {
            "regime": reg,
            "alphas": alphas,
            "weight_scale": prof_cfg.get("weight_scale", 1.0),
            "min_confidence": prof_cfg.get("min_confidence", 0.0),
        }
        logger.debug("[REGIME_ROUTER] {}", out)
        return out
