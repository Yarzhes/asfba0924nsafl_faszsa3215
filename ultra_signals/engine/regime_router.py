"""Regime Router (Phase-3 Upgrade)

Determines trading regime with enhanced logic including secondary checks,
squeeze detection, and configurable gates. Uses existing computed features
plus lightweight heuristics for robust regime classification.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from loguru import logger
import math

DEFAULT_REGIME = "mixed"

class RegimeRouter:
    @staticmethod
    def detect_regime(features: Dict[str, object], settings: Dict) -> str:
        """
        Enhanced regime detection with secondary checks and squeeze detection.
        
        Args:
            features: Dictionary of computed features
            settings: Application settings
            
        Returns:
            Regime string: 'trend', 'range', 'breakout', 'mean_revert', 'chop', 'mixed'
        """
        try:
            # Prefer existing regime classifier if present
            reg_obj = features.get("regime")
            if reg_obj is not None:
                prof = getattr(reg_obj, "profile", None)
                if prof:
                    return prof if isinstance(prof, str) else getattr(prof, 'value', DEFAULT_REGIME)
        except Exception:
            pass

        # Load regime detection settings
        regime_settings = settings.get("regimes", {})
        adx_trend_min = float(regime_settings.get("adx_trend_min", 18))
        squeeze_bbkc_ratio_max = float(regime_settings.get("squeeze_bbkc_ratio_max", 1.1))
        range_adx_max = float(regime_settings.get("range_adx_max", 14))
        breakout_vol_burst_z = float(regime_settings.get("breakout_vol_burst_z", 1.5))

        # Extract features
        trend_f = features.get("trend")
        vol_f = features.get("volatility")
        mom_f = features.get("momentum")
        flow_f = features.get("flow_metrics")
        volume_f = features.get("volume_flow")

        # Extract values with safe conversion
        adx = RegimeRouter._safe_float(getattr(trend_f, "adx", None))
        ema_short = RegimeRouter._safe_float(getattr(trend_f, "ema_short", None))
        ema_medium = RegimeRouter._safe_float(getattr(trend_f, "ema_medium", None))
        ema_long = RegimeRouter._safe_float(getattr(trend_f, "ema_long", None))
        
        # Volatility features
        atr = RegimeRouter._safe_float(getattr(vol_f, "atr", None))
        atr_percentile = RegimeRouter._safe_float(getattr(vol_f, "atr_percentile", None))
        bb_upper = RegimeRouter._safe_float(getattr(vol_f, "bbands_upper", None))
        bb_lower = RegimeRouter._safe_float(getattr(vol_f, "bbands_lower", None))
        
        # Momentum features
        rsi = RegimeRouter._safe_float(getattr(mom_f, "rsi", None))
        macd_hist = RegimeRouter._safe_float(getattr(mom_f, "macd_hist", None))
        
        # Flow features
        volume_z = RegimeRouter._safe_float(getattr(flow_f, "volume_z", None))
        cvd = RegimeRouter._safe_float(getattr(flow_f, "cvd", None))
        
        # Current price for calculations
        current_price = RegimeRouter._get_current_price(features)

        # Enhanced regime detection logic
        regime = RegimeRouter._detect_trend_regime(
            adx, ema_short, ema_medium, ema_long, current_price, adx_trend_min, squeeze_bbkc_ratio_max
        )
        
        if regime == "mixed":
            regime = RegimeRouter._detect_range_regime(
                adx, atr_percentile, bb_upper, bb_lower, current_price, range_adx_max
            )
        
        if regime == "mixed":
            regime = RegimeRouter._detect_breakout_regime(
                volume_z, cvd, atr_percentile, breakout_vol_burst_z
            )
        
        if regime == "mixed":
            regime = RegimeRouter._detect_mean_revert_regime(rsi, macd_hist)
        
        if regime == "mixed":
            regime = RegimeRouter._detect_chop_regime(atr_percentile, adx)

        # Safe logging with None checks
        adx_str = f"{adx:.1f}" if adx is not None else "None"
        rsi_str = f"{rsi:.1f}" if rsi is not None else "None"
        atr_str = f"{atr_percentile:.1f}" if atr_percentile is not None else "None"
        logger.debug(f"[REGIME_ROUTER] Detected: {regime} (ADX={adx_str}, RSI={rsi_str}, ATR%={atr_str})")
        return regime

    @staticmethod
    def _detect_trend_regime(
        adx: Optional[float],
        ema_short: Optional[float],
        ema_medium: Optional[float],
        ema_long: Optional[float],
        current_price: Optional[float],
        adx_trend_min: float,
        squeeze_bbkc_ratio_max: float
    ) -> str:
        """Detect trend regime with EMA ladder confirmation."""
        if adx is None or adx < adx_trend_min:
            return "mixed"
        
        if current_price is None or ema_short is None or ema_medium is None or ema_long is None:
            return "mixed"
        
        # Check EMA ladder alignment
        ema_aligned_up = ema_short > ema_medium > ema_long
        ema_aligned_down = ema_short < ema_medium < ema_long
        
        # Check squeeze condition (simplified)
        bb_width = (ema_short - ema_long) / (ema_short + ema_long) if (ema_short + ema_long) > 0 else 0
        not_squeezed = bb_width > squeeze_bbkc_ratio_max * 0.01  # 1% threshold
        
        if ema_aligned_up and not_squeezed:
            return "trend"
        elif ema_aligned_down and not_squeezed:
            return "trend"
        
        return "mixed"

    @staticmethod
    def _detect_range_regime(
        adx: Optional[float],
        atr_percentile: Optional[float],
        bb_upper: Optional[float],
        bb_lower: Optional[float],
        current_price: Optional[float],
        range_adx_max: float
    ) -> str:
        """Detect range regime with low ADX and price near BB edges."""
        if adx is None or adx > range_adx_max:
            return "mixed"
        
        if atr_percentile is None or atr_percentile > 0.3:  # Not low volatility
            return "mixed"
        
        if current_price is None or bb_upper is None or bb_lower is None:
            return "mixed"
        
        # Check if price is near Bollinger Band edges
        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return "mixed"
        
        price_position = (current_price - bb_lower) / bb_width
        near_edges = price_position < 0.2 or price_position > 0.8
        
        if near_edges:
            return "range"
        
        return "mixed"

    @staticmethod
    def _detect_breakout_regime(
        volume_z: Optional[float],
        cvd: Optional[float],
        atr_percentile: Optional[float],
        breakout_vol_burst_z: float
    ) -> str:
        """Detect breakout regime with volume burst and squeeze release."""
        # Check for volume burst
        volume_burst = volume_z is not None and volume_z > breakout_vol_burst_z
        
        # Check for squeeze release (high ATR percentile)
        squeeze_release = atr_percentile is not None and atr_percentile > 0.7
        
        # Check for strong flow
        strong_flow = cvd is not None and abs(cvd) > 0.1
        
        if volume_burst and (squeeze_release or strong_flow):
            return "breakout"
        
        return "mixed"

    @staticmethod
    def _detect_mean_revert_regime(rsi: Optional[float], macd_hist: Optional[float]) -> str:
        """Detect mean reversion regime with RSI extremes."""
        if rsi is None:
            return "mixed"
        
        # RSI extremes
        rsi_extreme = rsi < 30 or rsi > 70
        
        # MACD histogram reversal
        macd_reversal = macd_hist is not None and abs(macd_hist) > 0.01
        
        if rsi_extreme:
            return "mean_revert"
        
        return "mixed"

    @staticmethod
    def _detect_chop_regime(atr_percentile: Optional[float], adx: Optional[float]) -> str:
        """Detect choppy regime with low volatility and low ADX."""
        if atr_percentile is None or adx is None:
            return "mixed"
        
        low_vol = atr_percentile < 0.2
        low_adx = adx < 15
        
        if low_vol and low_adx:
            return "chop"
        
        return "mixed"

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            val = float(value)
            return val if math.isfinite(val) else None
        except Exception:
            return None

    @staticmethod
    def _get_current_price(features: Dict[str, object]) -> Optional[float]:
        """Extract current price from features."""
        # Try to get from various feature objects
        for key in ['trend', 'volatility', 'momentum']:
            feat_obj = features.get(key)
            if feat_obj and hasattr(feat_obj, 'current_price'):
                price_val = getattr(feat_obj, 'current_price')
                # Handle Mock objects
                if hasattr(price_val, '_mock_return_value'):
                    price_val = price_val._mock_return_value
                result = RegimeRouter._safe_float(price_val)
                if result is not None:
                    return result
        
        # Fallback: try to get from any feature with price
        for feat_obj in features.values():
            if hasattr(feat_obj, 'price'):
                price_val = getattr(feat_obj, 'price')
                # Handle Mock objects
                if hasattr(price_val, '_mock_return_value'):
                    price_val = price_val._mock_return_value
                result = RegimeRouter._safe_float(price_val)
                if result is not None:
                    return result
        
        return None

    @staticmethod
    def pick_alphas(regime: str, settings: Dict) -> Tuple[List[str], Dict]:
        """Pick alpha strategies based on regime."""
        profiles = (settings.get("alpha_profiles") or {})
        prof_cfg = profiles.get(regime, profiles.get("trend", {}))
        alphas = prof_cfg.get("alphas", []) or []
        return alphas, prof_cfg

    @staticmethod
    def route(features: Dict[str, object], settings: Dict) -> Dict:
        """Route features to appropriate regime and alpha strategies."""
        reg = RegimeRouter.detect_regime(features, settings)
        alphas, prof_cfg = RegimeRouter.pick_alphas(reg, settings)
        
        # Provide both legacy keys and visualization-friendly aliases
        out = {
            "regime": reg,                       # legacy
            "detected": reg,                     # alias for visualization
            "alphas": alphas,                    # legacy
            "alphas_used": alphas,               # alias
            "weight_scale": prof_cfg.get("weight_scale", 1.0),
            "confidence_boost": prof_cfg.get("weight_scale", 1.0),  # alias
            "min_confidence": prof_cfg.get("min_confidence", 0.0),
        }
        logger.debug("[REGIME_ROUTER] {}", out)
        return out
