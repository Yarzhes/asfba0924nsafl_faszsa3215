"""Adaptive Position Sizing & Risk Model v2.0 (Sprint 12)

Calculates dynamic position size, stop-loss, take-profit, and leverage using:
- Ensemble decision confidence (probability)
- ATR & volatility percentile
- Regime profile (trend / mean_revert / chop)
- Flow metrics (CVD strength, OI rate, liquidation clusters)

Non-intrusive: if risk_model.enabled=false, returns None to allow legacy sizing.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import math

from loguru import logger


@dataclass
class PositionRiskDecision:
    size_quote: float
    leverage: float
    stop_price: float
    take_profit: float
    entry_price: float
    risk_per_unit: float
    regime: str
    confidence: float
    atr: Optional[float]
    atr_mult_stop: float
    atr_mult_tp: float
    reasoning: Dict[str, Any]


class PositionSizing:
    @staticmethod
    def calculate(
        symbol: str,
        decision,  # EnsembleDecision
        features: Dict[str, object],
        settings: Dict,
        equity: float,
    ) -> Optional[PositionRiskDecision]:
        cfg = (settings.get("risk_model") or {})
        if not cfg.get("enabled", True):
            return None

        try:
            confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
            if confidence < cfg.get("min_confidence", 0.0):
                return None
        except Exception:
            confidence = 0.0

        # Extract ATR
        atr = None
        vol = features.get("volatility") if isinstance(features, dict) else None
        if vol is not None:
            atr = getattr(vol, "atr", None)
        try:
            atr = float(atr) if atr is not None else None
        except Exception:
            atr = None

        # Determine entry price (close of bar); fallback to ohlcv close
        ohlcv = features.get("ohlcv") or {}
        entry = float(ohlcv.get("close", 0.0))
        if entry <= 0:
            return None

        # ATR-based stop distance (fallback to % of price if ATR missing)
        atr_mult_stop = float(cfg.get("atr_multiplier_stop", 2.0))
        atr_mult_tp = float(cfg.get("atr_multiplier_tp", 3.0))
        if atr is None or not math.isfinite(atr) or atr <= 0:
            # fallback: assume pseudo-atr = 0.005 * price (0.5%)
            atr = entry * 0.005

        stop_dist = atr * atr_mult_stop
        tp_dist = atr * atr_mult_tp

        # Flow metrics influence (tighten TP on liquidation cluster)
        fm = features.get("flow_metrics")
        if fm is not None and cfg.get("tighten_tp_on_liq_cluster", True):
            try:
                liq_cluster = getattr(fm, "liq_cluster", None)
                if liq_cluster == 1:
                    factor = float(cfg.get("liq_tp_tighten_factor", 0.7))
                    tp_dist *= factor
            except Exception:
                pass

        # Direction-specific stop/TP prices
        side = getattr(decision, "decision", "FLAT")
        if side not in ("LONG", "SHORT"):
            return None

        if side == "LONG":
            stop_price = max(0.0, entry - stop_dist)
            tp_price = entry + tp_dist
        else:
            stop_price = entry + stop_dist
            tp_price = max(0.0, entry - tp_dist)

        # Risk per unit
        risk_per_unit = abs(entry - stop_price)
        if risk_per_unit <= 0:
            return None

        # Regime factor
        regime_profile = "mixed"
        reg_obj = features.get("regime")
        if reg_obj is not None:
            regime_profile = getattr(reg_obj, "profile", regime_profile)
            if not isinstance(regime_profile, str):
                try:
                    regime_profile = regime_profile.value
                except Exception:
                    regime_profile = "mixed"
        regime_factor = float((cfg.get("regime_risk") or {}).get(regime_profile, 1.0))

        # Volatility factor (inverse with ATR percentile if present)
        atr_pct = None
        if vol is not None:
            atr_pct = getattr(vol, "atr_percentile", None)
        try:
            atr_pct = float(atr_pct) if atr_pct is not None else None
        except Exception:
            atr_pct = None

        # Basic inverse scale: scale = 1 / (1 + (atr_pct/100)*k)
        k_vol = 1.0
        if atr_pct is not None:
            volatility_factor = 1.0 / (1.0 + (atr_pct / 100.0) * k_vol)
        else:
            volatility_factor = 1.0

        # Confidence weight (smooth ramp between min_confidence and 1.0)
        conf_min = float(cfg.get("min_confidence", 0.0))
        if confidence <= conf_min:
            confidence_weight = 0.0
        else:
            confidence_weight = min(1.0, (confidence - conf_min) / (1.0 - conf_min))

        base_risk_pct = float(cfg.get("base_risk_pct", 0.01))
        pos_risk_pct = base_risk_pct * confidence_weight * volatility_factor * regime_factor

        # Convert to position quote size (risk amount, not notional exposure)
        risk_amount_quote = equity * pos_risk_pct
        # Quantity in base asset = risk_amount / risk_per_unit
        qty_base = risk_amount_quote / risk_per_unit if risk_per_unit > 0 else 0.0
        notional_quote = qty_base * entry

        # Leverage scaling
        max_leverage = float(cfg.get("max_leverage", 5))
        if confidence > 0.85:
            leverage = min(max_leverage, 8)
        elif confidence > 0.70:
            leverage = min(max_leverage, 5)
        else:
            leverage = min(max_leverage, 2)

        # Extreme volatility cut: if atr_pct very high reduce leverage
        if atr_pct is not None and atr_pct > 85:
            leverage = min(leverage, 2)
        if atr_pct is not None and atr_pct > 95:
            leverage = min(leverage, 1)

        reasoning = {
            "confidence_weight": round(confidence_weight, 4),
            "volatility_factor": round(volatility_factor, 4),
            "regime_factor": round(regime_factor, 4),
            "risk_pct": round(pos_risk_pct, 6),
            "risk_amount": round(risk_amount_quote, 2),
            "atr_pct": atr_pct,
        }

        decision_obj = PositionRiskDecision(
            size_quote=notional_quote,
            leverage=leverage,
            stop_price=stop_price,
            take_profit=tp_price,
            entry_price=entry,
            risk_per_unit=risk_per_unit,
            regime=regime_profile,
            confidence=confidence,
            atr=atr,
            atr_mult_stop=atr_mult_stop,
            atr_mult_tp=atr_mult_tp,
            reasoning=reasoning,
        )

        logger.debug(
            "[RISK_MODEL] symbol=%s side=%s entry=%.4f size_quote=%.2f lev=%.2f stop=%.4f tp=%.4f conf=%.3f regime=%s atr=%.4f details=%s",
            symbol, side, entry, notional_quote, leverage, stop_price, tp_price, confidence, regime_profile, atr, reasoning
        )
        return decision_obj

    @staticmethod
    def trail_stop(decision: PositionRiskDecision, current_price: float, side: str) -> float:
        """Simple ATR-multiple trailing stop logic placeholder."""
        if not decision or decision.atr is None:
            return decision.stop_price
        # For a favorable move > 1 * ATR, tighten by 0.5*ATR
        if side == "LONG" and current_price - decision.entry_price > decision.atr:
            return max(decision.stop_price, current_price - decision.atr * decision.atr_mult_stop * 0.5)
        if side == "SHORT" and decision.entry_price - current_price > decision.atr:
            return min(decision.stop_price, current_price + decision.atr * decision.atr_mult_stop * 0.5)
        return decision.stop_price
