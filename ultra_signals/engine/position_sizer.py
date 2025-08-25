from __future__ import annotations
"""
Sprint 15: Dynamic Position Sizer
---------------------------------
Calculates adaptive position size based on:
- Ensemble + orderflow confidence (0..1)
- ATR (volatility)
- Liquidation cluster proximity risk (liq_risk)

Returns a quote currency size (simplified). Real implementation would convert to contracts.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class PositionSizeResult:
    size_quote: float
    base_risk: float
    applied_conf: float
    atr: float
    liq_risk: float
    raw_formula: float
    clipped: bool

class PositionSizer:
    def __init__(self, account_equity: float, max_risk_pct: float, atr_window: int = 14, liq_risk_weight: float = 0.6):
        self.account_equity = float(account_equity)
        self.max_risk_pct = float(max_risk_pct)
        self.atr_window = int(atr_window)
        self.liq_risk_weight = float(liq_risk_weight)

    def calc_position_size(self, signal_conf: float, atr: float, liq_risk: float) -> PositionSizeResult:
        # Guard rails
        try:
            conf = max(0.0, min(1.0, float(signal_conf)))
        except Exception:
            conf = 0.0
        try:
            atr_f = float(atr)
        except Exception:
            atr_f = 0.0
        try:
            lr = max(0.0, float(liq_risk))
        except Exception:
            lr = 0.0
        base_risk = self.account_equity * self.max_risk_pct
        if atr_f <= 0:
            # fallback: treat atr as tiny so we still get a number
            atr_f = 1e-6
        # risk reduction multiplier due to liq risk
        liq_penalty = 1.0 + self.liq_risk_weight * lr
        raw = base_risk * conf / (atr_f * liq_penalty)
        clipped = False
        if raw > base_risk * 10:  # arbitrary sanity clip
            raw = base_risk * 10
            clipped = True
        if raw < 0:
            raw = 0.0
        return PositionSizeResult(
            size_quote=raw,
            base_risk=base_risk,
            applied_conf=conf,
            atr=atr_f,
            liq_risk=lr,
            raw_formula=raw,
            clipped=clipped,
        )
