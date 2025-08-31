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
from typing import Optional, Dict
from ultra_signals.core.custom_types import Signal


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


def determine_position_size(signal: Signal, settings: Dict) -> Signal:
    """
    Determines the position size for a given signal.

    In v0.1, this is a placeholder and always returns a size of 0.
    In future sprints, this function will implement actual sizing logic.

    Args:
        signal: The trading signal to be sized.
        settings: The global application settings.

    Returns:
        The `Signal` object, potentially updated with sizing information.
    """
    # For paper trading, notional size is always 0.
    signal.notional_size = 0.0
    signal.quantity = 0.0
    
    # Future logic might look like this:
    # risk_per_trade = settings['risk']['max_risk_per_trade'] # e.g., 0.01 (1%)
    # account_balance = get_account_balance() # from an exchange API
    # risk_amount = account_balance * risk_per_trade
    # distance_to_sl = abs(signal.entry_price - signal.stop_loss)
    # if distance_to_sl > 0:
    #     signal.quantity = risk_amount / distance_to_sl
    #     signal.notional_size = signal.quantity * signal.entry_price
    
    return signal


def apply_volatility_scaling(base_risk: float, atr_percentile: float, cfg: Dict) -> float:
    """
    Adjusts the base risk for a trade based on the market's volatility percentile.

    Args:
        base_risk: The initial risk amount for the trade.
        atr_percentile: The current ATR percentile (0-100).
        cfg: The 'vol_risk_scale' section of the config.

    Returns:
        The adjusted risk amount.
    """
    if atr_percentile < cfg.get("low_vol_pct", 30):
        # Boost risk in low volatility
        return base_risk * cfg.get("low_vol_boost", 1.0)
    elif atr_percentile > cfg.get("high_vol_pct", 70):
        # Cut risk in high volatility
        return base_risk * cfg.get("high_vol_cut", 1.0)
    else:
        # No adjustment in medium volatility
        return base_risk


def kelly_lite_multiplier(p: float, rr: float, cap: float = 0.75) -> float:
    """
    Kelly-lite multiplier used to scale risk up/down based on calibrated win-probability.

    k_raw = (p*RR - (1-p)) / RR
    k_lite = clamp(0.5 * k_raw, 0.25, cap)

    Notes:
      - We halve Kelly (0.5*) to be conservative.
      - We clamp to a minimum of 0.25 so extremely small allocations aren't starved,
        and to a maximum of `cap` (e.g., 0.75) to avoid over-sizing.
    """
    p = max(0.0, min(1.0, float(p)))
    rr = max(1e-9, float(rr))
    k_raw = (p * rr - (1.0 - p)) / rr
    k_lite = 0.5 * k_raw
    # clamp to [0.25, cap]
    return max(0.25, min(float(cap), k_lite))


def kelly_lite_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    win_prob: float,
    risk_reward: float,
    cap: float = 0.75
) -> float:
    """
    Calculate position size using Kelly-lite formula.
    
    Args:
        equity: Account equity
        entry_price: Entry price
        stop_price: Stop loss price
        win_prob: Win probability (0-1)
        risk_reward: Risk/reward ratio
        cap: Maximum position size as fraction of equity
        
    Returns:
        Position size in quote currency
    """
    k_mult = kelly_lite_multiplier(win_prob, risk_reward, cap)
    risk_amount = equity * k_mult
    price_risk = abs(entry_price - stop_price)
    
    if price_risk <= 0:
        return 0.0
        
    return risk_amount / price_risk


def atr_position_size(
    equity: float,
    entry_price: float,
    atr: float,
    risk_pct: float = 0.01,
    atr_multiplier: float = 1.5
) -> float:
    """
    Calculate position size based on ATR.
    
    Args:
        equity: Account equity
        entry_price: Entry price
        atr: Average True Range
        risk_pct: Risk percentage of equity
        atr_multiplier: ATR multiplier for stop loss
        
    Returns:
        Position size in quote currency
    """
    risk_amount = equity * risk_pct
    stop_distance = atr * atr_multiplier
    
    if stop_distance <= 0:
        return 0.0
        
    return risk_amount / stop_distance
