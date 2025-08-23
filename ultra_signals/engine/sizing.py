"""
Position Sizing Engine

This module is responsible for determining the appropriate size for a given
trading signal. In a live trading environment, this would involve calculating
the notional value of the position based on account equity, risk parameters,
and signal conviction.

For v0.1 (paper trading), this module acts as a simple pass-through and does
not assign any real capital.
"""

from typing import Dict
from ultra_signals.core.custom_types import Signal


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