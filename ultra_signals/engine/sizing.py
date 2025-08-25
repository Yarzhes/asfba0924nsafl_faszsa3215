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


# =========================
# Sprint 8 Additions (new)
# =========================

from typing import Optional, Tuple


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
    settings: Dict,
    ensemble_prob: float,
    rr_fallback: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Compute a Kelly-lite sized position.

    Returns:
        (quantity_base, notional_quote, k_lite_used)

    Parameters:
        equity:          Account equity in quote currency (e.g., USDT).
        entry_price:     Intended entry price.
        stop_price:      Stop-loss price (used to compute per-unit risk).
        settings:        Global settings dict (expects 'sizing' keys if present).
        ensemble_prob:   Calibrated probability of success for the trade (0..1).
        rr_fallback:     Optional fallback RR if you don't have a rolling estimate.

    Sizing logic:
        - Base risk amount = equity * base_risk_pct
        - Risk per unit     = |entry - stop|
        - Kelly-lite        = kelly_lite_multiplier(p, RR, cap)
        - Quantity (base)   = (base_risk * (1 + k_lite)) / risk_per_unit
        - Notional (quote)  = quantity * entry_price
    """
    sizing_cfg = settings.get("sizing", {}) if isinstance(settings, dict) else {}
    base_risk_pct = float(sizing_cfg.get("base_risk_pct", 0.01))  # default 1% risk
    kelly_cap = float(sizing_cfg.get("kelly_cap", 0.75))
    rr_default = float(
        rr_fallback if rr_fallback is not None else sizing_cfg.get("rr_fallback", 1.2)
    )

    # 1) base risk in quote currency
    equity = float(equity)
    base_risk_amt = max(0.0, equity * base_risk_pct)

    # 2) risk per unit (quote per 1 unit base)
    risk_per_unit = abs(float(entry_price) - float(stop_price))
    if risk_per_unit <= 0:
        # Degenerate case: no stop distance. Return zero to avoid division by zero.
        return 0.0, 0.0, 0.0

    # 3) Kelly-lite multiplier
    k_lite = kelly_lite_multiplier(ensemble_prob, rr_default, cap=kelly_cap)

    # 4) size in base units (conservative: scale base risk by (1 + k_lite))
    qty_base = (base_risk_amt * (1.0 + k_lite)) / risk_per_unit
    qty_base = max(0.0, float(qty_base))

    # 5) notional in quote currency
    notional_quote = qty_base * float(entry_price)

    return qty_base, notional_quote, k_lite


def size_signal_inplace_with_kelly(
    signal: Signal,
    settings: Dict,
    equity: float,
    ensemble_prob: float,
    atr_percentile: Optional[float] = None,
) -> Signal:
    """
    OPTIONAL helper that **updates the signal in-place** with Kelly-lite sizing.
    Safe to call from your event runner/backtester if you want to activate Sprint-8 sizing.

    Behavior:
      - Uses settings.sizing.{base_risk_pct, kelly_cap, rr_fallback}
      - Applies optional volatility scaling (settings.vol_risk_scale) to the base risk
        *before* Kelly-lite, if you pass `atr_percentile`.
      - Requires `signal.entry_price` and `signal.stop_loss` to be set.

    Returns:
      The same Signal instance with .quantity and .notional_size set.
    """
    # Preconditions
    entry = getattr(signal, "entry_price", None)
    stop = getattr(signal, "stop_loss", None)
    if entry is None or stop is None:
        # Cannot size without a stop distance; keep zero sizes to be safe
        signal.quantity = 0.0
        signal.notional_size = 0.0
        return signal

    # 1) If requested, adjust base risk with volatility scaling
    #    We do this by temporarily modifying base_risk_pct for the formula below.
    sizing_cfg = dict(settings.get("sizing", {})) if isinstance(settings, dict) else {}
    vol_cfg = settings.get("vol_risk_scale", {}) if isinstance(settings, dict) else {}

    base_risk_pct = float(sizing_cfg.get("base_risk_pct", 0.01))
    if atr_percentile is not None:
        # convert pct-of-equity to an amount using 1.0 equity for scaling, then back to pct
        scaled_amount = apply_volatility_scaling(1.0 * base_risk_pct, float(atr_percentile), vol_cfg)
        # scaled_amount is still a "pct" here, so just assign back
        base_risk_pct = float(scaled_amount)

    # Clone a temp settings with updated base_risk_pct
    sizing_cfg["base_risk_pct"] = base_risk_pct
    temp_settings = dict(settings)
    temp_settings["sizing"] = sizing_cfg

    # 2) Kelly-lite core sizing
    qty_base, notional_quote, _k = kelly_lite_size(
        equity=equity,
        entry_price=float(entry),
        stop_price=float(stop),
        settings=temp_settings,
        ensemble_prob=float(ensemble_prob),
        rr_fallback=None,  # use settings.sizing.rr_fallback by default
    )

    # 3) Populate the signal
    signal.quantity = qty_base
    signal.notional_size = notional_quote
    return signal


def estimate_quantity_from_risk(
    equity: float,
    base_risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """
    Minimal helper (non-Kelly) to estimate quantity from a fixed risk percent.
    Useful for quick tests or fallbacks.

    qty_base = (equity * base_risk_pct) / |entry - stop|
    """
    risk_amt = max(0.0, float(equity) * float(base_risk_pct))
    per_unit = abs(float(entry_price) - float(stop_price))
    if per_unit <= 0:
        return 0.0
    return risk_amt / per_unit
