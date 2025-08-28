from typing import Dict, Any


def sizing_adapter(base_risk: float, sigma_next: float, target_sigma: float, floor: float = 1e-6, cap: float = 10.0) -> float:
    """Return size multiplier given base_risk and volatility forecast.

    size ∝ (target_sigma / max(sigma_next, floor)) with hard caps.
    """
    effective_sigma = max(sigma_next or 0.0, floor)
    mult = target_sigma / effective_sigma
    mult = min(mult, cap)
    return float(base_risk * mult)


def stop_adapter(k: float, sigma_next: float, atr: float = None, min_stop: float = 0.0, max_stop: float = 1e6) -> float:
    """Return stop distance in price units: k * sigma_next (optionally blend with ATR)"""
    base = k * (sigma_next or 0.0)
    if atr:
        # simple blend 70% sigma + 30% ATR
        base = 0.7 * base + 0.3 * atr
    base = max(base, min_stop)
    base = min(base, max_stop)
    return float(base)
