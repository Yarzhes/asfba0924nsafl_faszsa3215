"""
Derivative-based features, such as liquidations.
"""
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultra_signals.core.feature_store import FeatureStore


@dataclass
class DerivativesFeatures:
    """
    Features derived from derivatives data.
    """
    # A simple score representing recent liquidation pressure.
    # Positive: bullish (more shorts liquidated).
    # Negative: bearish (more longs liquidated).
    liq_pulse: int = 0
    
    # Optional: could add rolling notional sums, counts, etc.
    buy_liq_notional_5m: float = 0.0
    sell_liq_notional_5m: float = 0.0


def compute_derivatives_features(
    store: "FeatureStore",
    symbol: str,
    timeframe_ms: int = 5 * 60 * 1000
) -> DerivativesFeatures:
    """
    Computes features based on recent liquidation events.

    Args:
        store: The feature store holding market data.
        symbol: The symbol to compute features for.
        timeframe_ms: The rolling window in milliseconds to consider for the pulse.

    Returns:
        A DerivativesFeatures object.
    """
    now = int(time.time() * 1000)
    cutoff = now - timeframe_ms

    # Prune old liquidations and get the recent list in one atomic operation
    recent_liquidations = store.prune_and_get_liquidations(symbol, cutoff)

    buy_liq_notional = 0.0
    sell_liq_notional = 0.0

    for ts, side, notional in recent_liquidations:
        if side == "BUY":  # A liquidated short results in a market buy
            buy_liq_notional += notional
        else: # A liquidated long results in a market sell
            sell_liq_notional += notional

    # --- Liquidation Pulse ---
    # A simple metric: +1 for net bullish liqs, -1 for net bearish, 0 for neutral
    liq_pulse = 0
    if buy_liq_notional > sell_liq_notional:
        liq_pulse = 1
    elif sell_liq_notional > buy_liq_notional:
        liq_pulse = -1
        
    return DerivativesFeatures(
        liq_pulse=liq_pulse,
        buy_liq_notional_5m=buy_liq_notional,
        sell_liq_notional_5m=sell_liq_notional,
    )