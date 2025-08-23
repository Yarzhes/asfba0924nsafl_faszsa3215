from collections import deque
from dataclasses import dataclass, field
from typing import Deque, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ultra_signals.core.feature_store import FeatureStore


@dataclass
class CvdFeatures:
    """Features related to Cumulative Volume Delta."""

    cvd: float = 0.0
    cvd_slope: float = 0.0


@dataclass
class CvdState:
    """Maintains state for calculating CVD."""

    current_cvd: float = 0.0
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))


def compute_cvd_features(
    store: "FeatureStore",
    symbol: str,
    state: CvdState,
    config: dict,
) -> CvdFeatures | None:
    """
    Computes Cumulative Volume Delta (CVD) and its slope from recent trades.
    Args:
        store: The feature store containing the latest market data.
        symbol: The symbol to compute features for.
        state: The state object for tracking CVD.
        config: Configuration dictionary.
    Returns:
        A CvdFeatures object or None if no new trades are available.
    """
    # We need a time window to prune trades, e.g., last 5 minutes
    lookback_seconds = config.get("cvd_lookback_seconds", 300)
    now_ms = pd.Timestamp.now().value // 1_000_000
    cutoff_ts = now_ms - (lookback_seconds * 1000)

    # Prune and get trades within our lookback window
    trades = store.prune_and_get_trades(symbol, cutoff_ts)
    if not trades:
        return None

    # Calculate delta from the latest trades since last update
    # This is a simplification. A real implementation would need to track the last processed trade ID.
    # For this example, we'll calculate delta on the whole recent trade list.
    delta = 0
    for ts, price, qty, is_buyer_maker in trades:
        # If a buyer is a maker, it's a seller-initiated (taker) trade
        if is_buyer_maker:
            delta -= qty
        # If a seller is a maker, it's a buyer-initiated (taker) trade
        else:
            delta += qty

    # This is an approximation of live CVD.
    # A robust implementation would stream CVD from an exchange or calculate from a full trade history.
    state.current_cvd += delta
    state.history.append(state.current_cvd)

    # --- CVD Slope ---
    cvd_slope = 0.0
    k = config.get("cvd_slope_period", 20)  # Lookback period for slope
    if len(state.history) >= k:
        cvd_then = state.history[-k]
        cvd_slope = (state.history[-1] - cvd_then) / k

    return CvdFeatures(
        cvd=state.current_cvd,
        cvd_slope=cvd_slope,
    )