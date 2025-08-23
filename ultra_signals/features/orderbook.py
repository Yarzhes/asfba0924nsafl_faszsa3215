"""
Orderbook-based features, computed from live book ticker data.
"""
from collections import deque
from dataclasses import dataclass, field
import math
from typing import Optional, Deque, TYPE_CHECKING

if TYPE_CHECKING:
    from ultra_signals.core.feature_store import FeatureStore


@dataclass
class OrderbookFeatures:
    """
    Snapshot of lite order book features.
    
    Uses top-of-book data (best bid/ask) as a proxy for full depth.
    """
    # Ratio of buying to selling pressure at the top of the book. > 1 means more buy volume.
    imbalance: Optional[float] = 0.0
    
    # Notional value of the best bid.
    bid_sum_top: float = 0.0
    
    # Notional value of the best ask.
    ask_sum_top: float = 0.0
    
    # Absolute difference between best ask and best bid.
    spread: float = 0.0
    
    # Estimated percentage price change to fill an order of a certain size.
    # Here, it's a simplification using only top-of-book.
    slip_est: float = 0.0


@dataclass
class OrderbookFeaturesV2:
    """
    A snapshot of advanced order book features based on full depth.
    """
    imbalance_ratio: Optional[float] = None
    book_flip_detected: bool = False
    estimated_slippage_pct: Optional[float] = None


@dataclass
class BookFlipState:
    """Maintains state for detecting a book-flip."""
    imbalance_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_state: Optional[str] = None # "bid-dominant" or "ask-dominant"
    persistence_ticks: int = 0


def compute_orderbook_features_v2(
    store: "FeatureStore",
    symbol: str,
    state: BookFlipState,
    config: dict,
) -> OrderbookFeaturesV2 | None:
    """
    Computes advanced order book features from full depth data.

    Args:
        store: The feature store containing the latest market data.
        symbol: The symbol to compute features for.
        state: The state object for tracking book-flip persistence.
        config: Configuration dictionary with parameters like thresholds.

    Returns:
        An OrderbookFeaturesV2 object or None if data is not available.
    """
    depth = store.get_depth(symbol)
    if not depth or not depth.bids or not depth.asks:
        return None

    # --- 1. Order Book Imbalance ---
    depth_levels = config.get("depth_levels_N", 10)
    total_bid_volume = sum(p * q for p, q in depth.bids[:depth_levels])
    total_ask_volume = sum(p * q for p, q in depth.asks[:depth_levels])

    if (total_bid_volume + total_ask_volume) == 0:
        imbalance_ratio = 0.5
    else:
        imbalance_ratio = total_bid_volume / (total_bid_volume + total_ask_volume)

    state.imbalance_history.append(imbalance_ratio)

    # --- 2. Book-Flip Detection ---
    book_flip_detected = False
    current_state = "bid-dominant" if imbalance_ratio > 0.5 else "ask-dominant"
    
    if state.last_state and current_state != state.last_state:
        # State changed, check delta
        min_delta = config.get("book_flip_min_delta", 0.15)
        prev_imbalance = state.imbalance_history[-2] if len(state.imbalance_history) > 1 else 0.5
        
        if abs(imbalance_ratio - prev_imbalance) >= min_delta:
            state.persistence_ticks = 1 # Start counting
        else:
            state.persistence_ticks = 0 # Reset if delta is too small
    elif state.last_state and current_state == state.last_state:
        # State is the same, increment persistence counter
        state.persistence_ticks += 1
    
    persistence_req = config.get("book_flip_persistence_ticks", 5)
    if state.persistence_ticks >= persistence_req:
        book_flip_detected = True
        state.persistence_ticks = 0 # Reset after detection

    state.last_state = current_state

    # --- 3. Simple Slippage Estimator ---
    slippage_pct = None
    trade_size = config.get("slippage_trade_size", 1000) # Example trade size in quote currency
    
    # Estimate slippage for a buy order
    cost = 0
    filled_qty = 0
    for price, qty in depth.asks:
        volume_at_level = price * qty
        if (cost + volume_at_level) >= trade_size:
            needed_qty = (trade_size - cost) / price
            cost += needed_qty * price
            filled_qty += needed_qty
            break
        cost += volume_at_level
        filled_qty += qty
    
    if filled_qty > 0:
        vwap_fill = cost / filled_qty
        best_ask = depth.asks[0][0]
        slippage_pct = (vwap_fill - best_ask) / best_ask if best_ask > 0 else 0.0

    return OrderbookFeaturesV2(
        imbalance_ratio=imbalance_ratio,
        book_flip_detected=book_flip_detected,
        estimated_slippage_pct=slippage_pct,
    )


def compute_orderbook_features(store: "FeatureStore", symbol: str) -> OrderbookFeatures | None:
    """
    Computes order book features from the latest book ticker snapshot.

    Args:
        store: The feature store containing the latest market data.
        symbol: The symbol to compute features for.

    Returns:
        An OrderbookFeatures object or None if data is not available.
    """
    ticker = store.get_book_ticker(symbol)
    if not ticker:
        return None

    bid, bid_qty, ask, ask_qty = ticker

    # --- Spread ---
    spread = ask - bid

    # --- Imbalance ---
    bid_notional = bid * bid_qty
    ask_notional = ask * ask_qty
    total_notional = bid_notional + ask_notional
    imbalance = bid_notional / ask_notional if ask_notional > 0 else 1.0

    # --- Slippage Estimate ---
    # Simplified: assumes our trade eats the entire top-of-book liquidity.
    # A more advanced version would use depth data.
    mid_price = (bid + ask) / 2
    slip_est = (ask - mid_price) / mid_price if mid_price > 0 else 0.0

    return OrderbookFeatures(
        imbalance=imbalance,
        bid_sum_top=bid_notional,
        ask_sum_top=ask_notional,
        spread=spread,
        slip_est=slip_est,
    )