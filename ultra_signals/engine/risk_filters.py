"""
Risk Management Filters

Provides the apply_filters function to check if a signal passes basic risk filters.
"""

from dataclasses import dataclass
from typing import Optional, Dict

from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import Signal

@dataclass
class FilterResult:
    passed: bool
    reason: str = ""
    details: Optional[Dict] = None

def apply_filters(signal: Signal, store: FeatureStore, settings: dict) -> FilterResult:
    """
    Checks if a signal passes basic risk filters.
    - Must check: warmup bars and spread (bid/ask) availability.
    - `store` is FeatureStore, use its stored OHLCV and book ticker.
    - `settings` is a dict (pydantic model dumped by .model_dump()).
    """
    # ----------------- 1. WARMUP CHECK -----------------
    warmup_periods = settings.get("features", {}).get("warmup_periods", 20)
    available_bars = store.get_warmup_status(signal.symbol, signal.timeframe)
    if available_bars < warmup_periods:
        return FilterResult(False, reason="WARMUP_INCOMPLETE")

    # ----------------- 2. BOOK TICKER VALIDATION -----------------
    book_ticker = store.get_book_ticker(signal.symbol)
    if not book_ticker:
        return FilterResult(False, reason="MISSING_BOOK_TICKER")

    # Ensure book_ticker is a tuple with at least 2 elements
    if not isinstance(book_ticker, tuple) or len(book_ticker) < 2:
        return FilterResult(False, reason="INVALID_BOOK_TICKER")

    bid, ask = book_ticker[:2]
    if bid <= 0 or ask <= 0:
        return FilterResult(False, reason="INVALID_PRICE")

    mid_price = (ask + bid) / 2
    if mid_price == 0:
        return FilterResult(False, reason="ZERO_MID_PRICE")

    # ----------------- 3. SPREAD CALCULATION -----------------
    spread = ask - bid
    spread_pct = spread / mid_price

    # ----------------- 4. MAX SPREAD FETCH (UPDATED) -----------------
    # Try to fetch max_spread_pct from top-level filters first.
    # If not found, fallback to engine → risk → max_spread_pct → default.
    max_spread_pct = settings.get("filters", {}).get("max_spread_pct")
    if max_spread_pct is None:
        max_spread_pct = (
            settings.get("engine", {})
                    .get("risk", {})
                    .get("max_spread_pct", {})
                    .get("default", 0.002)  # fallback hard default = 0.2%
        )

    # ----------------- 5. APPLY FILTER -----------------
    if spread_pct > max_spread_pct:
        return FilterResult(
            False,
            reason="SPREAD_TOO_WIDE",
            details={"spread": spread_pct, "max_allowed": max_spread_pct}
        )

    # ----------------- 6. PASSED -----------------
    return FilterResult(True)
