"""
Core Data Models for Market Events

This module defines the canonical, normalized data structures for events
received from various data sources. Using standardized models ensures that
the rest of the application (feature store, engine, etc.) can operate
on a consistent data format, regardless of the source.

Pydantic models are used for data validation and to provide a clear,
self-documenting structure.
"""

from pydantic import BaseModel, Field
from typing import Union, Literal, List, Tuple


class ForceOrderEvent(BaseModel):
    """
    Represents a liquidation event.
    """
    event_type: Literal["forceOrder"] = "forceOrder"
    timestamp: int      # Event time
    symbol: str         # Trading symbol
    side: str           # "BUY" or "SELL"
    price: float
    quantity: float


class BookTickerEvent(BaseModel):
    """
    Represents a best bid/ask update.
    """
    event_type: Literal["bookTicker"] = "bookTicker"
    timestamp: int      # Event time
    symbol: str         # Trading symbol
    best_bid: float = Field(alias="b")
    best_bid_qty: float = Field(alias="B")
    best_ask: float = Field(alias="a")
    best_ask_qty: float = Field(alias="A")

class MarkPriceEvent(BaseModel):
    """
    Represents a mark price update.
    """
    event_type: Literal["markPrice"] = "markPrice"
    timestamp: int = Field(alias="E")
    symbol: str = Field(alias="s")
    mark_price: float = Field(alias="p")
    funding_rate: float = Field(alias="r")
    next_funding_time: int = Field(alias="T")


class KlineEvent(BaseModel):
    """
    Represents a single kline (candlestick) event.

    This model normalizes kline data from different exchanges into a
    single, consistent format.
    """
    event_type: Literal["kline"] = "kline"
    timestamp: int  # Kline start time as a Unix timestamp (milliseconds)
    symbol: str     # Trading symbol (e.g., "BTCUSDT")
    timeframe: str  # Kline timeframe/interval (e.g., "1m", "5m")
    open: float     # Open price
    high: float     # High price
    low: float      # Low price
    close: float    # Close price
    volume: float   # Volume during the kline
    closed: bool    # True if this is the final update for this kline


class DepthEvent(BaseModel):
    """
    Represents a full order book depth update.
    """
    event_type: Literal["depthUpdate"] = "depthUpdate"
    timestamp: int      # Event time
    symbol: str         # Trading symbol
    bids: List[Tuple[float, float]] # List of [price, quantity] tuples
    asks: List[Tuple[float, float]] # List of [price, quantity] tuples


class AggTradeEvent(BaseModel):
    """
    Represents an aggregated trade event.
    """
    event_type: Literal["aggTrade"] = "aggTrade"
    timestamp: int      # Event time
    symbol: str         # Trading symbol
    price: float
    quantity: float
    is_buyer_maker: bool # True if the buyer is the market maker


# A union of all possible market events that the WebSocket client can yield.
# This allows for type-safe handling of different event types.
MarketEvent = Union[
    KlineEvent, BookTickerEvent, MarkPriceEvent, ForceOrderEvent, DepthEvent, AggTradeEvent
]