from typing import Dict, List
from .types import AggregatedBook, L2Book, PriceLevel, VenueInfo


class Aggregator:
    """Simple aggregator that holds per-venue L2 snapshots and provides
    helpers to walk depth for a target notional.
    """

    def __init__(self):
        self._books: Dict[str, L2Book] = {}

    def update(self, venue: str, book: L2Book):
        self._books[venue] = book

    def snapshot(self, symbol: str) -> AggregatedBook:
        return AggregatedBook(symbol=symbol, books=self._books.copy())

    @staticmethod
    def depth_cost(book: L2Book, side: str, target_notional: float) -> float:
        """Walk the book on side ('buy' means consume asks) and return
        VWAP price to fill target_notional. If not enough liquidity, return inf.
        """
        remaining = target_notional
        weighted = 0.0
        levels = book.asks if side == 'buy' else book.bids
        for lvl in levels:
            take = min(remaining, lvl.size * lvl.price)
            weighted += take
            remaining -= take
            if remaining <= 1e-12:
                break
        if remaining > 1e-8:
            return float('inf')
        # vwap is weighted / notional
        return weighted / target_notional
