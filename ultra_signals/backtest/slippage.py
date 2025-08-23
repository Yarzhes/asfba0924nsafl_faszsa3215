import numpy as np
from typing import Dict, Any

def get_slippage_model(config: Dict[str, Any]):
    """Factory function to get the configured slippage model."""
    model_name = config.get("slippage_model", "none")
    if model_name == "atr":
        return ATROrderSlippage(config)
    elif model_name == "book_proxy":
        return BookProxySlippage(config)
    elif model_name == "none":
        return NoSlippage()
    raise ValueError(f"Unknown slippage model: {model_name}")

class SlippageModel:
    """Base class for slippage models."""
    def calculate(self, price: float, **kwargs) -> float:
        raise NotImplementedError

class NoSlippage(SlippageModel):
    """A model that applies no slippage."""
    def calculate(self, price: float, **kwargs) -> float:
        return price

class ATROrderSlippage(SlippageModel):
    """
    Calculates slippage as a multiplier of the Average True Range (ATR).
    A simple but effective model for markets without deep order book data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.atr_multiplier = float(config.get("atr_slippage_multiplier", 0.1))

    def calculate(self, price: float, atr: float, side: str, **kwargs) -> float:
        """
        Args:
            price: The ideal execution price.
            atr: The current ATR value.
            side: 'BUY' or 'SELL'.
        """
        slippage_amount = atr * self.atr_multiplier
        return price + slippage_amount if side == 'BUY' else price - slippage_amount

class BookProxySlippage(SlippageModel):
    """
    A simplified model that mimics order book impact.
    Assumes slippage increases with trade size relative to top-of-book depth.
    """
    def __init__(self, config: Dict[str, Any]):
        # Base slippage for a small trade in basis points
        self.base_slippage_bps = float(config.get("book_base_slippage_bps", 0.5))
        # How much slippage scales with size
        self.size_sensitivity = float(config.get("book_size_sensitivity", 1.5))

    def calculate(self, price: float, trade_size: float, book_depth: float, side: str, **kwargs) -> float:
        """
        Args:
            price: The ideal execution price.
            trade_size: The size of the trade being executed.
            book_depth: The available liquidity at the top of the book.
            side: 'BUY' or 'SELL'.
        """
        if book_depth == 0:
            return price # Avoid division by zero
            
        size_ratio = trade_size / book_depth
        # Slippage (in bps) increases non-linearly with the size ratio
        slippage_bps = self.base_slippage_bps * (1 + size_ratio) ** self.size_sensitivity
        
        slippage_pct = slippage_bps / 10000.0
        
        if side == 'BUY':
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)
