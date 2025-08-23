"""
Abstract Base Class and factory for Open Interest data providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

class OIProvider(ABC):
    """Abstract Base Class for Open Interest data providers."""

    @abstractmethod
    async def fetch_oi_delta(self, symbol: str, windows: List[str]) -> Dict[str, float]:
        """
        Fetches the percentage change in Open Interest over specified time windows.

        Args:
            symbol: The market symbol (e.g., "BTCUSDT").
            windows: A list of window strings (e.g., ["1m", "5m", "15m"]).

        Returns:
            A dictionary mapping each window to its OI delta.
            Example: {"1m": 0.05, "5m": -0.12, "15m": 0.25}
            Should return zero values if the provider fails, not raise an exception.
        """
        pass

def make_provider(name: str, **kwargs) -> "OIProvider":
    """
    Factory function to create an instance of a specific OI provider.

    Args:
        name: The name of the provider (e.g., "coinglass", "mock").
        **kwargs: Provider-specific arguments (e.g., api_key).

    Returns:
        An instance of a class that implements the OIProvider interface.

    Raises:
        ValueError: If the provider name is unknown.
    """
    # NOTE: Implementation will be added in subsequent tasks.
    # For now, we only support a mock provider.
    if name == "mock":
        # This avoids a circular import issue
        from .mock import MockOIProvider
        return MockOIProvider(**kwargs)
    
    raise ValueError(f"Unknown OI provider specified: {name}")