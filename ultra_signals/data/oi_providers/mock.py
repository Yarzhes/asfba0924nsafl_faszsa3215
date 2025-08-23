"""
A mock Open Interest provider for testing and development.
"""
import asyncio
import random
from typing import Dict, List

from loguru import logger

from .base import OIProvider


class MockOIProvider(OIProvider):
    """
    A mock implementation of the OIProvider that returns random data.
    This is useful for testing the OI pipeline without making live API calls.
    """

    async def fetch_oi_delta(self, symbol: str, windows: List[str]) -> Dict[str, float]:
        """
        Returns a dictionary of slightly randomized OI percentage changes.
        """
        logger.debug(f"MockOIProvider fetching data for {symbol}...")
        
        # Simulate a network delay
        await asyncio.sleep(random.uniform(0.05, 0.2))

        deltas = {}
        for window in windows:
            # Generate a random float between -0.5 and 0.5
            random_delta = random.uniform(-0.5, 0.5)
            deltas[window] = round(random_delta, 4)
            
        logger.success(f"MockOIProvider for {symbol} returned: {deltas}")
        return deltas