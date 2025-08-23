import asyncio
import inspect
import logging
from typing import Dict, List, Optional

import httpx
from ultra_signals.core.custom_types import Symbol

logger = logging.getLogger(__name__)

# NOTE: The actual endpoint URL will be provided later.
# This is a placeholder for demonstration purposes.
FUNDING_RATE_API_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

class FundingProvider:
    """
    Manages fetching and caching of historical funding rate data.
    """

    def __init__(self, config: Dict):
        """
        Initializes the FundingProvider.
        
        :param config: The configuration dictionary for the funding provider.
        """
        self.refresh_interval_seconds = config.get("refresh_interval_minutes", 15) * 60
        self._cache: Dict[Symbol, List[Dict]] = {}
        self._client = httpx.AsyncClient()
        logger.info(
            f"FundingProvider initialized with a {self.refresh_interval_seconds / 60}-minute refresh interval."
        )

    async def _fetch_all_symbols(self) -> None:
        """
        Fetches funding rate history for all known symbols and updates the cache.
        """
        try:
            response = await self._client.get(FUNDING_RATE_API_URL, timeout=10)
            # Handle raise_for_status safely for both sync and async
            if hasattr(response, "raise_for_status") and callable(response.raise_for_status):
                maybe_coro = response.raise_for_status()
                if inspect.iscoroutine(maybe_coro):
                    await maybe_coro
                else:
                    response.raise_for_status()
            
            data = response.json()

            if not isinstance(data, list):
                logger.warning(f"Funding rate API did not return a list. Got: {type(data)}")
                return

            for item in data:
                symbol = item.get("symbol")
                if not symbol:
                    logger.warning("Found funding rate item with no symbol.")
                    continue

                history_item = {
                    "funding_rate": float(item.get("fundingRate", 0.0)),
                    "funding_time": int(item.get("fundingTime", 0)),
                }
                self._cache[symbol] = [history_item]

            logger.info(f"Successfully refreshed funding rate data for {len(data)} symbols.")

        except httpx.RequestError as e:
            logger.error(f"Failed to fetch funding rates: {e}. Using stale data.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during funding rate fetch: {e}")

    async def start(self) -> None:
        """
        Starts the background task for periodically refreshing funding data.
        """
        logger.info("FundingProvider background refresh task started.")
        while True:
            await self._fetch_all_symbols()
            await asyncio.sleep(self.refresh_interval_seconds)

    def get_history(self, symbol: Symbol) -> Optional[List[Dict]]:
        """
        Retrieves the cached funding rate history for a given symbol.

        :param symbol: The symbol to retrieve data for.
        :return: A list of historical funding rate data points, or None if not in cache.
        """
        return self._cache.get(symbol)