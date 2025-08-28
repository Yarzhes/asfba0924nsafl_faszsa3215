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
        # internal cache may hold per-symbol list OR per-venue dict of lists
        # structure: _cache[symbol] -> list[dict] OR {venue: [dict,...]}
        self._cache = {}
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

            # Try to handle either single-venue list or multi-venue dict
            for item in data:
                symbol = item.get("symbol")
                if not symbol:
                    logger.warning("Found funding rate item with no symbol.")
                    continue
                history_item = {
                    "funding_rate": float(item.get("fundingRate", 0.0)),
                    "funding_time": int(item.get("fundingTime", 0)),
                    "venue": item.get("venue") or item.get("exchange") or "unknown",
                }
                # if we already have a per-venue dict, append
                cur = self._cache.get(symbol)
                if isinstance(cur, dict):
                    v = history_item.get('venue') or 'unknown'
                    cur.setdefault(v, []).append(history_item)
                    self._cache[symbol] = cur
                else:
                    # start as simple list when no venue info
                    if history_item.get('venue') == 'unknown':
                        self._cache[symbol] = [history_item]
                    else:
                        # convert existing list into dict under inferred venue if present
                        d = {}
                        if isinstance(cur, list):
                            d.setdefault('unknown', []).extend(cur)
                        d.setdefault(history_item.get('venue'), []).append(history_item)
                        self._cache[symbol] = d

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
        cur = self._cache.get(symbol)
        if cur is None:
            return None
        if isinstance(cur, dict):
            # flatten: prefer last value per-venue concatenated
            out = []
            for v, lst in cur.items():
                out.extend(lst or [])
            # sort by funding_time
            try:
                out = sorted(out, key=lambda x: int(x.get('funding_time', 0)))
            except Exception:
                pass
            return out
        return cur

    def get_per_venue_history(self, symbol: Symbol) -> Optional[Dict[str, List[Dict]]]:
        cur = self._cache.get(symbol)
        if cur is None:
            return None
        if isinstance(cur, dict):
            return cur
        return {'unknown': cur}

    def get_predicted(self, symbol: Symbol) -> Optional[float]:
        """Return a best-effort predicted next funding rate for symbol (if available).
        This is a lightweight heuristic (last value or average delta) unless a real predictor
        is implemented in the provider subclass.
        """
        try:
            hist = self.get_history(symbol) or []
            if not hist:
                return None
            # naive: use last observed funding_rate
            last = hist[-1].get('funding_rate')
            return float(last) if last is not None else None
        except Exception:
            return None

    def load_replay(self, symbol: Symbol, records: List[Dict]) -> None:
        """
        Load a list of funding/OI records into the internal cache for replay/testing.

        Each record should be a dict containing at least:
          - 'funding_rate' (float)
          - 'funding_time' (int ms epoch) or 'funding_time_ms'
          - optional 'venue' and 'oi_notional'

        This replaces any existing cache entry for the symbol with a sorted list.
        """
        try:
            if not records:
                return
            out = []
            for r in records:
                fr = float(r.get('funding_rate') or r.get('fund') or 0.0)
                ft = r.get('funding_time') or r.get('funding_time_ms') or r.get('ts') or None
                try:
                    ft = int(ft) if ft is not None else None
                except Exception:
                    ft = None
                venue = r.get('venue') or r.get('exchange') or 'unknown'
                item = {'funding_rate': fr, 'funding_time': ft or 0, 'venue': venue}
                if 'oi_notional' in r:
                    try:
                        item['oi_notional'] = float(r.get('oi_notional'))
                    except Exception:
                        pass
                out.append(item)
            out = sorted(out, key=lambda x: int(x.get('funding_time', 0)))
            self._cache[symbol] = out
        except Exception:
            logger.exception('FundingProvider.load_replay failed')