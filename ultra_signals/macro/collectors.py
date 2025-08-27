"""Cross-Asset Data Collectors (Sprint 42)

Lightweight async collectors using free/no-key sources (yfinance scraping
via pandas-datareader / direct HTTP) and optional Deribit public API for
options implied volatility snapshots.

Design:
- All collectors return normalised pandas Series/DataFrames with a UTC
  DatetimeIndex.
- Caching layer writes raw CSV into settings.cross_asset.cache_dir.
- Rate limiting / simple TTL handled per fetch() call.

NOTE: This is a scaffold; real implementation will flesh out each source
with retry/backoff and schema validation. We keep it minimal at first to
avoid breaking existing runtime until wired.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import time
import asyncio

import pandas as pd
try:
    from pytrends.request import TrendReq  # type: ignore
except Exception:
    TrendReq = None  # optional
import httpx
from loguru import logger

YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range}"  # noqa
DERIBIT_INDEX_URL = "https://www.deribit.com/api/v2/public/get_index_price_volatility?index_name={index}"

# Basic mapping from window label to Yahoo range/interval
_YF_INTERVAL_MAP = {
    "1d": ("1d", "1m"),
    "5d": ("5d", "5m"),
    "1mo": ("1mo", "30m"),
    "6mo": ("6mo", "1d"),
}

@dataclass
class FetchResult:
    symbol: str
    df: pd.DataFrame
    ts: float
    ok: bool
    error: Optional[str] = None

class YahooCollector:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.AsyncClient(timeout=10)

    async def close(self):
        await self._client.aclose()

    async def fetch(self, symbol: str, interval: str = "5m", rng: str = "5d") -> FetchResult:
        url = YF_CHART_URL.format(symbol=symbol, interval=interval, range=rng)
        try:
            r = await self._client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            raw = r.json()
            res = raw.get("chart", {}).get("result", [])
            if not res:
                return FetchResult(symbol, pd.DataFrame(), time.time(), False, "empty result")
            data = res[0]
            ts_list = data.get("timestamp") or []
            indicators = data.get("indicators", {}).get("quote", [{}])[0]
            vol = indicators.get("volume", [])
            opens = indicators.get("open", [])
            highs = indicators.get("high", [])
            lows = indicators.get("low", [])
            closes = indicators.get("close", [])
            rows = []
            for i, ts in enumerate(ts_list):
                try:
                    rows.append((pd.to_datetime(ts, unit="s"), opens[i], highs[i], lows[i], closes[i], vol[i]))
                except Exception:
                    continue
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
            # cache
            if not df.empty:
                out = self.cache_dir / f"yf_{symbol.replace('/', '_')}.csv"
                try:
                    df.to_csv(out)
                except Exception:
                    pass
            return FetchResult(symbol, df, time.time(), True)
        except Exception as e:
            logger.debug(f"Yahoo fetch error {symbol}: {e}")
            return FetchResult(symbol, pd.DataFrame(), time.time(), False, str(e))

class DeribitIVCollector:
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=10)

    async def close(self):
        await self._client.aclose()

    async def fetch_iv(self, index: str = "btc_usd") -> Optional[float]:
        url = DERIBIT_INDEX_URL.format(index=index)
        try:
            r = await self._client.get(url)
            r.raise_for_status()
            js = r.json()
            return float(js.get('result', {}).get('volatility', None))
        except Exception as e:
            logger.debug(f"Deribit IV fetch error {index}: {e}")
            return None

async def fetch_deribit_iv(indices: List[str]) -> Dict[str, float]:
    col = DeribitIVCollector()
    try:
        tasks = [col.fetch_iv(idx) for idx in indices]
        out: Dict[str, float] = {}
        for coro, name in zip(asyncio.as_completed(tasks), indices):
            val = await coro
            if val is not None:
                out[name] = val
        return out
    finally:
        await col.close()

# Google Trends (blocking inside thread for simplicity)
def fetch_google_trends(keywords: List[str], geo: str = "") -> Dict[str, float]:  # pragma: no cover (network)
    out: Dict[str, float] = {}
    if not keywords or TrendReq is None:
        return out
    try:
        pytrends = TrendReq(hl='en-US', tz=0)
        pytrends.build_payload(keywords, timeframe='now 7-d', geo=geo)
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return out
        for kw in keywords:
            if kw in df.columns:
                try:
                    out[kw] = float(df[kw].iloc[-1])
                except Exception:
                    continue
        return out
    except Exception as e:
        logger.debug(f"Google Trends fetch error: {e}")
        return out

# Macro calendar parser placeholder (RSS / ICS etc.)
def parse_macro_calendar(raw_items: List[dict]) -> List[dict]:
    # Placeholder: simply echo; real implementation would normalize event time, importance, symbol impact.
    return raw_items or []

async def batch_fetch_yahoo(symbols: List[str], interval: str, rng: str, cache_dir: str) -> Dict[str, pd.DataFrame]:
    yc = YahooCollector(cache_dir)
    try:
        tasks = [yc.fetch(sym, interval=interval, rng=rng) for sym in symbols]
        out: Dict[str, pd.DataFrame] = {}
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if res.ok and not res.df.empty:
                out[res.symbol] = res.df
        return out
    finally:
        await yc.close()
