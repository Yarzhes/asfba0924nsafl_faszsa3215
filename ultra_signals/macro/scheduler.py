"""Async scheduler for periodic cross-asset data refresh (Sprint 42).

Usage:
    scheduler = MacroScheduler(settings, feature_store)
    await scheduler.start()

It populates FeatureStore._macro_external_frames (dict symbol->DataFrame) so
FeatureStore macro hook can compute features on next bar.

Lightweight; production version should include cancellation, jitter, backoff.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List
from loguru import logger
import pandas as pd

from ultra_signals.macro.collectors import batch_fetch_yahoo

class MacroScheduler:
    def __init__(self, settings: Dict, feature_store):
        self.settings = settings
        self.feature_store = feature_store
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        ca = (self.settings.get('cross_asset') or {}) if isinstance(self.settings, dict) else {}
        if not ca.get('enabled'):
            logger.info("MacroScheduler not started (cross_asset disabled)")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("MacroScheduler started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            logger.info("MacroScheduler stopped.")

    async def _run_loop(self):
        ca = self.settings.get('cross_asset')
        refresh_min = int(ca.get('refresh_min', 5))
        tickers_cfg = ca.get('tickers') or {}
        # Flatten list of tickers to fetch
        symbols: List[str] = []
        for key in ['equities','fx','rates','commodities','volatility']:
            vals = tickers_cfg.get(key) or []
            for v in vals:
                if v not in symbols:
                    symbols.append(v)
        interval = '5m'
        rng = '5d'
        while self._running:
            start = time.time()
            try:
                data = await batch_fetch_yahoo(symbols, interval, rng, ca.get('cache_dir', '.cache/cross_asset'))
                # Normalize to OHLCV style DataFrames (already by collector)
                # Inject into FeatureStore for macro engine consumption
                try:
                    setattr(self.feature_store, '_macro_external_frames', data)
                except Exception:
                    pass
                logger.debug(f"MacroScheduler fetched {len(data)} external series")
            except Exception as e:
                logger.warning(f"MacroScheduler fetch error: {e}")
            # sleep remaining
            elapsed = time.time() - start
            wait = max(10.0, refresh_min*60 - elapsed)
            await asyncio.sleep(wait)
