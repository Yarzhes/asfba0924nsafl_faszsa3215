"""Arbitrage feature view integration.

Provides a helper to run the ArbitrageCollector + ArbitrageAnalyzer and merge the
result into an existing FeatureStore cache under a synthetic timeframe 'arb'.

This keeps consistency with other feature groups without altering core store
logic extensively.
"""
from __future__ import annotations
import asyncio
import pandas as pd
from typing import Dict, Any, List
from loguru import logger
from ultra_signals.arbitrage.collector import ArbitrageCollector
from ultra_signals.arbitrage.analyzer import ArbitrageAnalyzer
from ultra_signals.arbitrage.models import ArbitrageFeatureSet

class ArbitrageFeatureView:
    def __init__(self, store, venues: Dict[str, Any], symbol_mapper, config: dict, venue_regions: Dict[str,str]):
        self._store = store
        self._collector = ArbitrageCollector(venues, symbol_mapper, config)
        self._analyzer = ArbitrageAnalyzer(config, venue_regions)
        self._config = config

    async def run_once(self, symbols: List[str]) -> List[ArbitrageFeatureSet]:
        snap = await self._collector.collect_all(symbols)
        out: List[ArbitrageFeatureSet] = []
        for sym in symbols:
            fs = self._analyzer.build_feature_set(
                symbol=sym,
                quotes=[q for q in snap['quotes'] if q.symbol == sym],
                depth=[d for d in snap['depth'] if d.symbol == sym],
                funding=[f for f in snap['funding'] if f.symbol == sym],
                ts=snap['ts'],
            )
            out.append(fs)
            # Inject into FeatureStore under synthetic timeframe 'arb'
            try:
                ts_dt = pd.to_datetime(fs.ts, unit='ms')
                bucket = self._store._feature_cache.setdefault(sym, {}).setdefault('arb', {})
                bucket.setdefault(ts_dt, {})['arbitrage'] = fs.to_feature_dict()
            except Exception as e:
                logger.debug('arb feature inject failed: {}', e)
        return out

    async def loop(self, symbols: List[str]):  # pragma: no cover (runtime loop)
        interval = float(self._config.get('poll_interval_sec', 2.0))
        while True:
            try:
                await self.run_once(symbols)
            except Exception as e:
                logger.exception('Arb loop error: {}', e)
            await asyncio.sleep(interval)
