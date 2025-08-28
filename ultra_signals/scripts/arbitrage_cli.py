"""Simple CLI runner to fetch arbitrage features and optionally post to Telegram.

Usage: run from repo root with python -u ultra_signals/scripts/arbitrage_cli.py
"""
import asyncio, sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
from ultra_signals.venues.bybit_perp import BybitPerpPaper
from ultra_signals.venues.okx_swap import OKXSwapPaper
from ultra_signals.venues.symbols import SymbolMapper
from ultra_signals.features.arbitrage_view import ArbitrageFeatureView
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.arbitrage.telegram_sender import TelegramSender

async def main():
    mapper = SymbolMapper()
    venues = {
        'binance_usdm': BinanceUSDMPaper(mapper),
        'bybit_perp': BybitPerpPaper(mapper),
        'okx_swap': OKXSwapPaper(mapper),
    }
    cfg = {'notional_buckets_usd': [5000,25000], 'min_after_cost_bps': 0.5}
    venue_regions = {k:'ASIA' for k in venues.keys()}
    store = FeatureStore(warmup_periods=2, settings={'features': {'warmup_periods':1}})
    view = ArbitrageFeatureView(store, venues, mapper, cfg, venue_regions)
    tsym = sys.argv[1] if len(sys.argv) > 1 else 'BTCUSDT'
    telegram = TelegramSender(dry_run=True)
    while True:
        feats = await view.run_once([tsym])
        if feats:
            fs = feats[0]
            await telegram.send(fs)
        await asyncio.sleep(5)

if __name__ == '__main__':
    asyncio.run(main())
