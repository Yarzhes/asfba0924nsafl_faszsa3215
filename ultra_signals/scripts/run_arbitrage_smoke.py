import asyncio
import sys, os
# allow running from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ultra_signals.arbitrage.collector import ArbitrageCollector
from ultra_signals.arbitrage.analyzer import ArbitrageAnalyzer
from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
from ultra_signals.venues.bybit_perp import BybitPerpPaper
from ultra_signals.venues.okx_swap import OKXSwapPaper
from ultra_signals.venues.symbols import SymbolMapper
from ultra_signals.features.arbitrage_view import ArbitrageFeatureView
from ultra_signals.core.feature_store import FeatureStore

async def main():
    mapper = SymbolMapper()
    venues = {
        'binance_usdm': BinanceUSDMPaper(mapper),
        'bybit_perp': BybitPerpPaper(mapper),
        'okx_swap': OKXSwapPaper(mapper),
    }
    cfg = {
        'notional_buckets_usd': [5000,25000],
        'min_after_cost_bps': 0.1,
        'geo_baskets': {'ASIA': ['binance_usdm','bybit_perp','okx_swap'], 'US': []},
    }
    venue_regions = {k:'ASIA' for k in venues.keys()}
    store = FeatureStore(warmup_periods=2, settings={'features': {'warmup_periods':1}})
    view = ArbitrageFeatureView(store, venues, mapper, cfg, venue_regions)
    feats = await view.run_once(['BTCUSDT'])
    print('Feature sets returned:', len(feats))
    if feats:
        fs = feats[0]
        print(fs.to_feature_dict())

if __name__ == '__main__':
    asyncio.run(main())
