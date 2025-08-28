import pytest, asyncio
from ultra_signals.arbitrage.models import ArbitrageFeatureSet
from ultra_signals.arbitrage.analyzer import ArbitrageAnalyzer
from ultra_signals.arbitrage.collector import ArbitrageCollector
from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
from ultra_signals.venues.bybit_perp import BybitPerpPaper
from ultra_signals.venues.okx_swap import OKXSwapPaper
from ultra_signals.venues.symbols import SymbolMapper
from ultra_signals.features.arbitrage_view import ArbitrageFeatureView
from ultra_signals.core.feature_store import FeatureStore

@pytest.mark.asyncio
async def test_arbitrage_pipeline_smoke():
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
    assert feats, 'Expected feature set'
    fs = feats[0]
    assert fs.executable_spreads, 'Should compute spreads'
    d = fs.to_feature_dict()
    assert any(k.startswith('arb_spread_exec_bps') for k in d.keys())
