from ultra_signals.dq import time_sync, validators, gap_filler, normalizer, aligners, venue_merge
from ultra_signals.guards import live_guards
import pandas as pd
import numpy as np
import time

class DummyFetcher:
    def __call__(self, symbol: str, ts: int):
        # return synthetic row
        return {"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}

def test_time_sync_skew_calc():
    def fake_server_time(v):
        return 1_000_000
    time_sync.configure(fake_server_time)
    time_sync.poll_venue('BINANCE', {})
    assert isinstance(time_sync.get_skew_ms('BINANCE'), float)


def test_validate_ohlcv_monotonicity():
    df = pd.DataFrame({
        'ts': [1000, 900, 1100],
        'open': [1,1,1], 'high': [1,1,1], 'low': [1,1,1], 'close':[1,1,1], 'volume':[1,1,1]
    })
    rep = validators.validate_ohlcv_df(df, 100, {'data_quality': {}}, 'BTCUSDT','BINANCE')
    assert not rep.ok or rep.warnings  # out-of-order generates warning or error


def test_gap_detection_and_heal():
    df = pd.DataFrame({
        'ts': [0, 200, 400],
        'open':[1,1,1], 'high':[1,1,1], 'low':[1,1,1], 'close':[1,1,1], 'volume':[0,0,0]
    })
    healed, greport = gap_filler.heal_gaps_ohlcv(df, 'BTCUSDT', 200, DummyFetcher(), {'data_quality': {'gap_policy': {'heal_backfill_bars': 10}}})
    assert greport.gaps_found >= 1


def test_align_funding_to_trades():
    funding = pd.DataFrame({'ts':[1000, 2000], 'rate':[0.01, 0.02]})
    trades = pd.DataFrame({'ts':[950, 1990, 2050]})
    aligned = aligners.align_funding_to_trades(funding, trades, {'data_quality': {'aligners': {'funding_window_sec': 2, 'require_within_sec': 3}}})
    assert 'trade_ts' in aligned


def test_venue_merge_spread_guard():
    df1 = pd.DataFrame({'ts':[1000,2000], 'bid':[10,10], 'ask':[11,11]})
    df2 = pd.DataFrame({'ts':[1000,2000], 'bid':[10.5,10.5], 'ask':[11.5,11.5]})
    composite, flags = venue_merge.composite_mid({'A': df1, 'B': df2}, {'data_quality': {'multi_venue': {'max_spread_bps': 30}}})
    assert 'mid' in composite and 'venue_spread_bps' in composite


def test_symbol_normalization():
    settings = {'data_quality': {'symbols': {'map_path': 'config/symbol_map.yaml'}}}
    sym = normalizer.normalize_symbol('BTC-USDT-SWAP','OKX', settings)
    assert sym == 'BTCUSDT'


def test_heartbeat_guard_trips():
    settings = {'data_quality': {'enabled': True, 'heartbeats': {'market_data_max_silence_sec': 0}}}
    # last seen long ago
    try:
        live_guards.heartbeat_guard('market_data', last_seen_ms=0, settings=settings)
    except live_guards.CircuitBreak:
        return
    assert False, 'Expected CircuitBreak'


def test_skew_circuit_break():
    # configure skew artificially large
    def fake_server_time(v):
        # server way ahead
        return int( time_sync._wall_clock_ms() + 10_000 )  # type: ignore[attr-defined]
    time_sync.configure(fake_server_time)
    time_sync.poll_venue('BINANCE', {'data_quality': {'enabled': True}})
    settings = {'data_quality': {'enabled': True, 'max_clock_skew_ms': 100}}
    try:
        time_sync.assert_within_skew(settings, venues=['BINANCE'])
    except Exception:
        return
    assert False, 'Expected skew circuit break'
