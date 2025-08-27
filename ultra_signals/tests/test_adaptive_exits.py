import pandas as pd
from ultra_signals.engine.adaptive_exits import generate_adaptive_exits

BASE_SETTINGS = {
    'risk': {
        'adaptive_exits': {
            'enabled': True,
            'atr_lookback': 5,
            'atr_mult_stop': 1.0,
            'atr_mult_target': 2.0,
            'min_stop_pct': 0.001,
            'max_stop_pct': 0.05,
            'regime_multiplier': {
                'high_vol': 2.0,
                'low_vol': 0.5,
                'chop': 0.8,
                'trending': 1.5,
            },
            'structural_confluence': True,
            'swing_lookback': 4,
            'breakeven_enable': True,
            'breakeven_trigger_rr': 1.2,
            'trailing_enable': True,
            'trailing_type': 'atr',
            'trailing_step_mult': 1.0,
            'partial_tp': {
                'enabled': True,
                'levels': [ {'rr':1.5,'pct':0.5}, {'rr':3.0,'pct':0.5} ]
            }
        }
    }
}


def _mock_ohlcv(n=30, start=100.0, step=0.5):
    rows = []
    price = start
    for i in range(n):
        high = price + step
        low = price - step
        close = price + (step/2 if i % 2 == 0 else -step/2)
        rows.append({'high': high, 'low': low, 'close': close})
        price += (0.2 if i % 3 else -0.1)
    return pd.DataFrame(rows)


def test_atr_calculation_and_basic_fields():
    df = _mock_ohlcv(40)
    exits = generate_adaptive_exits('X','LONG', price=float(df.iloc[-1]['close']), ohlcv_tail=df, regime_info={}, settings=BASE_SETTINGS)
    assert exits is not None
    assert exits['stop_price'] < float(df.iloc[-1]['close'])
    assert exits['target_price'] > float(df.iloc[-1]['close'])
    assert exits['meta']['atr'] > 0


def test_regime_multipliers_trending_vs_chop():
    df = _mock_ohlcv(40)
    price = float(df.iloc[-1]['close'])
    trend = generate_adaptive_exits('X','LONG', price, df, {'profile':'trend'}, BASE_SETTINGS)
    chop = generate_adaptive_exits('X','LONG', price, df, {'profile':'chop'}, BASE_SETTINGS)
    # trending widens => expect larger target distance (greater target_price)
    assert trend['target_price'] - price > chop['target_price'] - price or trend['meta']['atr_mult_target_eff'] > chop['meta']['atr_mult_target_eff']


def test_structural_snapping_closer_swing_low():
    df = _mock_ohlcv(40)
    # Force a very close swing low by editing last bar low
    price = float(df.iloc[-1]['close'])
    swing_low = price - 0.05
    df.iloc[-4:, df.columns.get_loc('low')] = swing_low  # create local cluster of lows
    exits = generate_adaptive_exits('X','LONG', price, df, {'profile':'trend'}, BASE_SETTINGS)
    assert exits['meta']['struct_used'] is True
    # stop should be near swing_low
    assert abs(exits['stop_price'] - swing_low) < 1e-6


def test_partial_takeprofits_levels():
    df = _mock_ohlcv(40)
    price = float(df.iloc[-1]['close'])
    exits = generate_adaptive_exits('X','LONG', price, df, {}, BASE_SETTINGS)
    assert len(exits['partial_tp']) == 2
    # ensure R:R mapping monotonic
    rr_prices = [p['price'] for p in exits['partial_tp']]
    assert rr_prices[0] < rr_prices[1]


def test_trailing_config_present():
    df = _mock_ohlcv(40)
    price = float(df.iloc[-1]['close'])
    exits = generate_adaptive_exits('X','SHORT', price, df, {}, BASE_SETTINGS)
    assert exits['trail_config']['enabled']
    assert exits['trail_config']['type'] == 'atr'
    assert exits['trail_config']['step'] > 0
