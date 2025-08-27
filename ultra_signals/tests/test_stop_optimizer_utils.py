import pandas as pd
from ultra_signals.opt.stop_optimizer import diff_tables, compute_before_after, bootstrap_expectancy_diff, micro_replay


def test_diff_tables_detects_change():
    old = {'BTCUSDT': {'5m': {'trend': {'mode':'atr','value':1.0}}}}
    new = {'BTCUSDT': {'5m': {'trend': {'mode':'atr','value':1.2}}}}
    changes = diff_tables(old, new)
    assert any(c['regime']=='trend' for c in changes)


def test_bootstrap_expectancy_diff():
    # synthetic base / opt
    base = pd.DataFrame({'pnl':[1,-1,1,-1]})
    opt = pd.DataFrame({'pnl':[2,-1,2,1]})
    p = bootstrap_expectancy_diff(base, opt, iters=50)
    assert 0.0 <= p <= 1.0


def test_compute_before_after_basic():
    trades = []
    for i in range(10):
        trades.append({'ts_entry':i,'ts_exit':i+1,'symbol':'X','tf':'5m','regime':'trend','entry_price':100,'exit_price':101,'side':'LONG','atr':2.0,'pnl': (1 if i%2 else -1)*5,'rr':1})
    df = pd.DataFrame(trades)
    table = {'X': {'5m': {'trend': {'mode':'atr','value':1.0}}}}
    out = compute_before_after(df, table, base_atr_mult=1.0, bootstrap=True, iters=20)
    assert not out.empty
    assert 'p_value_improve' in out.columns
