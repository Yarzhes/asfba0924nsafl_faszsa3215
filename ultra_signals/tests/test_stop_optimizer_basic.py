import pandas as pd
from ultra_signals.opt.stop_optimizer import optimize

SETTINGS = {
    'auto_stop_opt': {
        'enabled': True,
        'mode': 'atr',
        'grid': { 'atr_mults': [0.6, 0.8], 'pct_mults': [0.5, 1.0] },
    'constraints': { 'min_winrate': 0.0, 'max_mdd_pct': 100.0, 'min_trades': 1 },
        'validation': { 'slices': 3 },
        'objective': 'expectancy',
        'tie_breaker': 'pf'
    }
}

def _mock_trades():
    # generate synthetic trades with varying atr and pnl
    rows = []
    import random
    random.seed(0)
    for i in range(30):
        side = 'LONG'
        atr = 10 + (i % 5)
        pnl = (1 if i % 3 else -1) * (10 + i % 4)
        rows.append({
            'ts_entry': i,
            'symbol': 'BTCUSDT',
            'tf': '5m',
            'regime': 'trend' if i % 2 else 'chop',
            'entry_price': 1000 + i,
            'atr': atr,
            'pnl': pnl,
            'rr': pnl / (atr if atr else 1),
            'ae': atr * 0.5,
        })
    return pd.DataFrame(rows)


def test_optimize_returns_table():
    df = _mock_trades()
    out = optimize(df, SETTINGS, return_candidates=True)
    assert 'table' in out
    assert out['table']
    # ensure at least one bucket present
    any_bucket = next(iter(out['table'].values()))
    assert isinstance(any_bucket, dict)


def test_constraints_filtering():
    cfg = SETTINGS.copy()
    cfg['auto_stop_opt'] = dict(cfg['auto_stop_opt'])
    cfg['auto_stop_opt']['constraints'] = {'min_winrate': 0.9, 'min_trades': 100, 'max_mdd_pct': 5}
    df = _mock_trades()
    out = optimize(df, cfg, return_candidates=True)
    # Expect possibly empty table due to harsh constraints
    assert isinstance(out['table'], dict)


def test_inheritance_mixed():
    # create only mixed regime trades -> ensure regimes filled
    df = _mock_trades()
    df['regime'] = 'mixed'
    out = optimize(df, SETTINGS, return_candidates=False)
    for sym, tfs in out.items():
        for tf, regimes in tfs.items():
            assert 'trend' in regimes and 'chop' in regimes
