import pandas as pd, yaml, os, json
from pathlib import Path
from ultra_signals.opt.stop_optimizer import optimize, write_table

# Simplified integration: generate synthetic trades, run optimize, assert outputs.

def test_stop_opt_cli_like_flow(tmp_path):
    settings = {
        'auto_stop_opt': {
            'enabled': True,
            'mode': 'atr',
            'grid': {'atr_mults': [0.6,1.0], 'pct_mults': [0.5]},
            'constraints': {'min_trades':1},
            'validation': {'slices': 2, 'purge_bars':0,'embargo_bars':0},
            'objective': 'expectancy'
        },
        'risk': {'adaptive_exits': {'atr_mult_stop': 1.2}}
    }
    # synthetic trades
    rows = []
    for i in range(12):
        rows.append({'ts_entry': i, 'ts_exit': i+1, 'symbol': 'BTCUSDT', 'tf': '5m', 'regime': 'trend', 'entry_price': 100+i, 'exit_price': 101+i, 'side':'LONG', 'atr': 2.0, 'pnl': (1 if i%2 else -1)*5, 'rr': (1 if i%2 else -1) * 1.5})
    df = pd.DataFrame(rows)
    out = optimize(df, settings, return_candidates=True)
    assert out['table']
    # write table and ensure file created
    out_file = tmp_path / 'stop_table.yaml'
    write_table(out['table'], out_file.as_posix())
    assert out_file.exists()
