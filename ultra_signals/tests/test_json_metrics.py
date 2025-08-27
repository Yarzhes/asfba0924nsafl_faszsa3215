import pandas as pd
from ultra_signals.backtest.json_metrics import build_run_metrics, build_wf_metrics, CORE_FIELDS

def _sample_trades():
    # simple alternating wins/losses with rr column
    data = [
        {'pnl': 100, 'fees': 1.2, 'slippage_bps': 5, 'rr': 2.0},
        {'pnl': -50, 'fees': 0.8, 'slippage_bps': 4, 'rr': -1.0},
        {'pnl': 120, 'fees': 1.5, 'slippage_bps': 6, 'rr': 2.4},
        {'pnl': -30, 'fees': 0.5, 'slippage_bps': 3, 'rr': -0.6},
    ]
    return pd.DataFrame(data)

def test_build_run_metrics_fields():
    trades = _sample_trades()
    kpis = {
        'profit_factor': 2.5,
        'sharpe': 1.2,
        'max_drawdown': -80,
        'win_rate_pct': 50.0,
        'total_trades': len(trades),
        'total_pnl': float(trades['pnl'].sum()),
    }
    equity_curve = [0,100,50,170,140]
    settings = {'backtest': {'execution': {'initial_capital': 10000}}}
    m = build_run_metrics(kpis, trades, equity_curve, settings, 'BTCUSDT','5m')
    for f in CORE_FIELDS:
        assert f in m, f"Missing field {f} in run metrics"
    # sanity: values numeric
    for k,v in m.items():
        if k in CORE_FIELDS:
            assert isinstance(v,(int,float)), f"Field {k} not numeric"


def test_build_wf_metrics_fields():
    trades = _sample_trades()
    m = build_wf_metrics(trades, 'ETHUSDT','15m')
    for f in CORE_FIELDS:
        assert f in m, f"Missing field {f} in wf metrics"
    # ensure streak metrics reasonable
    assert m['max_consec_wins'] >= 1
    assert m['max_consec_losses'] >= 1


def test_empty_inputs_produce_zero_payloads():
    empty = pd.DataFrame()
    kpis = {'profit_factor':0,'sharpe':0,'max_drawdown':0,'win_rate_pct':0,'total_trades':0,'total_pnl':0}
    m1 = build_run_metrics(kpis, empty, [], {'backtest':{'execution':{'initial_capital':10000}}}, 'XRPUSDT','1h')
    m2 = build_wf_metrics(empty, 'XRPUSDT','1h')
    for m in (m1,m2):
        for v in m.values():
            assert v == 0 or isinstance(v,str) or v == m.get('symbol') or v == m.get('timeframe')
