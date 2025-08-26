import pandas as pd
from ultra_signals.calibration.objective import _profit_factor, evaluate_candidate


def test_grade_distribution_metrics(monkeypatch):
    # Build fake settings minimal
    settings = {
        'runtime': {'symbols':['BTCUSDT'], 'primary_timeframe':'5m'},
        'features': {'warmup_periods': 5},
        'backtest': {'start_date':'2023-01-01','end_date':'2023-01-05'},
        'walkforward': { 'window': {'train_period':'2d','test_period':'1d','advance_by':'1d'}, 'data_rules': {'purge_period':'0d'} },
    }
    # Monkeypatch WalkForwardAnalysis.run to return synthetic trades & kpi summary
    fake_trades = pd.DataFrame({
        'pnl':[1,-0.5, 2, -1, 0.5, 1.2],
        'ts_exit':[1,2,3,4,5,6],
        'bin':['A+','A','B','C','D','A+']
    })
    fake_kpis = pd.DataFrame({'profit_factor':[2.0,1.5]})
    class DummyWFA:
        def __init__(self, *a, **k): pass
        def run(self, *a, **k):
            return fake_trades, fake_kpis
    monkeypatch.setattr('ultra_signals.calibration.objective.WalkForwardAnalysis', DummyWFA)
    metrics = evaluate_candidate(settings, {})
    assert metrics['trades'] == 6
    # Distribution keys
    assert 'grade_Aplus_pct' in metrics
    assert abs(metrics['grade_Aplus_pct'] - (2/6)) < 1e-9
    assert metrics['grade_good_poor_ratio'] > 0
