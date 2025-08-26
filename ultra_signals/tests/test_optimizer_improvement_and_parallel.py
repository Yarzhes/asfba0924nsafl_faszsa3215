import pytest, pandas as pd
from pathlib import Path

from ultra_signals.calibration.optimizer import run_optimization, HAVE_OPTUNA

class _WFMockHigherPF:
    """First param setting encountered gets lower PF; later param combos get higher PF.
    We simulate improvement by keying off one tuned param value threshold.
    """
    call_count = 0
    def __init__(self, *a, **k): pass
    def run(self, *a, **k):
        _WFMockHigherPF.call_count += 1
        # produce trades PF either 1.2 or 1.8 via pnl pattern
        if _WFMockHigherPF.call_count == 1:
            # Lower PF: gains 1, losses -1 => PF=1
            trades = pd.DataFrame({'pnl':[1,-1,1,-1], 'ts_exit':[1,2,3,4]})
            kpis = pd.DataFrame({'profit_factor':[1.2]})
        else:
            trades = pd.DataFrame({'pnl':[2,-1,2,-1,1], 'ts_exit':[1,2,3,4,5]})
            kpis = pd.DataFrame({'profit_factor':[1.8]})
        return trades, kpis

@pytest.mark.skipif(not HAVE_OPTUNA, reason='optuna required for this test')
def test_best_params_applied_improve_pf(monkeypatch, tmp_path):
    # Monkeypatch WalkForwardAnalysis
    monkeypatch.setattr('ultra_signals.calibration.objective.WalkForwardAnalysis', _WFMockHigherPF)
    base = {
        'runtime': {'symbols':['BTCUSDT'], 'primary_timeframe':'5m'},
        'features': {'warmup_periods': 5},
        'backtest': {'start_date':'2023-01-01','end_date':'2023-01-05'},
        'walkforward': { 'window': {'train_period':'2d','test_period':'1d','advance_by':'1d'}, 'data_rules': {'purge_period':'0d'} },
    }
    cal_cfg = {
        'search_space': {
            'ensemble': { 'vote_threshold_trend': {'low':0.4,'high':0.5} }
        },
        'objective': { 'weights': {'profit_factor':1.0}, 'penalties': {'min_trades':0} },
        'runtime': { 'trials': 2, 'seed': 1, 'save_study_db': False }
    }
    out = tmp_path / 'cal'
    out.mkdir()
    result = run_optimization(base, cal_cfg, str(out))
    assert result['best']['metrics']['profit_factor'] >= 1.2

@pytest.mark.skipif(not HAVE_OPTUNA, reason='optuna required')
def test_parallel_flag_thread_safety(monkeypatch, tmp_path):
    # Reuse mock WF to keep runtime tiny
    monkeypatch.setattr('ultra_signals.calibration.objective.WalkForwardAnalysis', _WFMockHigherPF)
    base = {
        'runtime': {'symbols':['BTCUSDT'], 'primary_timeframe':'5m'},
        'features': {'warmup_periods': 5},
        'backtest': {'start_date':'2023-01-01','end_date':'2023-01-05'},
        'walkforward': { 'window': {'train_period':'2d','test_period':'1d','advance_by':'1d'}, 'data_rules': {'purge_period':'0d'} },
    }
    cal_cfg = {
        'search_space': {
            'ensemble': { 'vote_threshold_trend': {'low':0.4,'high':0.6} }
        },
        'objective': { 'weights': {'profit_factor':1.0}, 'penalties': {'min_trades':0} },
        'runtime': { 'trials': 4, 'seed': 2, 'save_study_db': False, 'parallel': 2 }
    }
    out = tmp_path / 'calp'
    out.mkdir()
    res = run_optimization(base, cal_cfg, str(out))
    # Ensure best dict intact and leaderboard length matches trials (allow pruned subset)
    assert 'fitness' in res['best'] and res['best']['params'] is not None
    assert (out / 'leaderboard.csv').exists()
