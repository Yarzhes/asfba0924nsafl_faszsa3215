import yaml, tempfile, os, json
from pathlib import Path
import pytest

from ultra_signals.calibration.optimizer import run_optimization

@pytest.mark.timeout(10)
def test_optimization_smoke(monkeypatch, tmp_path):
    # Minimal base settings skeleton required by evaluate_candidate / WF
    base = {
        'runtime': {'symbols':['BTCUSDT'], 'primary_timeframe':'5m'},
        'features': {'warmup_periods': 5},
        'backtest': {'start_date':'2023-01-01','end_date':'2023-01-12'},
        'walkforward': { 'window': {'train_period':'4d','test_period':'2d','advance_by':'2d'}, 'data_rules': {'purge_period':'0d'} },
    }
    cal_cfg = {
        'search_space': {
            'ensemble': { 'vote_threshold_trend': {'low':0.4,'high':0.5}, 'min_agree_trend': {'low':1,'high':2,'step':1} }
        },
        'objective': { 'weights': {'profit_factor':0.5,'winrate':0.5}, 'penalties': {'min_trades':0} },
        'runtime': { 'trials': 2, 'seed': 1, 'save_study_db': False }
    }
    out = tmp_path / 'cal'
    out.mkdir()
    result = run_optimization(base, cal_cfg, str(out))
    assert 'best' in result and result['best']['params'] is not None
    assert (out / 'leaderboard.csv').exists()
    assert (out / 'settings_autotuned.yaml').exists()


def test_pruner_presence_if_optuna_installed():
    try:
        import optuna
    except Exception:
        pytest.skip('optuna not installed')
    from ultra_signals.calibration.optimizer import HAVE_OPTUNA
    assert HAVE_OPTUNA is True
