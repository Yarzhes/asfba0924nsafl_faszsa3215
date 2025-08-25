import os, pytest
from ultra_signals.calibration.optimizer import run_optimization, HAVE_OPTUNA

def test_pruning_behavior(tmp_path):
    if not HAVE_OPTUNA:
        pytest.skip('optuna not installed')
    # Base settings minimal skeleton
    base = {
        'runtime': {'symbols':['BTCUSDT'], 'primary_timeframe':'5m'},
        'features': {'warmup_periods': 5},
        'backtest': {'start_date':'2023-01-01','end_date':'2023-01-10'},
        'walkforward': { 'window': {'train_period':'3d','test_period':'1d','advance_by':'1d'}, 'data_rules': {'purge_period':'0d'} },
    }
    # Search parameters: one that meaningfully affects gates to vary PF/winrate
    cal_cfg = {
        'search_space': {
            'ensemble': {
                # A broader range to create dispersion so median pruner can act
                'vote_threshold_trend': {'low':0.2,'high':0.8},
                'min_agree_trend': {'low':1,'high':3,'step':1}
            }
        },
        'objective': { 'weights': {'profit_factor':1.0}, 'penalties': {'min_trades':0} },
        'runtime': { 'trials': 5, 'seed': 123, 'save_study_db': False, 'pruner': {'median_warmup_steps': 1, 'force_prune_first_n': 2} }
    }
    out = tmp_path / 'cal'
    out.mkdir()
    result = run_optimization(base, cal_cfg, str(out))
    # Ensure leaderboard has status column
    lb = result['leaderboard']
    assert any('status' in r for r in lb)
    # At least one pruned & one complete trial expected (unless all extremely fast & identical)
    states = {r.get('status') for r in lb}
    assert 'PRUNED' in states, f"No pruned trials. States observed: {states}"
    assert 'COMPLETE' in states, f"No complete trials. States observed: {states}"
    # Best fitness should come from a COMPLETE trial
    best_trial_numbers = [r['trial'] for r in lb if r.get('fitness') == result['best']['fitness']]
    assert best_trial_numbers, 'Best fitness not present in leaderboard'
    best_status = [r['status'] for r in lb if r['trial'] == best_trial_numbers[0]][0]
    assert best_status == 'COMPLETE'
