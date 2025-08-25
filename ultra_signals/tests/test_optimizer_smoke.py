import os, yaml
from pathlib import Path
from ultra_signals.calibration.search_space import SearchSpace
from ultra_signals.calibration.objective import composite_fitness

# Minimal smoke for optimizer integration without running heavy WF.

def test_composite_penalties():
    metrics = {'profit_factor':2.0,'winrate':0.6,'sharpe':1.0,'max_drawdown':-0.05,'trades':10,'pf_oos_minus_is_proxy':-0.5}
    weights = {'profit_factor':0.35,'winrate':0.25,'sharpe':0.15,'max_drawdown':-0.15}
    penalties = {'min_trades':60,'overfit_gap':0.15}
    score = composite_fitness(metrics, weights, penalties)
    # Should be penalized below naive weighted positive part (which would be >0)
    assert score < 0.1


def test_search_space_sample_int_float(tmp_path):
    spec = {
        'ensemble': {'vote_threshold_trend': {'low':0.4,'high':0.5}, 'min_agree_trend': {'low':1,'high':3,'step':1}},
        'slippage': {'atr_multiplier': {'low':0.05,'high':0.06}}
    }
    try:
        import optuna
    except Exception:
        class Dummy:
            def suggest_int(self, *a, **k): return 2
            def suggest_float(self, *a, **k): return 0.055
            def suggest_categorical(self, *a, **k): return a[-1][0]
        trial = Dummy()
    else:
        trial = optuna.trial.FixedTrial({
            'ensemble.vote_threshold_trend':0.45,
            'ensemble.min_agree_trend':2,
            'slippage.atr_multiplier':0.055
        })
    ss = SearchSpace(spec)
    params = ss.sample(trial)
    assert ss.validate_within_bounds(params)
