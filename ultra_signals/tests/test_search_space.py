from ultra_signals.calibration.search_space import SearchSpace

spec = {
    'ensemble': {
        'vote_threshold_trend': {'low':0.4,'high':0.8},
        'min_agree_trend': {'low':1,'high':3,'step':1}
    },
    'slippage': {
        'atr_multiplier': {'low':0.05,'high':0.3}
    }
}

def test_search_space_bounds():
    try:
        import optuna
    except Exception:
        class Dummy:
            def suggest_int(self, *a, **k): return 2
            def suggest_float(self, *a, **k): return 0.2
            def suggest_categorical(self, *a, **k): return a[-1][0]
        trial = Dummy()
    else:
        trial = optuna.trial.FixedTrial({
            'ensemble.vote_threshold_trend':0.5,
            'ensemble.min_agree_trend':2,
            'slippage.atr_multiplier':0.15
        })
    ss = SearchSpace(spec)
    params = ss.sample(trial)
    assert 0.4 <= params['ensemble.vote_threshold_trend'] <= 0.8
    assert 1 <= params['ensemble.min_agree_trend'] <= 3
    assert 0.05 <= params['slippage.atr_multiplier'] <= 0.3
