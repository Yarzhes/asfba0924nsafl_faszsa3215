import pandas as pd
import numpy as np
from ultra_signals.analytics.hyperparam_tuning import BayesianRegimeTuner


def _make_df(n=600, seed=7):
    rng = np.random.default_rng(seed)
    regimes = np.where(rng.random(n) < 0.34, 'trend', np.where(rng.random(n) < 0.5, 'mean_revert', 'chop'))
    base = pd.DataFrame({
        'feature_a': rng.normal(size=n),
        'feature_b': rng.normal(size=n),
        'feature_c': rng.normal(size=n),
        'label': rng.choice([-1, 0, 1], size=n, p=[0.33, 0.34, 0.33]),
        'regime_profile': regimes,
    })
    return base


def test_tuner_random_fallback_runs(tmp_path):
    df = _make_df()
    tuner = BayesianRegimeTuner(output_dir=tmp_path.as_posix())
    # Force small n_calls for speed
    res = tuner.tune(df, 'trend', n_calls=5, cv_folds=2, min_samples=50)
    assert 'trend' in res.best_params
    assert len(res.trials['trend']) >= 1
    # ensure file persisted
    assert (tmp_path / 'hpt_params_trend.json').is_file()
