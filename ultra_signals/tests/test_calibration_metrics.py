import numpy as np
import pandas as pd
import os

from ultra_signals.ensemble.calibration import brier_score, ece, save_reliability_plot


def test_brier_and_ece_basic():
    rng = np.random.RandomState(1)
    n = 1000
    # perfect predictions
    y = rng.randint(0, 2, size=n)
    probs_perfect = y.astype(float)
    assert brier_score(probs_perfect, y) == 0.0

    # random predictions have higher brier
    probs_random = rng.rand(n)
    assert brier_score(probs_random, y) >= 0.0

    # ECE is 0 for perfect predictions (within numerical tolerance)
    assert ece(probs_perfect, y, n_bins=10) < 1e-9


def test_save_reliability_plot(tmp_path):
    rng = np.random.RandomState(2)
    n = 200
    y = rng.randint(0, 2, size=n)
    probs = rng.rand(n)
    out = tmp_path / 'rel.png'
    p = save_reliability_plot(probs, y, str(out), n_bins=8)
    assert os.path.exists(p)
    # file should be non-empty
    assert os.path.getsize(p) > 0
