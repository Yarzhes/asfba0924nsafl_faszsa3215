"""Tests for Sprint 13 Model Registry & Per-Regime Trainer.

Scope:
 - Train small models on synthetic labeled dataset per regime.
 - Ensure models saved to disk and manifest produced.
 - Validate metrics object structure.
"""
import os
import shutil
import pandas as pd
import numpy as np
from ultra_signals.analytics.model_registry import (
    ModelRegistry, PerRegimeTrainer, default_feature_filter
)


def _make_synth_dataset(n_per_regime=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    regimes = ["trend","mean_revert","chop"]
    for reg in regimes:
        for i in range(n_per_regime):
            base = 100 + i * 0.01
            # simple directional drift per regime (just to create weak signal)
            if reg == "trend":
                price = base + rng.normal(0, 0.2)
            elif reg == "mean_revert":
                price = 100 + rng.normal(0, 0.5)
            else:  # chop
                price = 100 + rng.normal(0, 0.3)
            feat1 = np.sin(i/10) + rng.normal(0,0.05)
            feat2 = np.cos(i/15) + rng.normal(0,0.05)
            feat3 = rng.normal(0,1)
            # create a label heuristic
            if feat1 > 0.4:
                label = 1
            elif feat1 < -0.4:
                label = -1
            else:
                label = 0
            rows.append({
                'feat1': feat1,
                'feat2': feat2,
                'feat3': feat3,
                'close': price,
                'label': label,
                'regime_profile': reg,
            })
    df = pd.DataFrame(rows)
    # fabricate a time index
    df.index = pd.to_datetime(np.arange(len(df)), unit='m', origin='2024-01-01')
    return df


def test_train_and_registry(tmp_path):
    dataset = _make_synth_dataset()
    # use tmp model dir
    model_dir = tmp_path / 'models'
    reg = ModelRegistry(str(model_dir))
    trainer = PerRegimeTrainer(reg)
    res = trainer.train_all(dataset, feature_filter=default_feature_filter, min_samples=50)

    # models should exist for all regimes
    listed = reg.list_models()
    assert all(k in listed for k in ['trend','mean_revert','chop'])
    # ensure manifest
    manifest_path = model_dir / 'manifest.json'
    assert manifest_path.is_file()

    # metrics shape
    m = res.metrics_per_regime['trend']
    assert hasattr(m, 'accuracy')
    assert isinstance(m.precision, dict)
    # feature columns recorded
    assert 'trend' in res.feature_cols_used

    # load back one model
    _ = reg.load('trend')
