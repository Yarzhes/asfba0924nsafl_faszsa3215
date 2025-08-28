import pandas as pd
import numpy as np
from ultra_signals.vol_forecast.pipeline import prepare_returns, forecast_vols
from ultra_signals.vol_forecast import realized, pipeline, adapters
from ultra_signals.vol_forecast import persistence, models, scheduler


def make_dummy_ohlcv(n=200, seed=0):
    rng = np.random.RandomState(seed)
    t = pd.date_range("2023-01-01", periods=n, freq="T")
    prices = 100 + np.cumsum(rng.normal(scale=0.2, size=n))
    df = pd.DataFrame({"close": prices}, index=t)
    return df


def test_prepare_returns_and_forecast():
    df = make_dummy_ohlcv()
    returns = prepare_returns(df)
    assert returns.shape[0] > 0
    res = forecast_vols(df, model_types=["garch", "egarch"], horizons=[1,3])
    assert "model_choice" in res
    assert "forecasts" in res
    # sigma values should be positive
    for v in res["forecasts"].values():
        assert v >= 0


def test_realized_estimators():
    # create OHLC with small noise
    df = make_dummy_ohlcv(500)
    # add fake high/low/open columns
    df["open"] = df["close"] + 0.01
    df["high"] = df["close"] + 0.05
    df["low"] = df["close"] - 0.05
    p = realized.parkinson(df, window=20)
    gk = realized.garman_klass(df, window=20)
    rs = realized.rogers_satchell(df, window=20)
    assert p.dropna().shape[0] > 0
    assert gk.dropna().shape[0] > 0
    assert rs.dropna().shape[0] > 0


def test_annualize_and_adapters():
    s = 0.001
    ann = pipeline.annualize_sigma(s, bars_per_year=525600)
    assert ann > s
    size = adapters.sizing_adapter(1.0, s, target_sigma=0.02)
    assert size > 0
    stop = adapters.stop_adapter(3.0, s, atr=0.002)
    assert stop > 0


def test_walk_forward_score_and_fallback():
    df = make_dummy_ohlcv(800)
    # add high/low/open for realized fallback if needed
    df["open"] = df["close"] + 0.01
    df["high"] = df["close"] + 0.05
    df["low"] = df["close"] - 0.05
    res = forecast_vols(df, model_types=["garch", "egarch"], horizons=[1])
    # should include model_choice and possibly model score
    assert "model_choice" in res
    assert "models" in res


def test_persistence_and_scheduler(tmp_path):
    # quick persistence smoke: fit minimal manager (EWMA fallback) and save
    df = make_dummy_ohlcv(100)
    r = prepare_returns(df)
    mgr = models.VolModelManager("garch")
    meta = mgr.fit(r)
    # save using persistence
    persistence.save_model(mgr, "TESTSYM", "1m")
    loaded = persistence.load_model("TESTSYM", "1m")
    assert loaded is not None
    # update and read registry
    persistence.update_registry("TESTSYM", "1m", meta, mgr.last_refit_ts)
    reg = persistence.read_registry()
    assert "TESTSYM__1m" in reg

