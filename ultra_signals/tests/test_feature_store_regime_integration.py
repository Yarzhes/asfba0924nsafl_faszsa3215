import pandas as pd
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.events import KlineEvent

SETTINGS = {
    "features": {
        "warmup_periods": 20,
        "trend": {"ema_short": 5, "ema_medium": 10, "ema_long": 20, "adx_period": 14},
        "momentum": {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
        "volatility": {"atr_period": 14, "bbands_period": 20, "bbands_std": 2.0},
        "volume_flow": {"vwap_window": 10, "vwap_std_devs": [1.0,2.0], "volume_z_window": 20},
    },
    "regime": {"enabled": True, "cooldown_bars": 4, "hysteresis_hits": 2}
}

def test_feature_store_regime_plumbing():
    fs = FeatureStore(warmup_periods=40, settings=SETTINGS)
    # Generate simple trending bars
    base = 100.0
    for i in range(60):
        evt = KlineEvent(
            event_type="kline",
            timestamp= 1_700_000_000_000 + i*60_000,
            symbol="TESTUSDT",
            timeframe="1m",
            open=base + i*0.2,
            high=base + i*0.2 + 0.5,
            low=base + i*0.2 - 0.5,
            close=base + i*0.2 + 0.1,
            volume=100 + i,
            closed=True,
        )
        fs.ingest_event(evt)
    feats = fs.get_latest_features("TESTUSDT", "1m")
    assert feats is not None, "Features missing after ingestion"
    reg = feats.get("regime")
    assert reg is not None, "Regime not computed"
    assert getattr(reg, "profile", None) is not None, "Regime profile missing"
    # Access via helper
    reg2 = fs.get_regime_state("TESTUSDT", "1m")
    assert reg2 is not None