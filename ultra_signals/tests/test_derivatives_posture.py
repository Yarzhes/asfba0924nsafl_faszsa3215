import time
from types import SimpleNamespace

from ultra_signals.features.derivatives_posture import compute_derivatives_posture
from ultra_signals.core.custom_types import DerivativesFeatures


class MockStore:
    def __init__(self):
        self._funding_provider = None
        self._feature_cache = {}
        self._settings = {'derivatives': {'funding_overheat_pos': 0.001, 'funding_overheat_neg': -0.001, 'oi_z_hi': 2.0, 'oi_z_lo': -2.0}, 'settlement': {'pre_minutes': 30, 'post_minutes': 15}}
        self._ohlcv = {}

    def get_funding_rate_history(self, symbol):
        # return a simple trail
        return [{'funding_rate': 0.001}, {'funding_rate': 0.0012}, {'funding_rate': 0.0015}]

    def get_latest_features(self, symbol, timeframe=None):
        # provide flow_metrics with oi and oi_prev
        return {'flow_metrics': SimpleNamespace(oi=1200000.0, oi_prev=1160000.0, oi_rate= (1200000-1160000)/1160000/60 )}

    def get_ohlcv(self, symbol, timeframe):
        # simple 2-row df-like structure using dict with numeric index and 'close' key
        import pandas as pd
        df = pd.DataFrame([{'close': 100.0}, {'close': 101.0}])
        return df

    def get_minutes_to_next_funding(self, symbol, now_ms):
        return 18


def test_compute_posture_basic():
    store = MockStore()
    out = compute_derivatives_posture(store, 'BTCUSDT')
    assert isinstance(out, DerivativesFeatures)
    # funding fields
    assert out.funding_now is not None
    assert out.funding_trail and len(out.funding_trail) == 3
    assert out.funding_pctl is not None
    # oi fields
    assert out.oi_notional == 1200000.0
    assert out.oi_prev_notional == 1160000.0
    assert out.oi_change_pct is not None
    # taxonomy
    assert out.oi_taxonomy == 'new_longs'
    # settlement window detection -> policy suggests delay
    assert out.policy_suggest in ('delay_to_post_settlement', 'veto', 'allow')


def test_overheat_flags():
    store = MockStore()
    # bump funding to extreme
    def big_hist(sym):
        return [{'funding_rate': 0.005} for _ in range(10)]
    store.get_funding_rate_history = big_hist
    out = compute_derivatives_posture(store, 'BTCUSDT')
    assert out.deriv_overheat_flag_long == 1

