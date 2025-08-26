import pandas as pd
from ultra_signals.engine.playbooks import make_trend_breakout, make_trend_pullback, make_mr_bollinger_fade, make_chop_flat
from ultra_signals.engine.execution_planner import select_playbook, build_plan, is_plan_timed_out
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import TrendFeatures, MomentumFeatures, VolatilityFeatures, RegimeFeatures, EnsembleDecision, SubSignal

class DummyFS(FeatureStore):
    def __init__(self, feats):
        self._f = feats
    def get_features(self, symbol, timeframe, ts, nearest=True):
        return self._f
    def get_latest_features(self, symbol, timeframe):
        return self._f


def _decision(decision='LONG', conf=0.7):
    return EnsembleDecision(ts=0, symbol='BTC', tf='5m', decision=decision, confidence=conf, subsignals=[
        SubSignal(ts=0, symbol='BTC', tf='5m', strategy_id='trend', direction=decision, confidence_calibrated=conf, reasons={})
    ], vote_detail={'orderflow': {'cvd':0.1,'cvd_chg':0.1,'liq_dom':'short','sweep_flag':True}}, vetoes=[])


def test_trend_breakout_selection():
    feats = {
        'trend': TrendFeatures(ema_short=110, ema_medium=100, ema_long=90, adx=25),
        'volatility': VolatilityFeatures(atr=5.0),
        'momentum': MomentumFeatures(rsi=55),
        'regime': RegimeFeatures(profile='trend'),
    }
    settings = {'playbooks': {'trend': {'breakout': {'enabled': True, 'entry': {'min_conf':0.6,'min_adx':22,'ema_sep_atr_min':1.5,'of_confirm':{'cvd':0.05}}, 'exit': {'stop_atr_mult':1.4,'tp_atr_mults':[1.8]}, 'risk': {'size_scale':1.1,'cooldown_bars':5,'rr_min':1.2}}, 'pullback': {'enabled':True}}}}
    pb = select_playbook(feats['regime'], feats, _decision(), settings)
    assert pb and pb.name.startswith('trend_breakout')
    plan = build_plan(pb, {**feats,'ohlcv':{'close':100}}, _decision(), settings)
    assert plan and plan['reason']=='trend_breakout'


def test_chop_abstain():
    feats = {
        'trend': TrendFeatures(ema_short=101, ema_medium=100, ema_long=99, adx=15),
        'volatility': VolatilityFeatures(atr=5.0),
        'momentum': MomentumFeatures(rsi=50),
        'regime': RegimeFeatures(profile='chop'),
    }
    settings = {'playbooks': {'chop': {'flat': {'enabled': True}}}}
    pb = select_playbook(feats['regime'], feats, _decision(), settings)
    assert pb and pb.abstain


def test_rr_gate_abstains():
    feats = {
        'trend': TrendFeatures(ema_short=110, ema_medium=108, ema_long=100, adx=25),
        'volatility': VolatilityFeatures(atr=10.0),
        'momentum': MomentumFeatures(rsi=55),
        'regime': RegimeFeatures(profile='trend'),
    }
    # stop 2.0 vs tp 2.1 => rr just 1.05 < rr_min=1.2 -> abstain
    settings = {'playbooks': {'trend': {'breakout': {'enabled': True,'entry': {'min_conf':0.6,'min_adx':22,'ema_sep_atr_min':0.5}, 'exit': {'stop_atr_mult':2.0,'tp_atr_mults':[2.1]}, 'risk': {'size_scale':1.0,'cooldown_bars':3,'rr_min':1.2}}, 'pullback': {'enabled':False}}}}
    pb = select_playbook(feats['regime'], feats, _decision(), settings)
    plan = build_plan(pb, {**feats,'ohlcv':{'close':100}}, _decision(), settings)
    assert plan is None


def test_mr_bb_fade_selection_and_plan_prices():
    feats = {
        'trend': TrendFeatures(ema_short=100, ema_medium=100, ema_long=100, adx=18),
        'volatility': VolatilityFeatures(atr=4.0),
        'momentum': MomentumFeatures(rsi=25),  # below lower extreme 28 -> qualifies
        'regime': RegimeFeatures(profile='mean_revert'),
    }
    settings = {'playbooks': {'mean_revert': {'bb_fade': {'enabled': True, 'entry': {'min_conf':0.55,'rsi_extreme':[28,72]}, 'exit': {'stop_atr_mult':1.0,'tp_atr_mults':[1.2,1.8]}, 'risk': {'size_scale':0.9,'cooldown_bars':4,'rr_min':1.1}}, 'vwap_revert': {'enabled': False}}}}
    pb = select_playbook(feats['regime'], feats, _decision(), settings)
    assert pb and pb.name.startswith('mr_bb_fade')
    plan = build_plan(pb, {**feats,'ohlcv':{'close':200}}, _decision(), settings)
    assert plan and plan['stop_price'] is not None and plan['tp_bands']
    # LONG decision -> stop below entry, tp above
    assert plan['stop_price'] < 200
    assert all(tp > 200 for tp in plan['tp_bands'])


def test_timeout_helper():
    plan = {
        'timeout_bars': 5
    }
    # Before timeout
    assert not is_plan_timed_out(plan, 3, 0.0)
    # At timeout with insufficient progress
    assert is_plan_timed_out(plan, 5, 0.05)
    # At timeout but good progress -> keep
    assert not is_plan_timed_out(plan, 5, 0.2)
