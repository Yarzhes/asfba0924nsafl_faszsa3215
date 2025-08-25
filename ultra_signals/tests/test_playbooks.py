import pandas as pd
from ultra_signals.engine.playbooks import make_trend_breakout, make_trend_pullback, make_mr_bollinger_fade, make_chop_flat
from ultra_signals.engine.execution_planner import select_playbook, build_plan
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
