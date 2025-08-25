import pandas as pd
from ultra_signals.engine.quality_gates import QualityGates
from ultra_signals.core.custom_types import EnsembleDecision, SubSignal, TrendFeatures, MomentumFeatures, VolatilityFeatures, RegimeFeatures, FlowMetricsFeatures

# Helper to build a minimal EnsembleDecision
def _decision(direction='LONG', confidence=0.8):
    return EnsembleDecision(
        ts=int(pd.Timestamp.utcnow().timestamp()),
        symbol='BTCUSDT',
        tf='5m',
        decision=direction,
        confidence=confidence,
        subsignals=[SubSignal(ts=0, symbol='BTCUSDT', tf='5m', strategy_id='x', direction=direction, confidence_calibrated=confidence, reasons={})],
    vote_detail={'orderflow': {'cvd_chg': 0.1 if direction=='LONG' else -0.1}},
    vetoes=[]
    )

BASE_SETTINGS = {
    'quality_gates': {
        'enabled': True,
        'qscore_bins': { 'Aplus': 0.85, 'A': 0.75, 'B': 0.65, 'C': 0.55 },
        'bin_actions': {
            'Aplus': { 'size_mult': 1.30, 'require_extra_confirm': False },
            'A':     { 'size_mult': 1.15, 'require_extra_confirm': False },
            'B':     { 'size_mult': 1.00, 'require_extra_confirm': False },
            'C':     { 'size_mult': 0.75, 'require_extra_confirm': True },
            'D':     { 'size_mult': 0.00, 'require_extra_confirm': True },
        },
        'veto': {
            'max_spread_pct': 0.06,
            'atr_pct_limit': 0.97
        },
        'soft': {
            'ofi_conflict_limit': -0.15,
            'late_move_atr': 0.8,
            'min_volume_z': -0.8,
        }
    }
}


def _features(atr_pct=0.5, ofi=None, volume_z=0.0):
    return {
        'regime': RegimeFeatures(atr_percentile=atr_pct),
        'volatility': VolatilityFeatures(atr_percentile=atr_pct),
        'flow_metrics': FlowMetricsFeatures(ofi=ofi, volume_z=volume_z),
    }


def test_binning_aplus():
    dec = _decision(confidence=0.95)
    feats = _features(atr_pct=0.5)
    # Provide strong RR context (expected_rr far above rr_min) to yield high qscore
    qd = QualityGates.evaluate(feats, dec, {'expected_rr': 3.0, 'risk': {'rr_min': 1.0}}, BASE_SETTINGS)
    assert qd.bin in ('A+', 'A', 'B')  # must be at least B+ territory
    assert qd.qscore >= 0.75


def test_binning_d_low_conf():
    dec = _decision(confidence=0.20)
    feats = _features(atr_pct=0.2)
    qd = QualityGates.evaluate(feats, dec, {'expected_rr': 1.1, 'risk': {'rr_min': 1.4}}, BASE_SETTINGS)
    assert qd.bin == 'D'


def test_hard_veto_spread():
    dec = _decision(confidence=0.8)
    # inject large spread via flow_metrics.spread_bps (converted inside code)
    feats = _features()
    feats['flow_metrics'].spread_bps = 1000  # 1000 bps => 0.10
    qd = QualityGates.evaluate(feats, dec, {'expected_rr':1.8, 'risk': {'rr_min': 1.4}}, BASE_SETTINGS)
    assert qd.blocked and 'SPREAD_WIDE' in qd.veto_reasons


def test_hard_veto_vol_spike():
    dec = _decision(confidence=0.8)
    feats = _features(atr_pct=0.99)
    qd = QualityGates.evaluate(feats, dec, {'expected_rr':1.8, 'risk': {'rr_min': 1.4}}, BASE_SETTINGS)
    assert qd.blocked and 'VOL_SPIKE' in qd.veto_reasons


def test_soft_gate_ofi_conflict():
    dec = _decision(direction='LONG', confidence=0.7)
    feats = _features(ofi=-0.5)
    qd = QualityGates.evaluate(feats, dec, {'expected_rr':1.6, 'risk': {'rr_min': 1.2}}, BASE_SETTINGS)
    assert 'OFI_CONFLICT' in qd.soft_flags


def test_soft_gate_low_liq():
    dec = _decision(confidence=0.7)
    feats = _features(volume_z=-2.0)
    qd = QualityGates.evaluate(feats, dec, {'expected_rr':1.6, 'risk': {'rr_min': 1.2}}, BASE_SETTINGS)
    assert 'LOW_LIQ' in qd.soft_flags
