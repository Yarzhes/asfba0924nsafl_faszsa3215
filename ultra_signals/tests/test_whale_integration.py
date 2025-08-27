import time
from ultra_signals.engine.gates.whale_gate import evaluate_whale_gate

BASE_SETTINGS = {
    'features': {
        'whale_risk': {
            'enabled': True,
            'composite_pressure_veto_thr': -5_000_000,
            'composite_pressure_boost_thr': 5_000_000,
            'deposit_spike_action': 'DAMPEN',
            'withdrawal_spike_action': 'BOOST',
            'boost_size_mult': 1.5,
            'dampen_size_mult': 0.5,
        }
    }
}

def test_whale_gate_composite_veto():
    wf = { 'composite_pressure_score': -6_000_000 }
    d = evaluate_whale_gate(wf, BASE_SETTINGS)
    assert d.action == 'VETO' and d.reason == 'NEG_SM_PRESSURE'


def test_whale_gate_boost():
    wf = { 'composite_pressure_score': 6_500_000 }
    d = evaluate_whale_gate(wf, BASE_SETTINGS)
    assert d.action == 'BOOST' and d.size_mult == 1.5
