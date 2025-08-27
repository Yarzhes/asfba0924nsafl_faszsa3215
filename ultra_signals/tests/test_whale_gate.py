from ultra_signals.engine.gates.whale_gate import evaluate_whale_gate
from ultra_signals.core.custom_types import WhaleFeatures

BASE_CFG = {
    'features': {
        'whale_risk': {
            'enabled': True,
            'deposit_spike_action': 'DAMPEN',
            'withdrawal_spike_action': 'BOOST',
            'block_sell_extreme_action': 'VETO',
            'block_buy_extreme_action': 'BOOST',
            'composite_pressure_boost_thr': 5_000_000,
            'composite_pressure_veto_thr': -5_000_000,
            'boost_size_mult': 1.3,
            'dampen_size_mult': 0.7,
        }
    }
}

def test_whale_gate_veto_block_sell():
    wf = WhaleFeatures(block_trade_notional_p99_z=-3.0)
    d = evaluate_whale_gate(wf, BASE_CFG)
    assert d.action == 'VETO' and d.reason == 'BLOCK_SELL_EXTREME'

def test_whale_gate_dampen_deposit_spike():
    wf = WhaleFeatures(exch_deposit_burst_flag=1)
    d = evaluate_whale_gate(wf, BASE_CFG)
    assert d.action == 'DAMPEN' and d.size_mult == 0.7

def test_whale_gate_boost_withdrawal_spike():
    wf = WhaleFeatures(exch_withdrawal_burst_flag=1)
    d = evaluate_whale_gate(wf, BASE_CFG)
    assert d.action == 'BOOST' and d.size_mult == 1.3

def test_whale_gate_boost_composite_pressure():
    wf = WhaleFeatures(composite_pressure_score=6_000_000)
    d = evaluate_whale_gate(wf, BASE_CFG)
    assert d.action == 'BOOST'

def test_whale_gate_none():
    wf = WhaleFeatures()
    d = evaluate_whale_gate(wf, BASE_CFG)
    assert d.action == 'NONE'