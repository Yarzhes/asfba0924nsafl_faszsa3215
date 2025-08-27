from ultra_signals.autoopt.spaces import AutoOptSpace

def test_tp_monotonic():
    space = AutoOptSpace()
    class T: pass
    p = space.sample(T())
    m1,m2,m3 = p['execution.tp_atr_mults']
    assert m1 < m2 < m3
    assert 1.4 <= m1 <= 2.2
    assert 2.0 <= m2 <= 3.0
    assert 2.6 <= m3 <= 4.0
