from ultra_signals.engine.gates.meta_gate import MetaGate

def test_meta_gate_missing_model_safe():
    settings={'meta_scorer':{'enabled':True,'model_path':'/nonexistent.joblib','missing_policy':'SAFE','thresholds':{'trend':0.6},'partial_band':{'trend':{'low':0.55,'high':0.6,'size_mult':0.5,'widen_stop_mult':1.1}}}}
    gate=MetaGate(settings)
    out=gate.evaluate('LONG','trend',{})
    assert out.action=='DAMPEN'
    assert out.reason.startswith('MISSING_MODEL')

def test_meta_gate_disabled():
    settings={'meta_scorer':{'enabled':False}}
    gate=MetaGate(settings)
    out=gate.evaluate('LONG','trend',{})
    assert out.action=='ENTER'
    assert out.reason=='META_DISABLED'
