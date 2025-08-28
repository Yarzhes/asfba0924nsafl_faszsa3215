from ultra_signals.risk.exec_adapter import ExecAdapter


def test_veto_on_high_liq_cost():
    a = ExecAdapter(liq_cost_max_pct_equity=0.005)
    sug = a.suggest(lvar_pct=0.001, liq_cost_pct=0.006, ttl_minutes=10)
    assert sug.size_multiplier == 0.0 and sug.exec_style == 'VETO'


def test_twap_on_high_lvar():
    a = ExecAdapter(liq_cost_max_pct_equity=0.01, lvar_max_pct_equity=0.02)
    sug = a.suggest(lvar_pct=0.03, liq_cost_pct=0.001, ttl_minutes=5)
    assert sug.size_multiplier == 0.5 and sug.exec_style == 'TWAP'
