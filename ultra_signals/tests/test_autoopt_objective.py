from ultra_signals.autoopt.objective import compute_risk_aware_score

def test_objective_constraints_fail_min_trades():
    m = {'trades':10,'profit_factor':2,'sortino':2,'winrate':0.55,'max_dd_pct':5,'cvar_95_pct':5,'ret_iqr':0.1}
    assert compute_risk_aware_score(m) == float('-inf')

def test_objective_constraints_pass():
    m = {'trades':100,'profit_factor':1.5,'sortino':1.2,'winrate':0.55,'max_dd_pct':5,'cvar_95_pct':5,'ret_iqr':0.1}
    s = compute_risk_aware_score(m)
    assert s != float('-inf') and isinstance(s,float)
