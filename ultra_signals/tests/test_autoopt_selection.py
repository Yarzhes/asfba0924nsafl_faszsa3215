from ultra_signals.autoopt.selection import rank_candidates

def test_rank_and_uplift():
    base = 1.0
    cands = [
        {'score':1.05,'max_dd_pct':5,'ret_iqr':0.1,'fees_funding_pct':0.02},
        {'score':1.20,'max_dd_pct':7,'ret_iqr':0.2,'fees_funding_pct':0.03},
        {'score':1.10,'max_dd_pct':4,'ret_iqr':0.05,'fees_funding_pct':0.01},
    ]
    sel = rank_candidates(cands, uplift_min=0.05, baseline_score=base)
    assert sel['champion']['score'] == 1.20
    assert sel['promote'] is True
