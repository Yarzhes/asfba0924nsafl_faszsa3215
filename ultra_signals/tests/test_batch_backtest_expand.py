from ultra_signals.apps.batch_backtest import expand_jobs, TOP10, ensure_metric_fields


def test_expand_jobs_top10_rule():
    cfg = {
        'symbols': TOP10 + ['NEWCOINUSDT'],
        'timeframes': ['5m','15m','1h'],
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'walk_forward': {'enabled': False}
    }
    jobs = expand_jobs(cfg, wf_only_top10=False)
    # NEWCOINUSDT should not have 5m job
    newcoin_tfs = [j[1] for j in jobs if j[0]=='NEWCOINUSDT']
    assert '5m' not in newcoin_tfs and '15m' in newcoin_tfs and '1h' in newcoin_tfs
    # A top10 symbol should include 5m
    btc_tfs = [j[1] for j in jobs if j[0]=='BTCUSDT']
    assert '5m' in btc_tfs


def test_expand_jobs_wf_only_top10():
    cfg = {
        'symbols': TOP10 + ['OTHERUSDT'],
        'timeframes': ['5m','15m','1h'],
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'walk_forward': {'enabled': True}
    }
    jobs = expand_jobs(cfg, wf_only_top10=True)
    # Only 15m jobs for top10 should be present
    assert all(j[4] for j in jobs), 'all jobs should be WF flagged'
    assert all(j[1]=='15m' for j in jobs), 'only 15m timeframe for WF-only-top10'
    symbols_in_jobs = {j[0] for j in jobs}
    assert 'OTHERUSDT' not in symbols_in_jobs


def test_ensure_metric_fields_padding():
    minimal = {'symbol':'BTCUSDT','timeframe':'5m','profit_factor':1.5}
    norm = ensure_metric_fields(minimal)
    # All core fields present
    from ultra_signals.backtest.json_metrics import CORE_FIELDS
    for f in CORE_FIELDS:
        assert f in norm, f"missing field {f}"
    # Non-provided numeric become 0
    assert norm['sortino'] == 0
    assert norm['profit_factor'] == 1.5
