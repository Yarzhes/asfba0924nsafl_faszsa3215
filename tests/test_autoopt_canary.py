from ultra_signals.autoopt.canary import CanaryConfig, CanaryController

def test_canary_promote_and_rollback():
    # rolling stats simulation
    stats_seq = [
        {'trades':5,'profit_factor':0,'sortino':0,'max_dd_pct':2,'uplift_vs_baseline':0},
        {'trades':25,'profit_factor':1.4,'sortino':1.3,'max_dd_pct':2,'uplift_vs_baseline':0.06},
        {'trades':40,'profit_factor':1.1,'sortino':1.0,'max_dd_pct':7,'uplift_vs_baseline':0.02},
    ]
    def provider():
        return stats_seq[min(provider.idx, len(stats_seq)-1)]
    provider.idx = 0
    ctrl = CanaryController(CanaryConfig(), provider)
    ctrl.start_canary(candidate_version=2, baseline_version=1)
    assert ctrl.state.status=='CANARY'
    # first evaluation insufficient trades
    provider.idx = 0
    assert ctrl.evaluate()=='CANARY'
    # promotion on second stats set
    provider.idx = 1
    assert ctrl.evaluate()=='FULL'
    # once full, further evaluation does not rollback automatically
    provider.idx = 2
    assert ctrl.evaluate()=='FULL'
    assert not ctrl.needs_rollback()
