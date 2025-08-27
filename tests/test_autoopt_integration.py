import json, os
from pathlib import Path
import yaml
from ultra_signals.autoopt.wf_runner import AutoOptWFRunner
from ultra_signals.autoopt.spaces import AutoOptSpace
from ultra_signals.autoopt.objective import compute_risk_aware_score


class DummyDA:
    def __init__(self, settings):
        self.settings = settings

class DummyEngine:
    def __init__(self, settings, fs=None):
        self.settings = settings

def test_early_stop_and_artifacts(tmp_path: Path, monkeypatch):
    # minimal fake settings
    settings = {
        'backtest':{'start_date':'2024-01-01','end_date':'2024-06-01'},
        'walkforward':{'train_days':30,'test_days':15,'purge_days':3},
        'features':{'warmup_periods':10},
    }
    # monkeypatch WalkForwardAnalysis._run_test_fold to return shrinking performance
    from ultra_signals.backtest import walkforward as wfmod
    def fake_run_test_fold(self,symbol,timeframe,start,end,engine,fold_index=0):
        import pandas as pd
        # create trades df with fold-dependent pnl
        pnl = 10 - fold_index  # decreasing
        df = pd.DataFrame({'pnl':[pnl, pnl/2]})
        return df, None
    monkeypatch.setattr(wfmod.WalkForwardAnalysis,'_run_test_fold',fake_run_test_fold)
    from ultra_signals.backtest.metrics import compute_kpis as real_kpis
    def fake_compute_kpis(df):
        # simple metrics mapping
        return {'profit_factor':1.0 + df['pnl'].mean()/100,'winrate':0.6,'sortino':1.1,'max_dd_pct':2,'cvar_95_pct':1,'turnover_penalty':0,'fees_funding_pct':0}
    monkeypatch.setattr('ultra_signals.backtest.metrics.compute_kpis', fake_compute_kpis)
    runner = AutoOptWFRunner(settings, lambda s: DummyDA(s), lambda s: DummyEngine(s))
    metrics = runner.evaluate(settings,'BTCUSDT','5m', early_stop={'patience':1,'improvement_delta':0.05})
    assert metrics.get('early_stopped') is True
    space = AutoOptSpace()
    class DummyTrial: pass
    params = space.sample(DummyTrial())
    # simulate CLI challenger export
    challengers_dir = tmp_path/'challengers'; challengers_dir.mkdir()
    (challengers_dir/'challenger_1.yaml').write_text(yaml.safe_dump({'score':1.0, **params}))
    assert (challengers_dir/'challenger_1.yaml').exists()