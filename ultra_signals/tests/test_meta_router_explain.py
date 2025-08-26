from ultra_signals.core.meta_router import MetaRouter
from pathlib import Path

def test_meta_router_explain_paths(tmp_path):
    root = tmp_path / 'profiles'
    (root / 'BTCUSDT').mkdir(parents=True)
    (root / 'defaults.yaml').write_text('ensemble: { vote_threshold: { trend: 0.60 } }\n')
    (root / 'BTCUSDT' / '5m.yaml').write_text('meta: { profile_id: BTCUSDT_5m, version: s20 }\nensemble: { vote_threshold: { trend: 0.58 }, confidence_floor: 0.64 }\nplaybooks: { trend: { breakout: { exit: { stop_atr_mult: 1.5 } } } }\n')
    base = { 'profiles': { 'min_required_version': 's19' } }
    mr = MetaRouter(base, root_dir=str(root), hot_reload=False)
    resolved = mr.resolve('BTCUSDT','5m')
    explain = mr.explain('BTCUSDT','5m')
    assert explain['profile_id'] == 'BTCUSDT_5m'
    # Ensure nested override key captured
    assert any('playbooks.trend.breakout.exit.stop_atr_mult' in k or 'playbooks.trend.breakout.exit' in k for k in explain['resolved_keys'])
    # Ensure fallback chain relative paths
    assert all(not p.startswith(str(root)) for p in explain['fallback_chain'])
