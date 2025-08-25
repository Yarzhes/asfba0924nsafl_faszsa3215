import os
from pathlib import Path
import yaml

from ultra_signals.core.profile_loader import load_profile, deep_merge, profile_id
from ultra_signals.core.meta_router import MetaRouter


def test_load_profile_precedence(tmp_path):
    root = tmp_path / "profiles"
    (root / "BTCUSDT").mkdir(parents=True)
    # Only defaults initially
    (root / "defaults.yaml").write_text("ensemble: { vote_threshold: { trend: 0.70 } }\n")
    # BTC default
    (root / "BTCUSDT" / "_default.yaml").write_text("ensemble: { vote_threshold: { trend: 0.65 } }\n")
    # Specific timeframe override
    (root / "BTCUSDT" / "5m.yaml").write_text("ensemble: { vote_threshold: { trend: 0.60 } }\n")

    out = load_profile(str(root), "BTCUSDT", "5m")
    assert out["profile"]["ensemble"]["vote_threshold"]["trend"] == 0.60
    # Used files should reflect precedence order (all three present)
    assert len(out["used_files"]) == 3


def test_deep_merge_and_validation():
    base = {"ensemble": {"vote_threshold": {"trend": 0.60, "mean_revert": 0.58}}}
    override = {"ensemble": {"vote_threshold": {"trend": 0.55}}}
    merged = deep_merge(base, override)
    assert merged["ensemble"]["vote_threshold"]["trend"] == 0.55
    assert merged["ensemble"]["vote_threshold"]["mean_revert"] == 0.58


def test_profile_id_metadata_passthrough(tmp_path):
    root = tmp_path / "profiles"
    (root / "ETHUSDT").mkdir(parents=True)
    (root / "defaults.yaml").write_text("meta: { profile_id: GLOBAL, version: v1 }\n")
    (root / "ETHUSDT" / "1h.yaml").write_text("meta: { profile_id: ETHUSDT_1h, version: v2 }\n")
    base = {"profiles": {"min_required_version": "v0"}}
    mr = MetaRouter(base)
    resolved = mr.resolve("ETHUSDT", "1h", str(root))
    meta = resolved.get("meta_router")
    assert meta["profile_id"] == "ETHUSDT_1h"
    assert meta["version"] == "v2"


def test_fallback_when_profile_missing(tmp_path):
    root = tmp_path / "profiles"
    (root).mkdir(parents=True)
    (root / "defaults.yaml").write_text("ensemble: { vote_threshold: { trend: 0.70 } }\n")
    base = {}
    mr = MetaRouter(base)
    resolved = mr.resolve("SOLUSDT", "5m", str(root))
    meta = resolved.get("meta_router")
    assert meta["missing"] is True
    assert meta["profile_id"].startswith("SOLUSDT_5m")  # auto composed id


def test_router_in_backtest_basket(tmp_path, monkeypatch):
    """Integration style: simulate 2 symbols with different profiles and ensure router resolves distinct IDs."""
    # Reuse existing profiles from repo if possible; else craft minimal ones here.
    root = tmp_path / "profiles"
    (root / "BTCUSDT").mkdir(parents=True)
    (root / "ETHUSDT").mkdir(parents=True)
    (root / "defaults.yaml").write_text("ensemble: { vote_threshold: { trend: 0.70 } }\n")
    (root / "BTCUSDT" / "5m.yaml").write_text("meta: { profile_id: BTC_5m, version: v1 }\nensemble: { vote_threshold: { trend: 0.60 } }\n")
    (root / "ETHUSDT" / "5m.yaml").write_text("meta: { profile_id: ETH_5m, version: v1 }\nensemble: { vote_threshold: { trend: 0.62 } }\n")
    base = {"runtime": {"symbols": ["BTCUSDT","ETHUSDT"], "primary_timeframe": "5m"}, "profiles": {"min_required_version": "v0"}}
    mr = MetaRouter(base, root_dir=str(root), hot_reload=True)
    r1 = mr.resolve("BTCUSDT","5m")
    r2 = mr.resolve("ETHUSDT","5m")
    assert r1['meta_router']['profile_id'] != r2['meta_router']['profile_id']


def test_hot_reload_live_mock(tmp_path):
    root = tmp_path / "profiles"
    (root / "BTCUSDT").mkdir(parents=True)
    (root / "defaults.yaml").write_text("meta: { profile_id: GLOBAL, version: v1 }\n")
    f = root / "BTCUSDT" / "5m.yaml"
    f.write_text("meta: { profile_id: BTC_5m, version: v1 }\nensemble: { vote_threshold: { trend: 0.60 } }\n")
    base = {"profiles": {"min_required_version": "v0"}}
    mr = MetaRouter(base, root_dir=str(root), hot_reload=True)
    first = mr.resolve("BTCUSDT","5m")
    # Modify version
    f.write_text("meta: { profile_id: BTC_5m, version: v2 }\nensemble: { vote_threshold: { trend: 0.61 } }\n")
    second = mr.resolve("BTCUSDT","5m")
    assert first['meta_router']['version'] != second['meta_router']['version']


def test_telegram_formatter_profile_footer(monkeypatch):
    from ultra_signals.transport.telegram_formatter import format_message
    class DummyDec:
        def __init__(self):
            self.decision = 'LONG'
            self.symbol = 'BTCUSDT'
            self.tf = '5m'
            self.confidence = 0.75
            self.vote_detail = {
                'weighted_sum': 1.2,
                'agree': 3,
                'total': 4,
                'profile': {'profile_id': 'BTCUSDT_5m', 'version': 's19_2025-08-26'}
            }
            self.vetoes = []
            self.subsignals = []
    msg = format_message(DummyDec())
    # underscores and dashes are escaped in MarkdownV2 output
    assert 'cfg=BTCUSDT\\_5m@s19\\_2025\\-08\\-26' in msg


def test_stale_profile_size_scaling(tmp_path):
    from ultra_signals.core.meta_router import MetaRouter
    # Create an old version profile relative to min_required_version
    root = tmp_path / 'profiles'
    (root / 'BTCUSDT').mkdir(parents=True)
    (root / 'defaults.yaml').write_text('meta: { profile_id: GLOBAL, version: s10 }\n')
    (root / 'BTCUSDT' / '5m.yaml').write_text('meta: { profile_id: BTCUSDT_5m, version: s10 }\n')
    base = { 'profiles': { 'min_required_version': 's19_2025-01-01', 'stale_size_factor': 0.5 } }
    mr = MetaRouter(base, root_dir=str(root), hot_reload=False)
    resolved = mr.resolve('BTCUSDT','5m')
    assert resolved['meta_router']['stale'] is True
