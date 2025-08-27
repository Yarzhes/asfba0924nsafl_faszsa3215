"""Publisher & artifact persistence.

Adds:
    * Versioned champion YAMLs under `profiles/auto/`.
    * Run-scoped artifacts (leaderboard, challengers) under provided run directory.
    * Release note with diff of key params / metrics vs previous champion.
    * Fingerprint for integrity.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable
from pathlib import Path
import json, hashlib, time, yaml

VERSION_FILE = 'profiles/auto/version_index.json'

def _load_versions(root: Path) -> Dict[str,int]:
    p = root / VERSION_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def _save_versions(root: Path, versions: Dict[str,int]):
    p = root / VERSION_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(versions, indent=2))


def fingerprint(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _diff_params(prev: Dict[str,Any]|None, new: Dict[str,Any]) -> Dict[str, Any]:
    if not prev:
        return {k:{'old':None,'new':v} for k,v in new.items()}
    delta = {}
    for k,v in new.items():
        if prev.get(k) != v:
            delta[k] = {'old': prev.get(k), 'new': v}
    return delta


def publish_champion(root: str|Path, symbol: str, tf: str, profile: str, params: Dict[str,Any], derived_settings: Dict[str,Any], metrics: Dict[str,Any], run_dir: Path|None=None):
    root = Path(root)
    versions = _load_versions(root)
    key = f"{symbol}_{tf}_{profile}"
    ver = versions.get(key,0) + 1
    versions[key] = ver
    cfg = {
        'symbol': symbol,
        'timeframe': tf,
        'profile': profile,
        'version': ver,
        'params': params,
        'metrics': metrics,
        'cfg_fingerprint': fingerprint(params),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    auto_dir = root / 'profiles' / 'auto'
    auto_dir.mkdir(parents=True, exist_ok=True)
    fname = f"auto_{symbol}_{tf}_{profile}_{time.strftime('%Y%m%d')}_v{ver}.yaml"
    out_path = auto_dir / fname
    prev_pointer = (root / 'profiles' / 'active' / f"{symbol}_{tf}.yaml")
    prev_cfg: Dict[str, Any] | None = None
    if prev_pointer.exists():  # attempt to load previous metrics/params for diff
        try:
            import yaml as _y
            prev_file = prev_pointer.read_text().split('file:')[-1].strip().split('\n')[0].strip()
            candidate = auto_dir / prev_file
            if candidate.exists():
                prev_cfg = _y.safe_load(candidate.read_text())
        except Exception:
            prev_cfg = None
    with out_path.open('w') as f:
        yaml.safe_dump(cfg,f, sort_keys=False)
    # Active pointer
    active_dir = root / 'profiles' / 'active'
    active_dir.mkdir(parents=True, exist_ok=True)
    pointer = active_dir / f"{symbol}_{tf}.yaml"
    pointer.write_text(f"# active champion\nfile: {out_path.name}\nversion: {ver}\n")
    # Release note
    rel_dir = root / 'profiles' / 'releases'
    rel_dir.mkdir(parents=True, exist_ok=True)
    rel = rel_dir / f"release_{symbol}_{tf}_{profile}_v{ver}.md"
    param_diff = _diff_params((prev_cfg or {}).get('params') if prev_cfg else None, params)
    rel.write_text(
        f"## AutoOpt Promotion\nSymbol: {symbol} {tf} profile={profile} version={ver}\n\n"
        f"Metrics: {metrics}\n\nParam Changes: {param_diff}\n"
    )
    _save_versions(root, versions)
    # copy artifacts into run dir if provided
    if run_dir:
        try:
            run_dir = Path(run_dir)
            (run_dir / 'champion.yaml').write_text(out_path.read_text())
        except Exception:
            pass
    return {'path': str(out_path), 'version': ver, 'diff': param_diff}

__all__ = ['publish_champion','fingerprint']
