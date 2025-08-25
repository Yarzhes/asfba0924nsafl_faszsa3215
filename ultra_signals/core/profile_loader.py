"""Sprint 20: Profile Loader

Loads layered symbol/timeframe profile YAML fragments and merges them with
baseline defaults. Precedence:
  1) profiles/{SYMBOL}/{TF}.yaml
  2) profiles/{SYMBOL}/_default.yaml
  3) profiles/defaults.yaml

Only override keys are stored; deep merge onto base settings dict.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import yaml
from loguru import logger
import copy

MERGE_SCALAR_REPLACE = (int, float, str, bool, type(None))

class ProfileNotFound(Exception):
    pass


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        logger.warning(f"Failed to load profile {path}: {e}")
        return {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_profile(root: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    root_path = Path(root)
    # Merge order: start from global defaults, then symbol _default, then specific timeframe (highest precedence)
    chain: List[Path] = [
        root_path / "defaults.yaml",
        root_path / symbol / "_default.yaml",
        root_path / symbol / f"{timeframe}.yaml",
    ]
    merged: Dict[str, Any] = {}
    used_files = []
    for p in chain:
        data = _load_yaml(p)
        if data:
            merged = deep_merge(merged, data)
            used_files.append(str(p))
    # missing if no file at all except maybe defaults? Define missing when only defaults present or zero
    missing = (len(used_files) == 0) or (used_files == [str(root_path / "defaults.yaml")])
    return {"profile": merged, "used_files": used_files, "missing": missing}


def profile_id(profile: Dict[str, Any], symbol: str, timeframe: str) -> str:
    meta = (profile or {}).get('meta', {})
    pid = meta.get('profile_id')
    if pid:
        return str(pid)
    version = meta.get('version', 'na')
    return f"{symbol}_{timeframe}_{version}"

__all__ = ['load_profile','deep_merge','profile_id']
