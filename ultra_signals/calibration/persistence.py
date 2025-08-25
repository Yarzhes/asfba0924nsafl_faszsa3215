"""Sprint 19: Calibration Persistence Helpers
"""
from __future__ import annotations
from typing import Dict, Any, List
import yaml, csv, json, subprocess
from pathlib import Path
from loguru import logger
import datetime

def save_leaderboard(rows: List[Dict[str, Any]], out_dir: str):
    if not rows:
        return
    path = Path(out_dir) / 'leaderboard.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    # unify headers
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = list(sorted(keys))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"Leaderboard written: {path}")


def save_best(best: Dict[str, Any], base_settings: Dict[str, Any], out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'best_params.yaml').write_text(yaml.safe_dump(best.get('params') or {}, sort_keys=False))
    snapshot = {
        'generated_at': datetime.datetime.utcnow().isoformat(),
        'fitness': best.get('fitness'),
        'metrics': best.get('metrics'),
        'params': best.get('params')
    }
    # capture git commit if available
    try:  # pragma: no cover
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=out).decode().strip()
        snapshot['git_commit'] = commit
    except Exception:
        snapshot['git_commit'] = None
    (out / 'best_snapshot.json').write_text(json.dumps(snapshot, indent=2))
    logger.info(f"Best params saved to {out}")


def save_autotuned_settings(tuned_params: Dict[str, Any], derived_settings: Dict[str, Any], out_dir: str):
    """Persist only the tuned param overrides and full resolved settings snapshot.

    Writes:
      - settings_autotuned.yaml (full settings with params applied)
      - tuned_params.yaml (just the flattened search-space style params)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tuned_params.yaml').write_text(yaml.safe_dump(tuned_params, sort_keys=False))
    (out / 'settings_autotuned.yaml').write_text(yaml.safe_dump(derived_settings, sort_keys=False))
    logger.info(f"Autotuned settings written to {out}")

__all__ = ['save_leaderboard', 'save_best', 'save_autotuned_settings']
