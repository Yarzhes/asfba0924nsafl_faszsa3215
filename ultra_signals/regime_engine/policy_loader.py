"""Policy map loader for regime → actions mapping.

Schema example (JSON):
{
  "trend_up": {"size_mult": 1.2, "veto": [], "stop_tpl": "wide_trend", "notes": "favorable continuation"},
  "chop_lowvol": {"size_mult": 0.6, "veto": ["breakout"], "stop_tpl": "tight_range"}
}
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from loguru import logger

DEFAULT_POLICY = {
    "trend_up": {"size_mult": 1.15, "veto": [], "stop_tpl": "trend_std"},
    "trend_down": {"size_mult": 1.15, "veto": [], "stop_tpl": "trend_std"},
    "chop_lowvol": {"size_mult": 0.65, "veto": ["breakout"], "stop_tpl": "range_tight"},
    "panic_deleverage": {"size_mult": 0.5, "veto": ["counter_trend_long"], "stop_tpl": "panic_wide"},
    "gamma_pin": {"size_mult": 0.6, "veto": ["breakout"], "stop_tpl": "pin_neutral"},
    "carry_unwind": {"size_mult": 0.7, "veto": [], "stop_tpl": "carry_risk"},
    "risk_on": {"size_mult": 1.0, "veto": [], "stop_tpl": "std"},
    "risk_off": {"size_mult": 0.75, "veto": ["aggressive_leverage"], "stop_tpl": "std_wide"}
}

def load_policy_map(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return DEFAULT_POLICY
    p = Path(path)
    if not p.exists():
        logger.warning(f"Policy map '{path}' not found. Using defaults.")
        return DEFAULT_POLICY
    try:
        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError('Policy JSON root must be object')
        # Merge defaults for missing regimes
        merged = {**DEFAULT_POLICY, **data}
        return merged
    except Exception as e:
        logger.error(f"Failed loading policy map {path}: {e}. Using defaults.")
        return DEFAULT_POLICY

__all__ = ["load_policy_map", "DEFAULT_POLICY"]
