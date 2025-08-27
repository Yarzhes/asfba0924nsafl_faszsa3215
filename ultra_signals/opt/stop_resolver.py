"""Stop Resolver (Sprint 37)

Lightweight lookup helper to convert (symbol, timeframe, regime) + ATR/price
into a stop distance in price units using the optimized stop table.

Design:
- Lazy-load YAML table once (cached module-level) with reload capability.
- Fallback chain: (symbol, tf, regime) -> (symbol, tf, 'mixed') -> (symbol, '*', regime)
  -> ('*','*','*') -> static default from settings risk.adaptive_exits.atr_mult_stop.
- Mode aware: if table entry mode=='atr' => distance = value * atr;
  if 'percent' => distance = price * (value/100 if value>1 else value) to allow both 0.5 (50%) or 0.50 (%) conventions.
- Returns distance (not absolute stop price). Caller subtracts/adds based on side.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import threading

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

_STOP_TABLE_CACHE: Optional[Dict[str, Any]] = None
_STOP_TABLE_PATH: Optional[Path] = None
_STOP_TABLE_LOCK = threading.Lock()


def load_stop_table(path: str) -> Dict[str, Any]:
    global _STOP_TABLE_CACHE, _STOP_TABLE_PATH
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    with _STOP_TABLE_LOCK:
        if _STOP_TABLE_CACHE is not None and _STOP_TABLE_PATH == p:
            return _STOP_TABLE_CACHE
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except Exception:
            data = {}
        _STOP_TABLE_CACHE = data
        _STOP_TABLE_PATH = p
        return data


def invalidate_cache():  # pragma: no cover (rare)
    global _STOP_TABLE_CACHE, _STOP_TABLE_PATH
    with _STOP_TABLE_LOCK:
        _STOP_TABLE_CACHE = None
        _STOP_TABLE_PATH = None


def _lookup(table: Dict[str, Any], symbol: str, tf: str, regime: Optional[str]):
    regime_key = regime or 'mixed'
    # direct
    try:
        v = table.get(symbol, {}).get(str(tf), {}).get(regime_key)
        if v is not None:
            return v
    except Exception:
        pass
    # mixed fallback
    try:
        v = table.get(symbol, {}).get(str(tf), {}).get('mixed')
        if v is not None:
            return v
    except Exception:
        pass
    # symbol any timeframe
    try:
        v = table.get(symbol, {}).get('*', {}).get(regime_key) or table.get(symbol, {}).get('*', {}).get('mixed')
        if v is not None:
            return v
    except Exception:
        pass
    # global wildcard
    try:
        v = table.get('*', {}).get('*', {}).get(regime_key) or table.get('*', {}).get('*', {}).get('mixed')
        if v is not None:
            return v
    except Exception:
        pass
    return None


def resolve_stop(symbol: str, tf: str, regime: Optional[str], atr: Optional[float], price: float, settings: Dict[str, Any]) -> Optional[float]:
    ao = (settings.get('auto_stop_opt') or {}) if isinstance(settings, dict) else {}
    if not ao.get('enabled'):
        return None
    path = ao.get('output_path') or 'models/stop_opt/stop_table.yaml'
    table = load_stop_table(path)
    entry = _lookup(table, symbol, tf, regime)
    # If missing, fallback to static adaptive_exits multiplier
    if not entry:
        # static fallback distance using ATR if available
        atr_mult = float(((settings.get('risk') or {}).get('adaptive_exits') or {}).get('atr_mult_stop', 1.2))
        if atr is not None and atr > 0:
            return atr_mult * atr
        # percent fallback 0.5% of price
        return price * 0.005
    mode = str(entry.get('mode') or ao.get('mode') or 'atr').lower()
    value = float(entry.get('value', 0.0))
    if mode == 'atr':
        if atr is None or atr <= 0:
            return None
        return value * atr
    # percent mode: treat value as percent number (0.30 => 0.30%) regardless of magnitude
    if price <= 0:
        return None
    pct = value / 100.0
    return price * pct

__all__ = ['resolve_stop','load_stop_table','invalidate_cache']
