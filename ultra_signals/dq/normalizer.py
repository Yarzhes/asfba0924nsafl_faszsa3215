"""Normalization utilities for OHLCV + symbol mapping (Sprint 39)."""
from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict
from loguru import logger

_CANON = ["ts","open","high","low","close","volume"]

@lru_cache(maxsize=1)
def _load_symbol_map(path: str | None) -> Dict[str, Dict]:  # pragma: no cover trivial IO
    if not path or not Path(path).is_file():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    out = {}
    for canon, meta in raw.items():
        venues = (meta or {}).get('venues', {}) or {}
        out[canon] = venues
    return out


def normalize_symbol(symbol: str, venue: str, settings: dict) -> str:
    dq = (settings or {}).get('data_quality', {})
    map_path = dq.get('symbols', {}).get('map_path')
    smap = _load_symbol_map(map_path)
    # If symbol appears as canonical key, return it; else reverse-lookup
    if symbol in smap:
        return symbol
    for canon, venues in smap.items():
        if venues.get(venue) == symbol:
            return canon
    return symbol


def to_canonical_ohlcv(df: pd.DataFrame, symbol: str, venue: str, settings: dict) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    dq = (settings or {}).get('data_quality', {})
    cols = dq.get('ohlcv_schema', _CANON)
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for c in cols:
        if c not in df.columns and c in lower_cols:
            rename_map[lower_cols[c]] = c
    if rename_map:
        df = df.rename(columns=rename_map)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"normalizer.missing_cols symbol={symbol} venue={venue} missing={missing}")
        for c in missing:
            if c == 'ts':
                continue
            df[c] = np.nan
    # enforce ts int ms
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['ts'])
    df['ts'] = df['ts'].astype('int64')
    # sort
    df = df.sort_values('ts')
    # clamp decimals
    price_max = dq.get('price_decimals_max', 8)
    vol_max = dq.get('volume_decimals_max', 8)
    q_price = 10 ** price_max
    q_vol = 10 ** vol_max
    for col in ['open','high','low','close']:
        if col in df:
            df[col] = (df[col] * q_price).round().div(q_price)
    if 'volume' in df:
        df['volume'] = (df['volume'] * q_vol).round().div(q_vol)
    return df.reset_index(drop=True)[cols]

__all__ = ['to_canonical_ohlcv','normalize_symbol']
