"""Alignment helpers for funding, OI, and basis (Sprint 39)."""
from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger

def _nearest_trade(ts: int, trades_ts: np.ndarray):
    if trades_ts.size == 0:
        return None
    idx = np.searchsorted(trades_ts, ts)
    candidates = []
    if idx < trades_ts.size:
        candidates.append(trades_ts[idx])
    if idx > 0:
        candidates.append(trades_ts[idx-1])
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - ts))


def _align_generic(events_df: pd.DataFrame, trades_df: pd.DataFrame, settings: dict, kind: str) -> pd.DataFrame:
    dq = (settings or {}).get('data_quality', {})
    al = dq.get('aligners', {})
    window = int(al.get('funding_window_sec', 60)) * 1000
    require = int(al.get('require_within_sec', 90)) * 1000
    trades_ts = trades_df['ts'].values if trades_df is not None and 'ts' in trades_df else np.array([])
    out = events_df.copy()
    snapped = []
    flags = []
    for ts in out['ts']:
        nearest = _nearest_trade(ts, trades_ts)
        if nearest is None:
            snapped.append(np.nan)
            flags.append('missing_trade')
            continue
        if abs(nearest - ts) <= window:
            snapped.append(nearest)
            if abs(nearest - ts) > require:
                flags.append('missing_strict')
            else:
                flags.append('ok')
        else:
            snapped.append(np.nan)
            flags.append('missing_trade')
    out['trade_ts'] = snapped
    out[f'{kind}_align_flag'] = flags
    return out


def align_funding_to_trades(funding_df: pd.DataFrame, trades_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if funding_df is None or len(funding_df) == 0:
        return funding_df
    return _align_generic(funding_df, trades_df, settings, 'funding')


def align_oi_to_trades(oi_df: pd.DataFrame, trades_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if oi_df is None or len(oi_df) == 0:
        return oi_df
    return _align_generic(oi_df, trades_df, settings, 'oi')


def compute_basis(spot_df: pd.DataFrame, perp_df: pd.DataFrame) -> pd.DataFrame:
    if spot_df is None or perp_df is None or len(spot_df) == 0 or len(perp_df) == 0:
        return pd.DataFrame()
    merged = pd.merge_asof(perp_df.sort_values('ts'), spot_df.sort_values('ts'), on='ts', direction='nearest', suffixes=('_perp','_spot'))
    if 'close_perp' not in merged or 'close_spot' not in merged:
        logger.warning('basis.compute missing close columns')
        return pd.DataFrame()
    merged['basis'] = (merged['close_perp'] - merged['close_spot']) / merged['close_spot']
    return merged[['ts','basis']]

__all__ = ['align_funding_to_trades','align_oi_to_trades','compute_basis']
