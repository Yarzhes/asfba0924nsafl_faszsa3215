"""Multi-venue composite mid-price builder (Sprint 39)."""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from loguru import logger


def composite_mid(venues_data: Dict[str, pd.DataFrame], settings: dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dq = (settings or {}).get('data_quality', {})
    mv = dq.get('multi_venue', {})
    mode = mv.get('merge_mode', 'median')
    max_spread_bps = float(mv.get('max_spread_bps', 12))
    frames = []
    for venue, df in venues_data.items():
        if df is None or len(df) == 0:
            continue
        if 'mid' not in df.columns and {'bid','ask'} <= set(df.columns):
            df = df.assign(mid=(df['bid']+df['ask'])/2)
        frames.append(df[['ts','mid']].rename(columns={'mid': f'mid_{venue}'}))
    if not frames:
        return pd.DataFrame(), {"empty": True}
    # pairwise merge on ts using outer join then forward fill small gaps
    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge_asof(merged.sort_values('ts'), f.sort_values('ts'), on='ts', direction='nearest', tolerance=1000)
    mid_cols = [c for c in merged.columns if c.startswith('mid_')]
    if mode == 'median':
        merged['mid'] = merged[mid_cols].median(axis=1)
    elif mode == 'weighted_by_liquidity':  # placeholder equal weights
        merged['mid'] = merged[mid_cols].mean(axis=1)
    else:  # best_bid_ask or fallback
        merged['mid'] = merged[mid_cols].mean(axis=1)
    # disagreement
    max_spread = (merged[mid_cols].max(axis=1) - merged[mid_cols].min(axis=1)) / merged['mid'] * 10000
    merged['venue_spread_bps'] = max_spread
    flags = {"disagreement_exceeded": bool((max_spread > max_spread_bps).any())}
    if flags["disagreement_exceeded"]:
        logger.warning(f"venue_merge.disagreement spread_bps_max={max_spread.max():.2f} threshold={max_spread_bps}")
    return merged[['ts','mid','venue_spread_bps']], flags

__all__ = ['composite_mid']
