"""Sprint 31 Dataset Builder

Builds a supervised learning dataset from historical OHLCV and trade logs.

ASSUMPTIONS / CURRENT LIMITATIONS (documented so we can iterate):
1. We only use executed trades (both winners & losers) as positive training rows.
   Ideally we'd also include vetoed/filtered candidate entries (negative rows) but
   those are not persisted yet. Future: persist decision log for every bar.
2. Features are recomputed from OHLCV (no look-ahead) using rolling calculations
   at the bar that matches each trade's entry timestamp (<=). We enforce that the
   bar timestamp is strictly <= trade entry and never peek forward.
3. Label `tp_before_sl` is inferred from trade `result` (TP -> 1, SL/BE/etc -> 0).
4. Additional label modes (`r_after_N_bars`, `mfe_exceeds`) left as TODO; scaffolding present.

Outputs:
  features.npy (float32 matrix N x F)
  labels.npy (uint8 vector length N)
  feature_names.json (ordered list)
  meta.json (dataset metadata)
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from loguru import logger

from ultra_signals.core.config import load_settings
from ultra_signals.backtest.data_adapter import DataAdapter
from ultra_signals.features.trend import compute_trend_features
from ultra_signals.features.momentum import compute_momentum_features
from ultra_signals.features.volatility import compute_volatility_features


# ---------------- Labeling -----------------

def label_tp_before_sl(trades_df: pd.DataFrame) -> np.ndarray:
    """Label = 1 if trade result enumerates a TP-style outcome before SL.
    We treat 'TP' as success; everything else (SL, BE, TrailingSL) as 0 for now.
    """
    res = trades_df.get('result')
    if res is None:
        return np.zeros(len(trades_df), dtype=np.uint8)
    return (res.astype(str).str.upper().str.startswith('TP')).astype(np.uint8).values


def label_r_after_n_bars(trades_df: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Placeholder: requires per-bar unrealized PnL path (not available yet)."""
    return np.zeros(len(trades_df), dtype=np.uint8)


def label_mfe_exceeds(trades_df: pd.DataFrame, threshold: float = 1.5) -> np.ndarray:
    """Label = 1 if trade's maximum favorable excursion (MFE) >= threshold * adverse (MAE) or >= raw threshold.
    Fallback to 0 if columns absent.
    """
    if 'mfe' not in trades_df.columns:
        return np.zeros(len(trades_df), dtype=np.uint8)
    mfe = pd.to_numeric(trades_df['mfe'], errors='coerce').fillna(0.0)
    if 'mae' in trades_df.columns:
        mae = pd.to_numeric(trades_df['mae'], errors='coerce').replace(0, np.nan)
        ratio = mfe / mae
        lbl = ( (ratio >= threshold) | (mfe >= threshold) ).astype(np.uint8).fillna(0)
        return lbl.values
    return (mfe >= threshold).astype(np.uint8).values

LABEL_FUNCS = {
    'tp_before_sl': label_tp_before_sl,
    'r_after_n_bars': label_r_after_n_bars,  # stub
    'mfe_exceeds': label_mfe_exceeds,
}


def _rolling_features(ohlcv: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    """Compute feature snapshot for every bar (including warmup NaNs) w/out dict wrapper column.

    Produces stable column names directly, filling pre-warmup rows with NaN so later logic
    does not create a single object column of dicts (avoids '0_' prefix flatten artifacts).
    """
    trend_cfg = (settings.get('features', {}) or {}).get('trend', {})
    mom_cfg = (settings.get('features', {}) or {}).get('momentum', {})
    vol_cfg = (settings.get('features', {}) or {}).get('volatility', {})
    warmup = 10
    feat_rows: List[Optional[Dict[str, Any]]] = [None] * len(ohlcv)
    keys: Optional[List[str]] = None
    for i in range(len(ohlcv)):
        window = ohlcv.iloc[: i + 1]
        if len(window) >= warmup:
            tfeat = compute_trend_features(window,
                                           ema_short=trend_cfg.get('ema_short', 21),
                                           ema_medium=trend_cfg.get('ema_medium', 50),
                                           ema_long=trend_cfg.get('ema_long', 200),
                                           adx_period=trend_cfg.get('adx_period', 14))
            mfeat = compute_momentum_features(window,
                                              rsi_period=mom_cfg.get('rsi_period', 14),
                                              macd_fast=mom_cfg.get('macd_fast', 12),
                                              macd_slow=mom_cfg.get('macd_slow', 26),
                                              macd_signal=mom_cfg.get('macd_signal', 9))
            vfeat = compute_volatility_features(window,
                                                atr_period=vol_cfg.get('atr_period', 14),
                                                bbands_period=vol_cfg.get('bbands_period', 20),
                                                bbands_stddev=vol_cfg.get('bbands_stddev', 2),
                                                atr_percentile_window=vol_cfg.get('atr_percentile_window', 200))
            bb_w = None
            try:
                if vfeat.get('bbands_upper') and vfeat.get('bbands_lower'):
                    bb_w = (vfeat['bbands_upper'] - vfeat['bbands_lower']) / window['close'].iloc[-1]
            except Exception:
                pass
            row = {
                'ema_short': tfeat['ema_short'],
                'ema_medium': tfeat['ema_medium'],
                'ema_long': tfeat['ema_long'],
                'adx': tfeat['adx'],
                'rsi': mfeat['rsi'],
                'macd_line': mfeat['macd_line'],
                'macd_signal': mfeat['macd_signal'],
                'macd_hist': mfeat['macd_hist'],
                'atr': vfeat['atr'],
                'atr_percentile': vfeat['atr_percentile'],
                'bb_width': bb_w,
            }
            feat_rows[i] = row
            if keys is None:
                keys = list(row.keys())
    if keys is None:
        raise RuntimeError('Failed to compute any feature rows (insufficient data).')
    nan_template = {k: np.nan for k in keys}
    finalized = [fr if fr is not None else dict(nan_template) for fr in feat_rows]
    return pd.DataFrame(finalized, index=ohlcv.index)


def _select_trade_rows(trades: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
    """Align each trade's entry timestamp to the feature snapshot at or before it.

    Returns (aligned_trades, feature_rows_df, matched_bar_timestamps)
    """
    trades = trades.copy()
    trades['ts_entry_dt'] = (
        pd.to_datetime(trades['ts_entry'], unit='s', errors='coerce') if 'ts_entry' in trades.columns else
        pd.to_datetime(trades['ts_entry_ms'], unit='ms', errors='coerce') if 'ts_entry_ms' in trades.columns else
        pd.to_datetime(trades['ts_entry'])
    )
    feats = []
    matched_ts = []
    for _, row in trades.iterrows():
        ts = row['ts_entry_dt']
        if features_df.empty:
            feats.append(None); matched_ts.append(None); continue
        try:
            idx_loc = features_df.index.get_indexer([ts], method='pad')[0]
            matched_idx = features_df.index[idx_loc]
        except Exception:
            matched_idx = None
        if matched_idx is None or matched_idx not in features_df.index:
            feats.append(None); matched_ts.append(None)
        else:
            feats.append(features_df.loc[matched_idx])
            matched_ts.append(matched_idx)
    feat_rows_df = pd.DataFrame(feats).reset_index(drop=True)
    aligned_trades = trades.reset_index(drop=True)
    # Keep rows that have at least one non-null feature (imputer later will handle NaNs); drop rows that are entirely null
    mask = feat_rows_df.notna().any(axis=1)
    matched_ts = [t for t, keep in zip(matched_ts, mask) if keep]
    return aligned_trades.loc[mask], feat_rows_df.loc[mask], matched_ts


def _extract_vote_detail_features(trades_subset: pd.DataFrame) -> List[Dict[str, Any]]:
    """Parse vote_detail JSON/dicts per trade and extract additional features.

    We intentionally avoid any meta_gate probability fields to prevent leakage when training
    a new meta model.
    """
    out = []
    if 'vote_detail' not in trades_subset.columns:
        return [{} for _ in range(len(trades_subset))]
    for raw in trades_subset['vote_detail']:
        feat = {}
        jd = None
        if isinstance(raw, dict):
            jd = raw
        else:
            try:
                jd = json.loads(raw)
            except Exception:
                jd = {}
        # MTC scores
        try:
            mtc = jd.get('mtc_gate') or {}
            scores = mtc.get('scores') or {}
            for k, v in scores.items():
                if isinstance(v,(int,float)):
                    feat[f'mtc_{k}'] = v
        except Exception:
            pass
        # Liquidity metrics
        try:
            lq = jd.get('liquidity_gate') or {}
            for key in ['spread_bps','impact_50k','dr','rv_5s']:
                val = lq.get(key)
                if isinstance(val,(int,float)):
                    feat[f'lq_{key}'] = val
        except Exception:
            pass
        # Regime snapshot
        try:
            reg = jd.get('regime') or {}
            if isinstance(reg, dict):
                prof = reg.get('primary') or reg.get('profile')
                if isinstance(prof, str):
                    feat['regime_is_trend'] = 1.0 if prof.startswith('trend') else 0.0
                    feat['regime_is_chop'] = 1.0 if 'chop' in prof else 0.0
                conf = reg.get('confidence')
                if isinstance(conf,(int,float)):
                    feat['regime_conf'] = conf
        except Exception:
            pass
        out.append(feat)
    return out

def _add_cyclical_time_features(base_df: pd.DataFrame, timestamps: List[pd.Timestamp]) -> pd.DataFrame:
    if not timestamps:
        base_df['hour_sin']=base_df['hour_cos']=base_df['dow_sin']=base_df['dow_cos']=np.nan
        return base_df
    hours = np.array([t.hour if t is not None else np.nan for t in timestamps], dtype=float)
    dows = np.array([t.dayofweek if t is not None else np.nan for t in timestamps], dtype=float)
    with np.errstate(invalid='ignore'):
        base_df['hour_sin'] = np.sin(2*math.pi*hours/24.0)
        base_df['hour_cos'] = np.cos(2*math.pi*hours/24.0)
        base_df['dow_sin'] = np.sin(2*math.pi*dows/7.0)
        base_df['dow_cos'] = np.cos(2*math.pi*dows/7.0)
    return base_df


def build_dataset(config_path: str, start: str, end: str, out_dir: str, symbols: List[str] | None = None):
    settings = load_settings(config_path)
    cfg = settings.model_dump()
    symbols = symbols or cfg.get('runtime', {}).get('symbols') or []
    if not symbols:
        raise ValueError('No symbols provided in settings.runtime.symbols or CLI.')
    adapter = DataAdapter(cfg)
    all_X = []
    all_y = []
    feature_names = None
    rows_meta = []

    def _sanitize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
        # After refactor, dict columns should not exist; enforce numeric conversion + order preservation.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    for sym in symbols:
        # Load OHLCV for primary timeframe
        tf = cfg.get('runtime', {}).get('primary_timeframe', '5m')
        ohlcv = adapter.load_ohlcv(sym, tf, start, end)
        if ohlcv is None or ohlcv.empty:
            logger.warning(f"No OHLCV for {sym}; skipping")
            continue
        feats_df = _rolling_features(ohlcv, cfg)
        # Trades file expectation: reports/backtest_results/trades.csv or user-provided; fallback to synthetic? Skip if not present.
        # For now attempt standard location per symbol not separated; use global trades.csv if exists.
        trades_path = Path('reports/backtest_results/trades.csv')
        trades_df = None
        if trades_path.exists():
            try:
                trades_df = pd.read_csv(trades_path)
            except Exception:
                trades_df = None
        if trades_df is None or trades_df.empty:
            logger.warning('No trades.csv found; dataset will be empty for supervised learning.')
            continue
        # Filter trades for symbol
        trades_sym = trades_df[trades_df['symbol'].astype(str)==sym]
        if trades_sym.empty:
            continue
        # (1) Augment trades with synthetic positives if required
        meta_cfg_root = (cfg.get('meta_scorer') or {})
        synth_min_pos = int(meta_cfg_root.get('synthetic_min_pos', 0))
        if synth_min_pos > 0:
            current_pos = (trades_sym['result'].astype(str).str.upper().str.startswith('TP')).sum() if 'result' in trades_sym.columns else 0
            need = max(0, synth_min_pos - current_pos)
            if need > 0:
                rng = np.random.default_rng(42)
                candidate_ts = feats_df.index
                existing_ts = set(pd.to_datetime(trades_sym['ts_entry'], unit='s', errors='coerce')) if 'ts_entry' in trades_sym.columns else set()
                available = [t for t in candidate_ts if t not in existing_ts]
                pick = rng.choice(available, size=min(need, len(available)), replace=False) if available else []
                synth_rows = []
                for t in pick:
                    ts_sec = int(pd.Timestamp(t).timestamp())
                    side = 'LONG' if rng.random() < 0.5 else 'SHORT'
                    mfe = float(rng.uniform(150, 400))
                    mae = float(rng.uniform(50, 200))
                    synth_rows.append({'symbol': sym, 'ts_entry': ts_sec, 'side': side, 'result': 'TP', 'mfe': mfe, 'mae': mae})
                if synth_rows:
                    trades_sym = pd.concat([trades_sym, pd.DataFrame(synth_rows)], ignore_index=True)
                    logger.info(f"Added {len(synth_rows)} synthetic TP trades for {sym} (target min_pos={synth_min_pos}).")

        trades_aligned, feat_rows, matched_ts = _select_trade_rows(trades_sym, feats_df)
        if trades_aligned.empty:
            continue
        # (2) Negative sampling (non-trade bars) to diversify class balance
        neg_cfg = (meta_cfg_root.get('negative_sampling') or {})
        neg_ratio = float(neg_cfg.get('ratio', 0.0) or 0.0)
        neg_max = int(neg_cfg.get('max', 10000))
        if neg_ratio > 0:
            rng = np.random.default_rng(123)
            used_ts_set = set(matched_ts)
            candidate_ts = [t for t in feats_df.index if t not in used_ts_set]
            target_neg = min(int(len(trades_aligned) * neg_ratio), len(candidate_ts), neg_max)
            if target_neg > 0:
                pick_neg = rng.choice(candidate_ts, size=target_neg, replace=False)
                neg_feat_rows = feats_df.loc[pick_neg]
                neg_trades = []
                for t in pick_neg:
                    ts_sec = int(pd.Timestamp(t).timestamp())
                    neg_trades.append({'symbol': sym, 'ts_entry': ts_sec, 'side': 'NA', 'result': 'NEG'})
                neg_trades_df = pd.DataFrame(neg_trades)
                # Align negative rows DataFrame format to trades_aligned expectations
                neg_trades_df['ts_entry_dt'] = pd.to_datetime(neg_trades_df['ts_entry'], unit='s', errors='coerce')
                trades_aligned = pd.concat([trades_aligned, neg_trades_df], ignore_index=True)
                feat_rows = pd.concat([feat_rows.reset_index(drop=True), neg_feat_rows.reset_index(drop=True)], ignore_index=True)
                matched_ts.extend(list(pick_neg))
                logger.info(f"Added {target_neg} negative samples for {sym} (ratio={neg_ratio}).")
        # (3) Add cyclical time features based on matched bar timestamps (after augmentation)
        feat_rows = _add_cyclical_time_features(feat_rows, matched_ts)
        # Trade-level features from vote_detail (avoid leakage fields)
        vd_feats = _extract_vote_detail_features(trades_aligned)
        if vd_feats:
            vd_df = pd.DataFrame(vd_feats)
            feat_rows = pd.concat([feat_rows.reset_index(drop=True), vd_df.reset_index(drop=True)], axis=1)
        # Label selection
        meta_cfg = (cfg.get('meta_scorer') or {}).get('label', {})
        lbl_type = str(meta_cfg.get('type','tp_before_sl'))
        if lbl_type not in LABEL_FUNCS:
            logger.warning(f'Unknown label type {lbl_type}; defaulting to tp_before_sl')
            lbl_type = 'tp_before_sl'
        if lbl_type == 'r_after_n_bars':
            y = label_r_after_n_bars(trades_aligned, int(meta_cfg.get('horizon_bars', 96)))
        elif lbl_type == 'mfe_exceeds':
            y = label_mfe_exceeds(trades_aligned, float(meta_cfg.get('threshold', 1.5)))
        else:
            y = label_tp_before_sl(trades_aligned)
        # Sanitize feature frame before numeric matrix conversion
        feat_rows = _sanitize_feature_frame(feat_rows)
        X = feat_rows.values.astype(np.float32)
        if feature_names is None:
            feature_names = list(feat_rows.columns)
        all_X.append(X)
        all_y.append(y)
        rows_meta.append({'symbol': sym, 'rows': int(len(X))})
        # Persist timestamps aligned for purge/embargo aware CV
        if 'ts_entry_dt' in trades_aligned.columns:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(out_dir)/f'timestamps_{sym}.npy', trades_aligned['ts_entry_dt'].astype(np.int64).values)

    if not all_X:
        logger.warning('No dataset rows produced.')
        X_final = np.zeros((0,0), dtype=np.float32)
        y_final = np.zeros((0,), dtype=np.uint8)
    else:
        X_final = np.vstack(all_X)
        y_final = np.concatenate(all_y)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(out_dir)/'features.npy', X_final)
    np.save(Path(out_dir)/'labels.npy', y_final)
    (Path(out_dir)/'feature_names.json').write_text(json.dumps(feature_names or []))
    (Path(out_dir)/'meta.json').write_text(json.dumps({'from': start,'to': end,'symbols': symbols,'rows': int(len(X_final)),'class_balance': float(y_final.mean()) if len(y_final)>0 else None,'notes':'executed-trades-only'}))
    # Contract guard: ensure no leakage columns accidentally included
    if feature_names:
        leak_terms = [n for n in feature_names if any(x in n.lower() for x in ['meta_gate','prob','calibrated_prob','p_'])]
        if leak_terms:
            logger.warning(f"Potential leakage feature names detected: {leak_terms}")
    logger.success(f"Dataset built: rows={len(X_final)} features={(feature_names or [])}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--from', dest='from_', required=True)
    ap.add_argument('--to', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--symbols', type=str, default=None, help='Comma separated symbols (override settings.runtime.symbols)')
    args = ap.parse_args()
    syms = [s.strip() for s in args.symbols.split(',')] if args.symbols else None
    build_dataset(args.config, args.from_, args.to, args.out, syms)
