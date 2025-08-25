"""Flow Metrics (Sprint 11 Feature Pack v2)

Computes advanced order-flow and microstructure metrics per bar.
All functions are defensive: failures are caught and result values are None so
FeatureStore resilience is preserved.

Inputs expected (best-effort):
- OHLCV DataFrame (for volume and price context)
- Recent trades (timestamp, price, qty, is_buyer_maker)
- Recent liquidations (timestamp, side, notional)
- Top-of-book snapshot (bid, ask, B, A)
- Optional open interest series (if later integrated)

Output: dict mapping metric names (matching FlowMetricsFeatures fields).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger

# Light EMA helper
_DEF_EPS = 1e-9

def _safe_last(series: List[Any]) -> Optional[Any]:
    return series[-1] if series else None


def _compute_cvd(trades: List[Tuple[int,float,float,bool]]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return cumulative volume delta, bar delta, buy volume, sell volume.
    trades: list of (ts, price, qty, is_buyer_maker) where is_buyer_maker=True means SELL aggressor (taker sells).
    We treat is_buyer_maker=False as aggressive buy.
    """
    if not trades:
        return None, None, None, None
    buy_vol = 0.0
    sell_vol = 0.0
    for _ts, _p, q, is_buyer_maker in trades:
        if is_buyer_maker:  # seller aggressive
            sell_vol += q
        else:
            buy_vol += q
    bar_delta = buy_vol - sell_vol
    # For cumulative we rely on running memory outside here; caller will stitch via state.
    return bar_delta, bar_delta, buy_vol, sell_vol


def _order_flow_imbalance(book_top: Optional[Dict[str,float]]) -> Optional[float]:
    if not book_top:
        return None
    try:
        bid_sz = float(book_top.get('B', 0.0))
        ask_sz = float(book_top.get('A', 0.0))
        total = bid_sz + ask_sz
        if total <= 0:
            return None
        return (bid_sz - ask_sz) / total
    except Exception:
        return None


def _depth_imbalance_cross(depth_a: Optional[Dict[str,float]], depth_b: Optional[Dict[str,float]]) -> Optional[float]:
    # Placeholder for multi-venue depth; if one missing, return None
    if not depth_a or not depth_b:
        return None
    try:
        bid_a = depth_a.get('B',0.0); bid_b = depth_b.get('B',0.0)
        total = bid_a + bid_b
        if total <= 0:
            return None
        return (bid_a - bid_b)/ total
    except Exception:
        return None


def _cross_spread(mid_a: Optional[float], mid_b: Optional[float]) -> Optional[float]:
    if mid_a is None or mid_b is None:
        return None
    try:
        return (mid_a - mid_b) / ((mid_a + mid_b)/2.0 + _DEF_EPS)
    except Exception:
        return None


def compute_flow_metrics(
    ohlcv: pd.DataFrame,
    trades: List[Tuple[int,float,float,bool]],
    liquidations: List[Tuple[int,str,float]],
    book_top: Optional[Dict[str,float]],
    settings: Dict[str,Any],
    state: Dict[str,Any]
) -> Dict[str,Any]:
    cfg = ((settings or {}).get('features', {}) or {}).get('flow_metrics', {})
    if not cfg.get('enabled', True):
        return {}

    out: Dict[str,Any] = {}
    try:
        # Volume z (reuse existing volume if present)
        vol_z = None
        try:
            if 'volume' in ohlcv.columns:
                vols = ohlcv['volume'].astype(float)
                if len(vols) >= 10:
                    mean = vols.rolling(30, min_periods=5).mean().iloc[-1]
                    std = vols.rolling(30, min_periods=5).std(ddof=0).iloc[-1]
                    v_last = vols.iloc[-1]
                    if std and std > 0:
                        vol_z = (v_last - mean) / std
                        out['volume_z'] = vol_z
        except Exception:
            pass

        # CVD
        try:
            bar_cvd, bar_delta, buy_vol, sell_vol = _compute_cvd(trades)
            if bar_cvd is not None:
                prev_cvd = state.get('cvd_last', 0.0)
                cvd_cum = prev_cvd + bar_cvd
                out['cvd'] = cvd_cum
                out['cvd_chg'] = bar_delta
                out['buy_volume'] = buy_vol
                out['sell_volume'] = sell_vol
                state['cvd_last'] = cvd_cum
        except Exception as e:
            logger.debug('CVD compute fail: {}', e)

        # Order-flow imbalance (top of book)
        try:
            ofi = _order_flow_imbalance(book_top)
            if ofi is not None:
                out['ofi'] = float(np.clip(ofi, -1, 1))
        except Exception:
            pass

        # Open interest stub (placeholder: expects externally fed state)
        try:
            oi_series = state.get('oi_series')  # list of floats
            if oi_series and len(oi_series) >= 2:
                oi = float(oi_series[-1])
                oi_prev = float(oi_series[-2])
                out['oi'] = oi
                out['oi_prev'] = oi_prev
                if oi_prev != 0:
                    out['oi_rate'] = (oi - oi_prev)/abs(oi_prev)
        except Exception:
            pass

        # Liquidation pulse
        try:
            liq_cfg = cfg.get('liquidations', {})
            cluster_min = int(liq_cfg.get('min_cluster_size', 3))
            window_ms = int(5 * 60 * 1000)
            now_ts = int(pd.Timestamp(ohlcv.index[-1]).value // 1_000_000)
            cutoff = now_ts - window_ms
            liqs_recent = [l for l in liquidations if l[0] >= cutoff]
            if liqs_recent:
                out['liq_events'] = len(liqs_recent)
                out['liq_notional_sum'] = float(sum(l[2] for l in liqs_recent))
                if len(liqs_recent) >= cluster_min:
                    out['liq_cluster'] = 1
                    side_bias = sum(1 if l[1]=='BUY' else -1 for l in liqs_recent)
                    out['liq_cluster_side'] = 'BUY' if side_bias>0 else 'SELL'
                    out['last_liq_cluster_ts'] = now_ts
                else:
                    out['liq_cluster'] = 0
            else:
                out['liq_events'] = 0
                out['liq_cluster'] = 0
        except Exception:
            pass

        # Depth imbalance (reuse ofi if no multi-venue yet)
        if 'ofi' in out:
            out['depth_imbalance'] = out['ofi']

        # Cross-exchange spread placeholder (needs external prices)
        try:
            spread_cfg = cfg.get('spread', {})
            max_bp = float(spread_cfg.get('max_bp', 5))
            # stub: if we had mid prices list in state
            mids = state.get('mids_multi')  # list of mids
            if mids and len(mids) >= 2:
                m_arr = np.array(mids, dtype=float)
                mn = m_arr.mean()
                dev = np.max(np.abs(m_arr - mn) / (mn + _DEF_EPS))
                spread_bps = dev * 10_000
                out['spread_bps'] = spread_bps
                out['spread_dev_flag'] = 1 if spread_bps > max_bp else 0
        except Exception:
            pass

        # Volume anomaly flag
        try:
            if vol_z is not None:
                thr = float(cfg.get('volume', {}).get('z_threshold', 2.5))
                out['volume_anom'] = 1 if abs(vol_z) >= thr else 0
        except Exception:
            pass

    except Exception as e:
        logger.exception('Flow metrics compute fatal: {}', e)

    return out
