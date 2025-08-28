"""Whale / Smart Money Feature Computation (Sprint 41)

Pure transformation layer combining pre-aggregated whale pipeline state into a
`WhaleFeatures` model (see `core.custom_types`). This module does NOT perform
network I/O; collectors populate lightweight in-memory state structures that
are passed here each bar.

Design:
    - Idempotent & defensive: any failure returns {} so FeatureStore stays resilient.
    - Accepts rolling windows already materialized (lists / deques) with
      per-event dict records produced by async collectors.
    - Computes z-scores with small epsilon & min sample guards.

Inputs (dict expected fields):
    state: dict containing sub-dicts for keys:
        'exchange_flows': {'records': [...], 'windows': {'s': 3600, 'm': 6*3600, 'l': 24*3600}}
            each record: { 'ts': epoch_ms, 'symbol': 'BTCUSDT', 'direction': 'DEPOSIT'|'WITHDRAWAL', 'usd': float }
        'blocks': {'records': [...]}  record: { 'ts': ms, 'symbol': str, 'notional': float, 'side': 'BUY'|'SELL', 'type': 'BLOCK'|'SWEEP' }
        'options': {'snapshot': {...}}  snapshot precomputed anomaly stats
        'smart_money': {'records': [...], 'hit_rate_30d': float}

Outputs:
    dict mapping WhaleFeatures field names -> values. Missing metrics omitted.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import statistics
import time
try:
    # optional integration with onchain feature mapping
    from ultra_signals.onchain.feature_map import map_snapshot_to_features
except Exception:
    map_snapshot_to_features = None

_EPS = 1e-9

def _z(v: float, mean: float, std: float) -> float:
    if std is None or std <= 0:
        return 0.0
    return (v - mean) / (std + _EPS)

def _window_filter(records: List[Dict[str, Any]], now_ms: int, window_sec: int) -> List[Dict[str, Any]]:
    if not records:
        return []
    cutoff = now_ms - window_sec * 1000
    return [r for r in records if r.get('ts', 0) >= cutoff]

def compute_whale_features(symbol: str, now_ms: int, state: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        # Config windows (fallback defaults short=1h, medium=6h, long=24h)
        win_cfg = (cfg.get('windows') or {})
        w_s = int(win_cfg.get('short_sec', 3600))
        w_m = int(win_cfg.get('medium_sec', 6*3600))
        w_l = int(win_cfg.get('long_sec', 24*3600))

        # -------- Exchange Net Flows --------
        exch = state.get('exchange_flows') or {}
        flow_records: List[Dict[str, Any]] = [r for r in (exch.get('records') or []) if r.get('symbol') == symbol]
        dep = [r for r in flow_records if r.get('direction') == 'DEPOSIT']
        wdr = [r for r in flow_records if r.get('direction') == 'WITHDRAWAL']
        def _net(window):
            d = _window_filter(dep, now_ms, window)
            w_ = _window_filter(wdr, now_ms, window)
            return sum(r.get('usd',0.0) for r in w_) - sum(r.get('usd',0.0) for r in d)
        net_s = _net(w_s)
        net_m = _net(w_m)
        net_l = _net(w_l)
        out['whale_net_inflow_usd_s'] = net_s
        out['whale_net_inflow_usd_m'] = net_m
        out['whale_net_inflow_usd_l'] = net_l

        # Inflow z (use deposits only baseline)
        dep_usd_series = [r.get('usd',0.0) for r in dep if r.get('usd')]
        if len(dep_usd_series) >= 10:
            mean_dep = statistics.fmean(dep_usd_series)
            std_dep = statistics.pstdev(dep_usd_series) if len(dep_usd_series) > 1 else 0.0
            last_dep_total = sum(r.get('usd',0.0) for r in _window_filter(dep, now_ms, w_s))
            out['whale_inflow_z_s'] = _z(last_dep_total, mean_dep, std_dep)
            last_dep_total_m = sum(r.get('usd',0.0) for r in _window_filter(dep, now_ms, w_m))
            out['whale_inflow_z_m'] = _z(last_dep_total_m, mean_dep, std_dep)

        # Burst flags: compare short window vs median of trailing chunk
        if dep_usd_series:
            short_dep = sum(r.get('usd',0.0) for r in _window_filter(dep, now_ms, w_s))
            short_wdr = sum(r.get('usd',0.0) for r in _window_filter(wdr, now_ms, w_s))
            baseline = statistics.median(dep_usd_series[-200:]) if len(dep_usd_series) >= 3 else (dep_usd_series[-1] if dep_usd_series else 0.0)
            if baseline > 0:
                if short_dep > baseline * float(cfg.get('deposit_burst_multiplier', 3.0)):
                    out['exch_deposit_burst_flag'] = 1
                if short_wdr > baseline * float(cfg.get('withdrawal_burst_multiplier', 3.0)):
                    out['exch_withdrawal_burst_flag'] = 1

        # -------- Block / Sweep detection (records pre-tagged) --------
        blocks_state = state.get('blocks') or {}
        block_records = [r for r in (blocks_state.get('records') or []) if r.get('symbol') == symbol]
        recent_blocks_5m = _window_filter(block_records, now_ms, 300)
        if recent_blocks_5m:
            out['block_trade_count_5m'] = len(recent_blocks_5m)
            notional_sum = sum(r.get('notional',0.0) for r in recent_blocks_5m)
            out['block_trade_notional_5m'] = notional_sum
            # z vs rolling p99 approx using historical notional list if provided
            hist_notionals = [r.get('notional',0.0) for r in block_records]
            if len(hist_notionals) >= 30:
                mean_b = statistics.fmean(hist_notionals)
                std_b = statistics.pstdev(hist_notionals) if len(hist_notionals) > 1 else 0.0
                out['block_trade_notional_p99_z'] = _z(notional_sum, mean_b, std_b)
        # Sweep / iceberg heuristics (pre-tagged type)
        if any(r.get('type') == 'SWEEP' and r.get('side') == 'SELL' for r in recent_blocks_5m):
            out['sweep_sell_flag'] = 1
        if any(r.get('type') == 'SWEEP' and r.get('side') == 'BUY' for r in recent_blocks_5m):
            out['sweep_buy_flag'] = 1
        # Iceberg replenish score: fraction of blocks flagged as ICEBERG in recent window
        icebergs = [r for r in recent_blocks_5m if r.get('type') == 'ICEBERG']
        if recent_blocks_5m:
            out['iceberg_replenish_score'] = len(icebergs) / len(recent_blocks_5m)

        # -------- Options anomalies (snapshot) --------
        opt = (state.get('options') or {}).get('snapshot') or {}
        for k_src, k_dst in [
            ('call_put_volratio_z', 'opt_call_put_volratio_z'),
            ('oi_delta_1h_z', 'opt_oi_delta_1h_z'),
            ('skew_shift_z', 'opt_skew_shift_z'),
            ('block_trade_flag', 'opt_block_trade_flag'),
        ]:
            v = opt.get(k_src)
            if v is not None:
                out[k_dst] = v

        # -------- Smart money wallet pressure --------
        sm = state.get('smart_money') or {}
        sm_records = [r for r in (sm.get('records') or []) if r.get('symbol') == symbol]
        buy_p = [r.get('usd',0.0) for r in sm_records if r.get('side') == 'BUY']
        sell_p = [r.get('usd',0.0) for r in sm_records if r.get('side') == 'SELL']
        if buy_p:
            out['smart_money_buy_pressure_s'] = sum(buy_p[-50:])
        if sell_p:
            out['smart_money_sell_pressure_s'] = sum(sell_p[-50:])
        hr = sm.get('hit_rate_30d')
        if hr is not None:
            out['smart_money_hit_rate_30d'] = hr

        # Composite pressure (withdrawals + block buy - deposits - block sells normalized)
        try:
            comp = 0.0
            comp += max(0.0, net_s)  # net withdrawals positive (bullish)
            comp += (out.get('smart_money_buy_pressure_s') or 0.0) * 0.25
            comp -= max(0.0, -net_s)  # net deposits negative (bearish)
            comp -= (out.get('smart_money_sell_pressure_s') or 0.0) * 0.25
            out['composite_pressure_score'] = comp
        except Exception:
            pass

        out['last_update_ts'] = now_ms
        # Count active groups
        active = sum(1 for k in ['exchange_flows','blocks','options','smart_money'] if state.get(k))
        out['sources_active'] = active
        # Merge external onchain snapshot if present (new Sprint 57 collector)
        try:
            onchain = state.get('onchain_snapshot')
            if onchain and map_snapshot_to_features:
                mapped = map_snapshot_to_features(onchain)
                # merge but keep existing keys (onchain augments)
                for k, v in mapped.items():
                    if k not in out:
                        out[k] = v
                    else:
                        # prefer existing non-zero or existing value
                        if (out.get(k) in (None, 0)) and v:
                            out[k] = v
        except Exception:
            pass
    except Exception:
        return {}
    return out

# Minimal manual test
if __name__ == '__main__':
    now = int(time.time()*1000)
    demo_state = {
        'exchange_flows': {'records': [
            {'ts': now-1000*400, 'symbol':'BTCUSDT', 'direction':'DEPOSIT', 'usd': 5_000_000},
            {'ts': now-1000*300, 'symbol':'BTCUSDT', 'direction':'WITHDRAWAL', 'usd': 7_000_000},
        ]},
        'blocks': {'records': [
            {'ts': now-60_000, 'symbol':'BTCUSDT', 'notional': 3_000_000, 'side':'SELL', 'type':'SWEEP'},
        ]},
        'options': {'snapshot': {'call_put_volratio_z': 2.1}},
        'smart_money': {'records': [ {'ts': now-30_000, 'symbol':'BTCUSDT','usd':800000,'side':'BUY'} ], 'hit_rate_30d':0.58}
    }
    print(compute_whale_features('BTCUSDT', now, demo_state, {'windows':{}}))