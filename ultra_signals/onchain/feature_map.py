"""Map aggregated on-chain cohort snapshot into whale feature dict.

Produces keys like whale_cex_net_inflow_usd_15m, whale_withdraw_spike_flag,
stablecoin_rotation_usd, cohort_concentration_idx, and entity_confidence stub.
"""
from __future__ import annotations
from typing import Dict, Any
from .aggregator import CohortAggregator, zscore


def map_snapshot_to_features(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    glob = snapshot.get('global', {})
    # map windows
    for w in ['15m', '1h', '24h']:
        key = f'whale_cex_net_inflow_usd_{w}'
        out[key] = glob.get(f'net_{w}', 0.0)

    # simple spike flags: compare latest inflow/outflow against mean across cohorts
    cohorts = snapshot.get('cohorts', {})
    # compute cohort concentration: max cohort net / total abs net
    nets = [abs(c.get('net_24h', 0.0)) for c in cohorts.values()]
    total_abs = sum(nets) or 1.0
    max_abs = max(nets) if nets else 0.0
    out['cohort_concentration_idx'] = max_abs / total_abs

    # stablecoin rotation and bridge inflow simple sums
    out['stablecoin_rotation_usd'] = sum(c.get('inflow_24h', 0.0) for name, c in cohorts.items() if 'stable' in name.lower())
    out['bridge_inflow_usd'] = sum(c.get('inflow_24h', 0.0) for name, c in cohorts.items() if 'bridge' in name.lower())

    # basic flags from zscore of 15m net against 24h mean across cohorts
    hist = [c.get('net_24h', 0.0) for c in cohorts.values()]
    curr = glob.get('net_15m', 0.0)
    out['whale_flow_pctl'] = 0.0
    out['whale_withdraw_spike_flag'] = 1 if curr < 0 and zscore(curr, hist) < -1.5 else 0
    out['whale_deposit_spike_flag'] = 1 if curr > 0 and zscore(curr, hist) > 1.5 else 0

    # entity confidence stub (max of confidences if present in snapshot)
    out['entity_confidence'] = 0.8

    return out


__all__ = ['map_snapshot_to_features']
