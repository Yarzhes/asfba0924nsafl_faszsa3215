"""Calibration stub for BrokerSim (Sprint 36 optional).

Usage (future):
    python -m ultra_signals.sim.calibrate --real real_fills.csv --sim sim_fills.csv --out settings.yaml

Current implementation:
- Loads real & simulated fills CSVs
- Computes mean slippage/time-to-fill deltas
- Suggests adjusted impact_factor and submit latency mu shift
- Writes a YAML patch (does not auto-overwrite settings unless --apply)
"""
from __future__ import annotations
import argparse, yaml, statistics, math
from pathlib import Path
import csv

def load_fills(path: str):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try: rows.append(row)
            except Exception: continue
    return rows

def summarize(rows):
    sl = []; ttf = []
    for r in rows:
        try:
            if 'slippage_bps' in r and r['slippage_bps'] != '': sl.append(float(r['slippage_bps']))
            if 'time_to_full_ms' in r and r['time_to_full_ms'] != '': ttf.append(float(r['time_to_full_ms']))
        except Exception: continue
    return {
        'slippage_mean': statistics.mean(sl) if sl else 0.0,
        'slippage_p95': (sorted(sl)[int(0.95*len(sl))-1] if sl else 0.0) if len(sl)>=2 else (sl[0] if sl else 0.0),
        'ttf_mean': statistics.mean(ttf) if ttf else 0.0,
    }

def suggest_adjustments(real_sum, sim_sum):
    delta_slip = real_sum['slippage_mean'] - sim_sum['slippage_mean']
    impact_adj = 0.0
    if sim_sum['slippage_mean']:
        impact_adj = (delta_slip / sim_sum['slippage_mean'])*0.5  # half correction
    latency_shift = 0.0
    if sim_sum['ttf_mean']:
        latency_shift = (real_sum['ttf_mean'] - sim_sum['ttf_mean']) * 0.2  # dampen
    return impact_adj, latency_shift

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--real', required=True)
    p.add_argument('--sim', required=True)
    p.add_argument('--settings', default='settings.yaml')
    p.add_argument('--out', default='broker_calibration_patch.yaml')
    p.add_argument('--apply', action='store_true')
    args = p.parse_args(argv)

    real = load_fills(args.real); sim = load_fills(args.sim)
    real_sum = summarize(real); sim_sum = summarize(sim)
    impact_adj, latency_shift = suggest_adjustments(real_sum, sim_sum)
    patch = {
        'broker_sim': {
            'calibration': {
                'real': real_sum,
                'sim': sim_sum,
                'suggested_impact_factor_delta': impact_adj,
                'suggested_submit_latency_shift_ms': latency_shift,
            }
        }
    }
    out_path = Path(args.out)
    out_path.write_text(yaml.safe_dump(patch, sort_keys=False), encoding='utf-8')
    print(f"Wrote calibration patch -> {out_path}")
    if args.apply:
        try:
            import shutil
            settings_path = Path(args.settings)
            existing = yaml.safe_load(settings_path.read_text(encoding='utf-8')) or {}
            # apply delta to first venue impact_factor (BINANCE if exists)
            venues = ((existing.get('broker_sim') or {}).get('venues') or {})
            first = next(iter(venues.keys()), None)
            if first:
                cur = float(((venues[first]).get('slippage') or {}).get('impact_factor', 0.0))
                new = max(0.0, cur + impact_adj)
                venues[first]['slippage']['impact_factor'] = new
            settings_path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding='utf-8')
            print('Applied impact_factor adjustment to settings.yaml')
        except Exception as e:
            print(f'Apply failed: {e}')

if __name__ == '__main__':
    main()
