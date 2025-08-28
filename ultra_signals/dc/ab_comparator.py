"""A/B comparator: run identical aggressive orders against synthetic orderbook and tick replayer.

Produces simple metrics: fill ratios, slip (bps), and placeholders for P&L/MaxDD comparisons.
"""
from __future__ import annotations
from typing import Iterable, Dict, Any
from ultra_signals.sim.orderbook import SyntheticOrderBook
from ultra_signals.dc.tick_replayer import TickReplayer
import csv
import math


def run_ab(prices_events: Iterable[Dict[str, Any]], orders: Iterable[Dict[str, Any]], levels: int = 10) -> Dict[str, Any]:
    """prices_events: sequence of market events (snapshot/delta/trade). orders: sequence of aggressive orders to test.

    Returns summary dict with tick vs synthetic fill stats.
    """
    # Build synthetic book from last bar-like event (simple heuristic)
    synth = SyntheticOrderBook(symbol='X', levels=levels)
    # create tick replayer and feed events
    tr = TickReplayer(max_levels=levels)
    tr.feed_from_iter(prices_events)

    results = {"tick": [], "synth": []}
    # For each order, place against synthetic and tick engines
    for o in orders:
        side = o.get('side')
        size = float(o.get('size') or 0)
        px = float(o.get('price') or 0)
        # Synthetic: rebuild from a small synthetic bar
        synth.rebuild_from_bar({'close': px, 'high': px, 'low': px})
        # synth ladder match (simple FIFO using SyntheticOrderBook.ladder())
        ladder = synth.ladder() if side == 'buy' else synth.ladder_bid()
        remaining = size
        filled = 0.0
        slippage_bps = 0.0
        for lvl_px, qty in ladder:
            take = min(remaining, qty)
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        results['synth'].append({'requested': size, 'filled': filled, 'slip_bps': slippage_bps})

        # Tick engine: push a trade event and replay one step
        import time
        ev = {"ts": int(time.time()*1000), "type": "trade", "side": side, 'size': size, 'price': px}
        tr.add_event(ev)
        fills = tr.replay()
        if fills:
            last = fills[-1]
            results['tick'].append({'requested': size, 'filled': last.get('filled_qty', 0.0), 'slip_bps': 0.0})
        else:
            results['tick'].append({'requested': size, 'filled': 0.0, 'slip_bps': None})

    # compute simple aggregate metrics
    def agg(arr):
        total_req = sum([a['requested'] for a in arr])
        total_filled = sum([a['filled'] for a in arr])
        fill_ratio = (total_filled / total_req) if total_req else None
        return {'total_requested': total_req, 'total_filled': total_filled, 'fill_ratio': fill_ratio}

    out = {'synth': agg(results['synth']), 'tick': agg(results['tick'])}
    # compute sim_error metrics: difference in fill_ratio -> simple PF like metric
    try:
        synth_fr = out['synth']['fill_ratio'] or 0.0
        tick_fr = out['tick']['fill_ratio'] or 0.0
        out['sim_error_fill_ratio'] = (synth_fr - tick_fr)
    except Exception:
        out['sim_error_fill_ratio'] = None

    # simple win-rate proxy: fraction of orders with full fill
    def winrate(arr):
        if not arr: return None
        return sum(1 for a in arr if a['filled'] >= a['requested']) / len(arr)

    out['synth']['winrate'] = winrate(results['synth'])
    out['tick']['winrate'] = winrate(results['tick'])
    # drawdown proxy from hypothetical cumulative filled vs requested
    def maxdd(arr):
        if not arr: return None
        cum = 0.0
        peak = 0.0
        maxdd = 0.0
        for a in arr:
            cum += (a['filled'] - a['requested'])
            peak = max(peak, cum)
            dd = peak - cum
            maxdd = max(maxdd, dd)
        return maxdd

    out['synth']['maxdd'] = maxdd(results['synth'])
    out['tick']['maxdd'] = maxdd(results['tick'])

    # CSV export for per-order results
    try:
        with open('ab_comparator_results.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['engine','requested','filled','fill_ratio'])
            for r in results['synth']:
                w.writerow(['synth', r['requested'], r['filled'], (r['filled']/r['requested']) if r['requested'] else None])
            for r in results['tick']:
                w.writerow(['tick', r['requested'], r['filled'], (r['filled']/r['requested']) if r['requested'] else None])
    except Exception:
        pass

    return out


__all__ = ['run_ab']
