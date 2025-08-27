"""Reconciliation logic (Sprint 25).

Algorithm summarised:
1. Load pending/acked/open orders.
2. Query venue adapters for open orders + positions.
3. Match by client_order_id; detect lost/orphan orders.
4. Rebuild positions from fills if divergence.
5. Ensure risk + offsets baseline rows.

This module is intentionally conservative; ambiguous states lead to pause flag.
"""
from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Tuple
from loguru import logger
from .db import fetchall, execute, tx
from .db import upsert_oco_link, fetch_oco_link

class ReconcileReport(dict):  # simple structured report
    pass

async def reconcile(venue_router, *, pause_cb=None) -> ReconcileReport:
    report: ReconcileReport = ReconcileReport(changes=[], warnings=[], errors=[], summary={})
    db_open = fetchall("SELECT * FROM orders_outbox WHERE status IN ('PENDING','ACKED','OPEN','PARTIAL')")
    db_by_id = {r['client_order_id']: r for r in db_open}
    venue_orders: Dict[str, Tuple[str, Any]] = {}
    venue_positions: Dict[str, Dict[str, Any]] = {}
    if venue_router:
        for vid, adapter in venue_router.venues.items():  # type: ignore
            try:
                oos = await adapter.open_orders()  # type: ignore
                for ack in oos:
                    if ack.client_order_id not in venue_orders:
                        venue_orders[ack.client_order_id] = (vid, ack)
                poss = await adapter.positions()  # type: ignore
                for p in poss:
                    cur = venue_positions.get(p.symbol, {"qty":0.0, "notional":0.0})
                    venue_positions[p.symbol] = {
                        "qty": cur["qty"] + p.qty,
                        "notional": cur["notional"] + abs(p.qty)*p.avg_px,
                    }
            except Exception as e:  # pragma: no cover
                report['warnings'].append(f"venue {vid} query failed {e}")
    # Orphans detection
    for cid, (vid, ack) in venue_orders.items():
        if cid not in db_by_id:
            # Insert synthetic row
            execute("INSERT OR IGNORE INTO orders_outbox(client_order_id, venue, symbol, side, type, qty, price, reduce_only, parent_id, status, venue_order_id, created_ts, updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,strftime('%s','now')*1000,strftime('%s','now')*1000)",
                    (cid, vid, getattr(ack,'symbol',None), getattr(ack,'side',None), getattr(ack,'type','LIMIT'), getattr(ack,'qty',None), getattr(ack,'price',None), 0, None, getattr(ack,'status','ACKED'), getattr(ack,'venue_order_id',None)))
            report['changes'].append(f"orphan_venue_order_added {cid}")
    # Lost orders
    for cid, row in db_by_id.items():
        if cid not in venue_orders and row.get('status') in ('PENDING','ACKED','OPEN','PARTIAL'):
            execute("UPDATE orders_outbox SET status='LOST', updated_ts=strftime('%s','now')*1000 WHERE client_order_id=?", (cid,))
            report['changes'].append(f"lost_order_marked {cid}")
    # Positions alignment: compare DB vs aggregate venue (simple)
    db_pos = fetchall("SELECT symbol, qty, avg_px FROM positions")
    pos_by_sym = {r['symbol']: r for r in db_pos}
    for sym, v in venue_positions.items():
        venue_qty = v['qty']
        venue_avg = (v['notional']/abs(venue_qty)) if abs(venue_qty) > 0 else 0.0
        db_row = pos_by_sym.get(sym)
        if not db_row or abs(db_row['qty'] - venue_qty) > 1e-8:
            execute("INSERT INTO positions(symbol, qty, avg_px, realized_pnl, updated_ts, venue, hedge) VALUES(?,?,?,?,strftime('%s','now')*1000,NULL,0) ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, avg_px=excluded.avg_px, updated_ts=excluded.updated_ts", (sym, venue_qty, venue_avg, 0.0))
            report['changes'].append(f"position_adjusted {sym} -> {venue_qty}")
    # Risk runtime ensure today row
    from datetime import datetime
    day = datetime.utcnow().strftime('%Y-%m-%d')
    execute("INSERT OR IGNORE INTO risk_runtime(day, realized_pnl, consecutive_losses, paused) VALUES(?,?,?,?)", (day, 0.0, 0, 0))
    report['summary'] = {
        'open_orders_db': len(db_open),
        'open_orders_venue': len(venue_orders),
        'positions_db': len(db_pos),
        'positions_venue': len(venue_positions),
    }
    if any(c.startswith('lost_order') for c in report['changes']):
        report['warnings'].append('lost_orders_present')
    # Rebuild OCO links heuristically: group orders sharing parent_id pattern
    try:
        children = fetchall("SELECT client_order_id,parent_id FROM orders_outbox WHERE parent_id IS NOT NULL")
        groups: dict[str,list[str]] = {}
        for c in children:
            groups.setdefault(c['parent_id'], []).append(c['client_order_id'])
        for parent, kids in groups.items():
            existing = fetch_oco_link(parent)
            if existing:
                continue
            # Simple heuristic: stop id contains 'SL', tps contain 'TP'
            stop_id = next((k for k in kids if 'SL' in k.upper()), None)
            tp_ids = [k for k in kids if 'TP' in k.upper()]
            if stop_id or tp_ids:
                upsert_oco_link(parent, stop_id, tp_ids)
                report['changes'].append(f"oco_rebuilt {parent}")
    except Exception:
        pass
    if pause_cb and report['warnings']:
        try:
            pause_cb('RECONCILE')
        except Exception:
            pass
    logger.info(f"[Reconcile] report={report}")
    return report

__all__ = ['reconcile','ReconcileReport']
