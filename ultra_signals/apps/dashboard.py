"""Live risk & PnL dashboard (Sprint 26).

Run:
    python -m ultra_signals.apps.dashboard --port 8787

Provides:
  GET /status          -> snapshot JSON
  GET /equity_curve    -> historical equity curve
  GET /alerts          -> recent alerts
  WS  /ws/stream       -> realtime push of snapshot + alerts

Lightweight HTML/JS frontend served at '/'.
"""
from __future__ import annotations
import argparse
import json
import asyncio
import time
from typing import Any, Dict
from loguru import logger

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
except Exception as e:  # pragma: no cover
    raise SystemExit("fastapi not installed. Add fastapi[all] uvicorn to requirements.")

from ultra_signals.persist.db import init_db, fetchall, fetchone
from ultra_signals.persist.migrations import apply_migrations
from ultra_signals.core.alerts import subscribe, unsubscribe, recent_alerts
from ultra_signals.events import gating as event_gating
try:  # Sprint 29 liquidity telemetry
    from ultra_signals.engine.gates import liquidity_gate as _lq_mod  # type: ignore
except Exception:  # pragma: no cover
    _lq_mod = None

HTML = """<!DOCTYPE html><html><head><title>Ultra Signals Dashboard</title>
<meta charset='utf-8'/>
<style>body{font-family:Arial;background:#111;color:#eee;margin:0;padding:0}header{background:#222;padding:10px;font-size:18px}#cards{display:flex;gap:12px;margin:12px;flex-wrap:wrap} .card{background:#1d1f27;padding:12px;border-radius:6px;min-width:160px} .pos-table, .ord-table{width:100%;border-collapse:collapse;font-size:12px} th,td{padding:4px 6px;border-bottom:1px solid #333} .green{color:#4caf50} .red{color:#ff5252} #alerts{position:fixed;top:0;right:0;max-width:320px} .alert{background:#272b38;margin:4px;padding:6px;border-left:4px solid #f39c12;font-size:12px} .warn{border-color:#f39c12} .err{border-color:#e74c3c} .info{border-color:#3498db} .banner{background:#b71c1c;padding:8px;text-align:center;color:#fff;display:none} </style>
</head><body><div class='banner' id='pauseBanner'>PAUSED</div><header>Ultra Signals – Live Dashboard</header>
<div id='cards'></div>
<h3 style='margin:12px'>Open Positions</h3><div style='margin:12px'><table class='pos-table' id='positions'><thead><tr><th>Symbol</th><th>Qty</th><th>Avg Px</th><th>UPL</th></tr></thead><tbody></tbody></table></div>
<h3 style='margin:12px'>Open Orders</h3><div style='margin:12px'><table class='ord-table' id='orders'><thead><tr><th>ID</th><th>Sym</th><th>Side</th><th>Qty</th><th>Px</th><th>Status</th></tr></thead><tbody></tbody></table></div>
<h3 style='margin:12px'>Alerts</h3><div id='alerts'></div>
<script>
function fmt(n){return (n||0).toFixed(2)}
const cardsEl=document.getElementById('cards');
function renderCards(s){cardsEl.innerHTML='';const items=[['Equity',fmt(s.equity)],['Realized',fmt(s.realized_pnl)],['Unrealized',fmt(s.unrealized_pnl)],['Drawdown %',fmt(s.drawdown_pct)],['Open Positions',s.positions.length],['Open Orders',s.orders.length]];items.forEach(it=>{const d=document.createElement('div');d.className='card';d.innerHTML='<div>'+it[0]+'</div><div style="font-size:20px">'+it[1]+'</div>';cardsEl.appendChild(d);});document.getElementById('pauseBanner').style.display=s.risk_flags && s.risk_flags.includes('PAUSED')?'block':'none';}
function renderTable(id, rows, cols){const tbody=document.querySelector('#'+id+' tbody');tbody.innerHTML='';rows.forEach(r=>{const tr=document.createElement('tr');cols.forEach(c=>{const td=document.createElement('td');let v=r[c];if(typeof v==='number'){v=fmt(v);}td.textContent=v;tr.appendChild(td);});tbody.appendChild(tr);});}
function addAlert(a){const wrap=document.getElementById('alerts');const d=document.createElement('div');d.className='alert '+(a.severity==='WARN'?'warn':(a.severity==='ERROR'?'err':'info'));d.textContent=new Date(a.ts).toLocaleTimeString()+" "+a.type+": "+a.message;wrap.prepend(d);}
async function boot(){const ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws/stream');ws.onmessage=(ev)=>{const data=JSON.parse(ev.data);renderCards(data);renderTable('positions', data.positions, ['symbol','qty','avg_px','upl']);renderTable('orders', data.orders, ['client_order_id','symbol','side','qty','price','status']);if(data.alert){addAlert(data.alert);} if(data.risk_flags && data.risk_flags.includes('PAUSED')){document.getElementById('pauseBanner').style.display='block';} else {document.getElementById('pauseBanner').style.display='none';}}
}
boot();
</script></body></html>"""

app = FastAPI()

def compute_snapshot() -> Dict[str, Any]:
    # Equity heuristic: sum realized_pnl + unrealized from positions (unrealized not stored: 0) – extend later
    positions = fetchall("SELECT symbol, qty, avg_px, realized_pnl, updated_ts FROM positions WHERE ABS(qty) > 0.0000001")
    orders = fetchall("SELECT client_order_id, symbol, side, qty, price, status FROM orders_outbox WHERE status NOT IN ('FILLED','CANCELED','REJECTED')")
    # risk state
    risk = fetchall("SELECT * FROM risk_runtime ORDER BY day DESC LIMIT 1")
    realized = risk[0]['realized_pnl'] if risk else 0.0
    equity = realized  # placeholder; would add starting_balance + realized + unrealized
    # Unrealized simple placeholder 0; can be extended with mark prices feed
    unrealized = 0.0
    # drawdown placeholder: max equity not tracked yet; show 0
    drawdown_pct = 0.0
    risk_flags = []
    if risk and risk[0].get('paused'):
        risk_flags.append('PAUSED')
    # Active event banner (first active window)
    now_ms = int(time.time()*1000)
    active_events = []
    upcoming_events = []
    event_stats = {}
    try:
        # Active events (cheap internal helper)
        evs = event_gating._load_active_events(now_ms, {'event_risk': {'pre_window_minutes':{},'post_window_minutes':{}}})
        for e in evs:
            active_events.append({'category': e.get('category'), 'importance': e.get('importance'), 'banner': f"{e.get('category')} ACTIVE (sev {e.get('importance')})"})
        # Lightweight upcoming markers: peek at cached windows payload (do not rebuild if absent)
        cache = event_gating._WINDOW_CACHE.get('global')  # type: ignore
        if cache:
            for s,e,ev in cache.get('windows', [])[:5]:  # first few windows
                if s > now_ms:  # future only
                    upcoming_events.append({'ts': s, 'category': ev.get('category'), 'importance': ev.get('importance')})
        # Gate stats (evaluations, vetoes, abstain pct)
        try:
            event_stats = event_gating.stats()
        except Exception:
            event_stats = {}
    except Exception:
        pass
    return {
        'ts': int(time.time()),
        'equity': equity,
        'realized_pnl': realized,
        'unrealized_pnl': unrealized,
        'drawdown_pct': drawdown_pct,
        'positions': positions,
        'orders': orders,
        'risk_flags': risk_flags,
        'active_events': active_events,
        'upcoming_events': upcoming_events,
        'event_stats': event_stats,
    # Sprint 29 liquidity gate latest decisions (dict of symbol-> outcome)
    'liquidity_gate': getattr(_lq_mod, '_LAST_LQ', {}) if _lq_mod else {},
    }

@app.get('/')
def index():  # pragma: no cover - simple HTML
    return HTMLResponse(HTML)

@app.get('/status')
def status():
    return JSONResponse(compute_snapshot())

@app.get('/equity_curve')
def equity_curve():
    rows = fetchall("SELECT ts,equity,drawdown FROM equity_curve ORDER BY ts ASC")
    return JSONResponse({'points': rows})

@app.get('/alerts')
def alerts():
    return JSONResponse({'alerts': recent_alerts(100)})

_ws_clients: set[WebSocket] = set()

async def _ws_sender(ws: WebSocket):
    try:
        while True:
            await asyncio.sleep(1)
            snap = compute_snapshot()
            await ws.send_text(json.dumps(snap))
    except Exception:
        pass

@app.websocket('/ws/stream')
async def stream(ws: WebSocket):  # pragma: no cover (integration)
    await ws.accept()
    _ws_clients.add(ws)
    # Send initial snapshot
    await ws.send_text(json.dumps(compute_snapshot()))
    # Subscribe to alerts for push
    sid = subscribe(lambda a: asyncio.create_task(ws.send_text(json.dumps({**compute_snapshot(), 'alert': a}))))
    try:
        while True:
            await ws.receive_text()  # client can ping; ignore content
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe(sid)
        _ws_clients.discard(ws)

def run(port: int):  # pragma: no cover
    import uvicorn
    logger.info(f"[Dashboard] starting on 0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')

def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8787)
    ap.add_argument('--db', type=str, default='live_state.db')
    args = ap.parse_args()
    init_db(args.db)
    apply_migrations()
    run(args.port)

if __name__ == '__main__':  # pragma: no cover
    main()
