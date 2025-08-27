"""Global alert bus (Sprint 26).

Provides a lightweight publish / subscribe mechanism for runtime alerts
with persistence into the `alerts` table and optional side-channel
delivery (console log, telegram).

Usage:
    from ultra_signals.core.alerts import publish_alert, subscribe
    token = subscribe(lambda alert: print(alert))
    publish_alert('RISK_PAUSE', 'Trading paused due to DAILY_LOSS')

Alert dict schema:
    {
        'ts': 1675120200123,   # ms
        'type': 'RISK_PAUSE',
        'message': 'Trading paused due to DAILY_LOSS',
        'severity': 'WARN',
        'meta': {...}
    }
"""
from __future__ import annotations
import json
import time
import threading
from typing import Callable, Dict, Any
from loguru import logger

_subs: dict[int, Callable[[Dict[str, Any]], None]] = {}
_next_id = 1
_lock = threading.RLock()

def subscribe(cb: Callable[[Dict[str, Any]], None]) -> int:
    global _next_id
    with _lock:
        sid = _next_id
        _next_id += 1
        _subs[sid] = cb
    return sid

def unsubscribe(sid: int):  # pragma: no cover - trivial
    with _lock:
        _subs.pop(sid, None)

def publish_alert(alert_type: str, message: str, *, severity: str = 'INFO', meta: Dict[str, Any] | None = None):
    alert = {
        'ts': int(time.time()*1000),
        'type': alert_type,
        'message': message,
        'severity': severity,
        'meta': meta or {}
    }
    # Persist (best-effort)
    try:
        from ultra_signals.persist.db import execute
        execute("INSERT INTO alerts(ts,type,message,severity,meta_json) VALUES(?,?,?,?,?)", (
            alert['ts'], alert['type'], alert['message'], alert['severity'], json.dumps(alert['meta']) if alert['meta'] else None
        ))
    except Exception as e:  # pragma: no cover
        logger.error(f"[Alerts] persist failed {e}")
    # Log
    log_fn = logger.info if severity in ('INFO','LOW') else logger.warning if severity in ('WARN','MEDIUM') else logger.error
    log_fn(f"[Alert] {alert_type} {message} meta={alert['meta']}")
    # Fan-out to subscribers
    with _lock:
        subs = list(_subs.values())
    for cb in subs:
        try:
            cb(alert)
        except Exception as e:  # pragma: no cover
            logger.warning(f"[Alerts] subscriber error {e}")
    # Optional: Telegram
    try:  # pragma: no cover (depends on user config)
        from ultra_signals.transport.telegram import maybe_send_telegram
        maybe_send_telegram(f"ALERT {alert_type}: {message}")
    except Exception:
        pass
    return alert

def recent_alerts(limit: int = 50):
    try:
        from ultra_signals.persist.db import fetchall
        rows = fetchall("SELECT ts,type,message,severity,meta_json FROM alerts ORDER BY ts DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            meta = None
            try:
                if r.get('meta_json'):
                    meta = json.loads(r['meta_json'])
            except Exception:
                meta = None
            out.append({k: r[k] for k in ('ts','type','message','severity')} | {'meta': meta})
        return out
    except Exception as e:  # pragma: no cover
        logger.error(f"[Alerts] recent fetch failed {e}")
        return []

__all__ = ['publish_alert','subscribe','unsubscribe','recent_alerts']
