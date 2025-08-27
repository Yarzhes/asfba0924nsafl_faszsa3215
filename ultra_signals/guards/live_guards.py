"""Live data quality guards (Sprint 39)."""
from __future__ import annotations
import time
from typing import Optional
from loguru import logger
from ..dq.time_sync import assert_within_skew, CircuitBreak
import json, os, time as _t

_last_md: dict[str, int] = {}
_last_order_evt: dict[str, int] = {}


def pre_tick_guard(symbol: str, venue: str, settings: dict) -> None:
    dq = (settings or {}).get('data_quality', {})
    if not dq.get('enabled', False):
        return
    assert_within_skew(settings, venues=[venue])


def pre_bar_guard(symbol: str, tf_ms: int, settings: dict) -> None:
    dq = (settings or {}).get('data_quality', {})
    if not dq.get('enabled', False):
        return
    # placeholder for bar start skew logic
    # Could compare expected bar boundary vs now


def post_fetch_guard(report, settings: dict) -> None:
    dq = (settings or {}).get('data_quality', {})
    if not dq.get('enabled', False):
        return
    if report and not report.ok:
        # Persist snapshot then raise circuit break to upstream caller
        try:
            _snapshot({'type':'post_fetch','ts':_now_ms(),'errors':report.errors,'warnings':report.warnings}, settings)
        except Exception:  # pragma: no cover - best effort IO
            pass
        raise CircuitBreak(f"validation_failed errors={report.errors}")


def heartbeat_guard(kind: str, last_seen_ms: int, settings: dict) -> None:
    dq = (settings or {}).get('data_quality', {})
    if not dq.get('enabled', False):
        return
    hb = dq.get('heartbeats', {})
    now_ms = int(time.time() * 1000)
    if kind == 'market_data':
        max_silence = hb.get('market_data_max_silence_sec', 20) * 1000
    else:
        max_silence = hb.get('order_events_max_silence_sec', 30) * 1000
    silence = now_ms - last_seen_ms
    if silence > max_silence:
        action = hb.get('action_on_silence', 'circuit_break')
        logger.error(f"heartbeat.silence kind={kind} silence_ms={silence} action={action}")
        _snapshot({'type':'heartbeat','kind':kind,'silence_ms':silence,'threshold_ms':max_silence,'action':action,'ts':_now_ms()}, settings)
        if action == 'circuit_break':
            raise CircuitBreak(f"heartbeat silence {silence}ms > {max_silence}ms")
        # 'flatten' or 'log' actions to be implemented by caller

def _now_ms():  # pragma: no cover trivial
    return int(_t.time()*1000)

def _snapshot(payload: dict, settings: dict):  # pragma: no cover simple IO
    dq = (settings or {}).get('data_quality', {})
    rdir = dq.get('report_dir')
    if not rdir:
        return
    try:
        os.makedirs(rdir, exist_ok=True)
        fname = f"guard_{payload.get('type')}_{payload.get('kind','')}_{payload['ts']}.json"
        with open(os.path.join(rdir, fname), 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

__all__ = [
    'pre_tick_guard','pre_bar_guard','post_fetch_guard','heartbeat_guard','CircuitBreak'
]
