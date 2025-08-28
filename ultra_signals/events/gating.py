"""Event gating logic with interval merging + cooldown.

Decides VETO / DAMPEN / NONE (and optionally force_close) for a symbol at time now_ms.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import bisect, time
from loguru import logger
from . import store


@dataclass
class GateDecision:
    action: str  # VETO|DAMPEN|NONE
    reason: str = ""
    size_mult: float = 1.0
    widen_stop_mult: float | None = None
    maker_only: bool | None = None
    category: str | None = None
    force_close: bool | None = None


def _severity_key(importance: int) -> str:
    return {3: 'HIGH', 2: 'MED', 1: 'LOW'}.get(int(importance), 'LOW')


_WINDOW_CACHE: Dict[str, Dict[str, Any]] = {}
_LAST_VETO_TS: Dict[str, int] = {}
_STATS: Dict[str, int] = {'evaluations':0,'vetoes':0}


def _build_windows(now_ms: int, settings: dict) -> Dict[str, Any]:
    cfg = settings.get('event_risk') or {}
    pre_cfg = cfg.get('pre_window_minutes', {})
    post_cfg = cfg.get('post_window_minutes', {})
    rows = store.load_events_window(now_ms - 12*60*60*1000, now_ms + 48*60*60*1000)
    windows: List[Tuple[int,int,Dict[str,Any]]] = []
    for ev in rows:
        imp = int(ev.get('importance') or 1)
        sev = _severity_key(imp)
        start = int(ev['start_ts']); end = int(ev['end_ts'])
        win_s = start - int(pre_cfg.get(sev,0))*60*1000
        win_e = end + int(post_cfg.get(sev,0))*60*1000
        if win_e < now_ms - 12*60*60*1000:
            continue
        windows.append((win_s, win_e, ev))
    windows.sort(key=lambda x: x[0])
    merged: List[Tuple[int,int,List[Dict[str,Any]]]] = []
    for s,e,ev in windows:
        if not merged or s > merged[-1][1]:
            merged.append((s,e,[ev]))
        else:
            ms,me,evs = merged[-1]
            merged[-1] = (ms, max(me,e), evs+[ev])
    starts = [m[0] for m in merged]
    payload = []
    for s,e,evs in merged:
        best = sorted(evs, key=lambda r: int(r.get('importance') or 1), reverse=True)[0]
        payload.append((s,e,best))
    return {'starts': starts, 'windows': payload, 'built_at': time.time(), 'event_count': len(rows)}


def _load_active_events(now_ms: int, settings: dict) -> List[Dict[str, Any]]:
    cache_key = 'global'
    c = _WINDOW_CACHE.get(cache_key)
    if not c or (time.time() - c['built_at']) > 60:
        try:
            built = _build_windows(now_ms, settings)
            # if no windows, just reset cache and return empty
            if not built['windows']:
                _WINDOW_CACHE[cache_key] = {'starts': [], 'windows': [], 'built_at': time.time(), 'event_count': 0}
                return []
            _WINDOW_CACHE[cache_key] = built
            c = built
        except Exception as e:
            logger.debug('[events] window build fail: {}', e)
            return []
    starts = c['starts']; payload = c['windows']
    i = bisect.bisect_right(starts, now_ms)
    hits: List[Dict[str,Any]] = []
    for idx in (i-1, i):
        if 0 <= idx < len(payload):
            s,e,ev = payload[idx]
            if s <= now_ms <= e:
                hits.append(ev)
    return hits


@lru_cache(maxsize=1024)
def _window_params_hash(cfg_key: str) -> int:  # compatibility helper
    return 1


class PolicyAdapter:
    """Simple adapter that maps a realtime toxicity score (0..1) to a GateDecision.

    Config options accepted in dict:
      - veto_th: float (>=) threshold to VETO
      - dampen_th: float (>=) threshold to DAMPEN
      - dampen_size: float size_mult for DAMPEN
      - category: optional string
    """
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.veto_th = float(cfg.get('veto_th', 0.9))
        self.dampen_th = float(cfg.get('dampen_th', 0.7))
        self.dampen_size = float(cfg.get('dampen_size', 0.5))
        self.category = cfg.get('category')

    def decide(self, tox_score: float, symbol: str | None = None, now_ms: int | None = None) -> GateDecision | None:
        try:
            if tox_score >= self.veto_th:
                return GateDecision(action='VETO', reason=f'VPIN_TOX:{tox_score:.3f}', category=self.category)
            if tox_score >= self.dampen_th:
                return GateDecision(action='DAMPEN', reason=f'VPIN_TOX:{tox_score:.3f}', size_mult=self.dampen_size, category=self.category)
        except Exception:
            return None
        return None


def evaluate(symbol: str, now_ms: int, venue: str | None, profile: dict | None, settings: dict) -> GateDecision:
    cfg = (settings.get('event_risk') or {}) if isinstance(settings, dict) else {}
    if not cfg.get('enabled', False):
        return GateDecision(action='NONE')

    try:
        events = _load_active_events(now_ms, settings)
    except Exception as e:
        logger.debug('[events] load failure treated as missing feed: {}', e)
        events = []
    else:
        # Stale cache guard: if cache says events active but DB has none near now, invalidate
        if events:
            try:
                near = store.load_events_window(now_ms - 30*60*1000, now_ms + 30*60*1000)
                if not near:
                    # Clear cache and treat as no events (DB likely rotated)
                    _WINDOW_CACHE.clear()
                    events = []
            except Exception:
                pass
    if not events:
        pol = str(cfg.get('missing_feed_policy', 'SAFE')).upper()
        # Missing feed -> clear previous veto timestamps to avoid phantom cooldown
        try:
            _LAST_VETO_TS.pop(symbol, None)
        except Exception:
            pass
        if pol == 'SAFE':
            med = (cfg.get('actions') or {}).get('MED', {})
            return GateDecision(action='DAMPEN', reason='MISSING_FEED_SAFE', size_mult=float(med.get('size_mult', 0.5)))
        return GateDecision(action='NONE')

    # PolicyAdapter: allow realtime feature-driven overrides (e.g., tox_score)
    try:
        # profile may contain fused microstructure features produced by feature store
        tox = None
        if isinstance(profile, dict):
            # common keys: 'tox_score', 'vpin_tox', 'vpin' etc.
            tox = profile.get('tox_score') or profile.get('vpin_tox') or profile.get('tox')
        # also allow a settings-provided lookup path for features e.g., settings['features_path']
        if tox is None:
            feats_path = cfg.get('realtime_feature_path')
            if feats_path and isinstance(settings, dict):
                # simple dotted path lookup
                parts = feats_path.split('.')
                node = settings
                for p in parts:
                    if isinstance(node, dict):
                        node = node.get(p)
                    else:
                        node = None; break
                tox = node
        if tox is not None:
            # ensure numeric
            try:
                t = float(tox)
            except Exception:
                t = None
            if t is not None:
                # Map tox_score (0..1) to GateDecision using thresholds in cfg
                pa = PolicyAdapter(cfg.get('policy_adapter', {}))
                dec = pa.decide(t, symbol=symbol, now_ms=now_ms)
                if dec:
                    _STATS['evaluations'] += 1
                    if dec.action == 'VETO':
                        _STATS['vetoes'] += 1
                    return dec
    except Exception:
        # Swallow adapter errors and continue to DB-driven logic
        pass

    act_cfg = cfg.get('actions', {})
    sym_over = (cfg.get('symbol_overrides') or {}).get(symbol, {})
    best: tuple[int, Dict[str, Any], Dict[str, Any]] | None = None
    for ev in events:
        imp = int(ev.get('importance') or 1)
        sev = _severity_key(imp)
        cat = ev.get('category')
        scope = ev.get('symbol_scope') or 'GLOBAL'
        if scope != 'GLOBAL' and symbol not in scope.split(','):
            continue
        if not best or imp > best[0]:
            override = ((sym_over.get('categories') or {}).get(cat) or {}) if sym_over else {}
            a_cfg = override if override else act_cfg.get(sev, {})
            best = (imp, ev, a_cfg)

    if not best:
        return GateDecision(action='NONE')

    imp, ev, a_cfg = best
    mode = str(a_cfg.get('mode', 'NONE')).upper()
    sev = _severity_key(imp)
    reason = f"{ev.get('category')}:{sev}"
    cooldown_min = int(cfg.get('cooldown_minutes_after_veto', 0))
    if mode == 'VETO':
        now_sec = now_ms // 1000
        last = _LAST_VETO_TS.get(symbol)
        if last and (now_sec - last) < cooldown_min * 60:
            # Only treat as cooldown if this evaluation still has an event window (it does) and same category importance
            decision = GateDecision(action='VETO', reason='COOLDOWN', category=ev.get('category'))
            _STATS['evaluations'] += 1; _STATS['vetoes'] += 1
            return decision
        _LAST_VETO_TS[symbol] = now_sec
        force_close = bool((cfg.get('close_existing') or {}).get('HIGH') and _severity_key(imp) == 'HIGH')
        decision = GateDecision(action='VETO', reason=reason, category=ev.get('category'), force_close=force_close)
        _STATS['evaluations'] += 1; _STATS['vetoes'] += 1
        return decision
    if mode == 'DAMPEN':
        decision = GateDecision(action='DAMPEN', reason=reason, size_mult=float(a_cfg.get('size_mult', 0.5)), widen_stop_mult=a_cfg.get('widen_stop_mult'), maker_only=a_cfg.get('maker_only'), category=ev.get('category'))
        _STATS['evaluations'] += 1
        return decision
    _STATS['evaluations'] += 1
    return GateDecision(action='NONE')


def stats() -> Dict[str, Any]:
    ev = _STATS.get('evaluations',0)
    vt = _STATS.get('vetoes',0)
    return {'evaluations': ev, 'vetoes': vt, 'abstain_pct': (vt/ev*100.0) if ev else 0.0}


def reset_caches():  # for tests
    _WINDOW_CACHE.clear(); _LAST_VETO_TS.clear(); _STATS.update({'evaluations':0,'vetoes':0})


__all__ = ['GateDecision','evaluate']
