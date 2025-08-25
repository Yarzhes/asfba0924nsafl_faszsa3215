"""Sprint 20: Meta Router

Resolves per-symbol/timeframe settings by layering profile overrides on a
minimal base settings structure. Returns a resolved settings dict plus an
explain block recording provenance and fallback chain.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import os
from loguru import logger
from .profile_loader import load_profile, deep_merge, profile_id
import copy
import threading
import time

class MetaRouter:
    def __init__(self, base_settings: Dict[str, Any], root_dir: str | None = None, hot_reload: bool = False, watch_interval: Optional[float] = None):
        self.base = base_settings or {}
        self.root_dir = root_dir
        self.hot_reload = hot_reload
        # cache: (symbol,timeframe) -> (resolved_dict, file_mtimes_tuple)
        self._cache: Dict[Tuple[str,str], Tuple[Dict[str, Any], Tuple[Tuple[str,float], ...]]] = {}
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        if self.hot_reload and self.root_dir:
            # determine interval from settings if not provided
            if watch_interval is None:
                watch_interval = float((self.base.get('profiles', {}) or {}).get('watch_interval_sec', 1.0))
            self._start_watcher(interval=watch_interval)

    def _files_signature(self, file_paths):
        sig = []
        for p in file_paths:
            try:
                stat = os.stat(p)
                sig.append((p, stat.st_mtime))
            except OSError:
                sig.append((p, 0.0))
        return tuple(sig)

    def _needs_reload(self, key: Tuple[str,str], sig: Tuple[Tuple[str,float], ...]) -> bool:
        if key not in self._cache:
            return True
        _, cached_sig = self._cache[key]
        return cached_sig != sig

    # ------------- Watcher (polling) -------------
    def _scan_all_signatures(self) -> Tuple[Tuple[str,float], ...]:
        if not self.root_dir:
            return tuple()
        sig = []
        try:
            for p in Path(self.root_dir).rglob('*.yaml'):
                try:
                    stat = p.stat()
                    sig.append((str(p), stat.st_mtime))
                except OSError:
                    sig.append((str(p), 0.0))
        except Exception:
            pass
        return tuple(sorted(sig))

    def _watch_loop(self, interval: float):
        last_sig = self._scan_all_signatures()
        while not self._stop_event.wait(interval):
            cur = self._scan_all_signatures()
            if cur != last_sig:
                # invalidate cache entirely; granular diff not required
                self._cache.clear()
                last_sig = cur
                try:
                    from loguru import logger
                    logger.debug('[MetaRouter] Profiles changed on disk; cache invalidated.')
                except Exception:
                    pass

    def _start_watcher(self, interval: float):
        if self._watch_thread and self._watch_thread.is_alive():
            return
        self._stop_event.clear()
        t = threading.Thread(target=self._watch_loop, args=(interval,), daemon=True)
        t.start()
        self._watch_thread = t

    def stop(self):  # pragma: no cover (only used in long-lived runtime)
        try:
            self._stop_event.set()
            if self._watch_thread and self._watch_thread.is_alive():
                self._watch_thread.join(timeout=1.0)
        except Exception:
            pass
    # ---------------------------------------------

    def resolve(self, symbol: str, timeframe: str, root_dir: str | None = None) -> Dict[str, Any]:
        root = root_dir or self.root_dir
        if not root:
            raise ValueError("MetaRouter.resolve called without root_dir configured.")
        result = load_profile(root, symbol, timeframe)
        used_files = result['used_files']
        sig = self._files_signature(used_files)
        key = (symbol, timeframe)
        use_cache = (not self.hot_reload) and key in self._cache
        if self.hot_reload and not self._needs_reload(key, sig):
            use_cache = True
        if use_cache:
            return self._cache[key][0]
        prof = result['profile']
        resolved = deep_merge(self.base, prof)
        meta = (prof.get('meta') or {})
        pid = profile_id(prof, symbol, timeframe)
        min_ver = (self.base.get('profiles', {}) or {}).get('min_required_version')
        version = meta.get('version')
        stale = False
        try:
            if min_ver and version and str(version) < str(min_ver):
                stale = True
        except Exception:
            stale = False
        explain = {
            'profile_id': pid,
            'version': version,
            'source': meta.get('source'),
            'data_commit': meta.get('data_commit'),
            'fallback_chain': used_files,
            'resolved_keys': list(prof.keys()),
            'missing': result.get('missing', False),
            'stale': stale,
            'min_required_version': min_ver,
            'hot_reload': self.hot_reload
        }
        resolved.setdefault('meta_router', explain)
        # cache it
        self._cache[key] = (resolved, sig)
        return resolved

__all__ = ['MetaRouter']
