"""Lightweight JSON file cache for HTTP responses (Sprint 46 scaffold).

Stores entries keyed by URL (or arbitrary key) with fields:
    fetched_at_ms, etag, last_modified, status, body (optional subset), ttl_ms

The goal is to provide polite conditional requests (If-None-Match / If-Modified-Since)
without heavy dependencies. For large bodies we can choose to skip storing body.
"""
from __future__ import annotations

from typing import Dict, Optional, Any
import json
import time
import os
from threading import RLock

_lock = RLock()


class HTTPCache:
    def __init__(self, path: str, default_ttl_sec: int = 900):
        self.path = path
        self.default_ttl_ms = default_ttl_sec * 1000
        self._data: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    def _load(self):
        if self._loaded:
            return
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        self._loaded = True

    # ------------------------------------------------------------------
    def _save(self):
        tmp = self.path + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, separators=(',', ':'), ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def get(self, key: str, now_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
        with _lock:
            self._load()
            now_ms = now_ms or int(time.time()*1000)
            entry = self._data.get(key)
            if not entry:
                return None
            ttl_ms = entry.get('ttl_ms', self.default_ttl_ms)
            if now_ms - entry.get('fetched_at_ms', 0) > ttl_ms:
                return None
            return entry

    # ------------------------------------------------------------------
    def put(self, key: str, response: Dict[str, Any]):
        with _lock:
            self._load()
            self._data[key] = response
            self._save()

    # ------------------------------------------------------------------
    def conditional_headers(self, key: str, now_ms: Optional[int] = None) -> Dict[str, str]:
        entry = self.get(key, now_ms=now_ms)
        if not entry:
            return {}
        hdrs: Dict[str,str] = {}
        if entry.get('etag'):
            hdrs['If-None-Match'] = entry['etag']
        if entry.get('last_modified'):
            hdrs['If-Modified-Since'] = entry['last_modified']
        return hdrs

__all__ = ["HTTPCache"]
