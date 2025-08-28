"""Entity Registry: versioned labels + confidence tiers.

File format (json):
{
  "version": "v1",
  "kind": "exchange",
  "confidence": "high",
  "addresses": ["0x...", ...],
  "meta": {"source": "github:...", "note": "manual curation"}
}

Provides load_latest_registry(dir_path, kind) and Registry class with
in-memory hot-reload support.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set
from loguru import logger


def _select_latest_file(dir_path: str, kind: str) -> Optional[str]:
    prefix = f"{kind}."
    best_ver = -1
    best_file = None
    try:
        for fn in os.listdir(dir_path):
            if not fn.startswith(prefix) or not fn.endswith('.json'):
                continue
            parts = fn.split('.')
            if len(parts) < 3:
                continue
            ver_token = parts[-2]
            if ver_token.startswith('v') and ver_token[1:].isdigit():
                vnum = int(ver_token[1:])
                if vnum > best_ver:
                    best_ver = vnum
                    best_file = os.path.join(dir_path, fn)
    except FileNotFoundError:
        return None
    return best_file


def load_latest_registry(dir_path: str, kind: str) -> Dict[str, Any]:
    path = _select_latest_file(dir_path, kind)
    if not path:
        logger.warning("No registry file for kind=%s in %s", kind, dir_path)
        return {"version": None, "addresses": set(), "meta": {}}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        addrs = set(data.get('addresses') or [])
        return {"version": data.get('version') or os.path.basename(path).split('.')[-2],
                "addresses": addrs,
                "confidence": data.get('confidence'),
                "meta": data.get('meta') or {},
                "_path": path,
                "_mtime": os.path.getmtime(path)}
    except Exception as e:
        logger.error("Failed loading registry %s: %s", path, e)
        return {"version": None, "addresses": set(), "meta": {}}


@dataclass
class Registry:
    dir_path: Optional[str]
    kind: str
    min_reload_sec: float = 10.0
    _last: Optional[Dict[str, Any]] = None
    _last_check: float = 0.0

    def get_addresses(self) -> Set[str]:
        if not self._last:
            self._last = load_latest_registry(self.dir_path, self.kind)
        return set(self._last.get('addresses') or [])

    def version(self) -> Optional[str]:
        if not self._last:
            self._last = load_latest_registry(self.dir_path, self.kind)
        return self._last.get('version')

    def maybe_reload(self) -> bool:
        now = time.time()
        if now - self._last_check < self.min_reload_sec:
            return False
        self._last_check = now
        if not self.dir_path:
            return False
        if not self._last:
            self._last = load_latest_registry(self.dir_path, self.kind)
            return True
        path = self._last.get('_path')
        if not path or not os.path.isfile(path):
            self._last = load_latest_registry(self.dir_path, self.kind)
            return True
        mtime = os.path.getmtime(path)
        if mtime > (self._last.get('_mtime') or 0):
            logger.info("Hot reload registry kind=%s from %s", self.kind, path)
            self._last = load_latest_registry(self.dir_path, self.kind)
            return True
        return False


__all__ = ['load_latest_registry', 'Registry']
