"""Address Registry Loader & Hot-Reload (Sprint 41)

Loads versioned JSON address lists from a directory. File naming convention:
  exchange_wallets.v1.json, smart_money_wallets.v2.json etc.

Each file schema:
  {
    "version": "v1",
    "addresses": ["0xabc...", ...]
  }

Provides:
  - load_address_registry(dir_path, kind) -> {"version": str, "addresses": set[str]}
  - HotReloadAddressRegistry watching mtimes for reload() calls.

This keeps collectors decoupled from disk I/O complexity.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set
from loguru import logger


def _select_latest_file(dir_path: str, kind: str) -> Optional[str]:
    prefix = f"{kind}."  # e.g., exchange_wallets.
    best_ver = -1
    best_file = None
    try:
        for fn in os.listdir(dir_path):
            if not fn.startswith(prefix) or not fn.endswith('.json'):
                continue
            parts = fn.split('.')
            if len(parts) < 3:
                continue
            ver_token = parts[-2]  # expecting v1
            if ver_token.startswith('v') and ver_token[1:].isdigit():
                vnum = int(ver_token[1:])
                if vnum > best_ver:
                    best_ver = vnum
                    best_file = os.path.join(dir_path, fn)
    except FileNotFoundError:
        return None
    return best_file


def load_address_registry(dir_path: str, kind: str) -> Dict[str, Any]:
    path = _select_latest_file(dir_path, kind)
    if not path:
        logger.warning("No address registry file found for kind='{}' in {}", kind, dir_path)
        return {"version": None, "addresses": set()}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        addrs = set(data.get('addresses') or [])
        ver = data.get('version') or os.path.basename(path).split('.')[-2]
        return {"version": ver, "addresses": addrs, "_path": path, "_mtime": os.path.getmtime(path)}
    except Exception as e:
        logger.error("Address registry load failed {}: {}", path, e)
        return {"version": None, "addresses": set()}


@dataclass
class HotReloadAddressRegistry:
    dir_path: str
    kind: str
    min_reload_interval_sec: float = 10.0
    _last: Dict[str, Any] | None = None
    _last_check: float = 0.0

    def get(self) -> Set[str]:
        if not self._last:
            self._last = load_address_registry(self.dir_path, self.kind)
        return set(self._last.get('addresses') or [])

    def version(self) -> Optional[str]:
        if not self._last:
            self._last = load_address_registry(self.dir_path, self.kind)
        return self._last.get('version')

    def maybe_reload(self) -> bool:
        now = time.time()
        if now - self._last_check < self.min_reload_interval_sec:
            return False
        self._last_check = now
        if not self._last:
            self._last = load_address_registry(self.dir_path, self.kind)
            return True
        path = self._last.get('_path')
        if not path or not os.path.isfile(path):
            self._last = load_address_registry(self.dir_path, self.kind)
            return True
        mtime = os.path.getmtime(path)
        if mtime > (self._last.get('_mtime') or 0):
            logger.info("Hot reloading address registry kind={} from {}", self.kind, path)
            self._last = load_address_registry(self.dir_path, self.kind)
            return True
        return False

__all__ = ['load_address_registry','HotReloadAddressRegistry']
