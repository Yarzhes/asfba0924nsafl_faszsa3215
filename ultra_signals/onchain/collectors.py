"""On-chain collectors (skeletons) supporting offline ingestion.

Collectors accept pre-normalized transfers (symbol, from, to, amount, usd, ts)
and perform entity tagging using `Registry` instances. Implementations for
specific chains should be added separately; these are reusable helpers.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import time
from loguru import logger
from .registry import Registry


class BaseCollector:
    def __init__(self, feature_store, cfg: Dict[str, Any]):
        self.store = feature_store
        self.cfg = cfg or {}
        addr_dir = cfg.get('addresses_dir')
        # Registries by cohort
        self.registries = {
            'exchange': Registry(addr_dir, 'exchange_wallets') if addr_dir else None,
            'smart': Registry(addr_dir, 'smart_money_wallets') if addr_dir else None,
            'bridge': Registry(addr_dir, 'bridges') if addr_dir else None,
            'stable': Registry(addr_dir, 'stablecoin_treasuries') if addr_dir else None,
        }
        self._last_reload_check = 0.0

    def _maybe_reload(self):
        if time.time() - self._last_reload_check > 10:
            self._last_reload_check = time.time()
            for r in self.registries.values():
                if r:
                    r.maybe_reload()

    def ingest_transfer(self, chain: str, symbol: str, from_addr: str, to_addr: str, usd: float, ts_ms: int):
        """Tag transfer and forward to feature store.

        Direction rules:
         - DEPOSIT: to_addr in exchange registry and from_addr not in same registry
         - WITHDRAWAL: from_addr in exchange registry and to_addr not in same registry
         - BRIDGE_INFLOW: to_addr in bridge registry
         - STABLE_ROTATION: either side in stable registry
        """
        try:
            self._maybe_reload()
            exch = set(self.registries['exchange'].get_addresses()) if self.registries['exchange'] else set()
            bridge = set(self.registries['bridge'].get_addresses()) if self.registries['bridge'] else set()
            stable = set(self.registries['stable'].get_addresses()) if self.registries['stable'] else set()

            dep = to_addr in exch and from_addr not in exch
            wdr = from_addr in exch and to_addr not in exch
            if dep:
                self.store.whale_add_exchange_flow(symbol, 'DEPOSIT', usd, ts_ms)
            elif wdr:
                self.store.whale_add_exchange_flow(symbol, 'WITHDRAWAL', usd, ts_ms)

            if to_addr in bridge or from_addr in bridge:
                self.store.whale_add_bridge_flow(symbol, usd, ts_ms)

            if to_addr in stable or from_addr in stable:
                self.store.whale_add_stable_rotation(symbol, usd, ts_ms)
        except Exception:
            logger.exception('Collector ingest error')


class OfflineCSVCollector(BaseCollector):
    def ingest_file(self, path: str):
        import csv
        now = int(time.time() * 1000)
        with open(path, 'r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                # expect columns: chain,symbol,from,to,usd,ts_ms
                try:
                    chain = r.get('chain') or r.get('chain_id') or 'unknown'
                    symbol = r.get('symbol') or r.get('token')
                    from_a = r.get('from') or r.get('from_addr')
                    to_a = r.get('to') or r.get('to_addr')
                    usd = float(r.get('usd') or 0)
                    ts = int(r.get('ts_ms') or now)
                    self.ingest_transfer(chain, symbol, from_a, to_a, usd, ts)
                except Exception:
                    logger.exception('Failed parse row in %s', path)


__all__ = ['BaseCollector', 'OfflineCSVCollector']
