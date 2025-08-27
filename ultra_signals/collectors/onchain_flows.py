"""On-Chain Large Transfer Collector (Ethereum / BTC / TRON skeleton).

Parses large transfers (>= configured USD threshold). Requires token price
estimation (external or last trade price) for USD normalization.

For simplicity this skeleton only exposes an API to feed pre-normalized events
into FeatureStore. Real implementation would use aiohttp & chain specific JSON
RPC endpoints.
"""
from __future__ import annotations
from typing import Dict, Any
import time
from .address_loader import HotReloadAddressRegistry
from loguru import logger

class OnChainFlowCollector:
    def __init__(self, feature_store, cfg: Dict[str, Any]):
        self.store = feature_store
        self.cfg = cfg or {}
    self._exchange_addresses = set((cfg.get('exchange_addresses') or []))
    # Optional hot-reload directory
    addr_dir = cfg.get('addresses_dir')
    self._registry = HotReloadAddressRegistry(addr_dir, 'exchange_wallets') if addr_dir else None
    self._last_reload_check = 0.0
    # Override multipliers (symbol specific)
    self._dep_mult_over = (cfg.get('deposit_burst_multiplier_overrides') or {})
    self._wdr_mult_over = (cfg.get('withdrawal_burst_multiplier_overrides') or {})

    def ingest_transfer(self, symbol: str, from_addr: str, to_addr: str, usd: float, ts_ms: int):
        try:
            # Hot reload addresses at most every 15s
            if self._registry and time.time() - self._last_reload_check > 15:
                self._last_reload_check = time.time()
                if self._registry.maybe_reload():
                    self._exchange_addresses = self._registry.get()
            dep = to_addr in self._exchange_addresses and from_addr not in self._exchange_addresses
            wdr = from_addr in self._exchange_addresses and to_addr not in self._exchange_addresses
            if dep:
                self.store.whale_add_exchange_flow(symbol, 'DEPOSIT', usd, ts_ms)
            elif wdr:
                self.store.whale_add_exchange_flow(symbol, 'WITHDRAWAL', usd, ts_ms)
        except Exception:
            logger.exception('OnChainFlowCollector ingest error')
