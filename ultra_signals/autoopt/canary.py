"""Canary rollout & rollback controller.

Lightweight state machine reading live stats (abstracted via provider callable)
to decide: continue canary, promote to full, or rollback to previous baseline.

States: IDLE -> CANARY -> FULL; on breach -> ROLLBACK then IDLE.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any
from loguru import logger

@dataclass
class CanaryConfig:
    canary_size_mult: float = 0.25
    min_canary_trades: int = 20
    canary_max_dd_pct: float = 6.0
    promote_if_pf_min: float = 1.3
    promote_if_sortino_min: float = 1.2
    promote_if_uplift: float = 0.05

@dataclass
class CanaryState:
    status: str = 'IDLE'  # IDLE|CANARY|FULL|ROLLBACK
    baseline_version: int | None = None
    candidate_version: int | None = None
    last_action: str | None = None


class CanaryController:
    def __init__(self, cfg: CanaryConfig, live_stats_provider: Callable[[], Dict[str, Any]]):
        self.cfg = cfg
        self.live_stats_provider = live_stats_provider
        self.state = CanaryState()

    def start_canary(self, candidate_version: int, baseline_version: int):
        if self.state.status not in ('IDLE','FULL'):
            logger.warning('Cannot start canary from state %s', self.state.status)
            return False
        self.state = CanaryState(status='CANARY', candidate_version=candidate_version, baseline_version=baseline_version, last_action='START_CANARY')
        return True

    def evaluate(self) -> str:
        stats = self.live_stats_provider() or {}
        if self.state.status != 'CANARY':
            return self.state.status
        trades = stats.get('trades',0)
        pf = stats.get('profit_factor',0)
        sortino = stats.get('sortino',0)
        dd = stats.get('max_dd_pct',0)
        uplift = stats.get('uplift_vs_baseline',0)
        if dd > self.cfg.canary_max_dd_pct:
            self.state.status='ROLLBACK'; self.state.last_action='DD_ROLLBACK'; return 'ROLLBACK'
        if trades < self.cfg.min_canary_trades:
            return 'CANARY'  # need more data
        if pf >= self.cfg.promote_if_pf_min and sortino >= self.cfg.promote_if_sortino_min and uplift >= self.cfg.promote_if_uplift:
            self.state.status='FULL'; self.state.last_action='PROMOTED_FULL'; return 'FULL'
        return 'CANARY'

    def needs_rollback(self) -> bool:
        return self.state.status == 'ROLLBACK'

__all__ = ['CanaryConfig','CanaryController','CanaryState']