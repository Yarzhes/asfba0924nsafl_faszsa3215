"""Execution guard checks (spread, latency, slippage, flip)."""
from __future__ import annotations
from typing import Dict, Any, List

class GuardResult(dict):
    @property
    def passed(self) -> bool:
        return not self.get('blocked')


def pre_trade_guards(symbol: str, side: str, *, book: Dict[str,float], decision_latency_ms: int, ex_cfg: Dict[str,Any], signal_flip: bool=False, est_slip_bps: float | None=None) -> GuardResult:
    bid = float(book['bid']); ask = float(book['ask'])
    mid = (bid+ask)/2 if bid and ask else (bid or ask)
    spread_pct = (ask-bid)/mid*100 if mid else 0.0
    max_spread_pct = ex_cfg.get('max_spread_pct',0.06)*100
    if spread_pct > max_spread_pct:
        return GuardResult(blocked=True, reason='spread', spread_pct=spread_pct)
    # Latency
    p99_budget = ((ex_cfg.get('latency_budget') or {}).get('p99') if isinstance(ex_cfg.get('latency_budget'), dict) else None) or 180
    if decision_latency_ms > p99_budget:
        return GuardResult(blocked=True, reason='latency', latency_ms=decision_latency_ms)
    # Flip guard
    if ex_cfg.get('cancel_if_flip', True) and signal_flip:
        return GuardResult(blocked=True, reason='flip')
    # Slippage guard
    max_slip = ex_cfg.get('max_slip_bps',12)
    if est_slip_bps is not None and est_slip_bps > max_slip:
        return GuardResult(blocked=True, reason='slippage', est_slip_bps=est_slip_bps)
    return GuardResult(blocked=False)

__all__=['pre_trade_guards','GuardResult']
