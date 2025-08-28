"""Simple policy engine mapping drift test outputs to actions.

This module exposes a PolicyEngine that consumes metrics and test states
and returns an Action record specifying degrade/pause/retrain decisions.
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import time


class ActionType(str, Enum):
    NONE = "none"
    SHRINK = "shrink"
    PAUSE = "pause"
    RETRAIN = "retrain"


@dataclass
class Action:
    type: ActionType
    size_mult: float = 1.0
    reason_codes: List[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.reason_codes is None:
            self.reason_codes = []
        if self.timestamp is None:
            self.timestamp = time.time()


class PolicyEngine:
    """Map metrics & test signals to actions using hysteresis.

    This is intentionally minimal: it receives a dictionary with keys like
    'sprt_state', 'pf_delta_pct', 'maxdd_p95_breach', 'ece', 'slip_bps',
    and returns an Action.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        # internal state for hysteresis per-symbol
        # shape: { symbol: { 'last_action': ActionType, 'last_ts': float } }
        self._state = {}

    @staticmethod
    def default_config() -> dict:
        return {
            'thresholds': {
                'pf_deep_drop': -0.25,
                'ece_high': 0.08,
                'slip_high_bps': 3.0,
            },
            'hysteresis': {
                'pause_min_seconds': 60 * 30,  # 30 minutes minimum pause before allowing resume
                'shrink_cooldown_seconds': 60 * 10,  # 10 minutes cooldown between shrinks
            }
        }

    def evaluate(self, metrics: dict) -> Action:
        codes = []
        size_mult = 1.0

        cfg = {**self.default_config(), **(self.cfg or {})}
        thr = cfg.get('thresholds', {})
        hyst = cfg.get('hysteresis', {})

        sprt_state = metrics.get("sprt_state")
        pf_delta = metrics.get("pf_delta_pct", 0.0)
        maxdd_breach = metrics.get("maxdd_p95_breach", False)
        ece = metrics.get("ece_live", 0.0)
        slip = metrics.get("slip_delta_bps", 0.0)
        symbol = metrics.get('symbol')
        now = metrics.get('now', time.time())

        # Simple rules: SPRT accept_h1 means win-rate dropped vs H0->H1 is worse
        if sprt_state == "accept_h1":
            codes.append("SPRT_fail_winrate")
            # degrade first
            size_mult *= 0.6

        # PF drop severe
        if pf_delta <= float(thr.get('pf_deep_drop', -0.25)):
            codes.append("PF_deep_drop")
            size_mult *= 0.5

        # MaxDD breach -> pause & retrain
        if maxdd_breach:
            codes.append("DD_p95_breach")
            return Action(ActionType.RETRAIN, size_mult=0.0, reason_codes=codes)

        # ECE high
        if ece >= float(thr.get('ece_high', 0.08)):
            codes.append("ECE_high")
            size_mult *= 0.75

        # Slippage
        if slip >= float(thr.get('slip_high_bps', 3.0)):
            codes.append("Slip_high")

        # Hysteresis rules: if the last action was PAUSE and pause_min_seconds
        # hasn't elapsed, keep paused unless a severe retrain event occurs.
        last = None
        if symbol:
            last = self._state.get(symbol)
            if last and last.get('last_action') == ActionType.PAUSE:
                min_pause = float(hyst.get('pause_min_seconds', 60 * 30))
                if (now - float(last.get('last_ts', 0))) < min_pause:
                    # keep paused
                    codes.append('ESCALATE_PAUSE_HYSTERESIS')
                    return Action(ActionType.PAUSE, size_mult=size_mult, reason_codes=codes)

        if len(codes) >= 2 or size_mult <= 0.4:
            codes.append("ESCALATE_PAUSE")
            act = Action(ActionType.PAUSE, size_mult=size_mult, reason_codes=codes)
            if symbol:
                self._state.setdefault(symbol, {})['last_action'] = ActionType.PAUSE
                self._state[symbol]['last_ts'] = now
            return act

        if size_mult < 0.99:
            # enforce shrink cooldown
            if symbol and (last and last.get('last_action') == ActionType.SHRINK):
                cooldown = float(hyst.get('shrink_cooldown_seconds', 60 * 10))
                if (now - float(last.get('last_ts', 0))) < cooldown:
                    # do not repeat shrink too quickly; return NONE
                    return Action(ActionType.NONE, size_mult=1.0, reason_codes=codes)
            act = Action(ActionType.SHRINK, size_mult=size_mult, reason_codes=codes)
            if symbol:
                self._state.setdefault(symbol, {})['last_action'] = ActionType.SHRINK
                self._state[symbol]['last_ts'] = now
            return act

        return Action(ActionType.NONE, size_mult=1.0, reason_codes=codes)
