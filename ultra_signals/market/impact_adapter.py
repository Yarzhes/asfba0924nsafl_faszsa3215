"""Maps lambda estimates to execution hints and size multipliers.

Simple policy: uses lambda_z and spread/depth proxies to produce impact_state,
target_participation_pct and prefer_passive flag.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ImpactHints:
    impact_state: str  # normal | elevated | high
    target_participation_pct: float
    prefer_passive: bool
    size_multiplier: float


class ImpactAdapter:
    def __init__(self, hi_th: float = 2.0, lo_th: float = 1.0, base_participation: float = 0.01):
        self.hi_th = float(hi_th)
        self.lo_th = float(lo_th)
        self.base_participation = float(base_participation)
        self._state = 'normal'

    def decide(self, lambda_z: Optional[float], spread_z: Optional[float] = None, depth_z: Optional[float] = None) -> ImpactHints:
        # basic fusion: if any z > hi -> elevated
        lz = float(lambda_z) if lambda_z is not None else 0.0
        sz = float(spread_z) if spread_z is not None else 0.0
        dz = float(depth_z) if depth_z is not None else 0.0

        score = max(lz, sz, dz)
        # hysteresis
        if self._state == 'normal':
            if score >= self.hi_th:
                self._state = 'high'
            elif score >= self.lo_th:
                self._state = 'elevated'
        elif self._state == 'elevated':
            if score >= self.hi_th:
                self._state = 'high'
            elif score < self.lo_th:
                self._state = 'normal'
        else:  # high
            if score < self.lo_th:
                self._state = 'normal'
            elif score < self.hi_th:
                self._state = 'elevated'

        if self._state == 'normal':
            return ImpactHints('normal', self.base_participation, False, 1.0)
        if self._state == 'elevated':
            return ImpactHints('elevated', max(0.001, self.base_participation * 0.6), True, 0.6)
        return ImpactHints('high', max(0.0005, self.base_participation * 0.2), True, 0.3)


__all__ = ['ImpactAdapter','ImpactHints']
