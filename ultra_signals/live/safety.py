"""Circuit breakers & runtime safety controls (fail-closed)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from loguru import logger
try:  # lazy import guard
    from ultra_signals.core.alerts import publish_alert  # type: ignore
except Exception:  # pragma: no cover
    publish_alert = lambda *a, **k: None  # type: ignore

@dataclass
class CircuitState:
    paused: bool = False
    reason: Optional[str] = None
    last_change_ms: int = 0

@dataclass
class SafetyManager:
    daily_loss_limit_pct: float
    max_consecutive_losses: int
    order_error_burst_count: int
    order_error_burst_window_sec: int
    data_staleness_ms: int
    state: CircuitState = field(default_factory=CircuitState)
    _day_pnl: float = 0.0
    _consec_losses: int = 0
    _order_errors: list = field(default_factory=list)  # list of ts_ms

    def record_fill(self, pnl_delta: float):
        if pnl_delta < 0:
            self._consec_losses += 1
        else:
            self._consec_losses = 0
        self._day_pnl += pnl_delta
        self._recalc()

    def record_order_error(self):
        now = int(time.time()*1000)
        self._order_errors.append(now)
        # prune
        cutoff = now - self.order_error_burst_window_sec*1000
        self._order_errors = [t for t in self._order_errors if t >= cutoff]
        self._recalc()

    def _recalc(self):
        if self._day_pnl < -abs(self.daily_loss_limit_pct):
            self._trip("DAILY_LOSS")
        elif self._consec_losses >= self.max_consecutive_losses:
            self._trip("CONSEC_LOSSES")
        elif len(self._order_errors) >= self.order_error_burst_count:
            self._trip("ORDER_ERRORS")

    def check_data_fresh(self, age_ms: int) -> bool:
        if age_ms > self.data_staleness_ms:
            self._trip("DATA_STALENESS")
            return False
        return True

    def kill_switch(self, reason: str = "MANUAL"):
        self._trip(reason)

    def resume(self):
        if self.state.paused:
            self.state = CircuitState(paused=False, reason=None, last_change_ms=int(time.time()*1000))
            logger.warning("[Safety] Resumed trading.")
            try:
                publish_alert('RISK_RESUME', 'Trading resumed')
            except Exception:  # pragma: no cover
                pass

    def _trip(self, reason: str):
        if not self.state.paused or self.state.reason != reason:
            self.state = CircuitState(paused=True, reason=reason, last_change_ms=int(time.time()*1000))
            logger.error(f"[Safety] Circuit breaker TRIPPED: {reason} – trading paused (fail-closed).")
            try:
                publish_alert('RISK_PAUSE', f'Paused due to {reason}', severity='WARN', meta={'reason': reason})
            except Exception:  # pragma: no cover
                pass

    def snapshot(self) -> Dict:
        return {
            "paused": self.state.paused,
            "reason": self.state.reason,
            "daily_pnl": self._day_pnl,
            "consecutive_losses": self._consec_losses,
            "order_errors_window": len(self._order_errors),
        }

    # ----- Persistence helpers -----
    def serialize(self) -> Dict:
        return {
            "state": {
                "paused": self.state.paused,
                "reason": self.state.reason,
                "last_change_ms": self.state.last_change_ms,
            },
            "day_pnl": self._day_pnl,
            "consec_losses": self._consec_losses,
            "order_errors": self._order_errors,
        }

    def restore(self, payload: Dict):  # pragma: no cover (simple restore)
        try:
            st = payload.get("state", {})
            self.state = CircuitState(paused=st.get("paused", False), reason=st.get("reason"), last_change_ms=st.get("last_change_ms", 0))
            self._day_pnl = float(payload.get("day_pnl", 0.0))
            self._consec_losses = int(payload.get("consec_losses", 0))
            self._order_errors = list(payload.get("order_errors", []))
        except Exception:
            pass

__all__ = ["SafetyManager"]
