from dataclasses import dataclass
from typing import List, Dict, Any
import time


@dataclass
class TelemetryEvent:
    ts: float
    symbol: str
    slice_index: int
    decision: Dict[str, Any]


class TelemetryLogger:
    """Very small in-memory telemetry collector for router events.

    In production this would push to a metrics/logging system.
    """

    def __init__(self):
        self.events: List[TelemetryEvent] = []

    def emit_router_choice(self, symbol: str, slice_index: int, decision):
        e = TelemetryEvent(ts=time.time(), symbol=symbol, slice_index=slice_index, decision={'expected_cost_bps': decision.expected_cost_bps, 'allocation': decision.allocation, 'reason': decision.reason})
        self.events.append(e)

    def emit_policy_action(self, symbol: str, slice_index: int, action: dict, metrics: dict | None = None):
        """Emit a compact policy action record for telemetry and dashboards.

        action: a dict-like representation with keys: type, size_mult, reason_codes, timestamp
        metrics: optional snapshot of metrics that triggered the action
        """
        try:
            decision = {
                'policy_action': {
                    'type': action.get('type'),
                    'size_mult': action.get('size_mult'),
                    'reason_codes': action.get('reason_codes'),
                    'timestamp': action.get('timestamp'),
                }
            }
            if metrics:
                decision['policy_metrics'] = metrics
            e = TelemetryEvent(ts=time.time(), symbol=symbol, slice_index=slice_index, decision=decision)
            self.events.append(e)
        except Exception:
            # best-effort only
            pass

    def get_events(self):
        return list(self.events)
