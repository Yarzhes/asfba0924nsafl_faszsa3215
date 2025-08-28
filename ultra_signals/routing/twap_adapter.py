from typing import Dict, Optional
import time
from .router import Router
from .types import AggregatedBook
from .telemetry import TelemetryLogger


class TWAPExecutor:
    """Simplified TWAP executor that at each slice queries the router for
    allocation and returns a list of child orders (venue, qty)
    """

    def __init__(self, router: Router, slices: int = 5):
        self.router = router
        self.slices = slices
        self.telemetry: Optional[TelemetryLogger] = None

    def set_telemetry(self, telemetry: TelemetryLogger):
        self.telemetry = telemetry

    def execute(self, agg_provider, side: str, total_notional: float, symbol: str, rtt_map: Dict[str, float]={}):
        per_slice = total_notional / self.slices
        results = []
        for i in range(self.slices):
            agg: AggregatedBook = agg_provider.snapshot(symbol)
            dec = self.router.decide(agg, side, per_slice, rtt_map=rtt_map)
            # emit telemetry if configured
            if self.telemetry is not None:
                try:
                    self.telemetry.emit_router_choice(symbol, i, dec)
                except Exception:
                    # telemetry must not break execution
                    pass
            # convert allocation pct to notional per venue
            child = {v: pct * per_slice for v, pct in dec.allocation.items()}
            results.append({'slice': i, 'decision': dec, 'child_notional': child, 'ts': time.time()})
        return results
