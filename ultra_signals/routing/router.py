from typing import Dict, List, Optional
from collections import defaultdict
from .types import AggregatedBook, RouterDecision, VenueInfo
from .cost_model import estimate_all_in_cost
from .health import HealthMonitor
try:
    from ultra_signals.tca.tca_engine import TCAEngine
except Exception:  # pragma: no cover - optional dependency at runtime
    TCAEngine = None


class Router:
    def __init__(self, venues: Dict[str, VenueInfo], health_monitor: Optional[HealthMonitor] = None, tca_engine: Optional[object] = None):
        self.venues = venues
        # optional health monitor used to exclude unhealthy venues
        self.health_monitor = health_monitor
        # simple circuit state: venue -> bool (True=open/circuited)
        self._circuit_open: Dict[str, bool] = {}
        # simple reject counters
        self._reject_counts = defaultdict(int)
        # optional TCA engine used to adjust venue cost estimates
        self.tca_engine = tca_engine

    def record_reject(self, venue: str, open_threshold: int = 3):
        """Record a reject for a venue and open the circuit if threshold exceeded."""
        self._reject_counts[venue] += 1
        if self._reject_counts[venue] >= open_threshold:
            self._circuit_open[venue] = True

    def reset_circuit(self, venue: str):
        """Reset circuit state and reject counters for a venue."""
        self._circuit_open.pop(venue, None)
        self._reject_counts.pop(venue, None)

    def decide(self, agg: AggregatedBook, side: str, target_notional: float, rtt_map: Dict[str, float]={}) -> RouterDecision:
        """Decide allocation across venues.

        Simple strategy:
        - compute all-in total_bps per venue
        - if best is significantly better (>1bps) pick it 100%
        - else split proportional to 1/total_bps among top 3 available venues
        """
        costs = {}
        for vname, book in agg.books.items():
            # skip if venue unknown
            venue_info = self.venues.get(vname)
            if not venue_info:
                continue
            # skip if circuit is open
            if self._circuit_open.get(vname, False):
                continue
            # skip if health monitor says unhealthy
            if self.health_monitor is not None and not self.health_monitor.is_healthy(vname):
                continue
            rtt = rtt_map.get(vname, 20.0)
            cb = estimate_all_in_cost(book, side, venue_info, target_notional, rtt_ms=rtt)
            # consult TCA engine if present to adjust expected cost
            total_bps = float(cb.total_bps)
            if self.tca_engine is not None:
                try:
                    total_bps = float(self.tca_engine.get_effective_cost_bps(vname, total_bps, rtt_ms=rtt))
                except Exception:
                    pass
            # store a shallow copy-like object with total_bps attribute expected by call sites
            class _CB:
                def __init__(self, total_bps):
                    self.total_bps = total_bps

            costs[vname] = _CB(total_bps)

        if not costs:
            return RouterDecision(allocation={}, expected_cost_bps=float('inf'), reason='no_healthy_venues')

        # sort by total_bps
        sorted_vs = sorted(costs.items(), key=lambda kv: kv[1].total_bps)
        best_name, best_cb = sorted_vs[0]

        # dominance threshold
        if len(sorted_vs) == 1 or (sorted_vs[1][1].total_bps - best_cb.total_bps) > 1.0:
            return RouterDecision(allocation={best_name: 1.0}, expected_cost_bps=best_cb.total_bps, reason=f'select_{best_name}')

        # else split among top-3 by inverse cost weight
        topk = sorted_vs[:3]
        inv = {n: max(1e-6, 1.0 / cb.total_bps) for n, cb in topk}
        s = sum(inv.values())
        allocation = {n: inv[n] / s for n in inv}
        # expected cost is weighted sum
        expected = sum(allocation[n] * costs[n].total_bps for n in allocation)

        return RouterDecision(allocation=allocation, expected_cost_bps=expected, reason='split_topk')
