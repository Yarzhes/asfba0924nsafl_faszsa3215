"""Execution adapter wrapping TickReplayer to provide fills to BrokerSim / OrderExecutor.

This adapter exposes a simple place_order(plan) API that uses an internal TickReplayer
fed by supplied market events (snapshots/deltas) and returns a fill summary synchronously.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from ultra_signals.dc.tick_replayer import TickReplayer, LatencyModel


class ExecutionAdapter:
    def __init__(self, tick_replayer: Optional[TickReplayer] = None, latency: Optional[LatencyModel] = None):
        self.replayer = tick_replayer or TickReplayer(latency=latency)

    def feed_events(self, events):
        self.replayer.feed_from_iter(events)

    def place_order(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Place an immediate aggressive order (market/taker) and return a simulated fill.

        Plan expects: symbol, side, size, client_id(optional). This is synchronous and
        will run the internal replayer until the next fill occurs.
        """
        # convert plan -> trade event at now
        import time
        ev = {"ts": int(time.time()*1000), "type": "trade", "side": plan.get('side'), 'size': plan.get('size'), 'price': plan.get('price')}
        self.replayer.add_event(ev)
        fills = self.replayer.replay()
        # return the last fill event or empty
        return fills[-1] if fills else {}


__all__ = ["ExecutionAdapter"]
