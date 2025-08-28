"""Small example demonstrating TWAPExecutor over a mocked aggregator.

Run:
  python examples/example_twap.py

This script builds two mocked venues, creates a Router + HealthMonitor,
then runs a TWAPExecutor to show per-slice allocations.
"""
import time
from ultra_signals.routing.aggregator import Aggregator
from ultra_signals.routing.types import PriceLevel, L2Book, VenueInfo
from ultra_signals.routing.health import HealthMonitor
from ultra_signals.routing.router import Router
from ultra_signals.routing.twap_adapter import TWAPExecutor


def make_book(mid: float = 100.0, depth_levels: int = 5, size: float = 1.0):
    bids = [PriceLevel(price=mid - i * 0.1, size=size) for i in range(depth_levels)]
    asks = [PriceLevel(price=mid + i * 0.1, size=size) for i in range(depth_levels)]
    return L2Book(bids=bids, asks=asks, ts_ms=int(time.time() * 1000))


def main():
    agg = Aggregator()
    agg.update('BINANCE', make_book(mid=100.0))
    agg.update('BYBIT', make_book(mid=99.8))

    venues = {
        'BINANCE': VenueInfo(venue='BINANCE', maker_bps=0.0, taker_bps=1.8, min_notional=1.0, lot_size=0.0001),
        'BYBIT': VenueInfo(venue='BYBIT', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
    }

    hm = HealthMonitor(stale_threshold_s=5.0)
    hm.heartbeat('BINANCE')
    hm.heartbeat('BYBIT')

    r = Router(venues, health_monitor=hm)
    twap = TWAPExecutor(r, slices=4)

    rtt_map = {'BINANCE': 12.0, 'BYBIT': 25.0}

    print('Running TWAP, total notional 40 over 4 slices')
    results = twap.execute(agg, 'buy', total_notional=40.0, symbol='BTCUSD', rtt_map=rtt_map)
    for s in results:
        dec = s['decision']
        print(f"slice {s['slice']}: reason={dec.reason}, expected_bps={dec.expected_cost_bps:.3f}")
        for v, n in s['child_notional'].items():
            print(f"  -> {v}: notional={n:.2f}")


if __name__ == '__main__':
    main()
