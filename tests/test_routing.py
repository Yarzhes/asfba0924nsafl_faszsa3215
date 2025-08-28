import time
import unittest
from ultra_signals.routing.aggregator import Aggregator
from ultra_signals.routing.types import PriceLevel, L2Book, VenueInfo
from ultra_signals.routing.cost_model import estimate_all_in_cost
from ultra_signals.routing.router import Router
from ultra_signals.routing.health import HealthMonitor
from ultra_signals.routing.twap_adapter import TWAPExecutor


def make_book(mid: float = 100.0, depth_levels: int = 5, size: float = 1.0):
    bids = [PriceLevel(price=mid - i * 0.1, size=size) for i in range(depth_levels)]
    asks = [PriceLevel(price=mid + i * 0.1, size=size) for i in range(depth_levels)]
    return L2Book(bids=bids, asks=asks, ts_ms=int(time.time() * 1000))


class TestRouting(unittest.TestCase):
    def test_cost_model_and_depth(self):
        book = make_book()
        v = VenueInfo(venue='TEST', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001)
        cb = estimate_all_in_cost(book, 'buy', v, target_notional=50.0, rtt_ms=10.0, impact_lambda=0.01)
        self.assertGreater(cb.gross_price, 0)
        self.assertGreaterEqual(cb.total_bps, cb.fees_bps)

    def test_router_split_behavior(self):
        agg = Aggregator()
        agg.update('A', make_book(mid=100.0))
        agg.update('B', make_book(mid=99.5))
        agg.update('C', make_book(mid=100.5))

        venues = {
            'A': VenueInfo(venue='A', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
            'B': VenueInfo(venue='B', maker_bps=0.0, taker_bps=1.8, min_notional=1.0, lot_size=0.0001),
            'C': VenueInfo(venue='C', maker_bps=0.0, taker_bps=2.2, min_notional=1.0, lot_size=0.0001),
        }

        r = Router(venues)
        dec = r.decide(agg.snapshot('BTCUSD'), 'buy', 10.0, rtt_map={'A': 10.0, 'B': 20.0, 'C': 15.0})
        self.assertIsInstance(dec.allocation, dict)
        if dec.allocation:
            s = sum(dec.allocation.values())
            self.assertAlmostEqual(s, 1.0, places=6)

    def test_health_monitor(self):
        hm = HealthMonitor(stale_threshold_s=0.5)
        hm.heartbeat('X')
        self.assertTrue(hm.is_healthy('X'))
        time.sleep(0.6)
        self.assertFalse(hm.is_healthy('X'))

    def test_twap_executor_runs(self):
        agg = Aggregator()
        agg.update('A', make_book())
        agg.update('B', make_book())
        venues = {
            'A': VenueInfo(venue='A', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
            'B': VenueInfo(venue='B', maker_bps=0.0, taker_bps=2.5, min_notional=1.0, lot_size=0.0001),
        }
        r = Router(venues)
        t = TWAPExecutor(r, slices=3)
        res = t.execute(agg, 'buy', total_notional=30.0, symbol='BTCUSD')
        self.assertEqual(len(res), 3)

    def test_circuit_breaker_excludes_venue(self):
        agg = Aggregator()
        agg.update('A', make_book())
        agg.update('B', make_book())
        venues = {
            'A': VenueInfo(venue='A', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
            'B': VenueInfo(venue='B', maker_bps=0.0, taker_bps=2.5, min_notional=1.0, lot_size=0.0001),
        }
        from ultra_signals.routing.health import HealthMonitor
        from ultra_signals.routing.router import Router

        hm = HealthMonitor(stale_threshold_s=10.0)
        hm.heartbeat('A')
        hm.heartbeat('B')
        r = Router(venues, health_monitor=hm)
        # open circuit on B
        r.record_reject('B', open_threshold=1)
        dec = r.decide(agg.snapshot('BTCUSD'), 'buy', 5.0)
        # B should be excluded
        self.assertNotIn('B', dec.allocation)

    def test_telemetry_emits_per_slice(self):
        from ultra_signals.routing.telemetry import TelemetryLogger
        agg = Aggregator()
        agg.update('A', make_book())
        agg.update('B', make_book())
        venues = {
            'A': VenueInfo(venue='A', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
            'B': VenueInfo(venue='B', maker_bps=0.0, taker_bps=2.5, min_notional=1.0, lot_size=0.0001),
        }
        from ultra_signals.routing.health import HealthMonitor
        from ultra_signals.routing.router import Router
        from ultra_signals.routing.twap_adapter import TWAPExecutor

        hm = HealthMonitor(stale_threshold_s=10.0)
        hm.heartbeat('A')
        hm.heartbeat('B')
        r = Router(venues, health_monitor=hm)
        t = TWAPExecutor(r, slices=4)
        tel = TelemetryLogger()
        t.set_telemetry(tel)
        _ = t.execute(agg, 'buy', total_notional=40.0, symbol='BTCUSD')
        ev = tel.get_events()
        self.assertEqual(len(ev), 4)
        # each event should have allocation and expected_cost_bps
        for e in ev:
            self.assertIn('allocation', e.decision)
            self.assertIn('expected_cost_bps', e.decision)


if __name__ == '__main__':
    unittest.main()
