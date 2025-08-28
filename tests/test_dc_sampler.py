import unittest
from ultra_signals.dc.sampler import DirectionalChangeSampler


class TestDCSampler(unittest.TestCase):

    def test_basic_up_down(self):
        s = DirectionalChangeSampler(theta_pct=0.01, start_price=100.0)  # 1%
        # move up 1.2% -> DC_UP
        events = s.on_price(101.2)
        types = [e.type.name for e in events]
        self.assertIn("DC_UP", types)

        # now push higher -> OS events
        events = s.on_price(102.0)
        types = [e.type.name for e in events]
        self.assertIn("OS", types)

        # now reverse down by >1% from last extreme -> DC_DOWN
        # last_extreme was 102.0, drop 1.5% -> ~100.47
        events = s.on_price(100.4)
        types = [e.type.name for e in events]
        self.assertIn("DC_DOWN", types)


if __name__ == "__main__":
    unittest.main()
