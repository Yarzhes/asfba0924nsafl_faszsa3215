import unittest
import random
from ultra_signals.dc.replay import replay_prices


class TestDCProperties(unittest.TestCase):

    def test_higher_theta_fewer_events(self):
        random.seed(0)
        prices = [100.0]
        for _ in range(200):
            prices.append(prices[-1] * (1 + random.uniform(-0.002, 0.002)))
        thetas = [0.001, 0.002, 0.005]
        outputs = replay_prices(prices, thetas)
        counts = {th: len(evs) for th, evs in outputs.items()}
        # monotone: more theta -> fewer events
        self.assertTrue(counts[0.001] >= counts[0.002] >= counts[0.005])

    def test_multi_theta_subset_like(self):
        # For synthetic trend moves, larger theta should detect subset of flips
        prices = [100.0, 101.0, 100.0, 99.0, 100.5, 98.5, 102.0, 100.0]
        thetas = [0.005, 0.01]
        outputs = replay_prices(prices, thetas)
        small = outputs[0.005]
        large = outputs[0.01]
        # every event in large should coincide roughly with some event in small
        # check that count_large <= count_small
        self.assertTrue(len(large) <= len(small))


if __name__ == "__main__":
    unittest.main()
