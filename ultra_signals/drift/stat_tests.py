"""Statistical test primitives used for drift detection.

Includes a simple SPRT for binomial win-rate drift and a CUSUM for mean PnL
changes. The implementations are minimal but tested.
"""
from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class SPRT:
    """Sequential Probability Ratio Test for binomial outcomes.

    This SPRT tracks successes / trials and exposes a log-likelihood ratio.
    We use simple Bernoulli likelihoods with H0=p0 and H1=p1 and stop when
    the LLR crosses A=log((1-beta)/alpha) or B=log(beta/(1-alpha)).
    """

    p0: float = 0.5
    p1: float = 0.62
    alpha: float = 0.01
    beta: float = 0.1

    s: int = 0
    n: int = 0

    def update(self, success: bool) -> None:
        self.n += 1
        if success:
            self.s += 1

    def llr(self) -> float:
        # log likelihood ratio for observed s successes in n trials
        if self.n == 0:
            return 0.0
        # Avoid log(0)
        p0, p1 = max(1e-12, self.p0), max(1e-12, self.p1)
        s = self.s
        n = self.n
        llr = s * math.log(p1 / p0) + (n - s) * math.log((1 - p1) / (1 - p0))
        return llr

    def bounds(self) -> tuple[float, float]:
        A = math.log((1 - self.beta) / self.alpha)
        B = math.log(self.beta / (1 - self.alpha))
        return A, B

    def decision(self) -> Optional[str]:
        a, b = self.bounds()
        v = self.llr()
        if v >= a:
            return "accept_h1"
        if v <= b:
            return "accept_h0"
        return None


@dataclass
class CUSUM:
    """One-sided CUSUM for detecting increases in mean (or decreases by
    switching sign). Uses small-sample update with drift and threshold.
    """

    threshold: float = 1.0
    drift: float = 0.0
    s_pos: float = 0.0
    s_neg: float = 0.0

    def update(self, x: float) -> Optional[str]:
        # positive CUSUM (detect mean increase)
        self.s_pos = max(0.0, self.s_pos + x - self.drift)
        # negative CUSUM (detect mean decrease)
        self.s_neg = max(0.0, self.s_neg - x - self.drift)
        if self.s_pos >= self.threshold:
            return "pos"  # mean increased
        if self.s_neg >= self.threshold:
            return "neg"  # mean decreased
        return None
