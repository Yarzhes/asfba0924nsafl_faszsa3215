"""Orderbook / Microstructure health snapshot (Sprint 29).

Provides lightweight container + helper to compute core micro-liquidity
metrics used by the Liquidity Gate. Implementation is intentionally
minimal so it can operate both with full L2 depth feeds or fall back to
synthetic inputs (proxies) supplied by backtester code.

All numeric fields are floats (bps where noted). Any field may be None
if unavailable – the gate module will degrade gracefully.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Optional, Sequence


@dataclass(slots=True)
class BookHealth:
    ts: int                     # epoch seconds
    symbol: str
    spread_bps: Optional[float] = None   # (ask-bid)/mid * 10_000
    dr: Optional[float] = None            # top of book depth ratio (bid-ask)/(bid+ask)
    dr_5bps: Optional[float] = None       # depth ratio within +-5 bps if available
    dr_10bps: Optional[float] = None      # depth ratio within +-10 bps if available
    impact_50k: Optional[float] = None    # bps move to fill $50k (or configured) notional
    rv_5s: Optional[float] = None         # realized volatility over last ~5s window (bps)
    mt: Optional[float] = None            # micro trendiness (slope / correlation proxy)
    cu_ratio: Optional[float] = None      # cancel/update burst ratio (optional)
    source: str = "raw"                  # 'raw' or 'proxy'

    @staticmethod
    def compute(
        ts: int,
        symbol: str,
        bid: Optional[float],
        ask: Optional[float],
        bid_qty: Optional[float] = None,
        ask_qty: Optional[float] = None,
        micro_prices: Optional[Sequence[float]] = None,
        micro_returns: Optional[Sequence[float]] = None,
        rv_returns: Optional[Sequence[float]] = None,
        impact_levels: Optional[Sequence[tuple[float, float]]] = None,
        notional_target: float = 50_000.0,
    ) -> "BookHealth":
        """Best-effort computation of health metrics.

        Parameters
        ----------
        micro_prices : sequence of recent (ascending) trade prices (<=6 points)
        micro_returns: optional pre-computed returns sequence for volatility
        impact_levels: optional [(price, cum_notional)] sorted by level depth
        """
        spread_bps = None
        dr = None
        impact_50k = None
        rv_5s = None
        mt = None
        try:
            if bid and ask and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                spread_bps = (ask - bid) / mid * 10_000
        except Exception:
            pass
        try:  # depth ratio
            if bid_qty is not None and ask_qty is not None and (bid_qty + ask_qty) > 0:
                dr = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        except Exception:
            pass
        try:  # simple impact: first level exceeding target
            if impact_levels:
                for px, cum_notional in impact_levels:
                    if cum_notional >= notional_target and bid and bid > 0:
                        impact_50k = abs(px - bid) / bid * 10_000
                        break
        except Exception:
            pass
        try:  # realized micro volatility sqrt(sum(r^2)) in bps
            rets = list(rv_returns) if rv_returns is not None else []
            if not rets and micro_returns:
                rets = list(micro_returns)
            if rets:
                rv_5s = sqrt(sum((r or 0.0) ** 2 for r in rets)) * 10_000
        except Exception:
            pass
        try:  # micro trendiness: simple linear slope proxy using last N prices
            if micro_prices and len(micro_prices) >= 3:
                xs = list(range(len(micro_prices)))
                n = float(len(xs))
                mean_x = (n - 1) / 2.0
                mean_y = sum(micro_prices) / n
                num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, micro_prices))
                den = sum((x - mean_x) ** 2 for x in xs) or 1.0
                slope = num / den
                # Normalize slope by last price to approximate bps/step
                last = micro_prices[-1]
                if last:
                    mt = slope / last * 10_000
        except Exception:
            pass
        return BookHealth(
            ts=ts,
            symbol=symbol,
            spread_bps=spread_bps,
            dr=dr,
            impact_50k=impact_50k,
            rv_5s=rv_5s,
            mt=mt,
        )

__all__ = ["BookHealth"]
