from __future__ import annotations
import time
import random
from typing import Dict, List, Optional, Tuple

from .types import AggregatedBook, VenueInfo
from .router import RouterDecision
from .cost_model import estimate_all_in_cost


class StrategySelector:
    """Simple selector that compares expected all-in cost for a few styles.

    This is a lightweight, testable approximation used by the VWAP slicer.
    It queries the routing cost model for MARKET-like fills and applies
    heuristics for LIMIT and TWAP styles.
    """

    def __init__(self, venues: Dict[str, VenueInfo], rtt_map: Dict[str, float] = {}):
        self.venues = venues
        self.rtt_map = rtt_map
        # small hysteresis to avoid flip-flop
        self._last_choice: Optional[str] = None

    def score_market(self, agg: AggregatedBook, side: str, notional: float) -> float:
        # ask router cost_model per venue and take best total_bps
        best = float('inf')
        for v, book in agg.books.items():
            vi = self.venues.get(v)
            if not vi:
                continue
            rtt = self.rtt_map.get(v, 20.0)
            cb = estimate_all_in_cost(book, side, vi, notional, rtt_ms=rtt)
            best = min(best, cb.total_bps)
        return best

    def score_limit(self, agg: AggregatedBook, side: str, notional: float) -> float:
        # limits are passive: assume lower fees (maker) and lower impact but some fill uncertainty
        best = float('inf')
        for v, book in agg.books.items():
            vi = self.venues.get(v)
            if not vi:
                continue
            # reuse estimate but replace taker fee with maker fee (assume maker_bps field exists)
            cb = estimate_all_in_cost(book, side, vi, notional)
            maker_fee = getattr(vi, 'maker_bps', max(0.0, vi.taker_bps - 0.5))
            est = cb.total_bps - cb.fees_bps + maker_fee
            # penalize for potential non-fill by adding a small execution risk premium
            est += 0.5
            best = min(best, est)
        return best

    def score_twap(self, agg: AggregatedBook, side: str, notional: float, slices: int = 6) -> float:
        # approximate TWAP cost by splitting and averaging market scores
        per = max(1e-6, notional / slices)
        s = 0.0
        for _ in range(slices):
            s += self.score_market(agg, side, per)
        return s / slices

    def choose(self, agg: AggregatedBook, side: str, notional: float, urgency: int = 0, features: Optional[dict] = None) -> str:
        """Return one of: 'MARKET', 'LIMIT', 'TWAP', 'VWAP'.

        urgency: 0=low, 1=normal, 2=high -> bias towards aggressive styles when high.
        """
        m = self.score_market(agg, side, notional)
        l = self.score_limit(agg, side, notional)
        t = self.score_twap(agg, side, notional)

        # VWAP treated as passive/twap hybrid; approximate as min(limit, twap)
        v = min(l, t)

        scores = {'MARKET': m, 'LIMIT': l, 'TWAP': t, 'VWAP': v}

        # urgency bias: if urgency high, prefer MARKET unless cost delta large
        if urgency >= 2:
            scores['MARKET'] -= 0.5

        # feature-based adjustments: if lambda high (impact risk), prefer passive; if VPIN toxic, prefer passive
        try:
            if features:
                lam = float(features.get('lambda', 0.0) or features.get('lam', 0.0))
                vpin_pctl = float(features.get('vpin_pctl', 0.0) or 0.0)
                spread_z = float(features.get('spread_z', 0.0) or 0.0)
                # high lambda -> penalize MARKET
                if lam > 0.001:
                    scores['MARKET'] += 0.5 + min(2.0, lam*1000)
                    scores['LIMIT'] -= 0.2
                # VPIN toxicity -> penalize aggressive
                if vpin_pctl > 0.9:
                    scores['MARKET'] += 1.0
                    scores['TWAP'] += 0.3
                # wide spread -> prefer limit/passive
                if spread_z and spread_z > 1.5:
                    scores['MARKET'] += 0.5
                    scores['LIMIT'] -= 0.5
        except Exception:
            pass

        # pick min score with small hysteresis
        choice = min(scores.items(), key=lambda kv: kv[1])[0]
        if self._last_choice and self._last_choice != choice:
            # require an improvement margin to switch
            if scores[choice] + 0.2 >= scores[self._last_choice]:
                choice = self._last_choice
        self._last_choice = choice
        return choice


class VWAPExecutor:
    """VWAP slicer that creates a schedule following a volume curve, enforces PR cap,
    adds jitter, and selects an execution style per slice.

    Assumptions made to keep this self-contained for tests:
    - volume_curve: sequence of positive floats summing to 1.0 representing
      percent-of-day volume per bin (e.g., 5m bins).
    - adv_per_second: average market notional per second (external input).
    """

    def __init__(
        self,
        venues: Dict[str, VenueInfo],
        volume_curve: Optional[List[float]] = None,
        pr_cap: float = 0.1,
        jitter_frac: float = 0.05,
        max_slice_notional: Optional[float] = None,
        rtt_map: Dict[str, float] = {},
        feature_provider: Optional[callable] = None,
        feature_writer: Optional[any] = None,
    ):
        self.venues = venues
        self.volume_curve = volume_curve or self._default_u_shape()
        self.pr_cap = pr_cap
        self.jitter_frac = jitter_frac
        self.max_slice_notional = max_slice_notional
        self.selector = StrategySelector(venues, rtt_map=rtt_map)
        # optional callable(symbol) -> dict of live features (lambda, vpin, spread_z ...)
        self.feature_provider = feature_provider
        # optional FeatureViewWriter-like object with write_record(record)
        self.feature_writer = feature_writer
        self.telemetry = None

    def set_telemetry(self, telemetry):
        self.telemetry = telemetry

    def _default_u_shape(self, bins: int = 78) -> List[float]:
        # simple U-shaped curve across the trading day (fraction per bin)
        # create a convex U by using inverted gaussian-like weights
        x = [i / (bins - 1) for i in range(bins)]
        weights = [(0.4 + 0.6 * ((4 * (xi - 0.5)) ** 2)) for xi in x]
        s = sum(weights)
        return [w / s for w in weights]

    def build_schedule(self, total_notional: float, start_ts: float, end_ts: float) -> List[Dict]:
        """Create a schedule of slices based on the volume curve between start and end ts.

        Returns a list of dicts: {ts, target_cum_frac, slice_frac}
        """
        bins = len(self.volume_curve)
        dt = (end_ts - start_ts) / bins
        schedule = []
        cum = 0.0
        for i, frac in enumerate(self.volume_curve):
            ts = start_ts + i * dt
            cum += frac
            schedule.append({'bin': i, 'ts': ts, 'slice_frac': frac, 'target_cum_frac': cum, 'dt': dt})
        return schedule

    def execute(self, agg_provider, side: str, total_notional: float, symbol: str, start_ts: Optional[float] = None, end_ts: Optional[float] = None, adv_per_second: Optional[float] = None, urgency: int = 1) -> List[Dict]:
        """Execute VWAP slicing.

        adv_per_second: if provided, used with pr_cap to bound slice notional.
        Returns list of slice results with chosen style and child notional.
        """
        now = time.time()
        start_ts = start_ts or now
        end_ts = end_ts or (now + 60 * 60)  # default 1h
        schedule = self.build_schedule(total_notional, start_ts, end_ts)

        results = []
        filled = 0.0
        for s in schedule:
            desired_cum = s['target_cum_frac'] * total_notional
            desired_slice = max(0.0, desired_cum - filled)

            # PR cap enforcement
            if adv_per_second and self.pr_cap > 0:
                max_slice = self.pr_cap * adv_per_second * s['dt']
                if self.max_slice_notional is not None:
                    max_slice = min(max_slice, self.max_slice_notional)
                if desired_slice > max_slice:
                    slice_notional = max_slice
                else:
                    slice_notional = desired_slice
            else:
                slice_notional = min(desired_slice, self.max_slice_notional) if self.max_slice_notional else desired_slice

            # jitter size and time
            jitter_amount = slice_notional * self.jitter_frac
            slice_notional = max(0.0, slice_notional + random.uniform(-jitter_amount, jitter_amount))
            ts_jitter = random.uniform(-0.5 * s['dt'], 0.5 * s['dt']) * self.jitter_frac
            exec_ts = s['ts'] + ts_jitter

            # pick execution style; allow feature-based decisions
            agg = agg_provider.snapshot(symbol)
            features = None
            if self.feature_provider is not None:
                try:
                    features = self.feature_provider(symbol)
                except Exception:
                    features = None
            style = self.selector.choose(agg, side, max(1e-6, slice_notional), urgency=urgency, features=features)

            # allocation: prefer to ask a Router if available (gives multi-venue split); otherwise fallback
            best_alloc = {}
            best_cost = float('inf')
            router_decision = None
            try:
                # agg_provider may expose a venue_router attribute
                router = getattr(agg_provider, 'venue_router', None)
                if router and hasattr(router, 'decide'):
                    # router.decide returns RouterDecision(allocation, expected_cost_bps, reason)
                    router_decision = router.decide(symbol, side, slice_notional, rtt_map=getattr(self.selector, 'rtt_map', {}))
                else:
                    router = None
            except Exception:
                router = None
            if router_decision:
                best_alloc = dict(router_decision.allocation or {})
                best_cost = float(getattr(router_decision, 'expected_cost_bps', best_cost) or best_cost)
            else:
                best_name = None
                for vname, book in agg.books.items():
                    vi = self.venues.get(vname)
                    if not vi:
                        continue
                    rtt = self.selector.rtt_map.get(vname, 20.0)
                    cb = estimate_all_in_cost(book, side, vi, max(1e-6, slice_notional) if isinstance(slice_notional, (int,float)) else 1e-6, rtt_ms=rtt)
                    if cb.total_bps < best_cost:
                        best_cost = cb.total_bps
                        best_name = vname
                if best_name:
                    best_alloc = {best_name: 1.0}

            # create a slice id for linkage with downstream fills
            slice_id = f"{int(exec_ts*1000)}.{s['bin']}"

            res = {
                'bin': s['bin'],
                'ts': exec_ts,
                'requested_slice_notional': desired_slice,
                'slice_notional': slice_notional,
                'style': style,
                'router_allocation': best_alloc,
                'expected_cost_bps': best_cost,
                'slice_id': slice_id,
                'router_decision': router_decision,
            }
            if self.telemetry:
                try:
                    self.telemetry.emit('vwap_slice', res)
                except Exception:
                    pass

            # persist a compact FeatureView record if writer provided
            if self.feature_writer is not None:
                try:
                    fv = {
                        'ts': int(exec_ts * 1000) if isinstance(exec_ts, float) else int(time.time() * 1000),
                        'symbol': symbol,
                        'components': {
                            'exec_strategy': style,
                            'router_allocation': best_alloc,
                            'requested_slice_notional': desired_slice,
                        },
                        # Flatten important execution metrics for fast queries
                        'expected_cost_bps': best_cost,
                        'realized_cost_bps': None,
                        'schedule_lag': 0.0,
                        'price': None,
                        'slice_id': slice_id,
                    }
                    self.feature_writer.write_record(fv)
                except Exception:
                    pass

            results.append(res)
            filled += slice_notional
            # stop early if filled
            if filled >= total_notional * 0.999999:
                break

        return results


__all__ = ['VWAPExecutor', 'StrategySelector']
