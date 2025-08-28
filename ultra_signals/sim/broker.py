"""Sprint 36: Real-Time Broker Simulator (initial minimal implementation)

This module provides a pluggable BrokerSim used by backtests / paper trading to
approximate realistic exchange microstructure effects:

- Latency pipeline (submit -> ack -> resting/match -> fills)
- Limit order queue position and partial fills
- Market order sweeping top-N synthetic orderbook levels with impact + jitter
- Maker/taker fee handling (rebates supported)
- Post-only rejections if order would cross spread
- Deterministic seeded RNG for reproducibility

Design goals:
- Keep API small & side-effect free so integration surfaces are predictable
- Deterministic given (settings hash, seed, bar sequence)
- Allow future extension (iceberg, cancel/modify races) without breaking interface

High-level usage:
    cfg = settings['broker_sim']
    ob_model = SyntheticOrderBook(...)
    sim = BrokerSim(cfg, ob_model, rng_seed=cfg.get('rng_seed',42))
    fills = sim.submit_order(order)

The simulator operates in *sim time* (ms) advanced explicitly via advance(ms).
For backtests we map each bar close to a time stride.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Iterable
import math, random, time

Side = Literal['BUY','SELL']
OrderType = Literal['MARKET','LIMIT','POST_ONLY']
LiquidityFlag = Literal['MAKER','TAKER']

# ----------------------------- Data Structures -----------------------------
@dataclass
class Order:
    id: str
    symbol: str
    side: Side                # BUY/SELL internal (LONG -> BUY, SHORT -> SELL)
    type: OrderType
    qty: float
    price: Optional[float] = None
    tif: str = 'GTC'
    post_only: bool = False
    ts_submit: Optional[int] = None  # client submit ts (ms, sim clock)
    remaining: Optional[float] = None

@dataclass
class FillEvent:
    ts: int
    order_id: str
    symbol: str
    qty: float
    price: float
    fee_bps: float
    liquidity: LiquidityFlag
    venue: str

# ----------------------------- Clock / RNG -----------------------------
class SimClock:
    def __init__(self, start_ms: int = 0, rng: Optional[random.Random]=None):
        self._now = int(start_ms)
        self.rng = rng or random.Random()
    def now(self) -> int:
        return self._now
    def advance(self, ms: int) -> int:
        self._now += int(ms)
        return self._now

# ----------------------------- Latency helpers -----------------------------
class LatencySampler:
    def __init__(self, cfg: Dict[str, Any], rng: random.Random):
        self.cfg = cfg or {}
        self.rng = rng
    def sample(self, kind: str) -> int:
        spec = (self.cfg or {}).get(kind) or {}
        if 'fixed' in spec:
            return int(spec['fixed'])
        dist = spec.get('dist')
        if dist == 'lognorm':
            # mu/log sigma in natural log domain, clamp to [min,max]
            mu = float(spec.get('mu', 4.5)); sigma = float(spec.get('sigma', 0.35))
            # Python random.lognormvariate uses underlying mu,sigma of log distribution
            val = self.rng.lognormvariate(mu, sigma)
            if 'min' in spec: val = max(val, float(spec['min']))
            if 'max' in spec: val = min(val, float(spec['max']))
            return int(val)
        return int(spec.get('min', 1))

class JitterSampler:
    def __init__(self, spec: Dict[str, Any], rng: random.Random):
        self.spec = spec or {}
        self.rng = rng
    def sample_bps(self) -> float:
        if not self.spec: return 0.0
        if 'dist' not in self.spec: return 0.0
        d = self.spec['dist']
        if d == 'normal':
            mean = float(self.spec.get('mean', 0.0)); std = float(self.spec.get('std', 1.0))
            clip = float(self.spec.get('clip', std*3))
            v = self.rng.gauss(mean, std)
            if v > clip: v = clip
            if v < -clip: v = -clip
            return v
        return 0.0

# ----------------------------- Orderbook abstraction -----------------------------
class AbstractOrderBook:
    def best_bid(self) -> float: raise NotImplementedError
    def best_ask(self) -> float: raise NotImplementedError
    def ladder(self) -> List[tuple]: raise NotImplementedError  # list[(price, qty)] ask side if buying, bid side if selling
    def advance(self, ms: int): pass

# ----------------------------- Broker Simulator -----------------------------
class BrokerSim:
    def __init__(self, settings: Dict[str, Any], ob_model: AbstractOrderBook, rng_seed: int=42, venue: str='SIM', lambda_provider: Optional[callable]=None):
        # basic config & RNG
        self.cfg = settings or {}
        self.venue = venue
        self.rng = random.Random(int(rng_seed))
        self.clock = SimClock(0, self.rng)

        # venue-level settings
        venue_cfg = ((self.cfg.get('venues') or {}).get(venue) or {})
        lat_cfg = (venue_cfg.get('latency_ms') or {})
        self.lat_sampler = LatencySampler(lat_cfg, self.rng)

        slip_cfg = (venue_cfg.get('slippage') or {})
        self.impact_factor = float(slip_cfg.get('impact_factor', 0.5))
        self.jitter_sampler = JitterSampler(slip_cfg.get('jitter_bps') or {}, self.rng)

        # optional external lambda provider: callable(symbol) -> lambda (ΔP/ΔQ)
        self.lambda_provider = lambda_provider
        # k_temp multiplier for temporary impact when using lambda
        self.k_temp = float(slip_cfg.get('k_temp', 1.0))

        vd = self.cfg.get('venue_defaults') or {}
        self.maker_fee_bps = float(vd.get('maker_fee_bps', -1.0))
        self.taker_fee_bps = float(vd.get('taker_fee_bps', 4.0))

        # runtime state
        self.orderbook = ob_model
        self.open_orders = {}
        self.fills = []
        # queue model: price -> list[(order_id, remaining_qty)] for LIMIT/POST_ONLY
        self._queues = {}

    # --------------- Public API ---------------
    def submit_order(self, order: Order) -> List[FillEvent]:
        order.ts_submit = self.clock.now()
        order.remaining = float(order.qty)
        # submission latency
        self.clock.advance(self.lat_sampler.sample('submit'))
        if order.type in ('MARKET',):
            return self._exec_market(order)
        else:  # LIMIT or POST_ONLY
            return self._place_limit(order)

    def cancel_order(self, order_id: str) -> bool:
        o = self.open_orders.get(order_id)
        if not o: return False
        # cancel latency
        self.clock.advance(self.lat_sampler.sample('cancel'))
        # race: if already filled remaining=0 just remove
        if o.remaining and o.remaining > 0:
            self.open_orders.pop(order_id, None)
            # remove from queue
            if o.price in self._queues:
                self._queues[o.price] = [q for q in self._queues[o.price] if q[0]!=order_id]
        else:
            self.open_orders.pop(order_id, None)
        return True

    def advance_time(self, ms: int):
        """Advance simulated time & progress queue fills by consuming trade flow proportionally."""
        if ms <=0: return
        step = int(ms)
        # simplistic flow: each ms consume a fraction of resting depth at best price
        for _ in range(step):
            self.clock.advance(1)
            self.orderbook.advance(1)
            self._progress_limit_fills()

    # --------------- Internal Mechanics ---------------
    def _exec_market(self, order: Order) -> List[FillEvent]:
        ladder = self.orderbook.ladder()
        if not ladder:
            return []
        need = order.remaining or 0.0
        consumed = []  # list[(price, qty)]
        for px, avail in ladder:
            if need <= 0: break
            take = min(avail, need)
            consumed.append((px, take))
            need -= take
        filled_qty = sum(q for _,q in consumed)
        if filled_qty <= 0:
            return []
        vwap = sum(p*q for p,q in consumed)/filled_qty
        # price impact: prefer lambda-based temporary impact when provider present
        impact = 0.0
        if self.lambda_provider is not None:
            try:
                lam = float(self.lambda_provider(order.symbol) or 0.0)
            except Exception:
                lam = 0.0
            # ΔP_temp = k_temp * λ * slice_volume
            impact = self.k_temp * lam * filled_qty
            if order.side == 'BUY':
                exec_px = vwap + impact
            else:
                exec_px = vwap - impact
        else:
            # fallback: price impact adjustment based on bar proxy: approximate using total swept fractional of book
            total_depth = sum(q for _,q in ladder)
            depth_frac = filled_qty / max(1e-9, total_depth)
            impact = self.impact_factor * depth_frac * (ladder[-1][0] - ladder[0][0]) if len(ladder)>=2 else 0.0
            if order.side == 'BUY':
                exec_px = vwap + impact
            else:
                exec_px = vwap - impact
        # jitter
        jitter_bps = self.jitter_sampler.sample_bps()
        exec_px *= (1 + (jitter_bps/10_000.0) * (1 if order.side=='BUY' else -1))
        fill_ts = self.clock.now() + self.lat_sampler.sample('match')
        self.clock.advance(self.lat_sampler.sample('match'))
        fee_bps = self.taker_fee_bps
        fill = FillEvent(ts=fill_ts, order_id=order.id, symbol=order.symbol, qty=filled_qty, price=exec_px, fee_bps=fee_bps, liquidity='TAKER', venue=self.venue)
        self.fills.append(fill)
        order.remaining = 0.0
        return [fill]

    def _place_limit(self, order: Order) -> List[FillEvent]:
        # Post-only rejection
        if order.post_only or order.type == 'POST_ONLY':
            best_ask = self.orderbook.best_ask(); best_bid = self.orderbook.best_bid()
            if order.side=='BUY' and order.price is not None and best_ask and order.price >= best_ask:
                # crosses
                if (self.cfg.get('policies') or {}).get('post_only_reject_if_cross', True):
                    return []
            if order.side=='SELL' and order.price is not None and best_bid and order.price <= best_bid:
                if (self.cfg.get('policies') or {}).get('post_only_reject_if_cross', True):
                    return []
        # enqueue
        price = float(order.price)
        qlist = self._queues.setdefault(price, [])
        queue_qty_before = sum(q for _,q in qlist)
        # queue position at tail
        qlist.append((order.id, order.qty))
        self.open_orders[order.id] = order
        order.remaining = float(order.qty)
        order._queue_ahead = queue_qty_before  # type: ignore
        return []

    def _progress_limit_fills(self):
        # For each price level with resting orders, simulate taker flow hitting it with probability
        prices = list(self._queues.keys())
        if not prices: return
        for price in prices:
            q = self._queues.get(price) or []
            if not q: continue
            # simplistic flow intensity: random fraction
            flow = (0.02 + 0.05*self.rng.random()) * sum(rem for _, rem in q)
            # iterate queue
            new_q = []
            for oid, rem in q:
                if flow <= 0:
                    new_q.append((oid, rem))
                    continue
                consume = min(rem, flow)
                flow -= consume
                order = self.open_orders.get(oid)
                if order is None: # canceled
                    continue
                order.remaining = max(0.0, (order.remaining or 0.0) - consume)
                if order.remaining <= 1e-9:
                    # full fill
                    self._emit_fill(order, consume if consume>0 else rem, price, 'MAKER')
                else:
                    # partial fill event chunk only if chunk >= min ratio
                    min_ratio = float((self.cfg.get('policies') or {}).get('partial_fill_min_ratio', 0.1))
                    if consume / max(1e-9, order.qty) >= min_ratio:
                        self._emit_fill(order, consume, price, 'MAKER')
                    new_q.append((oid, order.remaining))
            # replace queue
            self._queues[price] = [item for item in new_q if item[1] > 1e-9]

    def _emit_fill(self, order: Order, qty: float, price: float, liq: LiquidityFlag):
        fee = self.maker_fee_bps if liq=='MAKER' else self.taker_fee_bps
        ts = self.clock.now()
        fill = FillEvent(ts=ts, order_id=order.id, symbol=order.symbol, qty=qty, price=price, fee_bps=fee, liquidity=liq, venue=self.venue)
        self.fills.append(fill)
        if order.remaining is not None and order.remaining <= 1e-9:
            self.open_orders.pop(order.id, None)

# ---------------- Convenience factory ----------------

def map_side(signal_side: str) -> Side:
    if str(signal_side).upper() in ('LONG','BUY'): return 'BUY'
    return 'SELL'

__all__ = ['BrokerSim','Order','FillEvent','SimClock','map_side']
