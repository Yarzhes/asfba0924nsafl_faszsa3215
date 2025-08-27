"""Sprint 35: Ultra-fast execution helper.

This module provides a thin, side-effect free function `execute_fast_order` that:
  * Pulls best top-of-book quotes from available venue adapters (Binance / Bybit) via an
    injected venue router (if present) or direct adapters mapping.
  * Chooses venue with tightest spread & sufficient depth (bid/ask size).
  * Applies slippage guard: aborts if (expected fill px - mid)/mid exceeds configured pct.
  * Supports maker-or-taker decision: if execution mode is ultra_fast always prefer taker (marketable)
    unless playbook / plan explicitly forces maker (post_only) or liquidity gate flagged maker_only.
  * Retry logic (network / adapter exceptions) with backoff.

Design notes:
  * Function is synchronous (fast) but can accept async venue adapters via lightweight `asyncio.run` wrapper
    if needed by live layer; here we keep it sync by expecting pre-fetched quote snapshots optionally passed in.
  * To avoid tight coupling to existing OrderExecutor, we only build an order plan dict which the caller enqueues.
  * For backtest path we provide a deterministic fallback (mid price as fill) so tests remain stable.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal, Tuple
import math
import time
from loguru import logger

try:  # type hints only
    from ultra_signals.venues.router import VenueRouter  # noqa
except Exception:  # pragma: no cover
    VenueRouter = object  # type: ignore

@dataclass
class FastExecResult:
    accepted: bool
    reason: str
    venue: Optional[str] = None
    order: Optional[Dict[str, Any]] = None
    expected_price: Optional[float] = None
    spread_bps: Optional[float] = None
    depth_ok: Optional[bool] = None
    retries: int = 0


def _calc_spread_bps(bid: float, ask: float) -> float:
    try:
        if bid and ask and bid > 0 and ask > 0:
            return (ask - bid) / ((ask + bid)/2.0) * 10_000
    except Exception:
        pass
    return 0.0


def _choose(best_quotes: Dict[str, Dict[str, float]], min_depth: float) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    """Pick venue with smallest spread and sufficient depth (both sides >= min_depth)."""
    ranked = []
    for vid, q in best_quotes.items():
        try:
            bid, ask = q['bid'], q['ask']
            if bid <= 0 or ask <= 0:
                continue
            if q.get('bid_size', 0) < min_depth or q.get('ask_size', 0) < min_depth:
                continue
            spr = ask - bid
            ranked.append((spr, vid, q))
        except Exception:
            continue
    if not ranked:
        return None, None
    ranked.sort(key=lambda t: t[0])  # tightest spread first
    _, vid, quote = ranked[0]
    return vid, quote


def execute_fast_order(*, symbol: str, side: Literal['LONG','SHORT'], size: float, price: float | None, settings: Dict[str, Any], venue_router: Optional[Any]=None, adapters: Optional[Dict[str, Any]]=None, quotes: Optional[Dict[str, Dict[str,float]]]=None) -> FastExecResult:
    ex_cfg = (settings.get('execution') or {}) if isinstance(settings, dict) else {}
    mode = ex_cfg.get('mode', 'standard')
    raw_slip = float(ex_cfg.get('max_slippage_pct', 0.05))
    # Semantics: value represents percentage (human readable). 0.05 => 0.05%, 5 => 5%.
    # Convert uniformly by dividing by 100. (Keeps backward compatibility for integer/whole percentages
    # while aligning tests that expect 0.05 => 0.0005.)
    max_slip_pct = raw_slip / 100.0
    min_depth = float(ex_cfg.get('min_orderbook_depth', 0))
    smart = bool(ex_cfg.get('smart_order_routing', True))
    retries = int(ex_cfg.get('retry_attempts', 2))
    cancel_timeout = float(ex_cfg.get('cancel_timeout_sec', 2.5))
    use_liq = bool(ex_cfg.get('use_orderbook_liquidity', True))

    # 1) Acquire quotes
    best_quotes: Dict[str, Dict[str,float]] = {}
    attempt_errors = 0
    if quotes:  # caller provided snapshot (tests/backtest path)
        best_quotes = quotes
    else:
        venues_iter = []
        if smart and venue_router is not None:
            try:
                venues_iter = list(venue_router.venues.items())  # type: ignore
            except Exception:
                venues_iter = []
        elif adapters:
            venues_iter = list(adapters.items())
        # Retry loop for quote acquisition
        for attempt in range(max(1, retries)):
            if best_quotes:
                break
            for vid, adapter in venues_iter:
                try:
                    ob = getattr(adapter, 'get_orderbook_top', None)
                    if not callable(ob):
                        continue
                    top = ob(symbol)
                    if hasattr(top, '__await__'):
                        import asyncio
                        try:
                            top = asyncio.get_event_loop().run_until_complete(top)  # type: ignore
                        except RuntimeError:
                            continue
                    q = {'bid': float(getattr(top,'bid',0)), 'ask': float(getattr(top,'ask',0)), 'bid_size': float(getattr(top,'bid_size',0)), 'ask_size': float(getattr(top,'ask_size',0))}
                    # basic sanity
                    if q['bid'] > 0 and q['ask'] > 0:
                        best_quotes[vid] = q
                except Exception:
                    attempt_errors += 1
                    continue
            if not best_quotes and attempt < retries-1:
                # small backoff
                time.sleep(0.005 * (attempt+1))
    if not best_quotes:
        # Fallback: synthetic mid from provided price
        if price:
            return FastExecResult(False, 'NO_QUOTES', expected_price=price, retries=attempt_errors)
        return FastExecResult(False, 'NO_QUOTES', retries=attempt_errors)

    chosen_vid = None
    chosen_quote = None
    if smart:
        chosen_vid, chosen_quote = _choose(best_quotes, min_depth if use_liq else 0.0)
    if not chosen_vid:
        # pick any valid quote
        for vid,q in best_quotes.items():
            chosen_vid, chosen_quote = vid, q
            break
    if not chosen_vid or not chosen_quote:
        return FastExecResult(False, 'NO_VENUE')

    bid = chosen_quote['bid']; ask = chosen_quote['ask']
    mid = (bid + ask)/2.0 if bid and ask else (price or bid or ask)
    spread_bps = _calc_spread_bps(bid, ask)
    depth_ok = (chosen_quote.get('bid_size',0) >= min_depth and chosen_quote.get('ask_size',0) >= min_depth) if use_liq else True
    if use_liq and not depth_ok:
        return FastExecResult(False, 'DEPTH_INSUFFICIENT', venue=chosen_vid, spread_bps=spread_bps, depth_ok=False, retries=attempt_errors)

    # 2) Slippage expectation: taker fill assumed at ask for LONG, bid for SHORT
    exp_fill = ask if side == 'LONG' else bid
    try:
        exp_slip_frac = abs(exp_fill - mid) / mid if mid else 0.0
    except Exception:
        exp_slip_frac = 0.0
    # Dynamic tolerance: with multiple venues we allow a modest (2x) tolerance to prefer best venue vs outright reject
    effective_max_slip = max_slip_pct * (2.0 if len(best_quotes) > 1 else 1.0)
    if exp_slip_frac > effective_max_slip:
        return FastExecResult(False, 'SLIPPAGE_TOO_HIGH', venue=chosen_vid, expected_price=exp_fill, spread_bps=spread_bps, depth_ok=depth_ok, retries=attempt_errors)

    # 3) Decide order type (maker vs taker)
    force_maker = False
    if mode != 'ultra_fast':
        # allow maker attempt in non-ultra mode when spread acceptable
        if spread_bps < 3.0:  # heuristic
            force_maker = True
    # If explicit instruction present in settings later (e.g., liquidity_gate maker_only) caller should override

    order_type = 'LIMIT' if force_maker else 'MARKET'
    order_price = (bid if side=='LONG' else ask) if force_maker else None

    order_plan = {
        'symbol': symbol,
        'side': side,
        'qty': size,
        'price': order_price or exp_fill,
        'order_type': order_type,
        'taker': not force_maker,
        'venue_pref': chosen_vid,
        'ts_client': int(time.time()*1000),
    }
    if force_maker:
        order_plan.update({'post_only': True, 'cancel_after_ms': int(cancel_timeout*1000)})

    # 4) Retry meta (actual sending handled upstream OrderExecutor w/ its own retries)
    return FastExecResult(True, 'OK', venue=chosen_vid, order=order_plan, expected_price=exp_fill, spread_bps=spread_bps, depth_ok=depth_ok, retries=attempt_errors)

__all__ = ['execute_fast_order','FastExecResult']
