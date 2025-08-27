"""VenueRouter deciding data & order venue with failover and stickiness."""
from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
import time
from loguru import logger
from .health import HealthRegistry


class VenueRouter:
    def __init__(self, venues: Dict[str, Any], symbol_mapper, cfg: Dict[str, Any]):
        self.venues = venues  # id -> adapter
        self.symbol_mapper = symbol_mapper
        self.cfg = cfg or {}
        self.health = HealthRegistry(cfg.get("health", {}))
        self._symbol_data_venue: Dict[str, str] = {}
        self._symbol_order_venue: Dict[str, str] = {}
        self._primary_order = cfg.get("primary_order", list(venues.keys()))
        self._data_order = cfg.get("data_order", list(venues.keys()))
        self._prefer_lower_fee = bool(cfg.get("prefer_lower_fee_on_tie", True))
        self._fees = cfg.get("fees", {}) or {}
        # Colocation bias: prefer keeping orders & data on same venue if score difference below threshold
        self._colocation_bias_diff = float(cfg.get("colocation_bias_score_diff", 0.10))

    def _health_rank(self, vid: str) -> float:
        snap = self.health.ensure(vid)
        return snap.score(self.health.cfg)

    def decide_data_venue(self, symbol: str, timeframe: str) -> Optional[str]:
        # stickiness
        current = self._symbol_data_venue.get(symbol)
        if current and self._health_rank(current) >= self.health.cfg.get("red_threshold", 0.35):
            return current
        # choose best by score among preferred list
        cand = sorted(self._data_order, key=lambda v: self._health_rank(v), reverse=True)
        for vid in cand:
            if self._health_rank(vid) >= self.health.cfg.get("red_threshold", 0.35):
                self._symbol_data_venue[symbol] = vid
                return vid
        return None

    def _fee(self, vid: str) -> float:
        f = self._fees.get(vid, {})
        return float(f.get("taker", f.get("maker", 0)))

    def decide_order_venue(self, symbol: str, side: str) -> Optional[str]:
        current = self._symbol_order_venue.get(symbol)
        if current and self._health_rank(current) >= self.health.cfg.get("red_threshold", 0.35):
            return current
        # Co-location preference with existing data venue
        data_vid = self._symbol_data_venue.get(symbol)
        ranked = sorted(self._primary_order, key=lambda v: (self._health_rank(v), -self._fee(v) if self._prefer_lower_fee else 0), reverse=True)
        if data_vid in ranked and self._health_rank(data_vid) >= self.health.cfg.get("red_threshold", 0.35):
            # Accept data venue if within bias threshold of top candidate
            top = ranked[0]
            if (self._health_rank(top) - self._health_rank(data_vid)) <= self._colocation_bias_diff:
                self._symbol_order_venue[symbol] = data_vid
                return data_vid
        for vid in ranked:
            if self._health_rank(vid) >= self.health.cfg.get("red_threshold", 0.35):
                self._symbol_order_venue[symbol] = vid
                return vid
        return None

    async def place_order(self, plan: Dict[str, Any], client_order_id: str) -> Dict[str, Any]:
        symbol = plan.get("symbol")
        primary = self.decide_order_venue(symbol, plan.get("side"))
        if not primary:
            raise RuntimeError("NO_HEALTHY_VENUE")
        adapter = self.venues[primary]
        try:
            started = time.perf_counter()
            ack = await adapter.place_order(plan, client_order_id)
            self.health.record_order_latency(primary, (time.perf_counter() - started)*1000)
            return {"venue": primary, "ack": ack}
        except Exception as e:
            logger.warning(f"[VenueRouter] primary venue {primary} failed {e}; trying failover")
            self.health.record_error(primary)
            # failover
            for vid in self._primary_order:
                if vid == primary:
                    continue
                if self._health_rank(vid) < self.health.cfg.get("red_threshold", 0.35):
                    continue
                try:
                    started2 = time.perf_counter()
                    ack2 = await self.venues[vid].place_order(plan, client_order_id)
                    self.health.record_order_latency(vid, (time.perf_counter() - started2)*1000)
                    self._symbol_order_venue[symbol] = vid
                    return {"venue": vid, "ack": ack2, "failover": True}
                except Exception as e2:  # pragma: no cover (multi failure rare)
                    self.health.record_error(vid)
                    logger.error(f"[VenueRouter] failover venue {vid} also failed {e2}")
            raise

    def snapshot(self):
        return {
            "health": self.health.snapshot(),
            "data_routes": dict(self._symbol_data_venue),
            "order_routes": dict(self._symbol_order_venue),
        }

    def all_order_venues_red(self) -> bool:
        thresh = self.health.cfg.get("red_threshold", 0.35)
        return all(self._health_rank(v) < thresh for v in self._primary_order)

    def all_data_venues_red(self) -> bool:
        thresh = self.health.cfg.get("red_threshold", 0.35)
        return all(self._health_rank(v) < thresh for v in self._data_order)

__all__ = ["VenueRouter"]
