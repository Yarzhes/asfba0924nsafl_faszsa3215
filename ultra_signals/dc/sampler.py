"""Simple percent-threshold Directional Change sampler.

This implements a per-symbol, per-theta streaming sampler which emits DC and
OS events. It's intentionally minimal to integrate with the project's
streaming data flows.
"""
from typing import List, Optional, Callable, Dict
from .events import EventType, DCEvent, OSEvent
import math
import time


class DirectionalChangeSampler:
    def __init__(
        self,
        start_price: float,
        theta_pct: Optional[float] = None,
        theta_atr_mult: Optional[float] = None,
        atr_provider: Optional[Callable[[], float]] = None,
        symbol: Optional[str] = None,
        hysteresis_frac: float = 0.25,
        min_os_range: float = 0.0,
    ):
        """Start a sampler for a symbol.

        - theta_pct: direct percent threshold (e.g., 0.001 == 0.1%)
        - theta_atr_mult: if set, multiply current ATR (from atr_provider) by this
        - atr_provider: function returning ATR for symbol (most-recent)
        - hysteresis_frac: extra fraction of theta to reduce flip-flop
        - min_os_range: minimum OS (fraction) to consider active (dead-zone)
        """
        self.start_price = start_price
        self.symbol = symbol
        self.theta_pct = theta_pct
        self.theta_atr_mult = theta_atr_mult
        self.atr_provider = atr_provider
        self.hysteresis_frac = hysteresis_frac
        self.min_os_range = min_os_range

        self.anchor = start_price
        self.state = "flat"  # 'flat', 'up', 'down'
        self.last_extreme = start_price
        self.next_event_id = 1
        self.events: List = []

        # Overshoot lifecycle
        self.os_active = False
        self.os_start_px: Optional[float] = None
        self.os_extreme: Optional[float] = None
        self.os_ticks = 0
        self.last_timestamp: Optional[float] = None

    def _compute_theta(self) -> float:
        if self.theta_atr_mult is not None and self.atr_provider is not None:
            atr = self.atr_provider()
            return (atr * self.theta_atr_mult) / self.anchor if self.anchor else 0.0
        return self.theta_pct or 0.0

    def on_price(self, px: float, tick_volume: int = 1, timestamp: Optional[float] = None) -> List:
        if timestamp is None:
            timestamp = time.time()
        emitted = []
        self.last_timestamp = timestamp

        theta = self._compute_theta()
        if theta <= 0:
            return emitted

        # apply hysteresis as extra margin
        hysteresis = theta * self.hysteresis_frac

        # helper thresholds
        up_thresh_anchor = self.anchor * (1 + theta)
        down_thresh_anchor = self.anchor * (1 - theta)

        # If flat, look for initial DC
        if self.state == "flat":
            if px >= up_thresh_anchor:
                ev = DCEvent(
                    event_id=self.next_event_id,
                    type=EventType.DC_UP,
                    anchor_px=self.anchor,
                    dc_px=px,
                    os_px_extreme=None,
                    os_range=None,
                    os_duration_ticks=0,
                    os_start_px=None,
                    os_end_px=None,
                    symbol=self.symbol,
                    tick_volume=tick_volume,
                    elapsed_wall_time_s=0.0,
                    theta=theta,
                    direction="up",
                )
                self.next_event_id += 1
                self.state = "up"
                self.last_extreme = px
                self.anchor = px
                # start OS after DC
                self.os_active = True
                self.os_start_px = px
                self.os_extreme = px
                self.os_ticks = 0
                emitted.append(ev)
            elif px <= down_thresh_anchor:
                ev = DCEvent(
                    event_id=self.next_event_id,
                    type=EventType.DC_DOWN,
                    anchor_px=self.anchor,
                    dc_px=px,
                    os_px_extreme=None,
                    os_range=None,
                    os_duration_ticks=0,
                    os_start_px=None,
                    os_end_px=None,
                    symbol=self.symbol,
                    tick_volume=tick_volume,
                    elapsed_wall_time_s=0.0,
                    theta=theta,
                    direction="down",
                )
                self.next_event_id += 1
                self.state = "down"
                self.last_extreme = px
                self.anchor = px
                self.os_active = True
                self.os_start_px = px
                self.os_extreme = px
                self.os_ticks = 0
                emitted.append(ev)
            return emitted

        # In-trend behavior
        if self.state == "up":
            # update extreme
            if px > self.last_extreme:
                self.last_extreme = px
            # threshold to flip down uses hysteresis: require a slightly larger move
            flip_thresh = self.last_extreme * (1 - (theta + hysteresis))
            if px <= flip_thresh:
                # finalize OS before flipping
                if self.os_active:
                    os_range = (self.os_extreme - self.os_start_px) / self.os_start_px if self.os_start_px else None
                    os_ev = OSEvent(
                        event_id=self.next_event_id,
                        type=EventType.OS,
                        anchor_px=self.os_start_px,
                        dc_px=None,
                        os_px_extreme=self.os_extreme,
                        os_range=os_range,
                        os_duration_ticks=self.os_ticks,
                        os_start_px=self.os_start_px,
                        os_end_px=self.os_extreme,
                        symbol=self.symbol,
                        tick_volume=None,
                        elapsed_wall_time_s=timestamp - (self.last_timestamp or timestamp),
                    )
                    self.next_event_id += 1
                    emitted.append(os_ev)
                # emit DC down
                ev = DCEvent(
                    event_id=self.next_event_id,
                    type=EventType.DC_DOWN,
                    anchor_px=self.last_extreme,
                    dc_px=px,
                    os_px_extreme=None,
                    os_range=None,
                    os_duration_ticks=0,
                    os_start_px=None,
                    os_end_px=None,
                    symbol=self.symbol,
                    tick_volume=tick_volume,
                    elapsed_wall_time_s=0.0,
                    theta=theta,
                    direction="down",
                )
                self.next_event_id += 1
                self.state = "down"
                self.anchor = px
                self.last_extreme = px
                # start new OS
                self.os_active = True
                self.os_start_px = px
                self.os_extreme = px
                self.os_ticks = 0
                emitted.append(ev)
            else:
                # still in up OS — update OS stats
                if px > self.os_extreme:
                    self.os_extreme = px
                self.os_ticks += 1
                # emit OS snapshot only if exceeds min_os_range
                cur_range = (self.os_extreme - self.os_start_px) / self.os_start_px if self.os_start_px else 0.0
                if cur_range >= self.min_os_range:
                    os_ev = OSEvent(
                        event_id=self.next_event_id,
                        type=EventType.OS,
                        anchor_px=self.os_start_px,
                        dc_px=None,
                        os_px_extreme=self.os_extreme,
                        os_range=cur_range,
                        os_duration_ticks=self.os_ticks,
                        os_start_px=self.os_start_px,
                        os_end_px=self.os_extreme,
                        symbol=self.symbol,
                        tick_volume=None,
                        elapsed_wall_time_s=timestamp - (self.last_timestamp or timestamp),
                    )
                    self.next_event_id += 1
                    emitted.append(os_ev)

        elif self.state == "down":
            if px < self.last_extreme:
                self.last_extreme = px
            flip_thresh = self.last_extreme * (1 + (theta + hysteresis))
            if px >= flip_thresh:
                if self.os_active:
                    os_range = (self.os_start_px - self.os_extreme) / self.os_start_px if self.os_start_px else None
                    os_ev = OSEvent(
                        event_id=self.next_event_id,
                        type=EventType.OS,
                        anchor_px=self.os_start_px,
                        dc_px=None,
                        os_px_extreme=self.os_extreme,
                        os_range=os_range,
                        os_duration_ticks=self.os_ticks,
                        os_start_px=self.os_start_px,
                        os_end_px=self.os_extreme,
                        symbol=self.symbol,
                        tick_volume=None,
                        elapsed_wall_time_s=timestamp - (self.last_timestamp or timestamp),
                    )
                    self.next_event_id += 1
                    emitted.append(os_ev)
                ev = DCEvent(
                    event_id=self.next_event_id,
                    type=EventType.DC_UP,
                    anchor_px=self.last_extreme,
                    dc_px=px,
                    os_px_extreme=None,
                    os_range=None,
                    os_duration_ticks=0,
                    os_start_px=None,
                    os_end_px=None,
                    symbol=self.symbol,
                    tick_volume=tick_volume,
                    elapsed_wall_time_s=0.0,
                    theta=theta,
                    direction="up",
                )
                self.next_event_id += 1
                self.state = "up"
                self.anchor = px
                self.last_extreme = px
                self.os_active = True
                self.os_start_px = px
                self.os_extreme = px
                self.os_ticks = 0
                emitted.append(ev)
            else:
                if px < self.os_extreme:
                    self.os_extreme = px
                self.os_ticks += 1
                cur_range = (self.os_start_px - self.os_extreme) / self.os_start_px if self.os_start_px else 0.0
                if cur_range >= self.min_os_range:
                    os_ev = OSEvent(
                        event_id=self.next_event_id,
                        type=EventType.OS,
                        anchor_px=self.os_start_px,
                        dc_px=None,
                        os_px_extreme=self.os_extreme,
                        os_range=cur_range,
                        os_duration_ticks=self.os_ticks,
                        os_start_px=self.os_start_px,
                        os_end_px=self.os_extreme,
                        symbol=self.symbol,
                        tick_volume=None,
                        elapsed_wall_time_s=timestamp - (self.last_timestamp or timestamp),
                    )
                    self.next_event_id += 1
                    emitted.append(os_ev)

        return emitted

