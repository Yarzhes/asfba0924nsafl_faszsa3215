from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time


class EventType(Enum):
    DC_UP = "DC_UP"
    DC_DOWN = "DC_DOWN"
    OS = "OS"


@dataclass
class BaseEvent:
    event_id: int
    type: EventType
    anchor_px: float
    dc_px: Optional[float]
    # Overshoot lifecycle
    os_px_extreme: Optional[float]
    os_range: Optional[float]
    os_duration_ticks: Optional[int]
    os_start_px: Optional[float]
    os_end_px: Optional[float]
    # generic metadata
    symbol: Optional[str]
    tick_volume: Optional[int]
    elapsed_wall_time_s: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class DCEvent(BaseEvent):
    theta: float = 0.0
    direction: str = "up"  # 'up' or 'down'
    # whether this DC was suppressed by dead-zone rules
    suppressed: bool = False


@dataclass
class OSEvent(BaseEvent):
    # OSEvent will have start/end info set when OS closes
    pass
