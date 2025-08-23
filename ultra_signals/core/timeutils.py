"""
Time-Related Utilities
----------------------

This module provides helper functions for time-based calculations, such as
determining proximity to funding events, which is critical for risk management
in perpetual futures trading.

Funding events often cause short-term volatility and unpredictable price
movements, so it's often prudent to avoid entering new positions immediately
before or after they occur.
"""

from datetime import datetime, timedelta, timezone
from typing import Tuple

FUNDING_HOURS_UTC = [0, 8, 16]

def next_prev_funding_ts(now_ts: int) -> Tuple[int, int]:
    """
    Calculates the timestamps of the next and previous funding events.

    Args:
        now_ts: The current timestamp in milliseconds.

    Returns:
        A tuple containing (previous_funding_ts_ms, next_funding_ts_ms).
    """
    dt_now = datetime.fromtimestamp(now_ts / 1000, tz=timezone.utc)
    
    all_funding_times = []
    for day_offset in [-1, 0, 1]:
        for hour in FUNDING_HOURS_UTC:
            dt = dt_now.replace(hour=hour, minute=0, second=0, microsecond=0)
            all_funding_times.append(dt + timedelta(days=day_offset))

    future_times = sorted([t for t in all_funding_times if t > dt_now])
    past_times = sorted([t for t in all_funding_times if t <= dt_now])

    next_funding_dt = future_times[0]
    prev_funding_dt = past_times[-1]

    return int(prev_funding_dt.timestamp() * 1000), int(next_funding_dt.timestamp() * 1000)


def within_avoid_window(now_ts: int, funding_ts: int, minutes: int) -> bool:
    """
    Checks if `now_ts` is within `minutes` of the `funding_ts`.
    """
    if minutes <= 0:
        return False
    
    delta_ms = abs(now_ts - funding_ts)
    window_ms = minutes * 60 * 1000
    
    return delta_ms <= window_ms


def is_funding_imminent(current_timestamp_ms: int, avoid_minutes: int) -> bool:
    """
    Checks if the current time is within a specified window around the NEAREST
    funding event (past or future). This provides a more robust check.
    """
    if avoid_minutes <= 0:
        return False

    prev_ts, next_ts = next_prev_funding_ts(current_timestamp_ms)

    in_next_window = within_avoid_window(current_timestamp_ms, next_ts, avoid_minutes)
    in_prev_window = within_avoid_window(current_timestamp_ms, prev_ts, avoid_minutes)

    return in_next_window or in_prev_window
def convert_to_timedelta(time_str: str) -> timedelta:
    """
    Converts a time string (e.g., "6m", "5d", "1y") into a timedelta object.

    :param time_str: The time string.
    :return: A timedelta object.
    """
    unit = time_str[-1].lower()
    value = int(time_str[:-1])

    if unit == 'd':
        return timedelta(days=value)
    elif unit == 'm':
        # dateutil.relativedelta is more robust for months
        from dateutil.relativedelta import relativedelta
        return relativedelta(months=value)
    elif unit == 'y':
        from dateutil.relativedelta import relativedelta
        return relativedelta(years=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'w':
        return timedelta(weeks=value)
    else:
        raise ValueError(f"Unsupported time unit in '{time_str}'. Use d, m, y, h, or w.")
