from typing import List, Dict, Any
from .sampler import DirectionalChangeSampler
from .feature_view import FeatureView


def replay_prices(prices: List[float], thetas: List[float]) -> Dict[float, List[Dict[str, Any]]]:
    """Replay a price series through one sampler per theta and collect events."""
    outputs = {}
    for th in thetas:
        s = DirectionalChangeSampler(start_price=prices[0], theta_pct=th, symbol=f"TH={th}")
        evs = []
        for px in prices:
            out = s.on_price(px)
            for e in out:
                # convert dataclass to dict-ish
                evs.append({k: getattr(e, k, None) for k in e.__dict__.keys()})
        outputs[th] = evs
    return outputs


def simple_ab_compare(prices: List[float], thetas: List[float]) -> Dict[str, Any]:
    """Produce a small A/B comparator: DC vs simple time-based returns.

    Returns counts and mean OS ranges per theta and a time-bar baseline of
    fixed-interval returns (e.g., every N samples).
    """
    outputs = replay_prices(prices, thetas)
    fv = FeatureView()
    res = {}
    for th, evs in outputs.items():
        res[th] = {
            "event_count": len(evs),
            "mean_os": (sum([e.get("os_range") or 0 for e in evs]) / len(evs)) if evs else None,
        }

    # simple time-bar baseline: count returns crossing theta as naive events
    baseline = {"time_bar_events": 0}
    last_px = prices[0]
    for px in prices[1:]:
        ret = (px - last_px) / last_px
        if abs(ret) > max(thetas):
            baseline["time_bar_events"] += 1
        last_px = px

    return {"dc": res, "baseline": baseline}
