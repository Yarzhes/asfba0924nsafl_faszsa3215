from typing import List, Dict, Any


class Fusion:
    """Combine multiple DC streams into simple signals like turn probability.

    This is an initial, pluggable fusion step — later we can add weights,
    hysteresis, and symbol-aware scaling.
    """

    def __init__(self, thetas: List[float]):
        self.thetas = sorted(thetas)

    def turn_probability(self, recent_events_by_theta: Dict[float, List[Dict[str, Any]]]) -> Dict[str, Any]:
        # Count DC events across scales in the last window
        agree_ct = 0
        total = 0
        for theta, evs in recent_events_by_theta.items():
            total += 1
            # if last event for theta is DC (not OS) then it's a vote
            if evs and evs[-1].get("type") in ("DC_UP", "DC_DOWN"):
                agree_ct += 1
        prob = agree_ct / total if total else 0.0
        return {"agree_ct": agree_ct, "total": total, "turn_prob": prob}
