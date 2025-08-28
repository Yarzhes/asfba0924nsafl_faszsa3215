"""Simple divergence detector between topic sentiment and derivatives posture.

This module implements prototypical divergence rules described in Sprint 58.
It is intentionally small and rule-based so it can be used without heavy ML
dependencies. Rules can be expanded and made configurable later.
"""
from typing import Dict, Any


class DivergenceDetector:
    def __init__(self, funding_threshold: float = 0.0003, oi_threshold_pct: float = 0.02):
        # funding_threshold: absolute funding rate magnitude considered "strong"
        self.funding_threshold = funding_threshold
        self.oi_threshold_pct = oi_threshold_pct

    def detect(self, symbol: str, topic_scores: Dict[str, Dict[str, float]],
               funding: Dict[str, Any], oi: Dict[str, Any]) -> Dict[str, Any]:
        """Return divergence metrics and flags for a single symbol.

        funding is expected to contain funding_now (float, e.g., 0.0004 positive means longs pay)
        oi is expected to contain oi_rate or oi_change_pct (float, e.g., 0.03 = 3% build)
        topic_scores: {topic: {score_s, score_m, z, pctl}}
        """
        # derive an aggregated sentiment: mean over topic score_s weighted by importance
        if not topic_scores:
            return {}
        # simple aggregation: average of topic short scores
        sent_vals = [v.get("score_s", 0.0) for v in topic_scores.values()]
        avg_sent = sum(sent_vals) / len(sent_vals)

        funding_now = funding.get("funding_now") if funding else None
        oi_rate = oi.get("oi_rate") if oi else oi.get("oi_change_pct") if oi else None

        res = {
            "avg_sent": avg_sent,
            "funding_now": funding_now,
            "oi_rate": oi_rate,
            "sent_vs_funding_div_long": 0.0,
            "sent_vs_funding_div_short": 0.0,
            "contrarian_flag_long": 0,
            "contrarian_flag_short": 0,
            "reason_codes": [],
        }

        # rule: bullish social & negative funding => contrarian long candidate
        if avg_sent > 0.3 and funding_now is not None and funding_now < -self.funding_threshold:
            res["sent_vs_funding_div_long"] = min(1.0, (avg_sent - 0.3) * abs(funding_now) / self.funding_threshold)
            res["contrarian_flag_long"] = 1
            res["reason_codes"].append("BULL_SOC_NEG_FUNDING")

        # rule: bearish social & positive funding => contrarian short candidate
        if avg_sent < -0.3 and funding_now is not None and funding_now > self.funding_threshold:
            res["sent_vs_funding_div_short"] = min(1.0, (abs(avg_sent) - 0.3) * funding_now / self.funding_threshold)
            res["contrarian_flag_short"] = 1
            res["reason_codes"].append("BEAR_SOC_POS_FUNDING")

        # euphoria spike + OI build-up => euphoria_flag (veto/size reduction)
        if any(v.get("pctl", 0) > 90 for v in topic_scores.values()):
            if oi_rate is not None and oi_rate > self.oi_threshold_pct:
                res["reason_codes"].append("EUPHORIA_OI_BUILDUP")
                res["contrarian_flag_long"] = 1

        return res
