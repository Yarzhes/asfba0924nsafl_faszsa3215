"""Hedge reporting utilities (Sprint 22).

Collects time-series of portfolio beta, hedge notional, hedge PnL (placeholder),
turnover and cost attribution for later comparison with a non-hedged path.

Intentionally minimal now; extend with real fills integration when execution
layer wiring is added.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable
import json
import math
from pathlib import Path
import csv


@dataclass
class HedgeSnapshot:
    ts: int
    beta_p: float
    hedge_notional: float
    action: str
    reason: str
    cost: float = 0.0
    pnl: float = 0.0  # placeholder until fills wired in


@dataclass
class HedgeReportCollector:
    history: List[HedgeSnapshot] = field(default_factory=list)

    def record(self, snap: HedgeSnapshot) -> None:
        self.history.append(snap)

    def summary(self) -> Dict[str, float]:
        if not self.history:
            return {}
        inside_band = sum(1 for s in self.history if s.reason == "IN_BAND")
        return {
            "points": len(self.history),
            "pct_in_band": inside_band / len(self.history),
            "hedge_actions": sum(1 for s in self.history if s.action in ("OPEN", "ADJUST", "CLOSE")),
        }

    # --------- NEW METRICS HELPERS ---------
    @staticmethod
    def _equity_metrics(equity_curve: Iterable[Dict[str, float]], timeframe: Optional[str] = None) -> Dict[str, float]:
        ec = list(equity_curve or [])
        if len(ec) < 2:
            return {"points": len(ec)}
        values = [float(x["equity"]) for x in ec]
        returns = []
        for i in range(1, len(values)):
            prev = values[i-1]
            if prev > 0:
                returns.append(values[i]/prev - 1.0)
        if not returns:
            return {"points": len(ec)}
        mean_r = float(sum(returns)/len(returns))
        std_r = float(math.sqrt(sum((r-mean_r)**2 for r in returns)/(len(returns)-1))) if len(returns) > 1 else 0.0
        # Determine periods per year from timeframe
        periods_per_year = float(len(returns))  # fallback
        if timeframe:
            try:
                tf = timeframe.lower()
                if tf.endswith('m'):
                    mins = int(tf[:-1])
                    periods_per_year = (365*24*60)/mins
                elif tf.endswith('h'):
                    hrs = int(tf[:-1])
                    periods_per_year = (365*24)/hrs
                elif tf.endswith('d'):
                    days = int(tf[:-1])
                    periods_per_year = 365/days
            except Exception:
                pass
        sharpe_annual = (mean_r/std_r*math.sqrt(periods_per_year)) if std_r > 0 else 0.0
        ret_annual = (1+mean_r)**periods_per_year - 1 if mean_r > -0.999 else 0.0
        # max drawdown
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (v/peak - 1.0)
            if dd < max_dd:
                max_dd = dd
        total_ret = values[-1]/values[0] - 1.0 if values[0] != 0 else 0.0
        return {
            "points": len(values),
            "total_return": total_ret,
            "sharpe_annual": sharpe_annual,
            "max_drawdown": max_dd,
            "vol": std_r,
            "avg_return_bar": mean_r,
            "return_annual": ret_annual,
        }

    @staticmethod
    def compare(hedged_curve: Iterable[Dict[str, float]], unhedged_curve: Iterable[Dict[str, float]], timeframe: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        m_hedged = HedgeReportCollector._equity_metrics(hedged_curve, timeframe=timeframe)
        m_un = HedgeReportCollector._equity_metrics(unhedged_curve, timeframe=timeframe)
        comp = {}
        for k in set(m_hedged.keys()) | set(m_un.keys()):
            if k in ("points",):
                continue
            hv = m_hedged.get(k)
            uv = m_un.get(k)
            if hv is not None and uv is not None:
                comp[k] = {"hedged": hv, "unhedged": uv, "delta": hv - uv}
        return {"hedged": m_hedged, "unhedged": m_un, "delta": comp}

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        data = {
            "snapshots": [s.__dict__ for s in self.history],
            "summary": self.summary(),
        }
        p.write_text(json.dumps(data, indent=2))

    def to_csv(self, path: str | Path) -> None:
        p = Path(path)
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "beta_p", "hedge_notional", "action", "reason", "cost", "pnl"])
            for s in self.history:
                w.writerow([s.ts, s.beta_p, s.hedge_notional, s.action, s.reason, s.cost, s.pnl])
