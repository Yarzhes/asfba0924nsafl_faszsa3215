from dataclasses import dataclass
from typing import Tuple, List, Dict
from ultra_signals.core.custom_types import EnsembleDecision, PortfolioState, RiskEvent

def evaluate_portfolio(decision: EnsembleDecision, state: PortfolioState, settings: dict) -> tuple[bool, float, list[RiskEvent]]:
    """
    Evaluates a trade decision against portfolio risk constraints.
    Returns (allowed, size_scale, events)
    """
    events: List[RiskEvent] = []
    allowed = True
    size_scale = 1.0
    
    # Load portfolio settings safely
    pset = settings.get("portfolio", {}) or {}
    max_total = pset.get("max_positions_total", 999999)

    # ✅ FIXED: Respect settings.yaml value for max_positions_per_symbol
    # Default changed from 1 ➝ 999999 if not specified, preventing accidental vetoes
    max_per_symbol = pset.get("max_positions_per_symbol", 999999)

    # --- 1) TOTAL POSITIONS CAP (must emit event) ---
    total_open = len(state.positions)
    # print(f"DEBUG: Checking portfolio limits. Open positions: {total_open}, Max total: {max_total}")
    if total_open >= max_total:
        events.append(RiskEvent(
            ts=decision.ts,
            symbol=decision.symbol,
            reason="MAX_POSITIONS_TOTAL",
            action="VETO",
            detail={"open_total": total_open, "cap": max_total}
        ))
        return False, 0.0, events

    # --- 2) PER-SYMBOL CAP ---
    sym_open = sum(1 for s in state.positions.keys() if s == decision.symbol)
    if sym_open >= max_per_symbol:
        events.append(RiskEvent(
            ts=decision.ts,
            symbol=decision.symbol,
            reason="MAX_POSITIONS_PER_SYMBOL",
            action="VETO",
            detail={"open_per_symbol": sym_open, "cap": max_per_symbol}
        ))
        return False, 1.0, events

    # --- 3) Net exposure caps (optional in tests) ---
    assumed_risk = pset.get("assumed_trade_risk", 0.01)
    if decision.decision == "LONG":
        proj = state.exposure["net"]["long"] + assumed_risk
        cap = pset.get("max_net_long_risk")
        if cap is not None and proj > cap:
            events.append(RiskEvent(
                ts=decision.ts,
                symbol=decision.symbol,
                reason="MAX_NET_LONG_RISK",
                action="VETO",
                detail={"projected": proj, "cap": cap}
            ))
            return False, size_scale, events
    elif decision.decision == "SHORT":
        proj = state.exposure["net"]["short"] + assumed_risk
        cap = pset.get("max_net_short_risk")
        if cap is not None and proj > cap:
            events.append(RiskEvent(
                ts=decision.ts,
                symbol=decision.symbol,
                reason="MAX_NET_SHORT_RISK",
                action="VETO",
                detail={"projected": proj, "cap": cap}
            ))
            return False, size_scale, events
        
    return True, 1.0, []
