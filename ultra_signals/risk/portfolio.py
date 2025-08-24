from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any

import pandas as pd

from ultra_signals.core.custom_types import EnsembleDecision, RiskEvent


# -----------------------------
# Minimal portfolio state class
# -----------------------------
@dataclass
class Portfolio:
    initial_capital: float
    max_positions_total: int
    max_positions_per_symbol: int

    trades: List[dict] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)
    positions: Dict[str, Any] = field(default_factory=dict)

    # Running equity
    current_equity: float = field(init=False)
    # Alias some tests look for (runner.portfolio.equity)
    equity: float = field(init=False)

    # Safe default exposure structure used by risk checks
    # exposure["net"]["long"] and ["net"]["short"] should always exist
    exposure: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"net": {"long": 0.0, "short": 0.0}, "cluster": {}, "symbol": {}}
    )

    # Default fractional sizing if runner/settings don’t provide one
    default_size_pct: float = 0.01  # 1% of equity per trade (matches tests expecting ~0.9524 on 10k @ 105)

    def __post_init__(self) -> None:
        self.current_equity = float(self.initial_capital)
        self.equity = self.current_equity  # keep alias in sync

    # -----------------
    # Trading utilities
    # -----------------
    def position_size(self, symbol: str, price: float) -> float:
        """
        Simple position sizing: use default_size_pct of current equity divided by price.
        """
        if price <= 0:
            return 0.0
        notional = self.current_equity * float(self.default_size_pct)
        return round(notional / float(price), 4)

    def open_position(self, symbol: str, side: str, price: float, ts: pd.Timestamp, size: float) -> None:
        """
        Record an open position; keep fields that runner/tests touch.
        """
        self.positions[symbol] = SimpleNamespace(
            symbol=symbol,
            side=side,
            entry_price=float(price),
            size=float(size),
            bars_held=0,
            stop=None,
            tp=None,
            opened_at=ts,
        )

    def close_position(self, symbol: str, price: float, ts: pd.Timestamp, reason: str = "EXIT") -> None:
        """
        Close position, compute PnL, store a trade record, update equity.
        """
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        px = float(price)
        if pos.side == "LONG":
            pnl = (px - pos.entry_price) * pos.size
        else:  # "SHORT"
            pnl = (pos.entry_price - px) * pos.size

        trade = {
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": px,
            "size": pos.size,
            "pnl": float(pnl),
            "exit_time": ts,
            "reason": reason,
            "bars_held": getattr(pos, "bars_held", 0),
        }
        self.trades.append(trade)

        # update equity and alias
        self.current_equity += float(pnl)
        self.equity = self.current_equity


# --------------------------------------
# Portfolio risk evaluation / gate logic
# --------------------------------------
def evaluate_portfolio(decision: EnsembleDecision, state: Portfolio, settings: dict) -> Tuple[bool, float, List[RiskEvent]]:
    """
    Evaluate a trade decision against portfolio constraints.

    Returns: (allowed: bool, size_scale: float, events: list[RiskEvent])
    Emitted veto reasons used by tests:
      * "MAX_POSITIONS_TOTAL"
      * "MAX_POSITIONS_PER_SYMBOL"
      * "MAX_NET_LONG_RISK"
      * "MAX_NET_SHORT_RISK"
    """
    events: List[RiskEvent] = []
    allowed = True
    size_scale = 1.0

    pset = (settings or {}).get("portfolio", {}) or {}
    max_total = int(pset.get("max_positions_total", 999_999))
    max_per_symbol = int(pset.get("max_positions_per_symbol", 999_999))
    assumed_risk = float(pset.get("assumed_trade_risk", 0.01))
    max_net_long = float(pset.get("max_net_long_risk", 1e9))
    max_net_short = float(pset.get("max_net_short_risk", 1e9))

    positions = getattr(state, "positions", {}) or {}
    symbol = getattr(decision, "symbol", None)

    # 1) TOTAL POSITIONS CAP
    total_open = len(positions)
    if total_open >= max_total:
        events.append(RiskEvent(
            ts=decision.ts,
            symbol=symbol,
            reason="MAX_POSITIONS_TOTAL",
            action="VETO",
            detail={"open_total": total_open, "cap": max_total},
        ))
        return False, 0.0, events

    # 2) PER-SYMBOL CAP
    if symbol is not None:
        open_for_symbol = sum(1 for s in positions.keys() if s == symbol)
        if open_for_symbol >= max_per_symbol:
            events.append(RiskEvent(
                ts=decision.ts,
                symbol=symbol,
                reason="MAX_POSITIONS_PER_SYMBOL",
                action="VETO",
                detail={"open_per_symbol": open_for_symbol, "cap": max_per_symbol},
            ))
            return False, 0.0, events

    # 3) NET EXPOSURE CAPS
    exposure = getattr(state, "exposure", {}) or {}
    net = exposure.get("net", {}) if isinstance(exposure, dict) else {}
    curr_long = float(net.get("long", 0.0))
    curr_short = float(net.get("short", 0.0))

    if decision.decision == "LONG":
        projected = curr_long + assumed_risk
        if projected > max_net_long:
            events.append(RiskEvent(
                ts=decision.ts,
                symbol=symbol,
                reason="MAX_NET_LONG_RISK",
                action="VETO",
                detail={"current": curr_long, "assumed_add": assumed_risk, "cap": max_net_long},
            ))
            return False, 0.0, events

    elif decision.decision == "SHORT":
        projected = curr_short + assumed_risk
        if projected > max_net_short:
            events.append(RiskEvent(
                ts=decision.ts,
                symbol=symbol,
                reason="MAX_NET_SHORT_RISK",
                action="VETO",
                detail={"current": curr_short, "assumed_add": assumed_risk, "cap": max_net_short},
            ))
            return False, 0.0, events

    # Passed all checks
    return allowed, size_scale, events
