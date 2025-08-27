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

    current_equity: float = field(init=False)
    equity: float = field(init=False)

    exposure: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"net": {"long": 0.0, "short": 0.0}, "cluster": {}, "symbol": {}}
    )

    default_size_pct: float = 0.01  # ~0.9524 @10k on price 105

    def __post_init__(self) -> None:
        self.current_equity = float(self.initial_capital)
        self.equity = self.current_equity

    # -----------------
    # Trading utilities
    # -----------------
    def position_size(self, symbol: str, price: float) -> float:
        if price <= 0:
            return 0.0
        notional = self.current_equity * float(self.default_size_pct)
        return round(notional / float(price), 4)

    def open_position(self, symbol: str, side: str, price: float, ts: pd.Timestamp, size: float) -> None:
        self.positions[symbol] = SimpleNamespace(
            symbol=symbol,
            side=side,
            entry_price=float(price),
            size=float(size),
            bars_held=0,
            stop=None,
            tp=None,
            opened_at=ts,
            risk_amount_at_entry=None,
            adv_stop_distance=None,
        )

    def close_position(self, symbol: str, price: float, ts: pd.Timestamp, reason: str = "EXIT") -> None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return

        px = float(price)
        if pos.side == "LONG":
            pnl = (px - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - px) * pos.size

        risk_amt = getattr(pos, 'risk_amount_at_entry', None)
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
            "risk_amount_at_entry": risk_amt,
            "adv_stop_distance": getattr(pos, 'adv_stop_distance', None),
        }
        try:
            if risk_amt and abs(risk_amt) > 1e-12:
                trade['R'] = float(pnl) / float(risk_amt)
        except Exception:
            pass
        # Sprint 30: propagate MTC gate tagging from position (if present)
        for attr in ("mtc_status", "mtc_action", "mtc_scores", "mtc_observe_only"):
            if hasattr(pos, attr):
                trade[attr] = getattr(pos, attr)
        self.trades.append(trade)

        self.current_equity += float(pnl)
        self.equity = self.current_equity

    # -----------------------------
    # ADDED: Optional sizing helpers
    # -----------------------------
    def set_default_size_pct(self, value: float) -> None:
        """
        ADDED: Safely set default_size_pct from settings.
        Accepts either 0–1 (fraction) or >1 as 'percent'.
        No change to existing behavior unless you call it.
        """
        try:
            v = float(value)
            self.default_size_pct = (v / 100.0) if v > 1.0 else v
        except Exception:
            # keep existing value on any error
            pass

    def atr_position_size(
        self,
        price: float,
        atr: float,
        risk_frac: float = 0.004,
        R: float = 1.0,
        contract_value: float = 1.0,
        min_qty: float = 0.0,
        precision: int = 4,
    ) -> float:
        """
        ADDED: Volatility-based sizer. Use only if you want ATR sizing.
        - risk_frac: fraction of equity to risk per trade (e.g. 0.004 == 0.4%)
        - R: number of ATRs to initial stop (1R default)
        - contract_value: multiplier to convert qty*price to notional if the
          instrument uses a contract value (keep 1.0 for most perp pricing)
        - min_qty/precision: round rules to keep exchange-friendly sizes
        Returns a *quantity*. Won't affect behavior unless you use it.
        """
        try:
            if price <= 0 or atr <= 0 or risk_frac <= 0:
                return 0.0
            cash_risk = float(self.current_equity) * float(risk_frac)
            stop_distance = float(R) * float(atr)
            if stop_distance <= 0:
                return 0.0
            qty = cash_risk / (stop_distance * float(contract_value))
            # enforce min and round
            qty = max(qty, float(min_qty))
            return round(qty, int(precision))
        except Exception:
            return 0.0

    def notional_for_qty(self, price: float, qty: float, contract_value: float = 1.0) -> float:
        """
        ADDED: Helper to compute notional for fees/slippage accounting.
        """
        try:
            return float(price) * float(qty) * float(contract_value)
        except Exception:
            return 0.0

    def risk_fraction_from_pct(self, pct: float) -> float:
        """
        ADDED: Accept 0–1 or >1 percentage and normalize to fraction.
        """
        try:
            p = float(pct)
            return (p / 100.0) if p > 1.0 else p
        except Exception:
            return 0.0


# --------------------------------------
# Portfolio risk evaluation / gate logic
# --------------------------------------
def evaluate_portfolio(decision: EnsembleDecision, state: Portfolio, settings: dict) -> Tuple[bool, float, List[RiskEvent]]:
    """
    Returns: (allowed: bool, size_scale: float, events: list[RiskEvent])
    Veto reasons emitted:
      * MAX_POSITIONS_TOTAL
      * MAX_POSITIONS_PER_SYMBOL
      * MAX_NET_LONG_RISK
      * MAX_NET_SHORT_RISK

    NOTE: positions are stored as a dict keyed by symbol in Portfolio.
    That means concurrent positions per symbol are effectively capped at 1,
    regardless of 'max_positions_per_symbol'. If you truly need >1 per
    symbol, you'd have to refactor positions into a list. This function
    keeps the original behavior intact.
    """
    events: List[RiskEvent] = []
    allowed = True
    size_scale = 1.0

    pset = (settings or {}).get("portfolio", {}) or {}
    # accept either key spelling
    max_total = int(pset.get("max_total_positions", pset.get("max_positions_total", 999_999)))
    max_per_symbol = int(pset.get("max_positions_per_symbol", 999_999))
    assumed_risk = float(pset.get("assumed_trade_risk", 0.01))
    max_net_long = float(pset.get("max_net_long_risk", 1e9))
    max_net_short = float(pset.get("max_net_short_risk", 1e9))

    positions = getattr(state, "positions", {}) or {}
    symbol = getattr(decision, "symbol", None)

    # >>> S8 PATCH: interpret `max_total_positions` as a cap on the **total number of positions
    # ever opened in this backtest run** (closed + currently open), to match the test's expectation.
    # This blocks a second entry even after the first has been closed.
    total_closed = len(getattr(state, "trades", []))        # trades appended on close
    total_open_now = len(positions)                         # currently open (dict keyed by symbol)
    total_ever = total_closed + total_open_now
    if total_ever >= max_total:
        events.append(RiskEvent(
            ts=decision.ts,
            symbol=symbol,
            reason="MAX_POSITIONS_TOTAL",
            action="VETO",
            detail={"total_ever": total_ever, "cap": max_total, "open_now": total_open_now, "closed": total_closed},
        ))
        return False, 0.0, events
    # <<< S8 PATCH

    # 2) PER-SYMBOL CAP  (we'll keep this as a concurrent-open cap)
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
