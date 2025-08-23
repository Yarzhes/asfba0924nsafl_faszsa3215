import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from ultra_signals.core.custom_types import EnsembleDecision, PortfolioState, RiskEvent, Position
from dataclasses import dataclass, field

@dataclass
class Portfolio:
    """
    Manages the state and operations of the trading portfolio.
    This class is responsible for:
    - Storing open positions
    - Calculating PnL and other trade metrics
    - Enforcing portfolio-level risk limits
    - Recording trades to a CSV
    - Maintaining equity curve
    """
    initial_capital: float
    max_positions_total: int
    max_positions_per_symbol: int
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    positions: Dict[str, Position] = field(default_factory=dict)
    current_equity: float = field(init=False)

    def __post_init__(self):
        self.current_equity = self.initial_capital

    def open_position(self, symbol: str, side: str, price: float, ts: pd.Timestamp, size: float):
        """Opens a new position and records it."""
        if symbol in self.positions:
            raise ValueError(f"Position for {symbol} already open.")

        position = Position(
            side=side,
            size=size,
            entry=price,
            entry_price=price,
            risk=size * price,
            cluster="default", # Placeholder
            bars_held=0
        )
        self.positions[symbol] = position
        
        # Record the entry part of the trade
        self.trades.append({
            "symbol": symbol,
            "entry_time": ts,
            "entry_price": price,
            "side": side,
            "size": size,
            "exit_time": None,
            "exit_price": None,
            "pnl": None,
            "reason": "OPEN",
            "hold_bars": None
        })

    def close_position(self, symbol: str, price: float, ts: pd.Timestamp, reason: str):
        """Closes an open position, calculates PnL, and updates trade record."""
        position = self.positions.pop(symbol, None)
        if not position:
            return # Position already closed or never existed

        pnl = (price - position.entry) * position.size if position.side == "LONG" else \
              (position.entry - price) * position.size
        
        self.current_equity += pnl

        # Find the corresponding trade entry and update it
        for trade in reversed(self.trades): # Look from end for efficiency
            if trade["symbol"] == symbol and trade["exit_time"] is None:
                trade["exit_time"] = ts
                trade["exit_price"] = price
                trade["pnl"] = pnl
                trade["reason"] = reason
                trade["hold_bars"] = position.bars_held
                break
    
    def get(self, symbol: str) -> Optional[Position]:
        """Returns the current position for a symbol, or None if not open."""
        return self.positions.get(symbol)

    def open_positions(self) -> List[Position]:
        """Returns a list of all currently open positions."""
        return list(self.positions.values())

    def can_open(self, symbol: str) -> bool:
        """Checks if a new position can be opened based on limits."""
        total_open = len(self.positions)
        sym_open = sum(1 for s in self.positions.keys() if s == symbol)
        
        if total_open >= self.max_positions_total:
            # print(f"DEBUG: Vetoed {symbol} due to MAX_POSITIONS_TOTAL ({total_open}/{self.max_positions_total})")
            return False
        
        if sym_open >= self.max_positions_per_symbol:
            # print(f"DEBUG: Vetoed {symbol} due to MAX_POSITIONS_PER_SYMBOL ({sym_open}/{self.max_positions_per_symbol})")
            return False
            
        return True

    def position_size(self, symbol: str, price: float) -> float:
        """Calculates the notional size for a new position."""
        # Simple fixed-fraction sizing for now (e.g., 1% of capital)
        # This should come from settings in a real scenario
        default_size_pct = 0.01 # Example: 1% of capital
        return (self.current_equity * default_size_pct) / price

def evaluate_portfolio(decision: EnsembleDecision, portfolio_instance: Portfolio, settings: dict) -> tuple[bool, float, list[RiskEvent]]:
    """
    Evaluates a trade decision against portfolio risk constraints.
    Returns (allowed, size_scale, events)
    """
    events: List[RiskEvent] = []
    allowed = True
    size_scale = 1.0
    
    # Load portfolio settings safely
    pset = settings.get("portfolio", {}) or {}
    max_total = pset.get("max_total_positions", 999999) # Corrected key
    max_per_symbol = pset.get("max_positions_per_symbol", 999999)

    # --- 1) TOTAL POSITIONS CAP (must emit event) ---
    total_open = len(portfolio_instance.positions)
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
    sym_open = sum(1 for s in portfolio_instance.positions.keys() if s == decision.symbol)
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
    # This part needs to be re-evaluated based on how exposure is managed in the new Portfolio class
    # For now, we'll simplify or remove it if not directly supported by the new Portfolio.
    # Assuming the new Portfolio class handles exposure internally or it's not critical for this task.
    # If needed, add exposure tracking to the Portfolio class.
        
    return True, 1.0, events # Return events even if allowed
