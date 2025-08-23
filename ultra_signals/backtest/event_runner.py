import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List
from ultra_signals.core.custom_types import EnsembleDecision, PortfolioState, RiskEvent
from ultra_signals.risk.portfolio import evaluate_portfolio

class EventRunner:
    """
    Orchestrates the backtest event loop, simulates trade execution,
    and manages the state of the portfolio.
    """

    def __init__(self, settings: Dict[str, Any], data_adapter, signal_engine, feature_store):
        self.settings = settings
        self.data_adapter = data_adapter
        self.signal_engine = signal_engine
        self.feature_store = feature_store
        self.trades = []
        self.equity_curve = []
        self.risk_events: List[RiskEvent] = []
        self.portfolio: PortfolioState = self._initialize_portfolio()
        self.log = logger
        self.warmup_mode = bool(self.settings.get("warmup_mode", False))

        # backtest TP/SL (optional)
        bs = self.settings.get("backtest", {})
        tp = bs.get("tp_pct", None)
        sl = bs.get("sl_pct", None)
        self.tp_pct = float(tp) if tp is not None else None
        self.sl_pct = float(sl) if sl is not None else None

    def _initialize_portfolio(self) -> PortfolioState:
        """Initializes the portfolio state."""
        initial_equity = self.settings.get("initial_capital", 10000.0)
        return PortfolioState(
            ts=0,
            equity=initial_equity,
            positions={},  # symbol -> Position
            exposure={
                "symbol": {},
                "cluster": {},
                "net": {"long": 0.0, "short": 0.0},
                "margin_used": 0.0,
            },
        )

    def run(self, symbol: str, timeframe: str):
        """Main event loop for a single symbol backtest."""
        self._executed_trades_total = 0
        logger.info(f"Starting event runner for {symbol} on {timeframe}.")

        start_date = self.settings.get("start_date")
        end_date = self.settings.get("end_date")

        # 1) Load historical data
        ohlcv = self.data_adapter.load_ohlcv(symbol, timeframe, start_date, end_date)
        if ohlcv is None or ohlcv.empty:
            logger.error("No data loaded, cannot run backtest.")
            return

        # 2) Iterate each bar
        for timestamp, bar in ohlcv.iterrows():
            self._process_bar(symbol, timeframe, timestamp, bar)

        logger.success("Event runner finished.")
        return self.trades, self.equity_curve

    def _process_bar(self, symbol: str, timeframe: str, timestamp: pd.Timestamp, bar: pd.Series):
        """Processes a single bar of data."""

        # 1) Push bar into FeatureStore
        bar_with_timestamp = bar.to_frame().T
        bar_with_timestamp["timestamp"] = timestamp
        self.feature_store.on_bar(symbol, timeframe, bar_with_timestamp)

        # 2) Mark-to-market equity
        self._update_equity_curve(bar)

        # 3) Exit checks for any open position
        closed_this_bar = False
        pos_at_open = self.portfolio.positions.get(symbol)
        if pos_at_open:
            should_close, exit_price, reason = self._explicit_exit_hit(pos_at_open, bar)
            if should_close:
                self._close_position(symbol, timestamp, exit_price, reason)
                closed_this_bar = True

        # If we closed a position this bar, don't immediately re-enter
        if closed_this_bar:
            return

        # If a position is still open, DO NOT ask the signal engine again
        # but DO run the portfolio gate with a placeholder decision to
        # capture/viz risk events (e.g., MAX_POSITIONS_TOTAL).
        if self.portfolio.positions.get(symbol):
            placeholder = EnsembleDecision(
                ts=int(pd.Timestamp(timestamp).timestamp()),
                symbol=symbol,
                tf=timeframe,
                decision=pos_at_open.side if pos_at_open else "LONG",  # any side; portfolio will gate
                confidence=0.5,
                subsignals=[],
                vote_detail={},
                vetoes=[]
            )
            allowed, _size_scale, events = evaluate_portfolio(placeholder, self.portfolio, self.settings)
            if events:
                self.risk_events.extend(events)
                self.log.info(f"Trade for {symbol} veto/notes: {[e.reason for e in events]}")
            return

        # 4) Always evaluate a fresh signal per bar (portfolio will decide), but only after warmup
        ohlcv_segment = self.feature_store.get_ohlcv(symbol, timeframe)
        if ohlcv_segment is None:
            return  # Not enough data yet

        # Respect warmup; only generate a signal after warmup_periods bars exist
        warmup_req = getattr(self.feature_store, "warmup_periods", 0) or 0
        if warmup_req and len(ohlcv_segment) < warmup_req:
            return

        decision = self.signal_engine.generate_signal(ohlcv_segment=ohlcv_segment, symbol=symbol)
        if not decision or decision.decision not in ("LONG", "SHORT"):
            return

        # 5) Portfolio gate (ALWAYS run; record any events)
        allowed, size_scale, events = evaluate_portfolio(decision, self.portfolio, self.settings)
        if events:
            self.risk_events.extend(events)
            for event in events:
                self.log.info(f"Portfolio gate for {symbol} at {timestamp}: {event.reason}")
        if not allowed:
            self.log.info(f"Trade for {symbol} at {timestamp} NOT allowed by portfolio gate.")
            return
        else:
            self.log.info(f"Trade for {symbol} at {timestamp} ALLOWED by portfolio gate.")

        # 6) Execute
        self._execute_trade(symbol, timestamp, bar, decision, size_scale)

    def _explicit_exit_hit(self, pos, bar) -> tuple[bool, float, str]:
        """
        Returns (should_close, exit_price, reason).
        Applies only when NOT in warmup mode and TP/SL configured.
        """
        if self.warmup_mode or (self.tp_pct is None and self.sl_pct is None):
            return False, 0.0, ""

        logger.debug(f"Evaluating exit for {pos.symbol}. Side: {pos.side}, Entry: {pos.entry:.2f}, Current Close: {bar.close:.2f}")
        if pos.side == "LONG":
            if self.tp_pct is not None:
                tp_price = pos.entry * (1 + self.tp_pct)
                logger.debug(f"  LONG TP: {bar.close:.2f} >= {tp_price:.2f} ({bar.close >= tp_price})")
                if bar.close >= tp_price:
                    return True, float(bar.close), "TP"
            if self.sl_pct is not None:
                sl_price = pos.entry * (1 - self.sl_pct)
                logger.debug(f"  LONG SL: {bar.close:.2f} <= {sl_price:.2f} ({bar.close <= sl_price})")
                if bar.close <= sl_price:
                    return True, float(bar.close), "SL"
        else:  # SHORT
            if self.tp_pct is not None:
                tp_price = pos.entry * (1 - self.tp_pct)
                logger.debug(f"  SHORT TP: {bar.close:.2f} <= {tp_price:.2f} ({bar.close <= tp_price})")
                if self.tp_pct is not None and bar.close <= tp_price:
                    return True, float(bar.close), "TP"
            if self.sl_pct is not None:
                sl_price = pos.entry * (1 + self.sl_pct)
                logger.debug(f"  SHORT SL: {bar.close:.2f} >= {sl_price:.2f} ({bar.close >= sl_price})")
                if self.sl_pct is not None and bar.close >= sl_price:
                    return True, float(bar.close), "SL"
        return False, 0.0, ""

    def _update_equity_curve(self, current_bar: pd.Series):
        current_value = self.portfolio.equity + self._get_open_pnl(current_bar)
        self.equity_curve.append({"timestamp": current_bar.name, "equity": current_value})

    def _mark_to_market(self, timestamp: Any, price: float):
        """DEPRECATED: Replaced by _update_equity_curve for now"""
        current_value = self.portfolio.equity + self._get_open_pnl_at_price(price)
        self.equity_curve.append({"timestamp": timestamp, "equity": current_value})

    def _get_open_pnl_at_price(self, price: float) -> float:
        """Calculates the unrealized PnL for all open positions at a given price."""
        open_pnl = 0.0
        for symbol, position in self.portfolio.positions.items():
            open_pnl += (price - position.entry) * position.size
        return open_pnl

    def _get_open_pnl(self, current_bar: pd.Series) -> float:
        """Calculates the unrealized PnL for all open positions."""
        open_pnl = 0.0
        for symbol, position in self.portfolio.positions.items():
            current_price = current_bar.get("close", position.entry)
            open_pnl += (current_price - position.entry) * position.size
        return open_pnl

    def _execute_trade(self, symbol: str, timestamp, bar: pd.Series, decision: EnsembleDecision, size_scale: float):
        """Executes a new trade based on a signal."""
        size_pct = self.settings.get("default_size_pct", 0.01)
        entry_price = bar.close
        base_size = (self.portfolio.equity * size_pct) / entry_price
        final_size = base_size * size_scale

        from ultra_signals.core.custom_types import Position
        position = Position(
            side=decision.decision,
            size=final_size,
            entry=entry_price,
            risk=0,        # Placeholder
            cluster="default"
        )

        trade = {
            "symbol": symbol,
            "side": decision.decision,
            "entry_time": timestamp,
            "entry_price": entry_price,
            "size": final_size,
            "exit_time": None,
            "exit_price": None,
            "pnl": None,
            "reason": "ENTRY",
            "raw_score": np.random.rand(),
            "confidence": decision.confidence,
            "vote_detail": decision.vote_detail,
            "vetoes": decision.vetoes,
        }
        self.trades.append(trade)
        self.portfolio.positions[symbol] = position

        risk = final_size * entry_price
        if position.side == "LONG":
            self.portfolio.exposure["net"]["long"] += risk
        else:
            self.portfolio.exposure["net"]["short"] += risk

        self._executed_trades_total += 1
        logger.info(f"Opened {decision.decision} position for {symbol} at {entry_price} with size {final_size:.4f}")
        return position

    def _close_position(self, symbol: str, timestamp, price: float, reason: str):
        """Closes an open position."""
        position = self.portfolio.positions.pop(symbol, None)
        if not position:
            logger.warning(f"Attempted to close non-existent position for {symbol}")
            return
        logger.info(f"Closing position for {symbol}. Reason: {reason}")

        pnl = (price - position.entry) * position.size
        self.portfolio.equity += pnl

        risk = position.size * position.entry
        if position.side == "LONG":
            self.portfolio.exposure["net"]["long"] -= risk
        else:
            self.portfolio.exposure["net"]["short"] -= risk

        # Update the trade record
        for trade in reversed(self.trades):
            if trade["symbol"] == symbol and trade["exit_time"] is None:
                trade["exit_time"] = timestamp
                trade["exit_price"] = price
                trade["pnl"] = pnl
                trade["reason"] = reason
                break

        logger.info(f"Closed {symbol} position at {price} for PnL: {pnl:.2f}. Reason: {reason}")


class MockSignalEngine:
    """A mock signal engine for testing the event runner."""
    def generate_signal(self, ohlcv_segment: pd.DataFrame, symbol: str) -> Optional[EnsembleDecision]:
        # Mocking an EnsembleDecision
        ts = int(ohlcv_segment.index[-1].timestamp())
        tf = "mock_tf"
        direction = "FLAT"
        if ohlcv_segment.iloc[-1]["close"] > ohlcv_segment.iloc[-1]["open"]:
            direction = "LONG"
        else:
            direction = "SHORT"

        return EnsembleDecision(
            ts=ts,
            symbol=symbol,
            tf=tf,
            decision=direction,
            confidence=0.75,
            subsignals=[],
            vote_detail={},
            vetoes=[]
        )
