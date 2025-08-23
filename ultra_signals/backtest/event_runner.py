import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List
from ultra_signals.core.custom_types import EnsembleDecision, RiskEvent, Position
from ultra_signals.risk.portfolio import evaluate_portfolio, Portfolio

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
        self.risk_events: List[RiskEvent] = []
        self.log = logger
        self.warmup_mode = bool(self.settings.get("warmup_mode", False))

        # Initialize the new Portfolio class
        initial_capital = self.settings["backtest"]["execution"].get("initial_capital", 10000.0)
        portfolio_settings = self.settings.get("portfolio", {})
        max_total_positions = portfolio_settings.get("max_total_positions", 999999)
        max_positions_per_symbol = portfolio_settings.get("max_positions_per_symbol", 999999)

        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_positions_total=max_total_positions,
            max_positions_per_symbol=max_positions_per_symbol
        )

    def run(self, symbol: str, timeframe: str):
        """Main event loop for a single symbol backtest."""
        logger.info(f"Starting event runner for {symbol} on {timeframe}.")

        start_date = self.settings["backtest"].get("start_date")
        end_date = self.settings["backtest"].get("end_date")

        # 1) Load historical data
        ohlcv = self.data_adapter.load_ohlcv(symbol, timeframe, start_date, end_date)
        if ohlcv is None or ohlcv.empty:
            logger.error("No data loaded, cannot run backtest.")
            return self.portfolio.trades, self.portfolio.equity_curve

        # 2) Iterate each bar
        for timestamp, bar in ohlcv.iterrows():
            self._process_bar(symbol, timeframe, timestamp, bar)

        logger.success("Event runner finished.")

        # Force-close all open positions at end of backtest
        if not ohlcv.empty:
            last_close = ohlcv.iloc[-1]["close"]
            last_ts    = ohlcv.index[-1]
            for symbol, pos in list(self.portfolio.positions.items()): # Iterate over a copy
                self.portfolio.close_position(symbol, last_close, last_ts, "EOD")
                self.log.info(f"Force-closing {symbol} open position at EOD.")

        return self.portfolio.trades, self.portfolio.equity_curve

    def _process_bar(self, symbol: str, timeframe: str, timestamp: pd.Timestamp, bar: pd.Series):
        """Processes a single bar of data."""

        # 1) Push bar into FeatureStore
        bar_with_timestamp = bar.to_frame().T
        bar_with_timestamp["timestamp"] = timestamp
        self.feature_store.on_bar(symbol, timeframe, bar_with_timestamp)

        # 2) Mark-to-market equity
        self.portfolio.equity_curve.append({"timestamp": timestamp, "equity": self.portfolio.current_equity})

        # 3) Exit checks for any open position
        pos = self.portfolio.get(symbol)
        close_px = bar["close"]
        ts = timestamp

        # 3) Exit checks for any open position
        if pos:
            # Increment bars_held
            pos.bars_held += 1
            features = self.feature_store.get_features(symbol, timestamp)
            reason = self.signal_engine.should_exit(symbol, pos, bar, features)
            self.log.debug(f"DEBUG ExitCheck: ts={ts}, side={pos.side}, price={close_px}, stop={getattr(pos, 'atr_mult_stop', 'N/A')}, tp={getattr(pos, 'atr_mult_tp', 'N/A')}, bars_held={pos.bars_held}, reason={reason}")
            if reason:
                self.portfolio.close_position(symbol, close_px, timestamp, reason)
                self.log.info(f"INFO Closed {symbol} pnl={self.portfolio.trades[-1]['pnl']:.2f} hold_bars={self.portfolio.trades[-1]['hold_bars']}")
                pos = None  # closed; fall through and allow a new open below

        # 4) Always evaluate a fresh signal per bar (portfolio will decide), but only after warmup
        ohlcv_segment = self.feature_store.get_ohlcv(symbol, timeframe)
        if ohlcv_segment is None:
            return  # Not enough data yet

        # Respect warmup; only generate a signal after warmup_periods bars exist
        warmup_req = getattr(self.feature_store, "warmup_periods", 0) or 0
        if warmup_req and len(ohlcv_segment) < warmup_req:
            return

        decision = self.signal_engine.generate_signal(ohlcv_segment=ohlcv_segment, symbol=symbol)
        action = decision.decision if decision else "FLAT"

        # Evaluate open/flip after exits
        if pos and ((pos.side == "LONG" and action == "SHORT") or
                    (pos.side == "SHORT" and action == "LONG")):
            self.log.info(f"INFO Flip: {pos.side}->{action} @ {close_px}, {ts}")
            self.portfolio.close_position(symbol, close_px, timestamp, "FLIP")
            self.log.info(f"INFO Closed {symbol} pnl={self.portfolio.trades[-1]['pnl']:.2f} hold_bars={self.portfolio.trades[-1]['hold_bars']}")
            pos = None

        if not pos and action in {"LONG", "SHORT"}:
            # 5) Portfolio gate (ALWAYS run; record any events)
            # The evaluate_portfolio function now returns events even if allowed
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
            size = self.portfolio.position_size(symbol, close_px) * size_scale # Apply size_scale
            self.portfolio.open_position(symbol, action, close_px, timestamp, size)
            self.log.info(f"INFO Opened {action} position for {symbol} at {close_px} with size {size:.4f}")


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

    def should_exit(self, symbol, pos, bar, features):
        # Mock should_exit for testing purposes
        return None
