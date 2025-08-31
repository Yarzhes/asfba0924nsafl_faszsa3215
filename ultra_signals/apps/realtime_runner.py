"""
Real-time Trading Signal Runner

This is the main entry point for the Ultra-Signals application. It orchestrates
the flow of data and logic from all the different components:

1.  Loads settings from a YAML file.
2.  Initializes the Binance WebSocket client to stream market data.
3.  Initializes the FeatureStore to hold time-series data.
4.  Enters a main loop that:
    - Listens for kline events from the WebSocket.
    - On each **closed** kline, it updates the FeatureStore.
    - Triggers the feature computation pipeline.
    - Runs the scoring engine to generate a potential signal.
    - Applies risk filters.
    - If a valid signal is produced, it sends a notification via Telegram.
5.  Handles graceful shutdown on keyboard interrupt (Ctrl+C).
"""
import argparse
import os
import asyncio
import time
from typing import Dict, Optional, Set
from dataclasses import dataclass

from loguru import logger
from collections import defaultdict

from ultra_signals.core.config import load_settings, ConfigError
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.events import KlineEvent
from ultra_signals.core.custom_types import FeatureVector
from ultra_signals.data.binance_ws import BinanceWSClient
from ultra_signals.data.funding_provider import FundingProvider
from ultra_signals.features import (
    compute_momentum,
    compute_trend,
    compute_volatility,
    compute_volume_flow_features,
    compute_orderbook_features,
    compute_derivatives_features,
    compute_funding_features,
    compute_orderbook_features_v2,
    compute_cvd_features,
    BookFlipState,
    CvdState,
)
from ultra_signals.engine.scoring import component_scores
from ultra_signals.engine.entries_exits import make_signal
from ultra_signals.engine.risk_filters import apply_filters
import ultra_signals.engine.sizing as sizing_module
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.transport.telegram import format_message, send_message
from ultra_signals.live.metrics import Metrics


@dataclass
class SymbolState:
    """Per-symbol state tracking for isolation and cooldown."""
    last_signal_ts: float = 0.0
    last_signal_side: str = "FLAT"
    last_confidence: float = 0.0
    cooldown_until: float = 0.0
    consecutive_signals: int = 0
    last_entry_price: Optional[float] = None


class ResilientSignalRunner:
    """Resilient signal runner with automatic recovery and per-symbol isolation."""
    
    def __init__(self, settings):
        self.settings = settings
        self.symbol_states: Dict[str, SymbolState] = defaultdict(SymbolState)
        self.shutdown_requested = False
        self.max_retries = 5
        self.base_backoff = 2.0
        self.max_backoff = 60.0
        self.heartbeat_interval = 30.0
        self.last_heartbeat = time.time()
        
        # Initialize components
        self.ws_client = BinanceWSClient(settings)
        self.funding_provider = FundingProvider(settings.funding_rate_provider.model_dump())
        self.feature_store = FeatureStore(
            warmup_periods=settings.features.warmup_periods,
            funding_provider=self.funding_provider,
        )
        self.metrics = Metrics()
        
        # Convert settings for RealSignalEngine
        settings_dict = settings.model_dump() if hasattr(settings, 'model_dump') else settings
        self.real_engine = RealSignalEngine(settings_dict, self.feature_store)
        
        # State management
        self.book_flip_states = defaultdict(BookFlipState)
        self.cvd_states = defaultdict(CvdState)
        
        # Track ready timeframes per symbol
        self.ready_timeframes: Dict[str, Set[str]] = defaultdict(set)
        
    async def start(self):
        """Start the resilient signal runner with automatic recovery."""
        logger.info("Starting resilient signal runner...")
        
        # Subscribe to streams
        stream_types = self.settings.runtime.timeframes + ["depth", "aggTrade"]
        self.ws_client.subscribe(self.settings.runtime.symbols, stream_types)
        
        # Start background tasks
        funding_task = asyncio.create_task(self.funding_provider.start())
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        try:
            await self._run_main_loop(funding_task)
        except asyncio.CancelledError:
            logger.info("Runner cancelled by user or system")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        finally:
            # Cleanup
            funding_task.cancel()
            heartbeat_task.cancel()
            await self.ws_client.stop()
            logger.info("Signal runner shutdown complete.")
    
    async def _run_main_loop(self, funding_task):
        """Main processing loop with resilient error handling."""
        retry_count = 0
        
        while not self.shutdown_requested:
            try:
                async for event in self.ws_client.start():
                    if self.shutdown_requested:
                        break
                        
                    await self._process_event(event)
                    retry_count = 0  # Reset on successful processing
                    
            except asyncio.CancelledError:
                logger.info("WebSocket connection cancelled - attempting reconnection")
                if self.shutdown_requested:
                    raise
                await self._handle_reconnection(retry_count)
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                if self.shutdown_requested:
                    break
                await self._handle_reconnection(retry_count)
                retry_count += 1
    
    async def _handle_reconnection(self, retry_count: int):
        """Handle reconnection with exponential backoff."""
        if retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded. Shutting down.")
            self.shutdown_requested = True
            return
            
        backoff_time = min(self.base_backoff * (2 ** retry_count), self.max_backoff)
        logger.warning(f"Reconnecting in {backoff_time:.1f}s (attempt {retry_count + 1}/{self.max_retries})")
        await asyncio.sleep(backoff_time)
    
    async def _heartbeat_monitor(self):
        """Monitor system health and detect stale connections."""
        while not self.shutdown_requested:
            try:
                current_time = time.time()
                if current_time - self.last_heartbeat > self.heartbeat_interval * 2:
                    logger.warning("Heartbeat timeout detected - forcing reconnection")
                    # Force reconnection by cancelling current WebSocket task
                    # This will be handled by the main loop
                    
                self.last_heartbeat = current_time
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5.0)
    
    async def _process_event(self, event):
        """Process a single market event with per-symbol isolation."""
        if event is None or not hasattr(event, 'event_type') or event.event_type != 'kline':
            return
            
        kline = event
        if kline.timeframe not in self.settings.runtime.timeframes:
            return
            
        # Update heartbeat
        self.last_heartbeat = time.time()
        
        # Ingest event into feature store
        self.feature_store.ingest_event(event)
        
        # Check warmup status for this symbol/timeframe
        if not self._is_timeframe_ready(kline.symbol, kline.timeframe):
            return
            
        # Process signal generation with per-symbol isolation
        await self._process_signal_generation(kline)
    
    def _is_timeframe_ready(self, symbol: str, timeframe: str) -> bool:
        """Check if a timeframe has sufficient data for reliable calculations."""
        try:
            ohlcv = self.feature_store.get_ohlcv(symbol, timeframe)
            if ohlcv is None or len(ohlcv) < self.settings.features.warmup_periods:
                return False
                
            # Mark as ready if we haven't already
            if timeframe not in self.ready_timeframes[symbol]:
                self.ready_timeframes[symbol].add(timeframe)
                logger.info(f"Timeframe {symbol}/{timeframe} now ready with {len(ohlcv)} bars")
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking timeframe readiness for {symbol}/{timeframe}: {e}")
            return False
    
    async def _process_signal_generation(self, kline: KlineEvent):
        """Generate signals with per-symbol isolation and cooldown."""
        symbol = kline.symbol
        current_time = time.time()
        
        # Check cooldown
        state = self.symbol_states[symbol]
        if current_time < state.cooldown_until:
            return
            
        try:
            # Generate signal
            ensemble_decision = self.real_engine.generate_signal(
                ohlcv_segment=self.feature_store.get_ohlcv(symbol, kline.timeframe),
                symbol=symbol
            )
            
            if not ensemble_decision or ensemble_decision.decision == "FLAT":
                return
                
            # Apply per-symbol cooldown and isolation logic
            if not self._should_send_signal(symbol, ensemble_decision, current_time):
                return
                
            # Update symbol state
            self._update_symbol_state(symbol, ensemble_decision, current_time)
            
            # Send notification
            await self._send_notification(ensemble_decision, kline)
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    def _should_send_signal(self, symbol: str, decision, current_time: float) -> bool:
        """Check if signal should be sent based on cooldown and isolation rules."""
        state = self.symbol_states[symbol]
        
        # Minimum time between signals (configurable)
        min_interval = self.settings.runtime.get('min_signal_interval_sec', 60.0)
        if current_time - state.last_signal_ts < min_interval:
            return False
            
        # Cooldown after consecutive signals
        if state.consecutive_signals >= 3:
            cooldown_duration = min(300.0, 60.0 * (2 ** (state.consecutive_signals - 3)))
            if current_time < state.cooldown_until:
                return False
                
        # Confidence threshold
        if decision.confidence < self.settings.runtime.min_confidence:
            return False
            
        return True
    
    def _update_symbol_state(self, symbol: str, decision, current_time: float):
        """Update per-symbol state tracking."""
        state = self.symbol_states[symbol]
        
        # Check if this is a new signal or continuation
        if (decision.decision != state.last_signal_side or 
            current_time - state.last_signal_ts > 300.0):  # 5 min gap
            state.consecutive_signals = 1
        else:
            state.consecutive_signals += 1
            
        state.last_signal_ts = current_time
        state.last_signal_side = decision.decision
        state.last_confidence = decision.confidence
        
        # Set cooldown if needed
        if state.consecutive_signals >= 3:
            cooldown_duration = min(300.0, 60.0 * (2 ** (state.consecutive_signals - 3)))
            state.cooldown_until = current_time + cooldown_duration
    
    async def _send_notification(self, ensemble_decision, kline: KlineEvent):
        """Send notification with enhanced trader-focused format."""
        try:
            # Add timeframe to ensemble_decision for Telegram formatting
            ensemble_decision.tf = kline.timeframe
            
            # Build pre-trade summary
            pre = {
                "p_win": float(ensemble_decision.confidence),
                "regime": self._extract_regime(ensemble_decision),
                "veto_count": len(getattr(ensemble_decision, 'vetoes', []) or []),
                "lat_ms": self.metrics.latency_tick_to_decision.snapshot(),
            }
            
            self.metrics.set_pre_trade_summary(pre)
            
            # Format and send message
            message = format_message(ensemble_decision, self.settings.model_dump())
            
            if not self.settings.transport.dry_run:
                await send_message(message, self.settings.transport.model_dump())
                logger.info(f"Signal sent for {kline.symbol}: {ensemble_decision.decision} @ {ensemble_decision.confidence:.3f}")
            else:
                logger.info(f"DRY RUN: Signal would be sent for {kline.symbol}")
                
        except Exception as e:
            logger.error(f"Error sending notification for {kline.symbol}: {e}")
    
    def _extract_regime(self, decision) -> Optional[str]:
        """Extract regime information from decision."""
        try:
            vd = decision.vote_detail or {}
            if isinstance(vd, dict):
                rg = vd.get('regime') or {}
                if isinstance(rg, dict):
                    return rg.get('regime_label') or rg.get('label')
        except Exception:
            pass
        return None


async def main_loop():
    """The main orchestration loop of the application."""
    parser = argparse.ArgumentParser(description="Ultra-Signals Realtime Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="settings.yaml",
        help="Path to the configuration file (default: settings.yaml)",
    )
    args = parser.parse_args()

    # --- 1. Load Settings ---
    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        logger.error(f"Failed to start due to configuration error: {e}")
        return

    # Canary mode override: force primary timeframe to 1m for higher tick rate without impacting other configs
    try:
        if (os.environ.get('TRADING_MODE') == 'CANARY' and 
            getattr(settings.runtime, 'primary_timeframe', None) != '1m'):
            if '1m' not in settings.runtime.timeframes:
                settings.runtime.timeframes.insert(0, '1m')
            settings.runtime.primary_timeframe = '1m'
            logger.info("[CANARY] Overriding primary_timeframe -> 1m for per-minute evaluation")
    except Exception as _e:
        logger.warning(f"[CANARY] Failed to apply primary timeframe override: {_e}")

    # --- 2. Initialize and Start Resilient Runner ---
    runner = ResilientSignalRunner(settings)
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")
        runner.shutdown_requested = True


def run():
    """Entry point to run the main async loop."""
    # Setup Loguru to be cleaner
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")


if __name__ == "__main__":
    run()
