"""
Binance USDⓈ-M Futures WebSocket Client

This module provides a high-performance, resilient client for subscribing to
Binance's real-time kline data streams.

Features:
- Combined Streams: Subscribes to multiple symbols and timeframes over a single
  WebSocket connection for efficiency (`/stream?streams=...`).
- Asynchronous API: Built with `asyncio` and the `websockets` library for
  non-blocking I/O.
- Automatic Reconnection: Handles connection drops gracefully with an
  exponential backoff and jitter algorithm to prevent hammering the server.
- Data Normalization: Parses raw JSON payloads from Binance into a
  standardized `KlineEvent` Pydantic model.
- Graceful Shutdown: Designed to be started and stopped cleanly by the
  application runner.
"""

import asyncio
import json
import random
from typing import AsyncIterator, Dict, List, Union

import websockets
from loguru import logger
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed, WebSocketException

from ultra_signals.core.events import BookTickerEvent, KlineEvent, MarkPriceEvent, MarketEvent
from ultra_signals.core.config import Settings

# --- Constants ---
BINANCE_WS_BASE_URL = "wss://fstream.binance.com/stream"

class BinanceWSClient:
    """
    Connects to Binance's WebSocket API and yields normalized market events.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the client with application settings.

        Args:
            settings: A validated Pydantic Settings object.
        """
        self._settings = settings
        self._ws_url = BINANCE_WS_BASE_URL
        self._subscriptions: List[str] = []
        self._connection = None
        self._is_running = False
        self._reconnect_attempts = 0

    def subscribe(self, symbols: List[str], timeframes: List[str]) -> None:
        """
        Defines the streams to subscribe to.

        This method must be called before `start()`.

        Args:
            symbols: A list of symbols, e.g., ["BTCUSDT", "ETHUSDT"].
            timeframes: A list of timeframes, e.g., ["1m", "5m"].
        """
        if not symbols or not timeframes:
            raise ValueError("Symbols and timeframes lists cannot be empty.")

        kline_streams = [f"{s.lower()}@kline_{tf}" for s in symbols for tf in timeframes]
        # Subscribe to all symbols for book ticker and mark price
        book_ticker_streams = [f"{s.lower()}@bookTicker" for s in symbols]
        mark_price_streams = [f"{s.lower()}@markPrice@1s" for s in symbols] # Or use !markPrice@arr for all symbols
        
        self._subscriptions.extend(kline_streams)
        self._subscriptions.extend(book_ticker_streams)
        self._subscriptions.extend(mark_price_streams)

        stream_query = "?streams=" + "/".join(self._subscriptions)
        self._ws_url += stream_query
        logger.info(f"Subscribing to {len(self._subscriptions)} streams for {len(symbols)} symbols.")
        logger.debug(f"Generated WebSocket URL: {self._ws_url}")

    async def start(self) -> AsyncIterator[MarketEvent]:
        """
        Starts the WebSocket connection and yields events.

        This method is an async generator that will run indefinitely, yielding
        `MarketEvent` objects as they are received. It handles all connection
        logic, including reconnections and ping/pong keepalives.

        Yields:
            MarketEvent: A validated Pydantic model of a supported market event.
        """
        self._is_running = True
        while self._is_running:
            try:
                # Set a timeout for the connection attempt
                async with asyncio.timeout(30):
                    async with websockets.connect(
                        self._ws_url,
                        ping_interval=25,  # Send a ping every 25 seconds
                        ping_timeout=20,   # Wait 20 seconds for a pong response
                        close_timeout=10,
                        extra_headers={"User-Agent": "UltraSignals/1.0"},
                    ) as ws:
                        self._connection = ws
                        self._reconnect_attempts = 0
                        logger.success("WebSocket connection established successfully.")

                        # Start a task to drain incoming messages
                        async for message in ws:
                            try:
                                event = self._parse_message(message)
                                if event:
                                    yield event
                            except (ValidationError, json.JSONDecodeError) as e:
                                logger.warning(f"Failed to parse WebSocket message: {e}. Message: {message[:150]}")

            except TimeoutError:
                logger.warning("WebSocket connection attempt timed out. Retrying...")
                await self._reconnect()
            except (ConnectionClosed, WebSocketException, OSError) as e:
                # Check for specific close codes if available
                close_code = getattr(e, 'code', 'N/A')
                reason = getattr(e, 'reason', 'N/A')
                logger.warning(f"WebSocket connection error: {e} (Code: {close_code}, Reason: {reason}). Attempting to reconnect...")
                if not self._is_running:
                    break
                await self._reconnect()
            except Exception as e:
                logger.error(f"An unexpected error occurred in the WebSocket client: {e}", exc_info=True)
                if not self._is_running:
                    break
                await self._reconnect()

    async def stop(self) -> None:
        """
        Signals the client to gracefully shut down.
        """
        logger.info("Stopping WebSocket client...")
        self._is_running = False
        if self._connection and self._connection.open:
            await self._connection.close()

    def _parse_message(self, message: str) -> MarketEvent | None:
        """
        Parses a raw JSON message from the WebSocket stream into a canonical event.

        Args:
            message: The raw JSON string from the stream.

        Returns:
            A `MarketEvent` (KlineEvent, BookTickerEvent, etc.) or `None` if the
            message is not a recognized event type.
        """
        data = json.loads(message)

        if 'stream' not in data or 'data' not in data:
            logger.trace(f"Received non-stream message: {data}")
            return None

        event_data = data['data']
        event_type = event_data.get('e')

        try:
            if event_type == 'kline':
                kline_data = event_data['k']
                return KlineEvent(
                    timestamp=kline_data['t'],
                    symbol=kline_data['s'],
                    timeframe=kline_data['i'],
                    open=float(kline_data['o']),
                    high=float(kline_data['h']),
                    low=float(kline_data['l']),
                    close=float(kline_data['c']),
                    volume=float(kline_data['v']),
                    closed=kline_data['x'],
                )
            elif event_type == 'bookTicker':
                # Map native event fields to pydantic model fields
                event_data['timestamp'] = event_data.pop('E')
                event_data['symbol'] = event_data.pop('s')
                return BookTickerEvent.model_validate(event_data)
            
            elif event_type == 'markPriceUpdate':
                return MarkPriceEvent.model_validate(event_data)
            
            else:
                # logger.trace(f"Received unhandled event type '{event_type}': {event_data}")
                return None
        except ValidationError as e:
            logger.warning(
                f"Validation error for event type '{event_type}': {e}. "
                f"Payload: {event_data}"
            )
            return None

    async def _reconnect(self) -> None:
        """

        Handles the reconnection logic with exponential backoff and jitter.
        """
        self._reconnect_attempts += 1
        backoff_time = (2 ** self._reconnect_attempts) + random.uniform(0, 1)
        # Use backoff from settings, but ensure it grows
        wait_time = max(self._settings.runtime.reconnect_backoff_ms / 1000.0, backoff_time)
        wait_time = min(wait_time, 60) # Cap wait time to 60 seconds
        
        logger.info(f"Reconnecting attempt {self._reconnect_attempts}, waiting {wait_time:.2f} seconds...")
        await asyncio.sleep(wait_time)

# --- Example Usage ---
async def main_example():
    """A simple example of how to use the BinanceWSClient."""
    from ultra_signals.core.config import load_settings
    
    try:
        settings = load_settings("settings.yaml")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return

    client = BinanceWSClient(settings)
    client.subscribe(settings.runtime.symbols, settings.runtime.timeframes)
    
    try:
        logger.info("Starting client... Press Ctrl+C to stop.")
        async for kline in client.start():
            # In a real app, you'd pass this event to a feature store or engine.
            logger.debug(f"Received event: {kline}")

    except asyncio.CancelledError:
        logger.warning("Client start task cancelled.")
    finally:
        await client.stop()
        logger.info("Client shutdown complete.")

if __name__ == "__main__":
    # To run this example:
    # 1. Make sure you have a `settings.yaml` file in the project root.
    # 2. Run `python -m data.binance_ws` from the `ultra_signals` directory.
    try:
        asyncio.run(main_example())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")