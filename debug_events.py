#!/usr/bin/env python3
"""
Debug Kline Events - Check what events are being received

This script focuses on analyzing the actual events coming from the WebSocket
to understand why no closed klines are being detected.
"""

import asyncio
import time
from loguru import logger

from ultra_signals.core.config import load_settings
from ultra_signals.data.binance_ws import BinanceWSClient
from ultra_signals.core.events import KlineEvent, MarketEvent


def setup_logging():
    """Setup simple console logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


async def debug_events():
    """Analyze the events being received."""
    setup_logging()
    
    settings = load_settings("settings.yaml")
    logger.info("Settings loaded successfully")
    
    # Initialize WebSocket client
    ws_client = BinanceWSClient(settings)
    test_symbol = "BTCUSDT"
    test_timeframe = "5m"
    
    # Subscribe to just one symbol for debugging
    ws_client.subscribe([test_symbol], [test_timeframe])
    
    logger.info(f"Starting event analysis for {test_symbol}/{test_timeframe}")
    
    # Counters for debugging
    total_events = 0
    kline_events = 0
    closed_klines = 0
    book_ticker_events = 0
    mark_price_events = 0
    other_events = 0
    
    # Track event types and properties
    event_samples = []
    
    try:
        # Set a timeout for this debug session
        start_time = time.time()
        max_runtime = 60  # 1 minute
        
        async for event in ws_client.start():
            total_events += 1
            
            # Check timeout
            if time.time() - start_time > max_runtime:
                logger.info("Debug session timeout reached")
                break
            
            # Analyze the event
            event_type = type(event).__name__
            
            if isinstance(event, KlineEvent):
                kline_events += 1
                logger.info(f"🕯️  Kline Event #{kline_events}: symbol={event.symbol}, tf={event.timeframe}, closed={event.closed}, ts={event.timestamp}")
                
                if event.closed:
                    closed_klines += 1
                    logger.success(f"✅ CLOSED KLINE #{closed_klines}: {event.symbol}/{event.timeframe} at {event.timestamp}")
                
                # Store sample for analysis
                if len(event_samples) < 5:
                    event_samples.append({
                        'type': event_type,
                        'symbol': event.symbol,
                        'timeframe': event.timeframe,
                        'timestamp': event.timestamp,
                        'closed': event.closed,
                        'open': event.open,
                        'close': event.close,
                        'volume': event.volume
                    })
            else:
                # Count other event types
                if hasattr(event, 'symbol'):
                    if 'BookTicker' in event_type:
                        book_ticker_events += 1
                        if book_ticker_events <= 3:  # Log first few
                            logger.debug(f"📖 Book Ticker: {event.symbol}")
                    elif 'MarkPrice' in event_type:
                        mark_price_events += 1
                        if mark_price_events <= 3:  # Log first few
                            logger.debug(f"💰 Mark Price: {event.symbol}")
                    else:
                        other_events += 1
                        logger.debug(f"❓ Other Event: {event_type}")
                else:
                    other_events += 1
                    logger.debug(f"❓ Unknown Event: {event_type}")
            
            # Log every 1000 events
            if total_events % 1000 == 0:
                logger.info(f"Processed {total_events} events - Klines: {kline_events} (closed: {closed_klines}), Book: {book_ticker_events}, Mark: {mark_price_events}, Other: {other_events}")
            
            # Stop early if we get some closed klines
            if closed_klines >= 3:
                logger.success("Got enough closed klines for analysis")
                break
    
    except Exception as e:
        logger.error(f"Error in debug session: {e}", exc_info=True)
    finally:
        await ws_client.stop()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("EVENT ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Total events processed: {total_events}")
        logger.info(f"Kline events: {kline_events}")
        logger.info(f"Closed klines: {closed_klines}")
        logger.info(f"Book ticker events: {book_ticker_events}")
        logger.info(f"Mark price events: {mark_price_events}")
        logger.info(f"Other events: {other_events}")
        logger.info("="*60)
        
        if event_samples:
            logger.info("\nSAMPLE EVENTS:")
            for i, sample in enumerate(event_samples):
                logger.info(f"  {i+1}. {sample}")
        
        # Diagnosis
        if kline_events == 0:
            logger.error("❌ NO KLINE EVENTS RECEIVED - WebSocket subscription may be incorrect")
        elif closed_klines == 0:
            logger.warning("⚠️  KLINE EVENTS RECEIVED BUT NONE CLOSED - May need to wait for candle completion")
        else:
            logger.success(f"✅ SUCCESS - Received {closed_klines} closed klines")


if __name__ == "__main__":
    try:
        asyncio.run(debug_events())
    except KeyboardInterrupt:
        logger.info("Debug session stopped by user")
