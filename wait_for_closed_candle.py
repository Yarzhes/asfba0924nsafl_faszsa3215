#!/usr/bin/env python3
"""
Wait for Closed Candle Test - Simplified Version
============================================== 
This script waits specifically for a closed=True kline event to verify 
that the WebSocket is correctly receiving candle completion events.
"""

import asyncio
from datetime import datetime
from loguru import logger

from ultra_signals.core.config import load_settings
from ultra_signals.data.binance_ws import BinanceWSClient
from ultra_signals.core.events import KlineEvent

def setup_logging():
    """Setup simple console logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

async def wait_for_closed_candle():
    """Wait for the next closed candle."""
    setup_logging()
    settings = load_settings("settings.yaml")
    
    logger.info("🕯️  Waiting for the next CLOSED candle...")
    logger.info(f"Current time: {datetime.now()}")
    
    # Create client
    client = BinanceWSClient(settings)
    test_symbol = "BTCUSDT"
    test_timeframe = "5m"
    
    # Subscribe to just one symbol
    client.subscribe([test_symbol], [test_timeframe])
    
    event_count = 0
    
    try:
        logger.info("✅ WebSocket starting - listening for events...")
        
        async for event in client.start():
            event_count += 1
            
            if isinstance(event, KlineEvent):
                if event.closed:
                    logger.success(f"🎯 FOUND CLOSED CANDLE!")
                    logger.info(f"   Symbol: {event.symbol}")
                    logger.info(f"   Timeframe: {event.timeframe}")
                    logger.info(f"   Close time: {datetime.fromtimestamp(event.timestamp/1000)}")
                    logger.info(f"   Events processed: {event_count}")
                    break
                else:
                    if event_count % 500 == 0:
                        logger.info(f"Processed {event_count} events - still waiting...")
            
            # Safety timeout
            if event_count > 50000:
                logger.warning("Timeout - no closed candle found")
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await client.close()

def main():
    asyncio.run(wait_for_closed_candle())

if __name__ == "__main__":
    main()
