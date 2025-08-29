#!/usr/bin/env python3
"""
Raw WebSocket Debug - Check exactly what Binance sends
====================================================
This script directly logs the raw WebSocket messages to see
if the 'x' field (closed flag) is ever set to true.
"""

import asyncio
import json
import time
from datetime import datetime
from loguru import logger
import websockets

def setup_logging():
    """Setup simple console logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

async def debug_raw_websocket():
    """Connect directly to Binance WebSocket and log raw messages."""
    setup_logging()
    
    # Binance WebSocket URL for 5m klines on BTCUSDT
    url = "wss://fstream.binance.com/stream?streams=btcusdt@kline_5m"
    
    logger.info("🔌 Connecting to Binance WebSocket...")
    logger.info(f"URL: {url}")
    logger.info(f"Current time: {datetime.now()}")
    
    # Calculate when next 5m candle should close
    now = datetime.now()
    minutes_until_next_5m = 5 - (now.minute % 5)
    seconds_until_next_5m = (minutes_until_next_5m * 60) - now.second
    next_close_time = now.replace(second=0, microsecond=0).replace(minute=now.minute + minutes_until_next_5m)
    
    logger.info(f"Next 5m candle should close at: {next_close_time}")
    logger.info(f"Seconds until next close: {seconds_until_next_5m}")
    
    message_count = 0
    closed_klines_found = 0
    
    try:
        async with websockets.connect(url) as websocket:
            logger.success("✅ Connected! Waiting for messages...")
            
            start_time = time.time()
            max_runtime = 300  # 5 minutes max
            
            async for message in websocket:
                message_count += 1
                
                try:
                    data = json.loads(message)
                    
                    # Check if this is a kline event
                    if 'stream' in data and 'data' in data:
                        stream = data['stream']
                        event_data = data['data']
                        
                        if 'e' in event_data and event_data['e'] == 'kline':
                            kline = event_data['k']
                            symbol = kline['s']
                            timeframe = kline['i']
                            is_closed = kline['x']  # This is the key field!
                            open_time = kline['t']
                            close_time = kline['T']
                            
                            if is_closed:
                                closed_klines_found += 1
                                logger.success(f"🎯 CLOSED KLINE #{closed_klines_found}!")
                                logger.info(f"   Symbol: {symbol}")
                                logger.info(f"   Timeframe: {timeframe}")
                                logger.info(f"   Open time: {datetime.fromtimestamp(open_time/1000)}")
                                logger.info(f"   Close time: {datetime.fromtimestamp(close_time/1000)}")
                                logger.info(f"   Close price: {kline['c']}")
                                logger.info(f"   Volume: {kline['v']}")
                                logger.info(f"   Messages processed: {message_count}")
                                # Found one - we can stop here
                                break
                            else:
                                # Log periodic updates
                                if message_count % 100 == 0:
                                    logger.info(f"Message #{message_count}: Open kline for {symbol} at {datetime.fromtimestamp(open_time/1000)}")
                                    logger.debug(f"  x (closed): {is_closed}, close_price: {kline['c']}")
                    
                    # Safety timeout
                    if time.time() - start_time > max_runtime:
                        logger.warning("Timeout reached - stopping")
                        break
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                except KeyError as e:
                    logger.error(f"Missing key in message: {e}")
                    logger.debug(f"Raw message: {message}")
                    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    logger.info(f"Summary: Processed {message_count} messages, found {closed_klines_found} closed klines")
    
    if closed_klines_found > 0:
        logger.success("✅ SUCCESS: Binance IS sending closed klines!")
    else:
        logger.warning("⚠️  No closed klines found - may need to wait longer")

def main():
    asyncio.run(debug_raw_websocket())

if __name__ == "__main__":
    main()
