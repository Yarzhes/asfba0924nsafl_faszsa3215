#!/usr/bin/env python3
"""
Bot Price Test - Quick verification of bot price data
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultra_signals.core.config import load_settings
from ultra_signals.data.binance_ws import BinanceWSClient

async def test_bot_prices():
    """Test bot price data for a short time"""
    print("🤖 Testing Trading Bot Price Feed:")
    print("=" * 50)
    
    try:
        # Load settings
        settings = load_settings("settings.yaml")
        
        # Initialize WebSocket client  
        ws_client = BinanceWSClient(settings)
        
        # Create a simple price tracker
        price_data = {}
        
        def on_kline(event):
            symbol = event.symbol
            close_price = float(event.kline.close)
            price_data[symbol] = close_price
            
            # Print first 5 symbols we receive
            if len(price_data) <= 5:
                print(f"Bot: {symbol:<12} ${close_price:>12,.4f}")
        
        # Connect and listen for 10 seconds
        print(f"Connecting to WebSocket...")
        await ws_client.start()
        
        # Subscribe to kline events
        ws_client.on_kline = on_kline
        
        print(f"Listening for price updates (10 seconds)...")
        await asyncio.sleep(10)
        
        print("\n📊 Summary:")
        print(f"Received price data for {len(price_data)} symbols")
        
        # Compare a few key symbols
        if 'BTCUSDT' in price_data:
            print(f"Bot BTC: ${price_data['BTCUSDT']:,.2f}")
        if 'ETHUSDT' in price_data:
            print(f"Bot ETH: ${price_data['ETHUSDT']:,.2f}")
            
        await ws_client.stop()
        print("✅ Bot price test completed")
        
    except Exception as e:
        print(f"❌ Error testing bot prices: {e}")

if __name__ == "__main__":
    asyncio.run(test_bot_prices())
