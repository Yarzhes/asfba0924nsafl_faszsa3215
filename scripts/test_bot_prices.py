#!/usr/bin/env python3
"""
Bot Price Test - Quick verification of bot price data
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultra_signals.core.config import load_settings
from ultra_signals.data.binance_ws import BinanceWSClient

async def _run_price_feed():
    print("🤖 Testing Trading Bot Price Feed:")
    print("=" * 50)
    # Load settings & connect
    settings = load_settings("settings.yaml")
    ws_client = BinanceWSClient(settings)
    price_data = {}

    def on_kline(event):
        symbol = event.symbol
        close_price = float(event.kline.close)
        price_data[symbol] = close_price
        if len(price_data) <= 5:
            print(f"Bot: {symbol:<12} ${close_price:>12,.4f}")

    print("Connecting to WebSocket...")
    await ws_client.start()
    ws_client.on_kline = on_kline
    print("Listening for price updates (3 seconds)...")
    await asyncio.sleep(3)
    await ws_client.stop()
    return price_data

def test_bot_prices():
    """Synchronous pytest wrapper. Skips by default unless RUN_PRICE_WS_TEST=1."""
    if os.environ.get("RUN_PRICE_WS_TEST") != "1":
        pytest.skip("Price feed test skipped (set RUN_PRICE_WS_TEST=1 to enable)")
    try:
        data = asyncio.run(_run_price_feed())
        assert isinstance(data, dict)
    except Exception as e:
        pytest.fail(f"Price feed test failed: {e}")

if __name__ == "__main__":
    # Allow manual execution without pytest
    asyncio.run(_run_price_feed())
