#!/usr/bin/env python3
"""
Test Telegram Signal Sending with actual system structure
"""

import sys
import os
import numpy as np
import asyncio
from typing import Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.core.custom_types import EnsembleDecision
from ultra_signals.transport.telegram import send_decision

def test_telegram_signal():
    """Test sending a signal to Telegram"""
    print("=== Testing Telegram Signal Sending ===")
    
    # Create a test ensemble decision
    decision = EnsembleDecision(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        confidence=85.0,
        score=0.75,
        regime="trend",
        entry_price=50000.0,
        stop_loss=49500.0,
        take_profit=51000.0,
        leverage=10,
        position_size=0.1,
        component_scores={
            'trend': 1.0,
            'momentum': 0.5,
            'volatility': 0.0,
            'orderbook': 0.0,
            'derivatives': 0.0,
            'pullback_confluence': 0.0,
            'breakout_confluence': 0.0,
            'patterns': 0.0,
            'pullback_confluence_rs': 0.0
        }
    )
    
    print(f"Created test decision: {decision.decision} {decision.symbol}")
    print(f"Score: {decision.score:.3f}, Confidence: {decision.confidence:.1f}%")
    print(f"Entry: {decision.entry_price}, SL: {decision.stop_loss}, TP: {decision.take_profit}")
    
    # Test settings
    settings = {
        'telegram': {
            'enabled': True,
            'bot_token': '8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs',
            'chat_id': '7072100094',
            'send_pre_summary': True,
            'send_blocked_signals_in_canary': True,
            'message_template': '{pair} | {side} | ENTRY:{entry} | SL:{sl} | TP:{tp} | Lev:{lev} | p:{p_win:.2f} | regime:{regime} | veto:{veto_flags} | code:{reason}',
            'dry_run': False
        }
    }
    
    # Test Telegram transport
    async def send_test():
        try:
            print("\nSending test decision to Telegram...")
            await send_decision(decision, settings)
            print("✅ Test decision sent to Telegram successfully!")
            print("Check your Telegram for the message.")
        except Exception as e:
            print(f"❌ Error testing Telegram: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the async function
    asyncio.run(send_test())

if __name__ == "__main__":
    test_telegram_signal()
