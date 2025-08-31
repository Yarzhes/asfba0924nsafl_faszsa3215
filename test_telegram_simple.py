#!/usr/bin/env python3
"""
Simple Telegram Test - Send a message directly
"""

import requests
import json

def test_telegram_direct():
    """Test sending a message directly to Telegram"""
    print("=== Testing Telegram Direct Message ===")
    
    bot_token = "8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs"
    chat_id = "7072100094"
    
    # Test message
    message = """🎯 Trading Helper Signal Test

BTCUSDT | LONG | ENTRY:50000 | SL:49500 | TP:51000 | Lev:10 | p:0.85 | regime:trend | veto:none | code:trend_breakout

This is a test signal to verify Telegram integration is working correctly."""

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message
    }
    
    try:
        print("Sending test message to Telegram...")
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print("✅ Test message sent successfully!")
                print("Check your Telegram for the message.")
                return True
            else:
                print(f"❌ Telegram API error: {result}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return False

if __name__ == "__main__":
    success = test_telegram_direct()
    if success:
        print("\n🎉 Telegram integration is working!")
        print("Your trading system should now send signals to Telegram.")
    else:
        print("\n❌ Telegram integration failed!")

