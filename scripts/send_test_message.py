#!/usr/bin/env python3
"""
Simple Telegram Test - Send One Message
"""

import os
import requests
import sys
from datetime import datetime

def send_test_message():
    """Send a single test message to Telegram."""
    
    # Get credentials from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    print(f"🤖 Sending test message to Telegram...")
    print(f"📱 Chat ID: {chat_id}")
    
    if not bot_token or not chat_id:
        print("❌ Missing Telegram credentials!")
        return False
    
    # Test message
    message = f"""🚀 Trading Helper - Telegram Test

✅ Connection Status: Working  
🤖 Bot Token: 8360503431:***  
💬 Chat ID: {chat_id}  
📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

🎯 System Ready: Your sniper mode signals will be sent to this chat during shadow testing.

This is a one-time test message."""
    
    # Send message
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print("✅ Message sent successfully!")
                print(f"📱 Message ID: {result['result']['message_id']}")
                print("📱 Check your Telegram app for the message.")
                return True
            else:
                print(f"❌ Telegram API error: {result.get('description', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"❌ Error details: {error_data}")
            except:
                print(f"❌ Response text: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

if __name__ == "__main__":
    success = send_test_message()
    if success:
        print("\n🎉 Telegram integration confirmed working!")
    else:
        print("\n❌ Telegram test failed.")
    sys.exit(0 if success else 1)
