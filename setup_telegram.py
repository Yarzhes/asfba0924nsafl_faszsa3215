#!/usr/bin/env python3
"""
Telegram Setup Helper Script

This script helps you set up your Telegram bot for receiving trading signals.
"""

import requests
import json
import os

def get_bot_info(bot_token):
    """Get bot information to verify the token is valid"""
    url = f"https://api.telegram.org/bot{bot_token}/getMe"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error getting bot info: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error connecting to Telegram: {e}")
        return None

def get_updates(bot_token):
    """Get recent updates to find your chat ID"""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error getting updates: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting updates: {e}")
        return None

def send_test_message(bot_token, chat_id):
    """Send a test message to verify the setup"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": "🎯 Trading Helper Test Message\n\nIf you see this message, your Telegram setup is working correctly!"
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("✅ Test message sent successfully!")
            return True
        else:
            print(f"❌ Error sending test message: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error sending test message: {e}")
        return False

def main():
    print("=== Telegram Setup Helper ===\n")
    
    # Bot token from your scripts
    bot_token = "8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs"
    
    print(f"Using bot token: {bot_token}")
    
    # Step 1: Verify bot token
    print("\n1. Verifying bot token...")
    bot_info = get_bot_info(bot_token)
    if bot_info and bot_info.get('ok'):
        bot_data = bot_info['result']
        print(f"✅ Bot verified: @{bot_data['username']} ({bot_data['first_name']})")
    else:
        print("❌ Bot token verification failed!")
        return
    
    # Step 2: Get updates to find chat ID
    print("\n2. Getting recent updates to find your chat ID...")
    updates = get_updates(bot_token)
    if updates and updates.get('ok'):
        results = updates['result']
        if results:
            print("📱 Recent messages found:")
            chat_ids = set()
            for update in results:
                if 'message' in update:
                    chat = update['message']['chat']
                    chat_id = chat['id']
                    chat_type = chat['type']
                    chat_title = chat.get('title', chat.get('first_name', 'Unknown'))
                    chat_ids.add(chat_id)
                    print(f"   - Chat ID: {chat_id} ({chat_type}: {chat_title})")
            
            if chat_ids:
                print(f"\nFound {len(chat_ids)} unique chat(s).")
                # Use the first chat ID found
                chat_id = list(chat_ids)[0]
                print(f"Using chat ID: {chat_id}")
            else:
                print("❌ No chat IDs found. Please send a message to your bot first.")
                return
        else:
            print("❌ No recent updates found. Please send a message to your bot first.")
            return
    else:
        print("❌ Could not get updates!")
        return
    
    # Step 3: Send test message
    print(f"\n3. Sending test message to chat ID {chat_id}...")
    if send_test_message(bot_token, chat_id):
        print("\n🎉 Telegram setup successful!")
        print(f"\nTo update your settings.yaml, replace the chat_id with: {chat_id}")
        print("\nYour current settings.yaml has:")
        print("  bot_token: 8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs")
        print("  chat_id: 1234567890  # Replace this with your actual chat ID")
    else:
        print("\n❌ Telegram setup failed!")

if __name__ == "__main__":
    main()

