#!/usr/bin/env python3
"""
Price Verification - Check current crypto prices from Binance API
"""

import requests
import json
from datetime import datetime

def check_market_prices():
    """Get current prices from Binance API"""
    try:
        response = requests.get('https://api.binance.com/api/v3/ticker/price', timeout=10)
        if response.status_code == 200:
            prices = response.json()
            
            # Filter for our 20 symbols
            our_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
                          'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'TONUSDT',
                          'TRXUSDT', 'DOTUSDT', 'NEARUSDT', 'ATOMUSDT', 'LTCUSDT',
                          'BCHUSDT', 'ARBUSDT', 'APTUSDT', 'MATICUSDT', 'SUIUSDT']
            
            print('🔍 Current Market Prices from Binance API:')
            print('=' * 50)
            print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}')
            print()
            
            symbol_prices = {}
            for price_data in prices:
                if price_data['symbol'] in our_symbols:
                    symbol = price_data['symbol']
                    price = float(price_data['price'])
                    symbol_prices[symbol] = price
                    print(f'{symbol:<12} ${price:>12,.4f}')
            
            print('=' * 50)
            print('✅ API connection successful - prices retrieved')
            
            # Show some key metrics
            btc_price = symbol_prices.get('BTCUSDT', 0)
            eth_price = symbol_prices.get('ETHUSDT', 0)
            print(f'\n📊 Key Metrics:')
            print(f'BTC: ${btc_price:,.2f}')
            print(f'ETH: ${eth_price:,.2f}')
            if btc_price > 0 and eth_price > 0:
                eth_btc_ratio = eth_price / btc_price
                print(f'ETH/BTC Ratio: {eth_btc_ratio:.4f}')
            
            return symbol_prices
            
        else:
            print('❌ Failed to get prices from Binance API')
            return None
    except Exception as e:
        print(f'❌ Error: {e}')
        return None

if __name__ == "__main__":
    check_market_prices()
