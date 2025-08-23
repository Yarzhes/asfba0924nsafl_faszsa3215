import pandas as pd
import numpy as np

def generate_data(symbol, start_date, end_date):
    """Generates dummy OHLCV data for a given symbol and date range."""
    dates = pd.date_range(start=start_date, end=end_date, freq='5T')
    count = len(dates)
    data = {
        'timestamp': dates,
        'open': np.random.uniform(100, 200, count).round(2),
        'high': np.random.uniform(200, 210, count).round(2),
        'low': np.random.uniform(90, 100, count).round(2),
        'close': np.random.uniform(100, 200, count).round(2),
        'volume': np.random.randint(100, 1000, count)
    }
    df = pd.DataFrame(data)
    df.to_csv(f"data/{symbol}_5m.csv", index=False)
    print(f"Generated data for {symbol}")

if __name__ == '__main__':
    symbols = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    for symbol in symbols:
        generate_data(symbol, start_date, end_date)