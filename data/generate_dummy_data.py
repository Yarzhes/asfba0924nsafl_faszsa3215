import pandas as pd
import numpy as np
import os

def generate_ohlcv_data(start_date, end_date, ticker):
    """Generates dummy OHLCV data for backtesting."""
    timeframe = "5min"
    date_range = pd.date_range(start=start_date, end=end_date, freq=timeframe)
    data_len = len(date_range)
    
    # Create some wavy price action
    price = (
        20000
        + (np.sin(np.linspace(0, 20, data_len)) * 100)
        + (np.sin(np.linspace(0, 5, data_len)) * 50)
        + np.random.randn(data_len).cumsum() * 3
    )
    
    df = pd.DataFrame(index=date_range)
    df["open"] = price - np.random.uniform(0, 5, size=data_len)
    df["high"] = df["open"] + np.random.uniform(0, 15, size=data_len)
    df["low"] = df["open"] - np.random.uniform(0, 15, size=data_len)
    df["close"] = (df["high"] + df["low"]) / 2 # Mid-point close
    df["volume"] = np.random.uniform(50, 200, size=data_len)
    df.index.name = "timestamp"

    # Ensure OHLC logic holds
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df

if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Generate data for BTCUSDT
    btcusdt_df = generate_ohlcv_data("2023-01-01", "2023-03-31", "BTCUSDT")
    btcusdt_path = "data/BTCUSDT_5m.csv"
    btcusdt_df.to_csv(btcusdt_path)
    print(f"Generated {len(btcusdt_df)} rows of data for BTCUSDT and saved to {btcusdt_path}")

    # Generate data for BTC-PERP (used in other configs)
    btcperp_df = generate_ohlcv_data("2023-01-01", "2023-03-31", "BTC-PERP")
    btcperp_path = "data/BTC-PERP_5m.csv"
    btcperp_df.to_csv(btcperp_path)
    print(f"Generated {len(btcperp_df)} rows of data for BTC-PERP and saved to {btcperp_path}")