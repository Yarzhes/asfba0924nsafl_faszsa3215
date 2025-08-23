# Features

This directory contains the pure functions responsible for calculating technical indicators and other features from the raw data provided by the `FeatureStore`. Each module takes in data (like OHLCV, trades, etc.) and returns a dictionary of computed feature values.

- `trend.py`: Trend-following indicators (EMAs, Supertrend, etc.).
- `momentum.py`: Momentum oscillators (RSI, MACD, etc.).
- `volatility.py`: Volatility measures (ATR, Bollinger Bands, etc.).
- `volume_flow.py`: Volume-based indicators (OBV, VWAP, CVD, etc.).
- `orderbook.py`: Order book analysis (imbalance, depth, etc.).
- `derivatives.py`: Futures-specific data (funding rates, OI, liquidations).
- `regime.py`: Market regime classification (e.g., trending vs. mean-reverting).