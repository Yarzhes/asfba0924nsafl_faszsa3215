# Data

This directory is responsible for all external data ingestion, including real-time websockets and batch/API calls.

- `binance_ws.py`: The client for connecting to Binance USDⓈ-M WebSocket streams, handling subscriptions, and normalizing incoming data into canonical event models.
- `rest_clients.py`: Clients for accessing supplementary data via REST APIs, such as historical funding rates.
- `oi_providers/`: Contains modules for fetching Open Interest and liquidation data from various third-party sources (e.g., Coinglass, Coinalyze).