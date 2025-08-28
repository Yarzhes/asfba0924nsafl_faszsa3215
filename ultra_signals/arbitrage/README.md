Arbitrage adapters and utilities
================================

This folder contains lightweight public collectors and helpers for the
Multi-Exchange Arbitrage Detection sprint.

What is included
- Async HTTP helper + rate limiter (`adapters/http.py`)
- Public REST adapters for Binance, Bybit, OKX, Coinbase, Kraken (`adapters/*`)
- VWAP depth walker integrated in `ArbitrageCollector.fetch_depth`
- Telegram sender wrapper with dry-run (`telegram_sender.py`)
- CLI runner for live testing (`ultra_signals/scripts/arbitrage_cli.py`)

Notes
- Adapters use public endpoints only; no API keys required.
- Rate limiting is rudimentary — adapt per-exchange limits in production.
- The collector will prefer L2 books when available on the venue adapter and
  will fall back to synthetic depth if the venue doesn't support L2 fetch.

Running live (dry-run):

```powershell
cd "c:\Users\Almir\Projects\Trading Helper"
python -u ultra_signals\scripts\arbitrage_cli.py BTCUSDT
```

To actually send Telegram messages, set a token and chat_id in the
`TelegramSender` initialization.
