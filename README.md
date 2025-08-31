# Ultra Signals — Crypto Futures Day-Trading Bot

A high-performance, resilient crypto futures day-trading bot with multi-timeframe analysis, real-time signal generation, and automated Telegram notifications.

## 🚀 Key Features

- **Multi-Timeframe Analysis**: 1m, 5m, 15m timeframe correlation
- **Resilient Architecture**: Automatic reconnection with exponential backoff
- **Per-Symbol Isolation**: Prevents signal bursts and ensures proper cooldown
- **Trader-Focused Notifications**: Clean Telegram messages with entry/SL/TP levels
- **Real-Time Processing**: WebSocket-based live market data processing
- **Risk Management**: Built-in position sizing and risk controls

## 🔧 Recent Improvements (v2.0)

### Stability & Reliability
- ✅ **Resilient Loop Semantics**: Automatic recovery from WebSocket disconnections
- ✅ **Heartbeat Monitoring**: Detects stale connections and forces reconnection
- ✅ **Exponential Backoff**: Intelligent retry logic with configurable limits
- ✅ **Graceful Shutdown**: Proper cleanup on user interruption

### Signal Quality & Isolation
- ✅ **Per-Symbol State Tracking**: Each symbol has independent cooldown and state
- ✅ **Anti-Burst Protection**: Prevents multiple signals for unrelated symbols
- ✅ **Configurable Cooldowns**: Minimum intervals and consecutive signal limits
- ✅ **Timeframe Warmup Enforcement**: Only uses fully warmed timeframes

### Trader Experience
- ✅ **Enhanced Telegram Messages**: Entry price, SL, TP1/TP2/TP3, leverage, risk %
- ✅ **Risk/Reward Ratios**: Automatic calculation and display
- ✅ **Clean Format**: No internal debug noise, only actionable information
- ✅ **Timestamp Tracking**: Clear signal timing information

## 📋 Prerequisites

- Python 3.8+
- Binance API credentials (for live trading)
- Telegram bot token and chat ID (for notifications)

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd trading-helper

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export ULTRA_SIGNALS_TRANSPORT__TELEGRAM__BOT_TOKEN="your_bot_token"
export ULTRA_SIGNALS_TRANSPORT__TELEGRAM__CHAT_ID="your_chat_id"
```

## 🚀 Running the Bot

### Live Mode
```bash
# Start the resilient signal runner
python ultra_signals/apps/realtime_runner.py --config settings.yaml
```

### Test Mode (Dry Run)
```bash
# Run with dry-run enabled (no actual trades)
python ultra_signals/apps/realtime_runner.py --config settings_canary.yaml
```

### Testing the Improvements
```bash
# Run the test suite to verify functionality
python test_resilient_runner.py
```

## ⚙️ Configuration

### Key Settings (`settings.yaml`)

```yaml
runtime:
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT, ...]
  timeframes: [1m, 5m, 15m]
  min_signal_interval_sec: 60.0  # Minimum time between signals
  max_consecutive_signals: 3     # Max signals before cooldown
  min_confidence: 0.65          # Minimum confidence threshold

execution:
  default_leverage: 10
  sl_atr_multiplier: 1.5        # ATR multiplier for stop loss
  default_risk_pct: 0.01        # 1% risk per trade

transport:
  telegram:
    enabled: true
    bot_token: "your_bot_token"
    chat_id: "your_chat_id"
    dry_run: false
```

## 📊 Telegram Message Format

The bot now sends clean, trader-focused messages:

```
📈 LONG BTCUSDT (5m)
Confidence: 75.0%

📍 Entry: $50000.0000
🛑 Stop Loss: $48500.0000
🎯 TP1: $51500.0000
🎯 TP2: $52250.0000
🎯 TP3: $53000.0000
⚡ Leverage: 10x
⚠️ Risk: 1.00%
📊 R:R = 1:1.00
🕐 Time: 2025-08-30 08:35:00 UTC
💡 Reason: Trend up + pullback to VWAP
```

## 🔍 Monitoring & Debugging

### Log Levels
- **INFO**: Signal generation, notifications sent
- **DEBUG**: Feature calculations, timeframe readiness
- **WARNING**: Reconnection attempts, insufficient data
- **ERROR**: Connection failures, processing errors

### Key Log Messages
- `Timeframe BTCUSDT/15m now ready with 250 bars` - Warmup complete
- `Signal sent for BTCUSDT: LONG @ 0.750` - Signal notification
- `Reconnecting in 4.0s (attempt 2/5)` - Recovery in progress

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_resilient_runner.py
```

Tests cover:
- ✅ Per-symbol state tracking
- ✅ SL/TP calculation accuracy
- ✅ Cooldown and isolation logic
- ✅ Timeframe readiness checking
- ✅ Message formatting

## 🚨 Troubleshooting

### Bot Stops Unexpectedly
- Check WebSocket connection logs
- Verify API credentials
- Monitor system resources

### No Signals Generated
- Check timeframe warmup status
- Verify confidence thresholds
- Review risk filter settings

### Telegram Notifications Not Working
- Verify bot token and chat ID
- Check network connectivity
- Review dry-run settings

## 📈 Performance Metrics

The resilient runner provides:
- **Uptime**: Automatic recovery from disconnections
- **Signal Quality**: Per-symbol isolation prevents noise
- **Latency**: Real-time processing with minimal delays
- **Reliability**: Exponential backoff and heartbeat monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This is a trading bot and involves financial risk. Use at your own discretion and ensure proper risk management.