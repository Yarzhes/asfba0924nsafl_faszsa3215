# 📱 Telegram Integration - READY FOR SHADOW TESTING

## ✅ Configuration Status

### Bot Credentials
- **Bot Token**: `8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs` ✅ Set
- **Chat ID**: `7072100094` ✅ Set  
- **Connection**: ✅ Tested and working

### Environment Variables
```powershell
# Set via scripts/setup_telegram.ps1
$env:TELEGRAM_BOT_TOKEN = "8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs"
$env:TELEGRAM_CHAT_ID = "7072100094"
```

### Configuration Files
- **✅ settings.yaml**: Telegram enabled, credentials from env vars
- **✅ shadow_test.yaml**: Real notifications enabled for shadow testing
- **✅ canary_mode.yaml**: Conservative settings for canary phase

---

## 🧪 Shadow Mode Telegram Messages

During shadow testing, you'll receive Telegram notifications for:

### Signal Format
```
🧪 SHADOW | BTCUSDT | LONG | ENTRY:45,230 | SL:44,100 | TP:47,500 | Lev:3x | p:0.67 | regime:BULL | veto:[] | sniper:ALLOWED
```

### Message Types
1. **🎯 ALLOWED SIGNALS**: Signals that pass sniper caps and MTF confirmation
2. **🚫 REJECTED SIGNALS**: Blocked by sniper caps or MTF disagreement  
3. **📊 METRICS UPDATES**: Hourly/daily counter summaries
4. **⚠️ SYSTEM ALERTS**: Circuit breakers, errors, status changes

### Shadow Mode Safety
- **NO ORDERS PLACED**: `shadow_mode: true` prevents all trading
- **REAL NOTIFICATIONS**: You'll see actual Telegram messages
- **FULL MONITORING**: All metrics, signals, and rejections tracked

---

## 🚀 Ready Commands

### 1. Start Shadow Test (120 minutes)
```bash
# Run this AFTER setting Telegram environment variables
.\scripts\setup_telegram.ps1
python scripts/run_shadow_test.py --duration 120
```

### 2. Monitor in Real-time  
```bash
python scripts/monitor_shadow.py --duration 120 --interval 30
```

### 3. Test Telegram Connection
```bash
python scripts/test_telegram.py
```

---

## 📱 Expected Telegram Activity

### During Shadow Test (120 minutes)
- **2-4 signal notifications** (caps: 2/hour, 6/day)
- **4-8 rejection notifications** (blocked signals)
- **4 metric summaries** (every 30 minutes)
- **1 completion report** (final summary)

### Message Timing
- **Real-time**: Signal decisions (allowed/rejected)
- **Every 30 min**: Counter status and metrics
- **Every hour**: Hourly cap reset notifications
- **End of test**: Complete shadow results summary

---

## 🔧 Troubleshooting

### If No Messages Received
1. **Check credentials**: Run `scripts/test_telegram.py`
2. **Verify bot token**: Confirm bot is active in @BotFather
3. **Check chat ID**: Ensure you've started a chat with the bot
4. **Environment vars**: Re-run `scripts/setup_telegram.ps1`

### If Too Many Messages
- Shadow mode will respect 2/hour, 6/day caps
- Actual frequency depends on market conditions
- Rejection messages help validate cap enforcement

---

## ✅ Pre-Shadow Checklist

- [x] **Telegram bot token set**: 8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs
- [x] **Chat ID configured**: 7072100094  
- [x] **Environment variables**: Set via `setup_telegram.ps1`
- [x] **Connection tested**: Test message sent successfully
- [x] **Shadow config updated**: Real notifications enabled
- [x] **Safety confirmed**: `shadow_mode: true` prevents orders

---

## 🎯 What's Next

You're now ready to execute the shadow test with full Telegram integration:

```bash
# Terminal A: Set environment and start shadow test
.\scripts\setup_telegram.ps1
python scripts/run_shadow_test.py --duration 120

# Terminal B: Monitor metrics in real-time  
python scripts/monitor_shadow.py --duration 120 --interval 30
```

**Expected Timeline**: 
- 0-120 min: Shadow testing with Telegram notifications
- 120-130 min: Results analysis and go/no-go decision
- If successful: Proceed to 60-minute BTCUSDT canary testing

**🚀 Your Telegram chat will be the primary interface for monitoring signal activity during shadow testing!**

---

*Generated: 2025-08-28 20:22 UTC*  
*Status: ✅ TELEGRAM READY FOR SHADOW MODE*
