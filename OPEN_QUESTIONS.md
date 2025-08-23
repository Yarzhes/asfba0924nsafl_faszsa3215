# Open Questions for Discussion

This document lists the remaining open questions that require clarification before or during implementation. These are intended to be discussed with the project lead or resolved in "Ask a Developer" mode.

---

### 1. Configuration & Universe
- **Q1.1: Top-20 Symbols:** The default list of 20 symbols in `CONFIG_SCHEMA.md` is a sensible starting point. Should this be reviewed and confirmed based on current market volume and volatility data before the first production deployment?
- **Q1.2: Depth Stream Granularity:** The initial design specifies `levels: 20` for the partial book depth stream. Is this the right balance between data granularity and CPU/network overhead? Would `levels: 10` be sufficient?
- **Q1.3: Funding Avoidance Window:** The default policy is to avoid signals within `±5 minutes` of a funding timestamp. Is this window appropriate, or should it be wider (e.g., `±10 minutes`) or narrower?
- **Q1.4: Dynamic Universe Thresholds:** For the dynamic universe selection, what are the ideal default values for `min_notional_usd` (24h volume) and `max_spread`?

### 2. Engine & Signal Logic
- **Q2.1: Minimum Confidence for Alerts:** The `thresholds.enter` setting defines when a signal is generated. Is there a separate, higher confidence bar that should be met before a Telegram alert is sent? For example, should we only send alerts for signals with `confidence > 65`?
- **Q2.2: Signal Throttling:** How should we handle repeated signals for the same symbol/direction? For example, if a `BTCUSDT LONG` signal is generated, should we enforce a cooldown period (e.g., 30 minutes) before another `BTCUSDT LONG` signal can be sent, even if the score remains high?

### 3. Transport (Telegram)
- **Q3.1: Message Format:** The default `parse_mode` is Markdown. Is this preferred over HTML? Markdown is simpler, but HTML allows for more complex formatting if needed later.
- **Q3.2: Rate-Limit Caps:** What are the actual rate limits of the target Telegram Bot API? The implementation should respect these, but we need the specific numbers (e.g., max 20 messages per minute to one group).

### 4. Backtesting
- **Q4.1: Canonical Data Source:** What is the official, trusted source for historical data for backtesting? Is it from a specific vendor, an internal database, or should it be manually downloaded from Binance?
- **Q4.2: Backtest Time Range:** What is the minimum historical time range that the system must be successfully backtested against before its first use (e.g., the last 2 years)?