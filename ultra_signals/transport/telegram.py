"""
Telegram Transport for Signal Notifications

This module provides the functionality to send formatted trading signals
to a Telegram chat. It handles message formatting and a "dry-run" mode
for safe testing.

Design Principles:
- Decoupling: This module is completely decoupled from the signal generation
  engine. It only knows how to format and send a `Signal` object.
- Resilience: The `send_message` function includes error handling to prevent
  the main application from crashing if Telegram is unreachable.
- Configurability: All Telegram-related settings (token, chat ID, enabled status)
  are sourced from the global application settings.
- Testability: A `dry_run` flag allows the runner to call this module without
  triggering actual network requests, which is critical for testing.
"""

import json
import re
from typing import Dict, List, Optional
import asyncio
import httpx
from loguru import logger

from ultra_signals.core.custom_types import EnsembleDecision


# --- helpers -----------------------------------------------------------------

def _escape_markdown_v2(text: str) -> str:
    """Escape characters for Telegram MarkdownV2."""
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)


def _chunk_for_telegram(text: str, limit: int = 4096) -> List[str]:
    """
    Telegram messages have a 4096 char limit. Split cleanly on newlines where possible.
    """
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        # try to break at last newline
        nl = text.rfind("\n", start, end)
        if nl == -1 or nl <= start:
            nl = end
        parts.append(text[start:nl])
        start = nl
    return parts


def _calculate_sl_tp(entry_price: float, decision: str, atr: float, settings: Dict) -> Dict[str, float]:
    """
    Calculate Stop Loss and Take Profit levels based on ATR.
    
    Args:
        entry_price: Entry price
        decision: "LONG" or "SHORT"
        atr: Average True Range value
        settings: Application settings for SL/TP configuration
    
    Returns:
        Dict with 'stop_loss', 'tp1', 'tp2', 'tp3', 'risk_amount' values
    """
    # Get SL multiplier from settings (default 1.5)
    sl_multiplier = settings.get('execution', {}).get('sl_atr_multiplier', 1.5)
    
    # Calculate stop loss
    if decision == "LONG":
        stop_loss = entry_price - (sl_multiplier * atr)
    else:  # SHORT
        stop_loss = entry_price + (sl_multiplier * atr)
    
    # Calculate risk amount
    risk_amount = abs(entry_price - stop_loss)
    
    # Calculate take profit levels (1R, 1.5R, 2R)
    if decision == "LONG":
        tp1 = entry_price + (1.0 * risk_amount)
        tp2 = entry_price + (1.5 * risk_amount)
        tp3 = entry_price + (2.0 * risk_amount)
    else:  # SHORT
        tp1 = entry_price - (1.0 * risk_amount)
        tp2 = entry_price - (1.5 * risk_amount)
        tp3 = entry_price - (2.0 * risk_amount)
    
    return {
        'stop_loss': stop_loss,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'risk_amount': risk_amount
    }


def _get(d, key, default_key):
    """Helper to get value from nested dict with fallback."""
    return d.get(key, d.get(default_key, None))


def format_message(decision: EnsembleDecision, settings: Dict, is_canary_debug: bool = False) -> str:
    """Clean trader-focused Telegram message with essential trade information."""
    
    # Direction icon and header
    icon = "📈" if decision.decision == "LONG" else "📉" if decision.decision == "SHORT" else "⚪"
    msg = f"{icon} *{decision.decision} {decision.symbol}* ({decision.tf})\n"
    
    # Confidence score
    try:
        msg += f"*Confidence: {decision.confidence:.1%}*\n\n"
    except Exception:
        msg += "\n"

    # Extract trade execution details
    try:
        vd = decision.vote_detail or {}
        risk_model = vd.get('risk_model', {})
        playbook = vd.get('playbook', {})
        
        # Get current price as entry
        entry_price = (_get(risk_model, 'entry_price', 'entry_price') or 
                      _get(playbook, 'entry_price', 'entry_price') or
                      _get(vd, 'current_price', 'price'))
        
        if entry_price:
            msg += f"📍 *Entry:* ${entry_price:.4f}\n"
            
            # Calculate SL/TP levels if not already provided
            atr = _get(risk_model, 'atr', 'atr') or _get(playbook, 'atr', 'atr')
            if atr:
                sl_tp_levels = _calculate_sl_tp(entry_price, decision.decision, atr, settings)
                
                # Stop loss
                msg += f"🛑 *Stop Loss:* ${sl_tp_levels['stop_loss']:.4f}\n"
                
                # Take profit levels
                msg += f"🎯 *TP1:* ${sl_tp_levels['tp1']:.4f}\n"
                msg += f"🎯 *TP2:* ${sl_tp_levels['tp2']:.4f}\n"
                msg += f"🎯 *TP3:* ${sl_tp_levels['tp3']:.4f}\n"
                
                # Risk/Reward ratio
                risk_amount = sl_tp_levels['risk_amount']
                reward_amount = abs(sl_tp_levels['tp1'] - entry_price)
                
                if risk_amount > 0:
                    rr_ratio = reward_amount / risk_amount
                    msg += f"📊 *R:R = 1:{rr_ratio:.2f}*\n"
        
        # Leverage
        leverage = (_get(risk_model, 'leverage', 'leverage') or 
                   _get(playbook, 'leverage', 'leverage') or
                   settings.get('execution', {}).get('default_leverage', 10))
        msg += f"⚡ *Leverage:* {leverage}x\n"
        
        # Risk percentage
        risk_pct = (_get(risk_model, 'risk_percentage', 'risk_pct') or 
                   _get(playbook, 'risk_percentage', 'risk_pct') or
                   settings.get('position_sizing', {}).get('max_risk_pct', 0.01))
        msg += f"⚠️ *Risk:* {risk_pct:.2%}\n"
        
        # Timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        msg += f"🕐 *Time:* {timestamp}\n"
        
        # Optional: Brief rationale (one line only)
        rationale = _get(vd, 'rationale', 'reason')
        if rationale and len(rationale) < 100:  # Keep it short
            msg += f"💡 *Reason:* {rationale}\n"
    
    except Exception as e:
        logger.debug(f"Error extracting trade details: {e}")
        # Fallback minimal message
        msg += "⚠️ Trade details unavailable\n"

    return msg


def format_arbitrage_message(fs) -> str:
    """Format arbitrage opportunity message."""
    if not fs.executable_spreads:
        return f"Arb: No spread {fs.symbol}"
    
    # Choose best by largest raw spread
    best = max(fs.executable_spreads, key=lambda e: e.raw_spread_bps)
    
    # Pick a standard bucket (25k) if present
    exec25 = best.exec_spread_bps_by_notional.get('25000') or next(iter(best.exec_spread_bps_by_notional.values()))
    
    parts = [
        f"Arb: Perp spread {best.raw_spread_bps:.2f} bps (exec {exec25:.2f} bps @ $25k) {best.venue_short.upper()}>{best.venue_long.upper()}"
    ]
    
    if fs.funding:
        # Naive funding diff: max - min current
        cur_rates = [f.current_rate_bps for f in fs.funding if f.current_rate_bps is not None]
        if len(cur_rates) >= 2:
            diff = max(cur_rates) - min(cur_rates)
            parts.append(f"Funding diff {diff:.2f} bps/8h")
    
    if fs.geo_premium:
        parts.append(f"Geo prem {fs.geo_premium.region_a} {fs.geo_premium.premium_bps:.2f} bps vs {fs.geo_premium.region_b}")
    
    return ' | '.join(parts)


async def send_message(text: str, settings: Dict):
    """
    Sends a message to the configured Telegram chat with bounded retries.

    Args:
        text: The message content (Markdown formatted).
        settings: The `transport` section of the global settings.
    """
    # --- DRY RUN support (from your docstring) -------------------------------
    # If settings['telegram'].dry_run is True, don't actually send—just log.
    tg_settings = settings.get('telegram', {})
    dry_run = bool(tg_settings.get('dry_run', False))
    if dry_run:
        logger.info("[Telegram dry-run] Message not sent:\n" + text)
        return
    # ------------------------------------------------------------------------

    if not tg_settings.get('enabled', False):
        logger.debug("Telegram transport is disabled in settings.")
        return

    token = tg_settings.get('bot_token')
    chat_id = tg_settings.get('chat_id')
    
    if not token or not chat_id:
        logger.warning("Telegram bot_token or chat_id not configured.")
        return

    # Rate limiting: Add jitter to prevent burst sends
    import random
    import time
    await asyncio.sleep(random.uniform(0.1, 0.5))

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # Split message if too long
    chunks = _chunk_for_telegram(text)
    
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(url, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('ok'):
                            logger.info(f"Telegram message sent successfully (chunk {i+1}/{len(chunks)})")
                            break
                        else:
                            logger.error(f"Telegram API error: {result.get('description', 'Unknown error')}")
                            break
                    elif response.status_code == 429:
                        # Rate limited - wait and retry
                        retry_after = response.json().get('parameters', {}).get('retry_after', 30)
                        logger.warning(f"Telegram rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after + random.uniform(0, 2))  # Add jitter
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Telegram HTTP error {response.status_code}: {response.text}")
                        break
                        
            except Exception as e:
                logger.error(f"Telegram send error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Failed to send Telegram message after all retries")


def send_decision(decision: EnsembleDecision, settings: Dict):
    """Send a formatted decision message to Telegram."""
    try:
        message = format_message(decision, settings)
        asyncio.create_task(send_message(message, settings))
        logger.info(f"Decision sent to Telegram: {decision.symbol} {decision.decision}")
    except Exception as e:
        logger.error(f"Error sending decision to Telegram: {e}")


if __name__ == "__main__":
    # Test the module
    print("Testing Telegram module...")
    
    # Mock decision for testing
    mock_decision = EnsembleDecision(
        symbol="BTCUSDT",
        decision="LONG",
        confidence=0.75,
        tf="5m",
        vote_detail={
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
    )
    
    mock_settings = {
        'execution': {
            'sl_atr_multiplier': 1.5,
            'default_leverage': 10
        },
        'position_sizing': {
            'max_risk_pct': 0.01
        }
    }
    
    # Test message formatting
    message = format_message(mock_decision, mock_settings)
    print("Formatted message:")
    print(message)
