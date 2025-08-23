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
from typing import Dict
import asyncio
import httpx
from loguru import logger

from ultra_signals.core.custom_types import EnsembleDecision


def format_message(decision: EnsembleDecision, settings: Dict) -> str:
    """
    Formats an EnsembleDecision object into a human-readable Markdown string.

    Escapes characters that are special in MarkdownV2.

    Args:
        decision: The `EnsembleDecision` object to format.
        settings: The global application settings.

    Returns:
        A formatted string ready to be sent via Telegram.
    """
    icon = "📈" if decision.decision == "LONG" else "📉"
    
    # Message header
    msg = (
        f"{icon} *New Ensemble Decision: {decision.decision} {decision.symbol}* ({decision.tf})\n\n"
        f"Ensemble Confidence: *{decision.confidence:.2%}*\n"
    )

    # Vote details
    if decision.vote_detail:
        vd = decision.vote_detail
        msg += (
            f"Vote: `{vd.get('agree', 0)}/{vd.get('total', 0)}` "
            f"| Profile: `{vd.get('profile', 'n/a')}` "
            f"| Wgt Sum: `{vd.get('weighted_sum', 0.0):.3f}`\n"
        )
    
    # Vetoes
    if decision.vetoes:
        msg += f"🚨 *VETOED*: {decision.vetoes[0]}\n"
    
    msg += f"--------------------------------------\n"
    
    # Sub-signal breakdown
    msg += "*Contributing Signals:*\n"
    for sub in decision.subsignals:
        sub_icon = "🟢" if sub.direction == "LONG" else "🔴" if sub.direction == "SHORT" else "⚪"
        # Escape special characters in strategy_id
        strategy_id_safe = sub.strategy_id.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]')
        msg += f"{sub_icon} `{strategy_id_safe}` ({sub.confidence_calibrated:.2f})\n"

    # Telegram's MarkdownV2 requires escaping certain characters
    # This is a basic implementation. A robust one would be more thorough.
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    for char in escape_chars:
        # Don't escape characters already part of markdown syntax we use
        if char not in ['*', '`']:
             msg = msg.replace(char, f"\\{char}")

    return msg

async def send_message(text: str, settings: Dict):
    """
    Sends a message to the configured Telegram chat with bounded retries.

    Args:
        text: The message content (Markdown formatted).
        settings: The `transport` section of the global settings.
    """
    tg_settings = settings['telegram']
    if not tg_settings['enabled']:
        logger.debug("Telegram transport is disabled in settings.")
        return

    token = tg_settings.get('bot_token')
    chat_id = tg_settings.get('chat_id')

    if not token or not chat_id:
        logger.error("Telegram sending enabled, but bot_token or chat_id is missing.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2",
    }
    
    max_retries = 3
    base_delay = 2 # seconds

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=15)
                
                if response.status_code == 429: # Rate limited
                    retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                    logger.warning(f"Rate limited by Telegram. Retrying in {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                response_data = response.json()
                msg_id = response_data.get('result', {}).get('message_id')
                logger.success(f"Successfully sent Telegram message {msg_id} to chat {chat_id}.")
                return

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Telegram API error (attempt {attempt + 1}/{max_retries}): "
                f"Status {e.response.status_code}, Response: {e.response.text[:200]}"
            )
        except httpx.RequestError as e:
            logger.error(f"Network error sending to Telegram (attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"Waiting {delay}s before retrying...")
            await asyncio.sleep(delay)
    
    logger.error("Failed to send message to Telegram after multiple retries.")