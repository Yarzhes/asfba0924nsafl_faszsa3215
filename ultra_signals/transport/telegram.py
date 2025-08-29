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
from typing import Dict, List, Optional
import asyncio
import httpx
from loguru import logger

from ultra_signals.core.custom_types import EnsembleDecision


# --- helpers -----------------------------------------------------------------

def _escape_markdown_v2(text: str) -> str:  # retained for backward compatibility (unused now)
    return text


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


def _format_veto_block(decision: EnsembleDecision) -> str:
    """
    Shows explicit top veto reason (Sprint 8 requirement) and full list if available.
    Sprint 9 adds new reasons like: NEWS_WINDOW, CVD_WEAK, DEPTH_THIN, LIQUIDATION_SPIKE, NEAR_FUNDING_WINDOW.
    """
    if not getattr(decision, "vetoes", None):
        return ""
    vetoes = list(decision.vetoes or [])
    block = ""
    if len(vetoes) > 0:
        block += f"🚨 *VETOED* — Top reason: `{vetoes[0]}`\n"
        if len(vetoes) > 1:
            block += f"All reasons: `{', '.join(vetoes)}`\n"
    else:
        block += "🚨 *VETOED*\n"
    return block


def _format_filter_details(decision: EnsembleDecision) -> str:
    """
    Optional Sprint-9 diagnostics section if your engine attaches details like:
    - cvd_slope
    - spread_bps, top_qty
    - liq_z
    - mins_to_funding
    """
    det = getattr(decision, "filter_details", None) or getattr(decision, "details", None)
    if not isinstance(det, dict) or len(det) == 0:
        return ""

    lines = []
    # Only show if present; keep short
    if "cvd_slope" in det:
        lines.append(f"CVD slope: `{det.get('cvd_slope'):0.3f}`")
    if "spread_bps" in det:
        lines.append(f"Spread(bps): `{det.get('spread_bps'):0.2f}`")
    if "top_qty" in det:
        lines.append(f"Top qty: `{det.get('top_qty'):0.1f}`")
    if "liq_z" in det:
        lines.append(f"Liq Z: `{det.get('liq_z'):0.2f}`")
    if "mins_to_funding" in det and det.get("mins_to_funding") is not None:
        lines.append(f"Mins→funding: `{int(det.get('mins_to_funding'))}`")

    if not lines:
        return ""

    return "*Filters:* " + " | ".join(lines) + "\n"


def format_message(decision: EnsembleDecision, settings: Dict, is_canary_debug: bool = False) -> str:
    """Simplified, high‑readability Telegram message.

    Removed: profile / regime / veto / latency / filter diagnostics to cut noise.
    Focus: direction, confidence, vote ratio, and clean entry/SL/TP block with deltas.
    """
    icon = "📈" if decision.decision == "LONG" else "📉" if decision.decision == "SHORT" else "⚪"
    msg = f"{icon} *{decision.decision} {decision.symbol}* ({decision.tf})\n"
    try:
        msg += f"Confidence: *{decision.confidence:.2%}*\n"
    except Exception:
        pass

    vd = decision.vote_detail or {}
    # Vote ratio
    try:
        a = vd.get('agree'); t = vd.get('total')
        if a is not None and t is not None:
            msg += f"Vote {a}/{t}\n"
    except Exception:
        pass

    # Entry / SL / TP block
    try:
        if all(k in vd for k in ('entry','sl','tp')):
            entry = float(vd.get('entry'))
            sl = float(vd.get('sl'))
            tp = float(vd.get('tp'))
            lev = vd.get('lev')
            risk_pct = vd.get('risk_pct')
            rr = vd.get('rr')
            sl_delta = ((sl-entry)/entry)*100 if entry else 0.0
            tp_delta = ((tp-entry)/entry)*100 if entry else 0.0
            msg += (
                f"Entry  {entry:.4f}\n"
                f"SL     {sl:.4f} ({sl_delta:+.2f}%)\n"
                f"TP     {tp:.4f} ({tp_delta:+.2f}%)\n"
            )
            tail = []
            if lev is not None:
                try: tail.append(f"Lev {float(lev):.1f}x")
                except Exception: pass
            if risk_pct is not None:
                try: tail.append(f"Risk {float(risk_pct):.2f}%")
                except Exception: pass
            if rr is not None:
                try: tail.append(f"RR {float(rr):.2f}")
                except Exception: pass
            if tail:
                msg += " | ".join(tail) + "\n"
    except Exception:
        pass

    # Sub‑signals single line
    subs = []
    for sub in getattr(decision, 'subsignals', []) or []:
        try:
            s_icon = '🟢' if sub.direction == 'LONG' else '🔴' if sub.direction == 'SHORT' else '⚪'
            subs.append(f"{s_icon}{sub.strategy_id}({float(getattr(sub,'confidence_calibrated',0.0)):.2f})")
        except Exception:
            continue
    if subs:
        msg += "Signals: " + ", ".join(subs) + "\n"

    # Reason for flat (only if explicitly asked for blocked debug)
    if is_canary_debug and decision.decision == 'FLAT':
        reason = vd.get('reason')
        if reason:
            msg += f"Reason: {reason}\n"

    # Plain text return (no Markdown parse mode to maximize readability)
    return msg


async def send_message(text: str, settings: Dict):
    """
    Sends a message to the configured Telegram chat with bounded retries.

    Args:
        text: The message content (Markdown formatted).
        settings: The `transport` section of the global settings.
    """
    # --- DRY RUN support (from your docstring) -------------------------------
    # If settings['telegram'].dry_run is True, don't actually send—just log.
    tg_settings = settings['telegram']
    dry_run = bool(tg_settings.get('dry_run', False))
    if dry_run:
        logger.info("[Telegram dry-run] Message not sent:\n" + text)
        return
    # ------------------------------------------------------------------------

    if not tg_settings['enabled']:
        logger.debug("Telegram transport is disabled in settings.")
        return

    token = tg_settings.get('bot_token')
    chat_id = tg_settings.get('chat_id')

    if not token or not chat_id:
        logger.error("Telegram sending enabled, but bot_token or chat_id is missing.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Chunk if too long
    messages = _chunk_for_telegram(text)

    max_retries = 3
    base_delay = 2  # seconds

    async with httpx.AsyncClient() as client:
        for idx, chunk in enumerate(messages, start=1):
            payload = {
                "chat_id": chat_id,
                "text": chunk,  # plain text for readability
            }

            for attempt in range(max_retries):
                try:
                    response = await client.post(url, json=payload, timeout=15)

                    if response.status_code == 429:  # Rate limited
                        retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                        logger.warning(f"Rate limited by Telegram. Retrying in {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    response_data = response.json()
                    msg_id = response_data.get('result', {}).get('message_id')
                    logger.success(f"Successfully sent Telegram message part {idx}/{len(messages)} id={msg_id} to chat {chat_id}.")
                    break  # next chunk

                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"Telegram API error (attempt {attempt + 1}/{max_retries}, part {idx}/{len(messages)}): "
                        f"Status {e.response.status_code}, Response: {e.response.text[:200]}"
                    )
                except httpx.RequestError as e:
                    logger.error(f"Network error sending to Telegram (attempt {attempt + 1}/{max_retries}, part {idx}/{len(messages)}): {e}")

                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Waiting {delay}s before retrying...")
                    await asyncio.sleep(delay)
            else:
                logger.error("Failed to send message to Telegram after multiple retries.")
                return


# --- convenience wrapper ------------------------------------------------------

async def send_decision(decision: EnsembleDecision, settings: Dict):
    """
    Convenience wrapper: format the decision and send it (honors dry_run).
    Also handles sending of blocked signals for canary debugging.
    """
    tg_settings = settings.get('telegram', {})
    send_blocked = tg_settings.get('send_blocked_signals_in_canary', False)
    is_canary_debug = send_blocked

    # Standard logic: only send non-flat decisions
    if decision.decision in ("LONG", "SHORT"):
        text = format_message(decision, settings, is_canary_debug=False)
        await send_message(text, settings)
    # Canary debug logic: if enabled, send FLAT decisions that were vetoed
    elif send_blocked and decision.decision == "FLAT" and getattr(decision, "vetoes", None):
        text = format_message(decision, settings, is_canary_debug=True)
        await send_message(text, settings)
