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

def _escape_markdown_v2(text: str) -> str:
    """
    Escape characters required by Telegram MarkdownV2.
    Keep '*' and '`' unescaped since we intentionally use them for emphasis/monospace.
    """
    escape_chars = r"_[]()~>#+-=|{}.!"
    for ch in escape_chars:
        text = text.replace(ch, f"\\{ch}")
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
    # Choose icon, including neutral for FLAT/abstain
    if decision.decision == "LONG":
        icon = "📈"
    elif decision.decision == "SHORT":
        icon = "📉"
    else:
        icon = "⚪"

    # Message header
    header_title = f"New Ensemble Decision: {decision.decision} {decision.symbol}"
    msg = (
        f"{icon} *{header_title}* ({decision.tf})\n\n"
        f"Ensemble Confidence: *{decision.confidence:.2%}*\n"
    )

    # Vote details (includes required Wgt Sum line)
    if getattr(decision, "vote_detail", None):
        vd = decision.vote_detail or {}
        # Required fields per Sprint 8 spec
        agree = vd.get("agree", 0)
        total = vd.get("total", 0)
        profile = vd.get("profile", "n/a")
        weighted_sum = float(vd.get("weighted_sum", 0.0))

        msg += (
            f"Vote: `{agree}/{total}` "
            f"| Profile: `{profile}` "
            f"| Wgt Sum: `{weighted_sum:.3f}`\n"
        )

        # Compact pre-trade summary line (if set by live runner)
        try:
            pre = vd.get('pre_trade') if isinstance(vd, dict) else None
            if pre and isinstance(pre, dict):
                pwin = pre.get('p_win')
                regime = pre.get('regime') or 'n/a'
                veto_ct = pre.get('veto_count', 0)
                lat = pre.get('lat_ms') or {}
                p50 = lat.get('p50') if isinstance(lat, dict) else None
                p90 = lat.get('p90') if isinstance(lat, dict) else None
                if pwin is not None:
                    msg += f"PRE: p={pwin:.2f} | reg={regime} | veto={veto_ct} | lat_p50={p50:.1f}ms p90={p90:.1f}ms\n"
                else:
                    msg += f"PRE: reg={regime} | veto={veto_ct}\n"
        except Exception:
            pass

        # If the ensemble abstained (FLAT), include abstain reason if present
        reason = vd.get("reason")
        if decision.decision == "FLAT" and reason:
            msg += f"Reason: `{str(reason)}`\n"

    # Vetoes (explicit top reason line required) + full list
    msg += _format_veto_block(decision)

    # Optional Sprint-9 filter diagnostics (if your engine supplies them)
    msg += _format_filter_details(decision)

    msg += f"--------------------------------------\n"

    # Sub-signal breakdown (format like: 🟢 strat_A_long (0.80))
    msg += "*Contributing Signals:*\n"
    for sub in getattr(decision, "subsignals", []) or []:
        sub_icon = "🟢" if sub.direction == "LONG" else "🔴" if sub.direction == "SHORT" else "⚪"
        # Keep plain (no backticks) to match the expected test string:
        # "🟢 strat_A_long (0.80)"
        # But we still sanitize visually dangerous characters lightly.
        strategy_id_safe = (
            str(sub.strategy_id)
            .replace('*', '')
            .replace('[', '')
            .replace(']', '')
        )
        conf = getattr(sub, "confidence_calibrated", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        msg += f"{sub_icon} {strategy_id_safe} ({conf_f:.2f})\n"

    # Escape for Telegram MarkdownV2 (keep * and ` intact)
    msg = _escape_markdown_v2(msg)

    # Append advanced sizer compact line (added post-escape to avoid double escaping multipliers formatting)
    try:
        adv = None
        if getattr(decision, 'vote_detail', None):
            adv = decision.vote_detail.get('advanced_sizer') if isinstance(decision.vote_detail, dict) else None
        if adv and isinstance(adv, dict) and decision.decision in ("LONG","SHORT"):
            parts = []
            rp = adv.get('risk_pct_effective')
            if rp is not None:
                parts.append(f"Sz={float(rp):.2f}%")
            for k,label in [('conv_meta','Meta'),('conv_mtc','MTC'),('dd_mult','DD'),('kelly_mult','Kelly')]:
                v = adv.get(k)
                try:
                    if v is not None:
                        parts.append(f"{label}={float(v):.2f}")
                except Exception:
                    continue
            line = " " + _escape_markdown_v2(" • ".join(parts)) + "\n"
            msg += line
    except Exception:
        pass

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
                "text": chunk,
                "parse_mode": "MarkdownV2",
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
    """
    text = format_message(decision, settings)
    await send_message(text, settings)
