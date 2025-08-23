import re
from ultra_signals.core.custom_types import EnsembleDecision

def _esc_md(s: str) -> str:
    # Escapes characters for Telegram MarkdownV2.
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", s)

def format_message(decision: EnsembleDecision, opts: dict = {}) -> str:
    """Formats an EnsembleDecision into a MarkdownV2-compatible Telegram message."""
    ws = decision.vote_detail.get("weighted_sum", 0.0)
    agree = decision.vote_detail.get("agree", 0)
    total = decision.vote_detail.get("total", 0)
    profile = decision.vote_detail.get("profile", "default")

    header_icon = "📈" if decision.decision == "LONG" else "📉" if decision.decision == "SHORT" else "FLAT"
    title = f"{header_icon} *New Ensemble Decision: {decision.decision} {_esc_md(decision.symbol)}* \\({_esc_md(decision.tf)}\\)"
    
    conf_pct = decision.confidence * 100
    conf = f"Ensemble Confidence: *{conf_pct:.2f}%*"
    
    vote = f"Vote: `{agree}/{total}` \\| Profile: `{profile}` \\| Wgt Sum: `{ws:.3f}`"

    parts = [title, "", conf, vote]

    if decision.vetoes:
        veto_reason = _esc_md(decision.vetoes[0])
        parts.append(f"*VETOED* — Top reason: `{veto_reason}`")
        
    parts.append("----") # MarkdownV2 horizontal rule
    parts.append("*Contributing Signals:*")
    
    for sub in decision.subsignals:
        direction_icon = "🟢" if sub.direction == "LONG" else "🔴" if sub.direction == "SHORT" else "⚪"
        strat_id = _esc_md(f"{sub.strategy_id}_{sub.direction.lower()}")
        conf_cal = sub.confidence_calibrated
        parts.append(f"{direction_icon} `{strat_id}` \\({conf_cal:.2f}\\)")

    return "\n".join(parts)