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
    prof_meta = decision.vote_detail.get("profile") if isinstance(decision.vote_detail, dict) else None
    if isinstance(prof_meta, dict):
        pid = prof_meta.get('profile_id') or 'trend'
        profile = str(pid)
    elif isinstance(prof_meta, str):
        profile = prof_meta
    else:
        profile = "trend"

    header_icon = "📈" if decision.decision == "LONG" else "📉" if decision.decision == "SHORT" else "FLAT"
    title = f"{header_icon} *New Ensemble Decision: {decision.decision} {_esc_md(decision.symbol)}* \\({_esc_md(decision.tf)}\\)"
    
    conf_pct = decision.confidence * 100
    conf = f"Ensemble Confidence: *{conf_pct:.2f}%*"
    
    vote = f"Vote: `{agree}/{total}` \\| Profile: `{_esc_md(profile)}` \\| Wgt Sum: `{ws:.3f}`"

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

    # Sprint 18 Quality Gates summary line
    q = decision.vote_detail.get("quality") if isinstance(decision.vote_detail, dict) else None
    if q:
        bin_ = _esc_md(str(q.get('bin')))
        qscore = float(q.get('qscore', 0.0))
        vetoes = q.get('vetoes') or []
        soft = q.get('soft_flags') or []
        if vetoes:
            parts.append(f"Quality: *VETO* `{','.join([_esc_md(v) for v in vetoes])}` q={qscore:.2f}→{bin_}")
        else:
            mult = q.get('size_multiplier')
            if mult:
                parts.append(f"Quality: {bin_} q={qscore:.2f} Size×{mult}")
            else:
                parts.append(f"Quality: {bin_} q={qscore:.2f}")
        if soft:
            parts.append(f"Soft Gates: `{_esc_md(','.join(soft))}`")

    # Profile config footer (expected by tests): cfg=profile_id@version
    if isinstance(prof_meta, dict):
        vid = prof_meta.get('version')
        if vid:
            parts.append(f"cfg={_esc_md(profile)}@{_esc_md(str(vid))}")

    return "\n".join(parts)