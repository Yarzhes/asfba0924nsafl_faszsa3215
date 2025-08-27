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

    # Sprint 30: MTC line (compact)
    try:
        mtc = decision.vote_detail.get('mtc_gate') if isinstance(decision.vote_detail, dict) else None
        if mtc:
            st = mtc.get('status')
            act = mtc.get('action')
            sc = mtc.get('scores') or {}
            c1 = sc.get('C1'); c2 = sc.get('C2')
            rs = mtc.get('reasons') or []
            # pick first non-stale/missing reason containing OK/WEAK/FAIL for context
            top_reason = None
            for r in rs:
                if any(x in r for x in ['TREND','MOM','VOL','STRUCT']):
                    top_reason = r
                    break
            if top_reason is None and rs:
                top_reason = rs[0]
            line = f"MTC: {st}"
            if c1 is not None or c2 is not None:
                line += f" C1={c1:.2f if c1 is not None else ''} C2={c2:.2f if c2 is not None else ''}" if c1 is not None and c2 is not None else ""
                if c1 is not None and c2 is None:
                    line += f" C1={c1:.2f}"
                if c2 is not None and c1 is None:
                    line += f" C2={c2:.2f}"
            if mtc.get('observe_only'):
                line += " (OBSERVE)"
            if top_reason:
                line += f" `{_esc_md(top_reason)}`"
            parts.append(line)
    except Exception:
        pass

    if decision.vetoes:
        veto_reason = _esc_md(decision.vetoes[0])
        parts.append(f"*VETOED* — Top reason: `{veto_reason}`")
    else:
        # Liquidity gate dampen badge (non-veto) for rapid situational awareness
        try:
            lq = decision.vote_detail.get("liquidity_gate") if isinstance(decision.vote_detail, dict) else None
            if lq and lq.get("action") == "DAMPEN":
                parts.append(f"⚠️ LQ DAMPEN size×{float(lq.get('size_mult') or 1.0):.2f} stop×{float(lq.get('widen_stop_mult') or 1.0):.2f}")
        except Exception:
            pass
        
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
            parts.append(f"VETO: {','.join([_esc_md(v) for v in vetoes])} | q={qscore:.2f} → {bin_}")
        else:
            mult = q.get('size_multiplier')
            if mult:
                gate_txt = "Gates: OK" if not soft else f"Gates: {len(soft)} soft"
                parts.append(f"Quality: {bin_} (q={qscore:.2f}) | {gate_txt} | Size×{mult}")
            else:
                gate_txt = "Gates: OK" if not soft else f"Gates: {len(soft)} soft"
                parts.append(f"Quality: {bin_} (q={qscore:.2f}) | {gate_txt}")
        if soft:
            parts.append(f"Soft Gates: `{_esc_md(','.join(soft))}`")

    # Sprint 45 Behavior context one-liner
    try:
        beh = decision.vote_detail.get('behavior') if isinstance(decision.vote_detail, dict) else None
        if isinstance(beh, dict):
            act = beh.get('behavior_action') or ('VETO' if beh.get('behavior_veto') else None)
            fomo = beh.get('beh_fomo_score_z'); eup = beh.get('beh_euphoria_score_z'); cap = beh.get('beh_capitulation_score_z')
            mult = None
            # choose layering precedence: explicit behavior_size_mult else top-level scaling record
            rm = decision.vote_detail.get('risk_model') if isinstance(decision.vote_detail, dict) else None
            if rm and rm.get('behavior_size_mult'):
                mult = rm.get('behavior_size_mult')
            elif beh.get('behavior_size_mult'):
                mult = beh.get('behavior_size_mult')
            flags = beh.get('flags') or []
            pieces = ['Behav:']
            if act:
                pieces.append(act)
            def _fmt(tag,val):
                try:
                    return f"{tag}{float(val):.1f}σ"
                except Exception:
                    return None
            for tag,val in [('F',fomo),('E',eup),('C',cap)]:
                fv = _fmt(tag,val)
                if fv:
                    pieces.append(fv)
            if mult and mult != 1 and mult is not None:
                pieces.append(f"Size×{float(mult):.2f}")
            if flags:
                pieces.append('/'.join(flags[:3]))
            parts.append(_esc_md(' '.join(pieces)))
    except Exception:
        pass

    # Profile config footer (expected by tests): cfg=profile_id@version
    if isinstance(prof_meta, dict):
        vid = prof_meta.get('version')
        if vid:
            parts.append(f"cfg={_esc_md(profile)}@{_esc_md(str(vid))}")

    # Sprint 40 Sentiment context line (if sentiment snapshot embedded in vote_detail)
    try:
        sent = decision.vote_detail.get('sentiment') if isinstance(decision.vote_detail, dict) else None
        if isinstance(sent, dict):
            sc = sent.get('sent_score_s'); z = sent.get('sent_z_s'); fg = sent.get('fg_index'); fund = sent.get('funding_z')
            dir_arrow = ''
            if z is not None:
                try:
                    if float(z) >= 0.75:
                        dir_arrow = '↑'
                    elif float(z) <= -0.75:
                        dir_arrow = '↓'
                except Exception:
                    pass
            line = 'Sentiment:'
            if sc is not None:
                line += f" {float(sc):+.2f}{dir_arrow}"
            if fg is not None:
                line += f" | F&G {int(fg)}"
            if fund is not None:
                try:
                    line += f" | Funding {float(fund):+.2f}σ"
                except Exception:
                    pass
            if z is not None:
                line += f" | z={float(z):+.2f}"
            if sent.get('extreme_flag_bull'):
                line += ' 🟢🔥'
            if sent.get('extreme_flag_bear'):
                line += ' 🔴⚠️'
            parts.append(_esc_md(line))
    except Exception:
        pass

    # Sprint 41 Whale / Smart Money context line (expects aggregated whale snapshot under vote_detail['whales'])
    try:
        wf = decision.vote_detail.get('whales') if isinstance(decision.vote_detail, dict) else None
        if isinstance(wf, dict):
            net24 = wf.get('whale_net_inflow_usd_l')
            block_z = wf.get('block_trade_notional_p99_z')
            opt_oi = wf.get('opt_oi_delta_1h_z')
            sm_press = wf.get('composite_pressure_score')
            parts.append(_esc_md(_fmt_whale_line(net24, block_z, opt_oi, sm_press)))
    except Exception:
        pass

    # Sprint 42 Macro / Cross-Asset context line
    try:
        macro = decision.vote_detail.get('macro') if isinstance(decision.vote_detail, dict) else None
        if isinstance(macro, dict):
            reg = macro.get('macro_risk_regime')
            risk_on = macro.get('risk_on_prob'); risk_off = macro.get('risk_off_prob')
            corr = macro.get('btc_spy_corr_1d') or macro.get('btc_spy_corr_4h')
            dxy_flag = macro.get('dxy_surge_flag'); vix_z = macro.get('btc_vix_proxy_z')
            cu = macro.get('carry_unwind_flag')
            pieces = []
            if reg:
                rop = (risk_on or 0.0) * 100
                pieces.append(f"Macro: {reg} ({rop:.0f}% RO)")
            if corr is not None:
                try:
                    arrow = '↑' if corr > 0 else '↓'
                    pieces.append(f"BTC-SPY {float(corr):+.2f}{arrow}")
                except Exception:
                    pass
            if dxy_flag:
                pieces.append('DXY⚡')
            if vix_z is not None:
                try:
                    pieces.append(f"BTC-VIX {float(vix_z):+.1f}σ")
                except Exception:
                    pass
            if cu:
                pieces.append('CarryUnwind!')
            if pieces:
                parts.append(_esc_md(' | '.join(pieces)))
    except Exception:
        pass

    # Sprint 43 Regime summary (if regime detail attached to vote_detail['regime'])
    try:
        reg = decision.vote_detail.get('regime') if isinstance(decision.vote_detail, dict) else None
        if isinstance(reg, dict):
            label = reg.get('regime_label') or reg.get('label')
            probs = reg.get('regime_probs') or {}
            haz = reg.get('transition_hazard')
            expv = reg.get('exp_vol_h')
            dirb = reg.get('dir_bias')
            if probs and not label:
                try:
                    label = max(probs.items(), key=lambda kv: kv[1])[0]
                except Exception:
                    pass
            if label or probs:
                top_prob = None
                if probs and label in probs:
                    top_prob = probs[label]
                pieces = []
                if label:
                    if top_prob is not None:
                        pieces.append(f"Regime: {label} {top_prob*100:.0f}%")
                    else:
                        pieces.append(f"Regime: {label}")
                if haz is not None:
                    try:
                        pieces.append(f"Haz {float(haz)*100:.0f}%")
                    except Exception:
                        pass
                if expv is not None:
                    try:
                        pieces.append(f"ExpVol {float(expv):.2f}")
                    except Exception:
                        pass
                if dirb is not None:
                    try:
                        arrow = '↗' if dirb>0.15 else '↘' if dirb<-0.15 else '→'
                        pieces.append(f"Dir {arrow}")
                    except Exception:
                        pass
                if pieces:
                    parts.append(_esc_md(' | '.join(pieces)))
    except Exception:
        pass

    return "\n".join(parts)


def _fmt_whale_line(net24, block_z, opt_oi, sm_press) -> str:
    """Compact whale summary line.

    Examples:
      Whales: -$220M exch inflow, Blocks +2.4σ sell, OI +1.1σ, SM +3.2
    """
    out = 'Whales:'
    try:
        if net24 is not None:
            try:
                val = float(net24)
                sign = '+' if val >= 0 else ''
                out += f" {sign}${val/1_000_000:.0f}M net"
            except Exception:
                pass
        if block_z is not None:
            out += f", Blocks {float(block_z):+.1f}σ"
        if opt_oi is not None:
            out += f", OI {float(opt_oi):+.1f}σ"
        if sm_press is not None:
            out += f", SM {float(sm_press):+.1f}"
    except Exception:
        pass
    return out