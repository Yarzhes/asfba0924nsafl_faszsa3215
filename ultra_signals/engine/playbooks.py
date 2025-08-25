from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

@dataclass
class EntryRule:
    min_conf: float
    min_adx: Optional[float] = None
    ema_sep_atr_min: Optional[float] = None
    rsi_extreme: Optional[Tuple[float, float]] = None  # (low, high)
    of_confirm: Optional[Dict[str, Any]] = None        # e.g. {"cvd": 0.05}

@dataclass
class ExitRule:
    stop_atr_mult: float
    tp_atr_mults: List[float]
    trail_after_rr: Optional[float] = None
    timeout_bars: Optional[int] = None
    max_hold_bars: Optional[int] = None

@dataclass
class RiskRule:
    size_scale: float = 1.0
    cooldown_bars: int = 0
    rr_min: Optional[float] = 1.2

@dataclass
class Playbook:
    name: str
    entry: EntryRule
    exit: ExitRule
    risk: RiskRule
    abstain: bool = False  # for chop flat

# --- Factory helpers ---------------------------------------------------------

def make_trend_breakout(cfg: Dict[str, Any]) -> Playbook:
    c = cfg or {}
    entry_c = c.get('entry', {})
    exit_c = c.get('exit', {})
    risk_c = c.get('risk', {})
    return Playbook(
        name='trend_breakout',
        entry=EntryRule(
            min_conf=float(entry_c.get('min_conf', 0.62)),
            min_adx=entry_c.get('min_adx'),
            ema_sep_atr_min=entry_c.get('ema_sep_atr_min'),
            rsi_extreme=tuple(entry_c.get('rsi_extreme')) if entry_c.get('rsi_extreme') else None,
            of_confirm=entry_c.get('of_confirm')
        ),
        exit=ExitRule(
            stop_atr_mult=float(exit_c.get('stop_atr_mult', 1.4)),
            tp_atr_mults=list(exit_c.get('tp_atr_mults', [1.8, 2.6, 3.4])),
            trail_after_rr=exit_c.get('trail_after_rr'),
            timeout_bars=exit_c.get('timeout_bars'),
        ),
        risk=RiskRule(
            size_scale=float(risk_c.get('size_scale', 1.15)),
            cooldown_bars=int(risk_c.get('cooldown_bars', 6)),
            rr_min=risk_c.get('rr_min', 1.4)
        )
    )

def make_trend_pullback(cfg: Dict[str, Any]) -> Playbook:
    c = cfg or {}
    entry_c = c.get('entry', {})
    exit_c = c.get('exit', {})
    risk_c = c.get('risk', {})
    return Playbook(
        name='trend_pullback',
        entry=EntryRule(
            min_conf=float(entry_c.get('min_conf', 0.60)),
            min_adx=entry_c.get('min_adx'),
            ema_sep_atr_min=entry_c.get('ema_sep_atr_min'),
            rsi_extreme=tuple(entry_c.get('rsi_extreme')) if entry_c.get('rsi_extreme') else None,
            of_confirm=entry_c.get('of_confirm')
        ),
        exit=ExitRule(
            stop_atr_mult=float(exit_c.get('stop_atr_mult', 1.2)),
            tp_atr_mults=list(exit_c.get('tp_atr_mults', [1.6, 2.2, 3.0])),
            trail_after_rr=exit_c.get('trail_after_rr'),
            timeout_bars=exit_c.get('timeout_bars'),
        ),
        risk=RiskRule(
            size_scale=float(risk_c.get('size_scale', 1.05)),
            cooldown_bars=int(risk_c.get('cooldown_bars', 6)),
            rr_min=risk_c.get('rr_min', 1.3)
        )
    )

def make_mr_bollinger_fade(cfg: Dict[str, Any]) -> Playbook:
    c = cfg or {}
    entry_c = c.get('entry', {})
    exit_c = c.get('exit', {})
    risk_c = c.get('risk', {})
    return Playbook(
        name='mr_bb_fade',
        entry=EntryRule(
            min_conf=float(entry_c.get('min_conf', 0.58)),
            rsi_extreme=tuple(entry_c.get('rsi_extreme')) if entry_c.get('rsi_extreme') else None,
            of_confirm=entry_c.get('of_confirm')
        ),
        exit=ExitRule(
            stop_atr_mult=float(exit_c.get('stop_atr_mult', 1.0)),
            tp_atr_mults=list(exit_c.get('tp_atr_mults', [1.2, 1.8, 2.4])),
            trail_after_rr=exit_c.get('trail_after_rr'),
            timeout_bars=exit_c.get('timeout_bars'),
        ),
        risk=RiskRule(
            size_scale=float(risk_c.get('size_scale', 0.9)),
            cooldown_bars=int(risk_c.get('cooldown_bars', 5)),
            rr_min=risk_c.get('rr_min', 1.2)
        )
    )

def make_mr_vwap_revert(cfg: Dict[str, Any]) -> Playbook:
    c = cfg or {}
    entry_c = c.get('entry', {})
    exit_c = c.get('exit', {})
    risk_c = c.get('risk', {})
    return Playbook(
        name='mr_vwap_revert',
        entry=EntryRule(
            min_conf=float(entry_c.get('min_conf', 0.60)),
            of_confirm=entry_c.get('of_confirm')
        ),
        exit=ExitRule(
            stop_atr_mult=float(exit_c.get('stop_atr_mult', 1.1)),
            tp_atr_mults=list(exit_c.get('tp_atr_mults', [1.4, 2.0])),
            timeout_bars=exit_c.get('timeout_bars'),
        ),
        risk=RiskRule(
            size_scale=float(risk_c.get('size_scale', 0.95)),
            cooldown_bars=int(risk_c.get('cooldown_bars', 5)),
            rr_min=risk_c.get('rr_min', 1.25)
        )
    )

def make_chop_flat(cfg: Dict[str, Any]) -> Playbook:
    return Playbook(
        name='chop_flat',
        entry=EntryRule(min_conf=1e9),  # impossible
        exit=ExitRule(stop_atr_mult=0, tp_atr_mults=[]),
        risk=RiskRule(size_scale=0.0, cooldown_bars=int((cfg or {}).get('risk', {}).get('cooldown_bars', 0)), rr_min=None),
        abstain=True
    )

# Registry helper
FACTORY_MAP = {
    'trend.breakout': make_trend_breakout,
    'trend.pullback': make_trend_pullback,
    'mean_revert.bb_fade': make_mr_bollinger_fade,
    'mean_revert.vwap_revert': make_mr_vwap_revert,
    'chop.flat': make_chop_flat,
}
