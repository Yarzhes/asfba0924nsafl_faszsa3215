"""Advanced Position Sizer (Sprint 32)

Blends conviction, capped Kelly, drawdown scaling, and volatility targeting.
Non-intrusive: if disabled in settings or insufficient inputs, returns zero size.

All risk percentages in config are expressed as PERCENT (e.g. 0.5 == 0.5%).

Return schema:
    {
      'qty': float,                 # base units
      'risk_pct_effective': float,  # % of equity actually risked (post clamps)
      'risk_amount': float,         # $ risk at stop
      'notional': float,            # qty * price
      'stop_distance': float,       # price distance used for sizing
      'multipliers': { ... },       # conviction/meta/kelly/dd
      'clamps': { 'portfolio': bool, 'symbol': bool, 'safety': bool },
      'reasons': [str,...]          # any clamp reasons
    }

Inputs expected (dict style):
  equity: float
  price: float
  stop_distance: float (optional; if missing will derive via ATR * target_R_multiple)
  p_meta: float (0-1)
  mtc_status: CONFIRM|PARTIAL|FAIL|None
  liquidity_gate_action: ENTER|DAMPEN|VETO|None
  atr: float
  drawdown: float (0..1) current peak drawdown fraction
  open_positions: list[{'symbol':..., 'risk_amount':..., 'side': 'LONG'|'SHORT'}]

Author: Sprint 32
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math

@dataclass
class SizerResult:
    qty: float
    risk_pct_effective: float
    risk_amount: float
    notional: float
    stop_distance: float
    breakdown: Dict[str, Any]
    clamped_by_portfolio: bool
    clamped_by_symbol: bool
    clamped_by_safety: bool

class AdvancedSizer:
    def __init__(self, settings: Dict[str, Any]):
        self.cfg = (settings or {}).get('sizer', {}) or {}
        self.enabled = bool(self.cfg.get('enabled', False))
        # State for drawdown recovery staging per symbol
        self._dd_state = {}

    @staticmethod
    def _clip(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def compute(self,
                symbol: str,
                direction: str,
                price: float,
                equity: float,
                features: Dict[str, Any]) -> SizerResult:
        if not self.enabled or price <= 0 or equity <= 0:
            return SizerResult(0.0, 0.0, 0.0, 0.0, 0.0, {'disabled': True}, False, False, False)

        cfg = self.cfg
        # Base risk pct (config uses percent units)
        base_risk_pct = float(cfg.get('base_risk_pct', 0.5))
        min_risk_pct = float(cfg.get('min_risk_pct', 0.1))
        max_risk_pct = float(cfg.get('max_risk_pct', 1.25))

        # 1) Neutral risk
        risk_pct = base_risk_pct

        breakdown = {
            'base_risk_pct': base_risk_pct,
        }

        # 2) Conviction multiplier
        conv_cfg = cfg.get('conviction', {}) or {}
        p_meta = self._safe_float(features.get('p_meta'))
        conv_meta = 1.0
        if conv_cfg.get('use_meta', True) and p_meta is not None:
            anchor = float(conv_cfg.get('meta_anchor', 0.55))
            span = float(conv_cfg.get('meta_span', 0.15)) or 0.15
            conv_meta = math.exp((p_meta - anchor) / span)
            conv_meta = self._clip(conv_meta, 0.5, 1.8)
        # MTC
        mtc_status = (features.get('mtc_status') or '').upper() if isinstance(features.get('mtc_status'), str) else None
        conv_mtc = 1.0
        if conv_cfg.get('use_mtc', True):
            if mtc_status == 'CONFIRM':
                conv_mtc = float(conv_cfg.get('mtc_bonus', 1.15))
            elif mtc_status == 'PARTIAL':
                conv_mtc = float(conv_cfg.get('mtc_partial', 0.75))
            elif mtc_status == 'FAIL':
                conv_mtc = 0.0
        # Liquidity gate
        lq_action = (features.get('liquidity_gate_action') or '').upper()
        conv_liq = 1.0
        if conv_cfg.get('use_liquidity', True) and lq_action == 'DAMPEN':
            conv_liq = float(conv_cfg.get('liquidity_dampen', 0.75))
        conv_mult = conv_meta * conv_mtc * conv_liq
        conv_mult = self._clip(conv_mult, 0.5, 2.0)

        breakdown.update({'conv_meta': conv_meta, 'conv_mtc': conv_mtc, 'conv_liq': conv_liq, 'conv_mult': conv_mult})

        # 3) Kelly fraction (capped) -- default disabled for staged rollout
        k_cfg = cfg.get('kelly', {}) or {}
        kelly_mult = 1.0
        if k_cfg.get('enabled', False) and p_meta is not None:
            win_R = float(k_cfg.get('win_R', 1.0))
            loss_R = float(k_cfg.get('loss_R', 1.0))
            edge = p_meta * win_R - (1 - p_meta) * loss_R
            denom = win_R * loss_R if win_R > 0 and loss_R > 0 else 1.0
            k_full = max(0.0, edge / denom)
            cap = float(k_cfg.get('cap_fraction', 0.25))
            kelly_mult = 1.0 + min(k_full, cap)
        breakdown['kelly_mult'] = kelly_mult

        # 4) Drawdown scaler with staged recovery
        dd_cfg = cfg.get('dd_scaler', {}) or {}
        dd_mult = 1.0
        if dd_cfg.get('enabled', True):
            dd = self._safe_float(features.get('drawdown'))  # fraction (0.05 == 5%)
            if dd is not None and dd > 0:
                thresholds = dd_cfg.get('thresholds', [])
                # pick smallest mult where dd >= threshold.dd
                for th in sorted(thresholds, key=lambda x: x.get('dd')):
                    try:
                        if dd >= float(th.get('dd')):
                            dd_mult = float(th.get('mult'))
                    except Exception:
                        continue
            # Recovery logic: only step up after 'recovery_steps' consecutive calls while drawdown improving
            rec_steps = 0
            try:
                rec_steps = int(dd_cfg.get('recovery_steps', 0) or 0)
            except Exception:
                rec_steps = 0
            if rec_steps > 0:
                state = self._dd_state.setdefault(symbol, {
                    'current_mult': dd_mult,
                    'target_mult': dd_mult,
                    'progress': 0,
                    'last_dd': dd or 0.0,
                })
                state['target_mult'] = dd_mult
                if dd is not None:
                    # If drawdown worsens -> adopt lower multiplier immediately
                    if dd > state.get('last_dd', 0.0) + 1e-9 and dd_mult < state['current_mult']:
                        state['current_mult'] = dd_mult
                        state['progress'] = 0
                    # If drawdown improves and target allows higher risk -> progress towards it
                    elif state['target_mult'] > state['current_mult']:
                        state['progress'] += 1
                        if state['progress'] >= rec_steps:
                            state['current_mult'] = state['target_mult']
                            state['progress'] = 0
                    state['last_dd'] = dd
                dd_mult = state['current_mult']
        breakdown['dd_mult'] = dd_mult

        # 5) Volatility targeting & stop distance
        vol_cfg = cfg.get('vol_target', {}) or {}
        atr = self._safe_float(features.get('atr'))
        stop_distance = self._safe_float(features.get('stop_distance'))
        if stop_distance is None:
            if vol_cfg.get('method', 'atr') == 'atr' and atr and atr > 0:
                stop_distance = atr * float(vol_cfg.get('target_R_multiple', 1.0))
        # Vol floor (bps of price)
        vol_floor_bps = float(vol_cfg.get('vol_floor_bps', 20))
        floor_dist = price * (vol_floor_bps / 10_000.0)
        if stop_distance is None or stop_distance < floor_dist:
            stop_distance = floor_dist
        breakdown['stop_distance'] = stop_distance

        # Raw proposed risk dollars
        pre_clip_risk_pct = risk_pct * conv_mult * kelly_mult * dd_mult
        # clip based on min/max base band now
        pre_clip_risk_pct = self._clip(pre_clip_risk_pct, min_risk_pct, max_risk_pct)

        risk_amount = equity * (pre_clip_risk_pct / 100.0)
        qty = 0.0
        if stop_distance > 0:
            qty = risk_amount / stop_distance
        notional = qty * price

        breakdown.update({
            'risk_pct_pre_clamp': pre_clip_risk_pct,
            'risk_amount_pre_clamp': risk_amount,
        })

        # 6) Portfolio & symbol clamps
        clamped_portfolio = clamped_symbol = clamped_safety = False
        reasons: List[str] = []
        per_sym_cfg = cfg.get('per_symbol', {}) or {}
        sym_cap = float(per_sym_cfg.get('max_risk_pct', 0.75))
        if pre_clip_risk_pct > sym_cap:
            scale = sym_cap / pre_clip_risk_pct if pre_clip_risk_pct > 0 else 0.0
            qty *= scale; risk_amount *= scale; notional *= scale
            pre_clip_risk_pct = sym_cap
            clamped_symbol = True
            reasons.append('PER_SYMBOL_CAP')

        port_cfg = cfg.get('portfolio', {}) or {}
        open_positions: List[Dict[str, Any]] = features.get('open_positions') or []
        max_gross = float(port_cfg.get('max_gross_risk_pct', 2.0)) / 100.0 * equity
        gross_after = risk_amount + sum(float(p.get('risk_amount', 0)) for p in open_positions)
        if gross_after > max_gross:
            # scale down so total == max_gross
            allowable = max(0.0, max_gross - sum(float(p.get('risk_amount', 0)) for p in open_positions))
            if allowable <= 0:
                qty = 0.0; risk_amount = 0.0; notional = 0.0; pre_clip_risk_pct = 0.0
            else:
                scale = allowable / risk_amount if risk_amount > 0 else 0.0
                qty *= scale; risk_amount = allowable; notional *= scale
                pre_clip_risk_pct *= scale
            clamped_portfolio = True
            reasons.append('PORTFOLIO_GROSS_CAP')

        # Net exposure clamp (approx via side counts)
        max_net_long_pct = float(port_cfg.get('max_net_long_pct', 3.0))
        max_net_short_pct = float(port_cfg.get('max_net_short_pct', 3.0))
        net_long = sum(p.get('risk_amount', 0) for p in open_positions if p.get('side') == 'LONG')
        net_short = sum(p.get('risk_amount', 0) for p in open_positions if p.get('side') == 'SHORT')
        if direction == 'LONG':
            if (net_long + risk_amount) > (max_net_long_pct/100.0)*equity:
                clamped_portfolio = True; reasons.append('NET_LONG_CAP'); qty=0; risk_amount=0; notional=0; pre_clip_risk_pct=0
        else:
            if (net_short + risk_amount) > (max_net_short_pct/100.0)*equity:
                clamped_portfolio = True; reasons.append('NET_SHORT_CAP'); qty=0; risk_amount=0; notional=0; pre_clip_risk_pct=0

        # 7) Safety constraints
        safety_cfg = cfg.get('safety', {}) or {}
        min_notional = float(safety_cfg.get('min_notional', 25))
        if notional < min_notional:
            qty = 0.0; risk_amount = 0.0; notional = 0.0; pre_clip_risk_pct = 0.0
            clamped_safety = True; reasons.append('MIN_NOTIONAL')

        # 8) Rounding
        rnd_cfg = cfg.get('rounding', {}) or {}
        step = float(rnd_cfg.get('step_size', 0.0001))
        if step > 0 and qty > 0:
            qty = math.floor(qty / step) * step
            notional = qty * price

        breakdown['reasons'] = reasons
        breakdown['risk_pct_effective'] = pre_clip_risk_pct
        breakdown['risk_amount'] = risk_amount
        breakdown['notional'] = notional
        breakdown['qty'] = qty

        # 9) Sprint 40 Sentiment dampen (applied after core clamps)
        try:
            sent = features.get('sentiment') if isinstance(features, dict) else None
            if isinstance(sent, dict):
                # Expect pre-computed size modifier included by engine; else check flags
                size_mod = sent.get('size_modifier')
                if size_mod is None:
                    # Fallback: crude dampen if extreme flags present
                    if sent.get('extreme_flag_bull') or sent.get('extreme_flag_bear'):
                        size_mod = 0.5
                if size_mod is not None and 0 < float(size_mod) < 1.0 and qty > 0:
                    sm = float(size_mod)
                    qty *= sm; risk_amount *= sm; notional *= sm; pre_clip_risk_pct *= sm
                    breakdown['sentiment_size_mod'] = sm
        except Exception:
            pass

        return SizerResult(qty, pre_clip_risk_pct, risk_amount, notional, stop_distance, breakdown,
                           clamped_portfolio, clamped_symbol, clamped_safety)

    @staticmethod
    def _safe_float(x) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None
