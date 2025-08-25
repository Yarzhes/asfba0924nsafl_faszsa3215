"""Sprint 19: Composite Objective for Auto-Calibration

Provides walk-forward evaluation for a candidate parameter set.
Simplified skeleton: integrates existing WalkForwardAnalysis, computes
aggregate metrics, and returns fitness scalar + detail dict.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List
from loguru import logger
import math
import pandas as pd
from ultra_signals.backtest.walkforward import WalkForwardAnalysis
from ultra_signals.backtest.data_adapter import DataAdapter
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.engine.real_engine import RealSignalEngine

# --- Metric helpers ---------------------------------------------------------

def _profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gains = trades[trades.pnl > 0].pnl.sum()
    losses = abs(trades[trades.pnl < 0].pnl.sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses

def _winrate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return (trades.pnl > 0).mean()

def _sharpe(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    daily = trades.groupby(trades.ts_exit // 86400).pnl.sum()
    if daily.std(ddof=0) == 0:
        return 0.0
    return (daily.mean() / (daily.std(ddof=0) + 1e-9)) * math.sqrt(365)

def _max_drawdown(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    eq = trades.pnl.cumsum()
    roll_max = eq.cummax()
    drawdown = (eq - roll_max)
    return drawdown.min()

# ---------------------------------------------------------------------------

def apply_params(base_settings: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied settings dict with candidate params applied.
    Keys are dot-form group.param inside search space; route to appropriate
    nested config paths (simplified mapping for key groups used in Sprint 19)."""
    import copy
    s = copy.deepcopy(base_settings)
    for full_key, val in params.items():
        group, key = full_key.split('.', 1)
        if group == 'ensemble':
            ens = s.setdefault('ensemble', {})
            # map custom names
            mapping = {
                'vote_threshold_trend': ('vote_threshold', 'trend'),
                'vote_threshold_mr': ('vote_threshold', 'mean_revert'),
                'confidence_floor': ('confidence_floor', None),
                'margin_of_victory': ('margin_of_victory', None),
                'min_agree_trend': ('min_agree', 'trend'),
                'min_agree_mr': ('min_agree', 'mean_revert'),
            }
            if key in mapping:
                base_key, sub = mapping[key]
                if sub:
                    ens.setdefault(base_key, {})[sub] = val
                else:
                    ens[base_key] = val
        elif group.startswith('playbooks_trend_breakout'):
            pb = s.setdefault('playbooks', {}).setdefault('trend', {}).setdefault('breakout', {})
            entry = pb.setdefault('entry', {})
            risk = pb.setdefault('exit', {})
            risk2 = pb.setdefault('risk', {})
            if key == 'min_adx': entry['min_adx'] = val
            elif key == 'ema_sep_atr_min': entry['ema_sep_atr_min'] = val
            elif key == 'stop_atr_mult': risk['stop_atr_mult'] = val
            elif key == 'tp1_atr_mult': 
                # adjust first tp
                tps = risk.get('tp_atr_mults', [1.8,2.6,3.4])
                if tps: tps[0] = val
                risk['tp_atr_mults'] = tps
            elif key == 'rr_min': risk2['rr_min'] = val
        elif group.startswith('playbooks_mr_fade'):
            pb = s.setdefault('playbooks', {}).setdefault('mean_revert', {}).setdefault('bb_fade', {})
            entry = pb.setdefault('entry', {})
            risk = pb.setdefault('exit', {})
            risk2 = pb.setdefault('risk', {})
            if key == 'rsi_low': entry.setdefault('rsi_extreme', [28,72])[0] = val
            elif key == 'rsi_high': entry.setdefault('rsi_extreme', [28,72])[1] = val
            elif key == 'stop_atr_mult': risk['stop_atr_mult'] = val
            elif key == 'tp1_atr_mult': 
                tps = risk.get('tp_atr_mults', [1.2,1.8,2.4])
                if tps: tps[0] = val
                risk['tp_atr_mults'] = tps
            elif key == 'rr_min': risk2['rr_min'] = val
        elif group == 'quality_gates':
            q = s.setdefault('quality_gates', {})
            bins = q.setdefault('qscore_bins', {})
            if key == 'q_Aplus': bins['Aplus'] = val
            elif key == 'q_A': bins['A'] = val
            elif key == 'q_C': bins['C'] = val
            elif key == 'max_spread_pct': q.setdefault('veto', {})['max_spread_pct'] = val
            elif key == 'atr_pct_limit': q.setdefault('veto', {})['atr_pct_limit'] = val
        elif group == 'sizing':
            pos = s.setdefault('position_sizing', {})
            if key == 'base_risk_pct': s.setdefault('risk_model', {})['base_risk_pct'] = val
            elif key == 'trend_size_scale': pos['trend_size_scale'] = val
            elif key == 'mr_size_scale': pos['mr_size_scale'] = val
            elif key == 'liq_risk_weight': pos['liq_risk_weight'] = val
        elif group == 'orderflow':
            of = s.setdefault('orderflow', {})
            mapping = {
                'cvd_weight': 'cvd_weight',
                'liquidation_weight': 'liquidation_weight',
                'sweep_weight': 'liquidity_sweep_weight'
            }
            if key in mapping:
                of[mapping[key]] = val
        elif group == 'slippage':
            sl = s.setdefault('backtest', {}).setdefault('execution', {})
            if key == 'atr_multiplier': sl['slippage_atr_multiplier'] = val
    return s

# ---------------------------------------------------------------------------

def evaluate_candidate(settings_dict: Dict[str, Any], wf_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run a walk-forward evaluation returning composite + per-fold metrics.

    We treat the returned trades from each fold as OOS performance. We do not
    currently simulate in-sample (train) performance separately; instead we
    approximate an *in-sample* profit factor for overfit-gap penalty using the
    max fold PF (aggressive upper bound). This still penalizes volatile trials
    where one fold dominates others (proxy for potential overfit).
    """
    adapter = DataAdapter(settings_dict)
    warmup = int(settings_dict.get('features', {}).get('warmup_periods', 100))
    fs = FeatureStore(warmup_periods=warmup, settings=settings_dict)

    def engine_factory():
        return RealSignalEngine(settings_dict, fs)

    wf = WalkForwardAnalysis(settings_dict, adapter, engine_factory)
    symbol = settings_dict.get('runtime', {}).get('symbols', ['BTCUSDT'])[0]
    tf = settings_dict.get('runtime', {}).get('primary_timeframe', '5m')

    try:
        trades_df, kpi_summary = wf.run(symbol, tf)
    except Exception as e:  # be robust inside optimization loop
        logger.exception(f"WF failure for candidate: {e}")
        trades_df, kpi_summary = pd.DataFrame(), pd.DataFrame()

    if trades_df is None or not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame(columns=['pnl', 'ts_exit'])

    # Per-fold PF list (OOS) from KPI summary if available
    fold_pfs: List[float] = []
    if isinstance(kpi_summary, pd.DataFrame) and 'profit_factor' in kpi_summary.columns:
        fold_pfs = [float(x) for x in kpi_summary['profit_factor'].fillna(0.0).tolist()]

    # Stability: 1 - stdev(PF) (bounded to [0,1] assuming PF stdev rarely >1; clamp)
    if fold_pfs:
        pf_std = float(pd.Series(fold_pfs).std(ddof=0))
        oos_stability = max(0.0, 1.0 - pf_std)
    else:
        oos_stability = 0.0

    # Aggregate metrics from combined trades
    metrics = {
        'profit_factor': _profit_factor(trades_df),
        'winrate': _winrate(trades_df),
        'sharpe': _sharpe(trades_df),
        'max_drawdown': _max_drawdown(trades_df),
        'trades': len(trades_df),
        'oos_stability': oos_stability,
        'fold_count': len(fold_pfs),
    }
    # Overfit gap proxy
    if fold_pfs:
        pf_is_proxy = max(fold_pfs)  # optimistic in-sample
        pf_oos = metrics['profit_factor']
        metrics['pf_is_proxy'] = pf_is_proxy
        metrics['pf_oos'] = pf_oos
        metrics['pf_oos_minus_is_proxy'] = pf_oos - pf_is_proxy
    else:
        metrics['pf_is_proxy'] = 0.0
        metrics['pf_oos'] = metrics['profit_factor']
        metrics['pf_oos_minus_is_proxy'] = 0.0
    return metrics


def composite_fitness(metrics: Dict[str, Any], weights: Dict[str, float], penalties: Dict[str, Any]) -> float:
    score = 0.0
    for k, w in weights.items():
        if k not in metrics:
            continue
        score += w * float(metrics[k])

    # Penalty: insufficient trades
    min_trades = penalties.get('min_trades')
    if min_trades and metrics.get('trades', 0) < int(min_trades):
        score -= 1.0

    # Penalty: overfit gap (using proxy metric pf_oos - pf_is_proxy)
    overfit_gap = penalties.get('overfit_gap')
    if overfit_gap is not None:
        diff = metrics.get('pf_oos_minus_is_proxy')
        if diff is not None and diff < -float(overfit_gap):
            score -= 0.5  # moderate penalty (tunable)
    return score

__all__ = ['evaluate_candidate', 'apply_params', 'composite_fitness']
