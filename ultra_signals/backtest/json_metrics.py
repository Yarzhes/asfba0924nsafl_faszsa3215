"""Utility functions for constructing normalized JSON metric payloads for backtest runs and walk-forward analysis.

These consolidate previously duplicated logic in `backtest_cli` for:
- Extra trade-level metrics (avg_win, avg_loss, win/loss ratio, expectancy, avg_rr)
- Streak metrics (max_consec_wins / losses)
- Risk-adjusted stats (Sortino, CAGR, Calmar, max_drawdown_pct)

All numeric outputs are sanitized (no NaN/inf values).
"""
from __future__ import annotations

from typing import Dict, List, Optional
import math
import pandas as pd

# Dimension (non-numeric) fields kept outside CORE_FIELDS to satisfy tests expecting CORE_FIELDS numeric
DIM_FIELDS = ['symbol','timeframe']
CORE_FIELDS = [  # strictly numeric metrics (tests assert numeric type)
    'profit_factor','sortino','sharpe','max_drawdown_pct',
    'win_rate_pct','total_trades','net_pnl','fees','slippage_bps',
    'cagr','calmar','expectancy','avg_win','avg_loss','win_loss_ratio','avg_rr','max_consec_wins','max_consec_losses'
]


def _extra_trade_metrics(trades: pd.DataFrame) -> Dict[str,float]:
    if trades is None or trades.empty or 'pnl' not in trades.columns:
        return {k:0.0 for k in ['avg_win','avg_loss','win_loss_ratio','expectancy','avg_rr','max_consec_wins','max_consec_losses']}
    pnl = trades['pnl'].astype(float)
    wins = pnl[pnl>0]; losses = pnl[pnl<0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    win_rate = len(wins)/len(pnl) if len(pnl) else 0.0
    loss_rate = 1 - win_rate
    win_loss_ratio = (avg_win/abs(avg_loss)) if avg_loss != 0 else 0.0
    expectancy = (win_rate*avg_win) + (loss_rate*avg_loss)
    rr_col = trades['rr'] if 'rr' in trades.columns else None
    avg_rr = float(rr_col.mean()) if rr_col is not None and not rr_col.empty else 0.0
    # streaks
    max_w = max_l = cur_w = cur_l = 0
    for v in pnl:
        if v > 0:
            cur_w += 1; max_w = max(max_w, cur_w); cur_l = 0
        elif v < 0:
            cur_l += 1; max_l = max(max_l, cur_l); cur_w = 0
        else:
            cur_w = cur_l = 0
    return {
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'expectancy': expectancy,
        'avg_rr': avg_rr,
        'max_consec_wins': max_w,
        'max_consec_losses': max_l
    }


def _calc_sortino(pnl: pd.Series) -> float:
    if pnl is None or len(pnl) < 2:
        return 0.0
    neg = pnl[pnl < 0]
    if neg.empty:
        return 0.0
    downside = neg.std(ddof=0) or 1e-9
    mean = pnl.mean()
    return (mean / downside) * math.sqrt(252)


def _equity_from_trades(trades: pd.DataFrame) -> List[float]:
    if trades is None or trades.empty or 'pnl' not in trades.columns:
        return []
    eq = [0.0]
    total = 0.0
    for v in trades['pnl'].astype(float):
        total += v
        eq.append(total)
    return eq


def _max_drawdown_pct(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    return abs(max_dd) / peak * 100 if peak else 0.0


def _cagr_calmar(equity: List[float], timeframe: str) -> Dict[str,float]:
    if not equity or len(equity) < 3:
        return {'cagr':0.0,'calmar':0.0}
    start_eq = equity[0]; end_eq = equity[-1]
    if start_eq <= 0 or end_eq <= 0:
        return {'cagr':0.0,'calmar':0.0}
    # timeframe minutes
    tf_minutes = 5
    try:
        if timeframe.endswith('m'):
            tf_minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            tf_minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            tf_minutes = int(timeframe[:-1]) * 60 * 24
    except Exception:
        pass
    bars = len(equity)
    years = max(1/365, (bars * tf_minutes)/(60*24*365))
    cagr = (end_eq/start_eq)**(1/years) - 1 if years > 0 else 0.0
    dd_pct = _max_drawdown_pct(equity)
    calmar = cagr / (dd_pct/100) if dd_pct > 0 else 0.0
    return {'cagr': cagr, 'calmar': calmar}


def _sanitize(d: Dict[str, float]) -> Dict[str, float]:
    for k,v in list(d.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            d[k] = 0.0
    return d


def build_run_metrics(kpis: Dict[str,float], trades: pd.DataFrame, equity_curve: List[float], settings: Dict, symbol: str, timeframe: str) -> Dict[str,float]:
    pnl_series = trades['pnl'].astype(float) if trades is not None and not trades.empty and 'pnl' in trades.columns else None
    sortino = _calc_sortino(pnl_series) if pnl_series is not None else 0.0
    extra = _extra_trade_metrics(trades)
    initial_cap = 10000.0
    try:
        initial_cap = float(((settings.get('backtest') or {}).get('execution') or {}).get('initial_capital', initial_cap))
    except Exception:
        pass
    maxdd_raw = kpis.get('max_drawdown') or 0.0
    max_drawdown_pct = (maxdd_raw / initial_cap * 100.0) if initial_cap else 0.0
    # If equity curve passed, prefer recalculating max_dd_pct from curve values if looks like equity ( > initial_cap )
    if equity_curve and equity_curve[0] > 0 and any(v < equity_curve[0] for v in equity_curve):
        max_drawdown_pct = _max_drawdown_pct(equity_curve)
    cagr_calmar = _cagr_calmar(equity_curve, timeframe) if equity_curve else {'cagr':0.0,'calmar':0.0}
    fees_sum = float(trades['fees'].sum()) if (trades is not None and 'fees' in trades.columns) else 0.0
    slip_bps = float(trades['slippage_bps'].mean()) if (trades is not None and 'slippage_bps' in trades.columns and not trades.empty) else 0.0
    payload = {
        'symbol': symbol,
        'timeframe': timeframe,
        'profit_factor': kpis.get('profit_factor'),
        'sortino': sortino,
        'sharpe': kpis.get('sharpe'),
        'max_drawdown_pct': max_drawdown_pct,
        'win_rate_pct': kpis.get('win_rate_pct'),
        'total_trades': kpis.get('total_trades'),
        'net_pnl': kpis.get('total_pnl'),
        'fees': fees_sum,
        'slippage_bps': slip_bps,
        'cagr': cagr_calmar['cagr'],
        'calmar': cagr_calmar['calmar'],
        'expectancy': extra['expectancy'],
        'avg_win': extra['avg_win'],
        'avg_loss': extra['avg_loss'],
        'win_loss_ratio': extra['win_loss_ratio'],
        'avg_rr': extra['avg_rr'],
        'max_consec_wins': extra['max_consec_wins'],
        'max_consec_losses': extra['max_consec_losses'],
    }
    # Sprint 30: PnL stratification by MTC status (if present)
    try:
        if trades is not None and not trades.empty and 'mtc_status' in trades.columns:
            strat = {}
            for st in ['CONFIRM','PARTIAL','FAIL',None]:
                subset = trades[trades['mtc_status']==st] if st is not None else trades[trades['mtc_status'].isna()]
                if subset is not None and not subset.empty:
                    strat_key = f"pnl_mtc_{str(st).lower()}" if st is not None else "pnl_mtc_unTagged"
                    strat[strat_key] = float(subset['pnl'].sum())
                    strat[strat_key+"_count"] = int(len(subset))
            payload.update(strat)
    except Exception:
        pass
    return _sanitize(payload)


def build_wf_metrics(trades: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str,float]:
    if trades is None or trades.empty or 'pnl' not in trades.columns:
        return {k:0.0 for k in CORE_FIELDS}
    pnl = trades['pnl'].astype(float)
    wins = pnl[pnl>0]; losses = pnl[pnl<0]
    gross_win = wins.sum(); gross_loss = -losses.sum()
    profit_factor = gross_win / gross_loss if gross_loss>0 else 0.0
    win_rate = len(wins)/len(pnl) if len(pnl)>0 else 0.0
    extra = _extra_trade_metrics(trades)
    # Sharpe / Sortino
    if len(pnl) > 1:
        mean = pnl.mean(); std = pnl.std(ddof=0) or 1e-9
        sharpe = (mean/std) * math.sqrt(252)
        neg = pnl[pnl<0]; dstd = neg.std(ddof=0) or 1e-9
        sortino = (mean/dstd) * math.sqrt(252)
    else:
        sharpe = sortino = 0.0
    # Equity & drawdown
    equity = _equity_from_trades(trades)
    max_dd_pct = _max_drawdown_pct(equity)
    # CAGR/Calmar approx
    cagr_calmar = _cagr_calmar(equity, timeframe)
    payload = {
        'symbol': symbol,
        'timeframe': timeframe,
        'profit_factor': profit_factor,
        'sortino': sortino,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd_pct,
        'win_rate_pct': win_rate*100,
        'total_trades': len(pnl),
        'net_pnl': float(pnl.sum()),
        'fees': 0.0,
        'slippage_bps': 0.0,
        'cagr': cagr_calmar['cagr'],
        'calmar': cagr_calmar['calmar'],
        'expectancy': extra['expectancy'],
        'avg_win': extra['avg_win'],
        'avg_loss': extra['avg_loss'],
        'win_loss_ratio': extra['win_loss_ratio'],
        'avg_rr': extra['avg_rr'],
        'max_consec_wins': extra['max_consec_wins'],
        'max_consec_losses': extra['max_consec_losses'],
    }
    return _sanitize(payload)
