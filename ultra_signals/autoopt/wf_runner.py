"""Walk-forward runner facade for autoopt.

Wraps existing WalkForwardAnalysis providing early-stop hooks and
aggregation for risk-aware scoring.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from loguru import logger
import pandas as pd
from ultra_signals.backtest.walkforward import WalkForwardAnalysis, _safe_to_datetime
from ultra_signals.backtest.metrics import compute_kpis

class AutoOptWFRunner:
    def __init__(self, base_settings: Dict[str,Any], data_adapter_factory, engine_factory):
        self.base_settings = base_settings
        self._data_adapter_factory = data_adapter_factory
        self._engine_factory = engine_factory

    def evaluate(self, params_applied_settings: Dict[str,Any], symbol: str, timeframe: str, early_stop: Dict[str,Any]|None=None) -> Dict[str,Any]:
        """Evaluate params via walk-forward.

        early_stop config (optional):
          {'score_best': float, 'patience': int}
        Placeholder: early stopping not yet incremental; full run executed.
        """
        wf = WalkForwardAnalysis(params_applied_settings, self._data_adapter_factory(params_applied_settings), lambda s=params_applied_settings: self._engine_factory(params_applied_settings))
        # If no early_stop config, fallback to full run
        if not early_stop:
            trades, kpis = wf.run(symbol,timeframe)
            return self._aggregate_metrics(trades, kpis)
        # Incremental fold execution (replicates WalkForwardAnalysis.run logic)
        settings_bt = params_applied_settings.get('backtest', {})
        start_date = _safe_to_datetime(settings_bt.get('start_date'))
        end_date = _safe_to_datetime(settings_bt.get('end_date'))
        windows = wf._generate_windows(start_date, end_date)
        all_trades = []
        kpi_reports = []
        best_score = float('-inf')
        stale = 0
        patience = early_stop.get('patience', 0)
        improvement_delta = early_stop.get('improvement_delta', 0.0)
        from ultra_signals.autoopt.objective import compute_risk_aware_score
        for i,(train_start, train_end, test_start, test_end) in enumerate(windows):
            # Reuse wf._run_test_fold to avoid duplicating engine creation.
            test_trades, _ = wf._run_test_fold(symbol, timeframe, test_start, test_end, None, fold_index=i)
            if not test_trades.empty:
                all_trades.append(test_trades)
                kpis_fold = compute_kpis(test_trades)
                kpis_fold['fold'] = i+1
                kpi_reports.append(kpis_fold)
            # Aggregate so far
            if all_trades:
                agg_metrics = self._aggregate_metrics(pd.concat(all_trades, ignore_index=True), pd.DataFrame(kpi_reports))
                score = compute_risk_aware_score(agg_metrics)
                if score > best_score + improvement_delta:
                    best_score = score
                    stale = 0
                else:
                    stale += 1
                if patience and stale >= patience:
                    agg_metrics['early_stopped'] = True
                    return agg_metrics
        if not all_trades:
            return {'trades':0,'profit_factor':0,'winrate':0,'sortino':0,'max_dd_pct':0,'cvar_95_pct':0,'ret_iqr':1.0,'early_stopped':False}
        final = self._aggregate_metrics(pd.concat(all_trades, ignore_index=True), pd.DataFrame(kpi_reports))
        final['early_stopped'] = False
        return final

    def _aggregate_metrics(self, trades, kpis):
        if trades is None or trades.empty:
            return {'trades':0,'profit_factor':0,'winrate':0,'sortino':0,'max_dd_pct':0,'cvar_95_pct':0,'ret_iqr':1.0}
        if isinstance(kpis,pd.DataFrame) and not kpis.empty:
            pf_median = float(kpis.get('profit_factor').median()) if 'profit_factor' in kpis else 0.0
            sortino_median = float(kpis.get('sortino').median()) if 'sortino' in kpis else 0.0
            ret_iqr = float(kpis.get('return').quantile(0.75) - kpis.get('return').quantile(0.25)) if 'return' in kpis else 0.0
        else:
            pf_median = sortino_median = ret_iqr = 0.0
        kpi_all = compute_kpis(trades)
        return {
            'trades': int(len(trades)),
            'profit_factor': float(kpi_all.get('profit_factor',0)),
            'profit_factor_median': pf_median,
            'winrate': float(kpi_all.get('winrate',0)),
            'sortino': float(kpi_all.get('sortino',0)),
            'sortino_median': sortino_median,
            'max_dd_pct': float(kpi_all.get('max_dd_pct',0)),
            'cvar_95_pct': float(kpi_all.get('cvar_95_pct',0)),
            'turnover_penalty': float(kpi_all.get('turnover_penalty',0)),
            'fees_funding_pct': float(kpi_all.get('fees_funding_pct',0)),
            'ret_iqr': ret_iqr,
        }

__all__ = ['AutoOptWFRunner']
