"""Search orchestration (random + Bayesian) for AutoOpt.

Responsibilities:
  * Sample parameter sets from `AutoOptSpace`.
  * Apply params into a derived settings dict (lightweight mapper, non-destructive).
  * Run walk-forward evaluation via `AutoOptWFRunner`.
  * Compute risk-aware score with constraints (reject -> -inf).
  * Maintain leaderboard (trial -> metrics + score + params).
  * Optional Optuna TPE phase after random seeds (if optuna installed).

This module intentionally keeps implementation lean; advanced features
like asynchronous workers, early upper-bound pruning, or per-profile
parallelization can layer on top later without changing the API.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import copy
import math
from loguru import logger

from .spaces import AutoOptSpace
from .objective import compute_risk_aware_score
from .wf_runner import AutoOptWFRunner

try:  # optional dependency
    import optuna  # type: ignore
    HAVE_OPTUNA = True
except Exception:  # pragma: no cover
    HAVE_OPTUNA = False


def _apply_params(base: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply sampled params into a deep-copied settings structure.

    Only touches known subtrees; anything else left unchanged ensuring
    safety (no accidental removal of existing config keys).
    """
    s = copy.deepcopy(base)
    ens = s.setdefault('ensemble', {})
    exec_cfg = s.setdefault('execution', {})
    risk_cfg = s.setdefault('risk', {})
    hedge_cfg = s.setdefault('hedge', {})

    # Ensemble mapping
    vt = ens.setdefault('vote_threshold', {})
    vt['trend'] = params.get('ensemble.vote_threshold_trend', vt.get('trend', 0.5))
    vt['mean_revert'] = params.get('ensemble.vote_threshold_mean_revert', vt.get('mean_revert', 0.5))
    ma = ens.setdefault('min_agree', {})
    ma['trend'] = params.get('ensemble.min_agree_trend', ma.get('trend', 1))
    ma['mean_revert'] = params.get('ensemble.min_agree_mean_revert', ma.get('mean_revert', 1))
    ens['margin_of_victory'] = params.get('ensemble.margin_of_victory', ens.get('margin_of_victory', 0.1))
    ens['confidence_floor'] = params.get('ensemble.confidence_floor', ens.get('confidence_floor', 0.5))

    # Execution
    exec_cfg['k1_ticks'] = params.get('execution.k1_ticks', exec_cfg.get('k1_ticks', 1))
    exec_cfg['taker_fallback_ms'] = int(params.get('execution.taker_fallback_ms', exec_cfg.get('taker_fallback_ms', 1000)))
    exec_cfg['max_chase_bps'] = params.get('execution.max_chase_bps', exec_cfg.get('max_chase_bps', 8))
    exec_cfg['stop_atr_mult'] = params.get('execution.stop_atr_mult', exec_cfg.get('stop_atr_mult', 1.6))
    tp_vals = params.get('execution.tp_atr_mults')
    if tp_vals:
        exec_cfg['tp_atr_mults'] = list(tp_vals)

    # Hedge
    beta_band = hedge_cfg.setdefault('beta_band', {})
    beta_band['min'] = params.get('hedge.beta_band_min', beta_band.get('min', -0.1))
    beta_band['max'] = params.get('hedge.beta_band_max', beta_band.get('max', 0.1))
    hedge_cfg['rebalance.min_notional'] = params.get('hedge.rebalance_min_notional', hedge_cfg.get('rebalance.min_notional', 0.005))
    hedge_cfg['corr_threshold_high'] = params.get('hedge.corr_threshold_high', hedge_cfg.get('corr_threshold_high', 0.5))

    # Risk
    risk_cfg['daily_loss_cap_pct'] = params.get('risk.daily_loss_cap_pct', risk_cfg.get('daily_loss_cap_pct', 3.0))
    risk_cfg['max_positions_total'] = params.get('risk.max_positions_total', risk_cfg.get('max_positions_total', 3))
    return s


def _score_candidate(settings: Dict[str, Any], runner: AutoOptWFRunner, symbol: str, timeframe: str) -> Tuple[float, Dict[str, Any]]:
    metrics = runner.evaluate(settings, symbol, timeframe)
    score = compute_risk_aware_score(metrics)
    metrics['score'] = score
    return score, metrics


def run_search(base_settings: Dict[str, Any], *, symbol: str, timeframe: str, space: AutoOptSpace | None = None, random_trials: int = 20, bayes_trials: int = 50, seed: int = 42, runner_factory=None) -> Dict[str, Any]:
    """Execute random + (optional) Bayesian search; return artifacts dict.

    Parameters
    ----------
    base_settings : dict
        Baseline config used as template.
    symbol, timeframe : str
        Target market context for WF evaluations.
    space : AutoOptSpace
        Parameter space (defaults to new instance).
    random_trials : int
        Number of pure random seeds before (optional) Bayesian phase.
    bayes_trials : int
        Number of TPE trials; ignored if optuna missing or <=0.
    runner_factory : callable
        factory returning AutoOptWFRunner; allows dependency injection for tests.
    """
    space = space or AutoOptSpace()
    leaderboard: List[Dict[str, Any]] = []
    best = {'score': float('-inf'), 'params': None, 'metrics': None, 'settings': None}
    # Build WF runner once (stateless per evaluate: underlying WFA builds fresh engine each run)
    if runner_factory:
        runner = runner_factory()
    else:
        # late import to avoid heavy modules at import time
        from ultra_signals.backtest.data_adapter import DataAdapter  # type: ignore
        from ultra_signals.engine.real_engine import RealSignalEngine  # type: ignore
        runner = AutoOptWFRunner(base_settings, lambda s: DataAdapter(s), lambda s: RealSignalEngine(s, None))

    # --- Random phase -----------------------------------------------------
    class DummyTrial:  # simple shim for space.sample
        def suggest_float(self, name, low, high):  # pragma: no cover (space uses random directly w/out trial)
            import random
            return random.uniform(low, high)
        def suggest_categorical(self, name, choices):  # pragma: no cover
            import random
            return random.choice(choices)

    for t in range(random_trials):
        params = space.sample(DummyTrial())
        derived = _apply_params(base_settings, params)
        score, metrics = _score_candidate(derived, runner, symbol, timeframe)
        row = {'phase': 'random', 'trial': t, 'score': score, **params, **metrics}
        leaderboard.append(row)
        if score > best['score']:
            best = {'score': score, 'params': params, 'metrics': metrics, 'settings': derived}
        logger.debug(f"[autoopt][random] t={t} score={score:.4f}")

    # --- Bayesian phase ---------------------------------------------------
    if HAVE_OPTUNA and bayes_trials > 0:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        def _objective(trial: 'optuna.Trial'):
            params = space.sample(trial)
            derived = _apply_params(base_settings, params)
            score, metrics = _score_candidate(derived, runner, symbol, timeframe)
            trial.set_user_attr('params', params)
            trial.set_user_attr('metrics', metrics)
            if score > best['score']:
                best.update({'score': score, 'params': params, 'metrics': metrics, 'settings': derived})
            return score

        study.optimize(_objective, n_trials=bayes_trials, show_progress_bar=False)
        for t in study.trials:
            entry = {'phase': 'bayes', 'trial': t.number, 'score': (t.value if t.value is not None else float('nan'))}
            if 'params' in t.user_attrs:
                entry.update(t.user_attrs['params'])
            if 'metrics' in t.user_attrs:
                entry.update(t.user_attrs['metrics'])
            leaderboard.append(entry)
    else:
        if bayes_trials > 0:
            logger.warning("Optuna not installed; skipping Bayesian phase.")

    # Sort final leaderboard (desc score)
    leaderboard.sort(key=lambda r: r['score'], reverse=True)
    return {'leaderboard': leaderboard, 'best': best}


__all__ = ['run_search', 'HAVE_OPTUNA']
