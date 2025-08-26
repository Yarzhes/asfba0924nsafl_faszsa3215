"""Sprint 19: Optimizer Wrapper (Optuna / fallback)
"""
from __future__ import annotations
from typing import Dict, Any, Callable, List
from loguru import logger
import time

try:
    import optuna
    HAVE_OPTUNA = True
except Exception:  # pragma: no cover
    HAVE_OPTUNA = False

from .search_space import SearchSpace
from .objective import apply_params, evaluate_candidate, composite_fitness
from .persistence import save_leaderboard, save_best, save_autotuned_settings


def _process_worker(storage_url: str, study_name: str, space_spec: Dict[str, Any], base_settings: Dict[str, Any], wf_cfg: Dict[str, Any], weights: Dict[str, float], penalties: Dict[str, Any], pruner_cfg: Dict[str, Any], force_prune_first_n: int, seed: int, trials: int, timeout: Any):  # pragma: no cover (subprocess)
    """Worker for process-based parallel optimization.

    Each process attaches to the same Optuna RDB storage and contributes trials.
    We keep objective logic minimal & robust; failures prune the trial.
    """
    try:
        import optuna
    except Exception:
        return
    sampler = optuna.samplers.TPESampler(seed=seed)
    median_warmup = int(pruner_cfg.get('median_warmup_steps', 10))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=median_warmup)
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    space = SearchSpace(space_spec)
    def _objective(trial: 'optuna.Trial'):
        if force_prune_first_n and trial.number < force_prune_first_n:
            raise optuna.TrialPruned()
        params = space.sample(trial)
        derived = apply_params(base_settings, params)
        metrics = evaluate_candidate(derived, wf_cfg)
        fitness = composite_fitness(metrics, weights, penalties)
        trial.set_user_attr('metrics', metrics)
        trial.report(fitness, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return fitness
    study.optimize(_objective, n_trials=trials, timeout=timeout, show_progress_bar=False, n_jobs=1)


def run_optimization(base_settings: Dict[str, Any], cal_cfg: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    search_spec = cal_cfg.get('search_space', {})
    wf_cfg = cal_cfg.get('walk_forward', {})
    obj_cfg = cal_cfg.get('objective', {})
    weights = (obj_cfg.get('weights') or {})
    penalties = (obj_cfg.get('penalties') or {})
    rt_cfg = cal_cfg.get('runtime', {}) or {}
    trials = int(rt_cfg.get('trials', 50))
    seed = int(rt_cfg.get('seed', 42))
    timeout = rt_cfg.get('timeout')
    parallel = int(rt_cfg.get('parallel', 1) or 1)
    parallel_mode = str(rt_cfg.get('parallel_mode', 'thread')).lower()

    space = SearchSpace(search_spec)
    leaderboard: List[Dict[str, Any]] = []
    best = {"fitness": float('-inf'), "params": None, "metrics": None, "derived_settings": None}

    if HAVE_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner_cfg = (cal_cfg.get('runtime', {}) or {}).get('pruner', {}) or {}
        median_warmup = int(pruner_cfg.get('median_warmup_steps', 10))
        force_prune_first_n = int(pruner_cfg.get('force_prune_first_n', 0))
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=median_warmup)
        storage = None
        if cal_cfg.get('runtime', {}).get('save_study_db', True) or parallel_mode == 'process':
            storage = f"sqlite:///{output_dir}/study.db"
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner, study_name=cal_cfg.get('runtime', {}).get('study_name'), storage=storage, load_if_exists=True)

        # Simple lock to protect shared 'best' when using multi-thread parallel (n_jobs>1)
        import threading
        _best_lock = threading.Lock()

        def _objective(trial: 'optuna.Trial'):
            # Deterministic early prune for testing / acceleration if configured
            if force_prune_first_n and trial.number < force_prune_first_n:
                raise optuna.TrialPruned()
            params = space.sample(trial)
            derived = apply_params(base_settings, params)
            metrics = evaluate_candidate(derived, wf_cfg)
            fitness = composite_fitness(metrics, weights, penalties)
            trial.set_user_attr('metrics', metrics)
            # Report intermediate value (single-step for now) to enable pruner decisions
            try:  # pragma: no cover (robustness)
                trial.report(fitness, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except optuna.TrialPruned:
                logger.debug(f"Trial {trial.number} pruned (fitness={fitness})")
                # Do not update best on pruned trials
                raise
            nonlocal best
            if fitness > best['fitness']:
                # Protect assignment when using n_jobs>1
                with _best_lock:
                    if fitness > best['fitness']:
                        best = {"fitness": fitness, "params": params, "metrics": metrics, "derived_settings": derived}
            return fitness

        if parallel > 1 and parallel_mode == 'process':
            # Spawn processes each doing a slice of trials (requires storage)
            if not storage:
                storage = f"sqlite:///{output_dir}/study.db"
            import math, multiprocessing
            per_worker = math.ceil(trials / parallel)
            workers = []
            remaining = trials
            for i in range(parallel):
                n_i = min(per_worker, remaining)
                if n_i <= 0:
                    break
                p = multiprocessing.Process(target=_process_worker, args=(storage, study.study_name, search_spec, base_settings, wf_cfg, weights, penalties, pruner_cfg, force_prune_first_n, seed + i, n_i, timeout))
                p.start()
                workers.append(p)
                remaining -= n_i
            for p in workers:
                p.join()
        else:
            # Use parallel threads if requested (Optuna uses threads for n_jobs>1)
            study.optimize(_objective, n_trials=trials, timeout=timeout, show_progress_bar=False, n_jobs=parallel)
        # Rebuild leaderboard including pruned trials & status for transparency
        leaderboard = []
        for t in study.trials:
            entry: Dict[str, Any] = {
                'trial': t.number,
                'status': t.state.name,
                'fitness': (t.value if t.value is not None else float('nan')),
            }
            if 'metrics' in t.user_attrs:
                entry.update(t.user_attrs['metrics'])
            entry.update({k: v for k, v in t.params.items()})
            leaderboard.append(entry)
    else:
        logger.warning("Optuna not installed; falling back to random search.")
        import random
        class DummyTrial:
            def suggest_int(self, name, low, high, step=1):
                return random.randrange(low, high+1, step)
            def suggest_float(self, name, low, high):
                return random.uniform(low, high)
            def suggest_categorical(self, name, choices):
                return random.choice(choices)
        for i in range(trials):
            t = DummyTrial()
            params = space.sample(t)
            derived = apply_params(base_settings, params)
            metrics = evaluate_candidate(derived, wf_cfg)
            fitness = composite_fitness(metrics, weights, penalties)
            leaderboard.append({'trial': i, 'fitness': fitness, **metrics, **params})
            if fitness > best['fitness']:
                best = {"fitness": fitness, "params": params, "metrics": metrics, "derived_settings": derived}

    # Persist artifacts here (so CLI thin wrapper simpler)
    try:
        save_leaderboard(leaderboard, output_dir)
        save_best(best, base_settings, output_dir)
        if best.get('derived_settings'):
            # Only tuned keys applied for settings_autotuned.yaml
            save_autotuned_settings(best['params'] or {}, best['derived_settings'], output_dir)
        # Optional plots if optuna and study available
        if HAVE_OPTUNA and 'study' in locals():  # pragma: no cover (visual artifacts)
            try:
                from optuna.visualization import plot_optimization_history, plot_param_importances
                hist = plot_optimization_history(study)
                hist.write_image(f"{output_dir}/opt_history.png") if hasattr(hist, 'write_image') else hist.write_html(f"{output_dir}/opt_history.html")
            except Exception as e:
                logger.warning(f"Could not write optimization history plot: {e}")
            try:
                from optuna.visualization import plot_param_importances
                imp = plot_param_importances(study)
                imp.write_image(f"{output_dir}/param_importances.png") if hasattr(imp, 'write_image') else imp.write_html(f"{output_dir}/param_importances.html")
            except Exception as e:
                logger.warning(f"Could not write param importance plot: {e}")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Persistence error: {e}")

    return {"best": best, "leaderboard": leaderboard}

__all__ = ['run_optimization']
