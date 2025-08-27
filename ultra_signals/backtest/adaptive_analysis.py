import copy
import os
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision


def run_static_vs_adaptive(
    base_settings: Dict[str, Any],
    data_adapter,
    static_decision: EnsembleDecision,
    adaptive_decision: EnsembleDecision,
    symbol: str,
    timeframe: str,
    output_dir: str = None
) -> Dict[str, Any]:
    """Run two backtests (static vs adaptive exits) and optionally save an equity comparison plot.

    Args:
        base_settings: Baseline settings dict (will be deep-copied per run).
        data_adapter: Object exposing load_ohlcv().
        static_decision: Decision object WITHOUT adaptive_exits in vote_detail.
        adaptive_decision: Decision object WITH adaptive_exits payload in vote_detail.
        symbol: Trading symbol.
        timeframe: Timeframe string (e.g. '5m').
        output_dir: Optional directory to save 'adaptive_vs_static.png'. If None, will attempt
                    to read from settings['backtest']['output_dir'].

    Returns:
        Dict with keys: static_trades, adaptive_trades, static_equity, adaptive_equity, pnl_static, pnl_adaptive
    """
    # --- Static run ---
    settings_static = copy.deepcopy(base_settings)
    fs_static = FeatureStore(warmup_periods=2, settings=settings_static)
    engine_static = _SingleDecisionEngine(static_decision, fs_static)
    runner_static = EventRunner(settings_static, data_adapter, engine_static, fs_static)
    trades_static, eq_static = runner_static.run(symbol, timeframe)

    # --- Adaptive run ---
    settings_adapt = copy.deepcopy(base_settings)
    # ensure adaptive block enabled
    settings_adapt.setdefault('risk', {}).setdefault('adaptive_exits', {})['enabled'] = True
    fs_adapt = FeatureStore(warmup_periods=2, settings=settings_adapt)
    engine_adapt = _SingleDecisionEngine(adaptive_decision, fs_adapt)
    runner_adapt = EventRunner(settings_adapt, data_adapter, engine_adapt, fs_adapt)
    trades_adapt, eq_adapt = runner_adapt.run(symbol, timeframe)

    pnl_static = trades_static[0]['pnl'] if trades_static else 0.0
    pnl_adaptive = trades_adapt[0]['pnl'] if trades_adapt else 0.0

    out_dir = output_dir or (base_settings.get('backtest', {}) or {}).get('output_dir') or base_settings.get('output_dir')
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            plt.figure(figsize=(7,4))
            xs_s = [row['timestamp'] for row in eq_static]
            ys_s = [row['equity'] for row in eq_static]
            xs_a = [row['timestamp'] for row in eq_adapt]
            ys_a = [row['equity'] for row in eq_adapt]
            plt.plot(xs_s, ys_s, label=f'Static (PnL {pnl_static:.2f})', color='gray')
            plt.plot(xs_a, ys_a, label=f'Adaptive (PnL {pnl_adaptive:.2f})', color='blue')
            plt.legend(); plt.title('Adaptive vs Static Equity'); plt.xlabel('Time'); plt.ylabel('Equity')
            fname = os.path.join(out_dir, f'adaptive_vs_static_{symbol}_{timeframe}.png')
            plt.tight_layout(); plt.savefig(fname)
            plt.close()
        except Exception as e:
            # non-fatal
            print(f"[adaptive_analysis] Plot generation failed: {e}")

    return {
        'static_trades': trades_static,
        'adaptive_trades': trades_adapt,
        'static_equity': eq_static,
        'adaptive_equity': eq_adapt,
        'pnl_static': pnl_static,
        'pnl_adaptive': pnl_adaptive,
        'event_metrics_static': runner_static.event_metrics,
        'event_metrics_adaptive': runner_adapt.event_metrics
    }


class _SingleDecisionEngine:
    """Simple engine that emits a supplied decision once then FLAT for remainder."""
    def __init__(self, decision: EnsembleDecision, feature_store):
        self._decision = decision
        self._emitted = False
        self.feature_store = feature_store
    def generate_signal(self, ohlcv_segment, symbol: str):
        if not self._emitted:
            self._emitted = True
            return self._decision
        return EnsembleDecision(ts=int(ohlcv_segment.index[-1].timestamp()), symbol=symbol, tf='5m', decision='FLAT', confidence=0.0, subsignals=[], vote_detail={}, vetoes=[])
    def should_exit(self, *a, **k):
        return None
