import pandas as pd
from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision

class StaticDataAdapter:
    def __init__(self, df):
        self._df = df
    def load_ohlcv(self, *a, **k):
        return self._df


def _settings():
    return {
        'start_date': '2023-01-01',
        'end_date': '2023-01-02',
        'risk': {'adaptive_exits': {'enabled': True}},
        'features': {'trend': {}, 'momentum': {}, 'volatility': {}, 'volume_flow': {}},
        'backtest': {'execution': {'initial_capital': 10000.0, 'default_size_pct': 0.05}},
    }

class OneShotEngine:
    """Emits one LONG decision embedding adaptive_exits then stays FLAT."""
    def __init__(self, decision):
        self._decision = decision
        self._emitted = False
        self.feature_store = None
    def generate_signal(self, ohlcv_segment: pd.DataFrame, symbol: str):
        if not self._emitted:
            self._emitted = True
            return self._decision
        return EnsembleDecision(ts=int(ohlcv_segment.index[-1].timestamp()), symbol=symbol, tf='5m', decision='FLAT', confidence=0.0, subsignals=[], vote_detail={}, vetoes=[])
    def should_exit(self, *a, **k):
        return None


def test_adaptive_runtime_metrics_trailing_breakeven_partial():
    # Price path moves up steadily to trigger partials, breakeven and trailing
    data = {
        'timestamp': pd.to_datetime([
            '2023-01-01 00:00','2023-01-01 00:05','2023-01-01 00:10','2023-01-01 00:15','2023-01-01 00:20'
        ]),
        'open':  [100,101,102,103,104],
        'high':  [101,102,103.5,104.5,105.5],
        'low':   [99,100.5,101.5,102.5,103.5],
        'close': [100,102,103,104,105]
    }
    df = pd.DataFrame(data).set_index('timestamp')

    adaptive_payload = {
        'stop_price': 98.0,
        'target_price': 106.0,
        'partial_tp': [
            {'rr': 1.0, 'price': 102.0, 'pct': 0.5},
            {'rr': 2.0, 'price': 104.0, 'pct': 0.5},
        ],
        'trail_config': {'enabled': True, 'step': 1.0},
        'breakeven': {'enabled': True, 'trigger_rr': 0.8},
        'meta': {'atr': 1.0}
    }
    decision = EnsembleDecision(
        ts=int(df.index[0].timestamp()),
        symbol='TESTAE',
        tf='5m',
        decision='LONG',
        confidence=0.9,
        subsignals=[],
        vote_detail={'adaptive_exits': adaptive_payload},
        vetoes=[]
    )

    settings = _settings()
    fs = FeatureStore(warmup_periods=2, settings=settings)
    da = StaticDataAdapter(df)
    engine = OneShotEngine(decision)
    engine.feature_store = fs
    runner = EventRunner(settings, da, engine, fs)
    trades, _eq = runner.run('TESTAE','5m')

    # Expect final close (either TP or partial final or EOD) - at least 1 trade
    assert len(trades) >= 1
    m = runner.event_metrics
    # At least one partial fill executed (first level 102) and maybe second (104)
    assert m['adaptive_partial_fills'] >= 1
    # Breakeven move should trigger once price passes 0.8R (~101.6) -> by second bar
    assert m['adaptive_breakeven_moves'] >= 1
    # Some trailing adjustments expected as price steps by >=1.0 each bar
    assert m['adaptive_trailing_adjustments'] >= 1
    # RR samples collected (trade exit)
    assert len(m['adaptive_rr_samples']) >= 1
    # Exit counts dictionary populated
    assert isinstance(m['adaptive_exit_counts'], dict)
