import pandas as pd
from types import SimpleNamespace

from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision

class DummySignalEngine:
    """Emits a single LONG decision with embedded risk_model output to test integration."""
    def __init__(self, decision):
        self._decision = decision
        self.feature_store = None

    def generate_signal(self, ohlcv_segment: pd.DataFrame, symbol: str):
        return self._decision

    def should_exit(self, *a, **k):
        return None


def _basic_settings():
    return {
        'start_date': '2023-01-01',
        'end_date': '2023-01-02',
        'risk_model': {
            'enabled': True,
            'base_risk_pct': 0.02,
            'max_leverage': 10,
            'min_confidence': 0.55,
            'regime_risk': {'trend':1.2,'mean_revert':1.0,'chop':0.5},
            'atr_multiplier_stop': 2.5,
            'atr_multiplier_tp': 3.0
        },
        'features': {'trend': {}, 'momentum': {}, 'volatility': {}, 'volume_flow': {}},
        'backtest': {
            'start_date': '2023-01-01',
            'end_date': '2023-01-02',
            'execution': {'initial_capital': 10000.0, 'default_size_pct': 0.05}
        }
    }


def test_risk_model_integration_sets_sl_tp_and_size():
    # OHLCV with 3 bars so engine called once
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:05', '2023-01-01 00:10']),
        'open': [100, 100, 100],
        'high': [101, 101, 101],
        'low': [99, 99, 99],
        'close': [100, 100, 100]
    }
    df = pd.DataFrame(data).set_index('timestamp')

    # Mock data adapter
    class DA:
        def load_ohlcv(self, *a, **k):
            return df
    da = DA()

    # Decision with risk_model payload
    decision = EnsembleDecision(
        ts=int(df.index[-1].timestamp()),
        symbol='TEST',
        tf='5m',
        decision='LONG',
        confidence=0.80,
        subsignals=[],
        vote_detail={'risk_model': {
            'position_size': 500.0,  # USD notional -> expect qty 5 if price 100
            'leverage': 5,
            'stop_loss': 97.5,      # 2.5 below entry
            'take_profit': 103.0,
            'confidence': 0.80,
            'regime': 'trend',
            'atr': 1.0,
            'atr_mult_stop': 2.5,
            'atr_mult_tp': 3.0
        }},
        vetoes=[]
    )

    settings = _basic_settings()
    fs = FeatureStore(warmup_periods=2, settings=settings)
    engine = DummySignalEngine(decision)
    engine.feature_store = fs
    runner = EventRunner(settings, da, engine, fs)

    trades, _eq = runner.run('TEST', '5m')

    # One trade should have been force-closed at EOD
    assert len(trades) == 1
    trade = trades[0]
    # Size ~5 qty
    assert abs(trade['size'] - 5.0) < 1e-6


def test_trailing_stop_moves_up():
    # Two bars, second advances price so trailing logic should tighten stop
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:05']),
        'open': [100, 102],
        'high': [101, 104],
        'low': [99, 101],
        'close': [100, 103]
    }
    df = pd.DataFrame(data).set_index('timestamp')

    class DA:
        def load_ohlcv(self, *a, **k):
            return df
    da = DA()

    decision = EnsembleDecision(
        ts=int(df.index[0].timestamp()),
        symbol='TEST2',
        tf='5m',
        decision='LONG',
        confidence=0.90,
        subsignals=[],
        vote_detail={'risk_model': {
            'position_size': 500.0,
            'leverage': 5,
            'stop_loss': 97.5,
            'take_profit': 103.0,
            'confidence': 0.90,
            'regime': 'trend',
            'atr': 1.0,
            'atr_mult_stop': 2.5,
            'atr_mult_tp': 3.0
        }},
        vetoes=[]
    )
    settings = _basic_settings()
    fs = FeatureStore(warmup_periods=2, settings=settings)
    engine = DummySignalEngine(decision)
    engine.feature_store = fs
    runner = EventRunner(settings, da, engine, fs)
    runner.run('TEST2', '5m')

    # Trade should have closed via SL after trailing moved it up (so exit price > original 97.5)
    trades = runner.portfolio.trades
    assert len(trades) == 1
    t = trades[0]
    assert t['reason'] == 'SL'
    assert t['exit_price'] > 97.5  # trailing tightened stop
