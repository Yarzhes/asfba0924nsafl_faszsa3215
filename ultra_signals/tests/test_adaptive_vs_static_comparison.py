import pandas as pd
from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision

class DataAdapter:
    def __init__(self, df):
        self._df = df
    def load_ohlcv(self, *a, **k):
        return self._df

class OneShotEngine:
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


def base_settings():
    return {
        'start_date': '2023-01-01',
        'end_date': '2023-01-02',
        'features': {'trend': {}, 'momentum': {}, 'volatility': {}, 'volume_flow': {}},
        'backtest': {'execution': {'initial_capital': 10000.0, 'default_size_pct': 0.05, 'sl_pct':0.02, 'tp_pct':0.02}},
    }


def _price_data():
    # Monotonic rising prices so holding longer is better
    ts = pd.date_range('2023-01-01 00:00', periods=6, freq='5min')
    prices = [100,101,102,103,104,105]
    data = {
        'timestamp': ts,
        'open': prices,
        'high': [p+0.5 for p in prices],
        'low':  [p-0.5 for p in prices],
        'close': prices
    }
    return pd.DataFrame(data).set_index('timestamp')


def test_adaptive_holds_longer_and_captures_more_pnl():
    df = _price_data()
    da = DataAdapter(df)

    # Static decision (no adaptive_exits) -> will exit at 2% TP around price 102
    static_decision = EnsembleDecision(
        ts=int(df.index[0].timestamp()), symbol='CMP', tf='5m', decision='LONG', confidence=0.9, subsignals=[], vote_detail={}, vetoes=[]
    )
    s_settings = base_settings()
    fs_static = FeatureStore(warmup_periods=2, settings=s_settings)
    eng_static = OneShotEngine(static_decision); eng_static.feature_store = fs_static
    runner_static = EventRunner(s_settings, da, eng_static, fs_static)
    trades_static, _ = runner_static.run('CMP','5m')
    assert len(trades_static) == 1
    pnl_static = trades_static[0]['pnl']

    # Adaptive decision overrides TP far away so it holds until EOD (price 105)
    adaptive_payload = {
        'stop_price': 98.0,
        'target_price': 999.0,  # effectively unreachable
        'partial_tp': [],
        'trail_config': {'enabled': False},
        'breakeven': {'enabled': False},
        'meta': {'atr': 1.0}
    }
    adaptive_decision = EnsembleDecision(
        ts=int(df.index[0].timestamp()), symbol='CMP', tf='5m', decision='LONG', confidence=0.9, subsignals=[], vote_detail={'adaptive_exits': adaptive_payload}, vetoes=[]
    )
    a_settings = base_settings(); a_settings['risk']={'adaptive_exits': {'enabled': True}}
    fs_adapt = FeatureStore(warmup_periods=2, settings=a_settings)
    eng_adapt = OneShotEngine(adaptive_decision); eng_adapt.feature_store = fs_adapt
    runner_adapt = EventRunner(a_settings, da, eng_adapt, fs_adapt)
    trades_adapt, _ = runner_adapt.run('CMP','5m')
    assert len(trades_adapt) == 1
    pnl_adapt = trades_adapt[0]['pnl']

    # Adaptive should have larger PnL because it holds till final bar
    assert pnl_adapt > pnl_static
