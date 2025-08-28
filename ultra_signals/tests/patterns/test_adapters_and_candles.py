import pandas as pd
import numpy as np
from ultra_signals.patterns.bar_adapters import HeikinAshiAdapter, RenkoAdapter
from ultra_signals.patterns.base import Bars, Bar
from ultra_signals.patterns.candle_patterns import CandlestickPatternLibrary


def mk_df(seq):
    idx = pd.date_range('2020-01-01', periods=len(seq), freq='T')
    df = pd.DataFrame(seq, index=idx)
    return df


def test_heikin_ashi_basic():
    seq = [
        {'open': 10, 'high': 11, 'low': 9, 'close': 10.5, 'volume': 1},
        {'open': 10.5, 'high': 11.5, 'low': 10, 'close': 11, 'volume': 1},
    ]
    df = mk_df(seq)
    bars = Bars.from_dataframe(df)
    ha = HeikinAshiAdapter()
    out = ha.transform(bars)
    assert len(out.bars) == 2
    assert out.bars[0].close == (10 + 11 + 9 + 10.5) / 4.0


def test_renko_fixed_box():
    # ascending close sequence to generate bricks
    seq = []
    for i in range(10):
        seq.append({'open': 100 + i, 'high': 100 + i + 0.5, 'low': 100 + i - 0.5, 'close': 100 + i, 'volume': 1})
    df = mk_df(seq)
    bars = Bars.from_dataframe(df)
    ren = RenkoAdapter()
    out = ren.transform(bars, box_size=1.0)
    assert len(out.bars) >= 5


def test_candlestick_hammer_detect():
    # hammer has long lower wick
    seq = [
        {'open': 10, 'high': 10.2, 'low': 9.0, 'close': 10.1, 'volume': 1},
    ]
    df = mk_df(seq)
    bars = Bars.from_dataframe(df)
    lib = CandlestickPatternLibrary()
    det = lib.detect(bars)
    names = [d.name for d in det]
    assert 'hammer' in names
