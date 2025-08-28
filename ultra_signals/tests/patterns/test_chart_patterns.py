import pandas as pd
from ultra_signals.patterns.chart_patterns import ChartPatternLibrary
from ultra_signals.patterns.base import Bars


def mk_df(seq):
    idx = pd.date_range('2020-01-01', periods=len(seq), freq='T')
    return pd.DataFrame(seq, index=idx)


def test_triangle_detect():
    # construct simple converging highs and lows
    seq = []
    highs = [10, 9.8, 9.6, 9.5, 9.52, 9.55, 9.57]
    lows = [9, 9.1, 9.25, 9.35, 9.4, 9.45, 9.5]
    closes = [(h + l)/2 for h, l in zip(highs, lows)]
    for o, h, l, c in zip(closes, highs, lows, closes):
        seq.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1})
    df = mk_df(seq)
    bars = Bars.from_dataframe(df)
    lib = ChartPatternLibrary(fractal_k=1)
    det = lib.detect(bars)
    names = [d.name for d in det]
    assert 'sym_triangle' in names


def test_wedge_detect():
    # rising wedge scenario
    seq = []
    highs = [10, 10.2, 10.4, 10.6, 10.8, 11.0]
    lows = [9, 9.3, 9.6, 9.9, 10.2, 10.5]
    closes = [(h + l)/2 for h, l in zip(highs, lows)]
    for o, h, l, c in zip(closes, highs, lows, closes):
        seq.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1})
    df = mk_df(seq)
    bars = Bars.from_dataframe(df)
    lib = ChartPatternLibrary(fractal_k=1)
    det = lib.detect(bars)
    names = [d.name for d in det]
    assert 'rising_wedge' in names


def test_flag_pennant_detect():
    # create a run then small consolidation
    seq = []
    # run up
    for i in range(10):
        seq.append({'open': 100 + i, 'high': 100 + i + 0.5, 'low': 100 + i - 0.5, 'close': 100 + i + 0.4, 'volume': 1})
    # consolidation
    for i in range(6):
        seq.append({'open': 110 + i*0.01, 'high': 110.02 + i*0.01, 'low': 109.98 + i*0.01, 'close': 110 + i*0.01, 'volume': 1})
    df = mk_df(seq)
    bars = Bars.from_dataframe(df)
    lib = ChartPatternLibrary(fractal_k=1)
    det = lib.detect(bars)
    names = [d.name for d in det]
    assert 'flag_pennant' in names


def test_cup_and_handle_detect():
    # create a cup (U-shape) then small handle
    seq = []
    # left rim
    for v in [110, 108, 106, 104]:
        seq.append({'open': v, 'high': v+1, 'low': v-1, 'close': v, 'volume': 1})
    # bottom
    for v in [102, 101, 101, 102]:
        seq.append({'open': v, 'high': v+0.5, 'low': v-0.5, 'close': v, 'volume': 1})
    # right rim
    for v in [104, 106, 108, 110]:
        seq.append({'open': v, 'high': v+1, 'low': v-1, 'close': v, 'volume': 1})
    # handle small pullback
    seq.append({'open': 109, 'high': 109.5, 'low': 108.5, 'close': 109, 'volume': 1})
    df = mk_df(seq)
    bars = Bars.from_dataframe(df)
    lib = ChartPatternLibrary(fractal_k=1)
    det = lib.detect(bars)
    names = [d.name for d in det]
    assert 'cup_and_handle' in names
