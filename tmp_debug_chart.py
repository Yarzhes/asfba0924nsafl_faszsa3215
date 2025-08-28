from ultra_signals.patterns.chart_patterns import ChartPatternLibrary
from ultra_signals.patterns.base import Bars
import pandas as pd

seq = []
for v in [110,108,106,104]:
    seq.append({'open': v, 'high': v+1, 'low': v-1, 'close': v, 'volume': 1})
for v in [102,101,101,102]:
    seq.append({'open': v, 'high': v+0.5, 'low': v-0.5, 'close': v, 'volume': 1})
for v in [104,106,108,110]:
    seq.append({'open': v, 'high': v+1, 'low': v-1, 'close': v, 'volume': 1})
seq.append({'open':109,'high':109.5,'low':108.5,'close':109,'volume':1})

idx = pd.date_range('2020-01-01', periods=len(seq), freq='T')
df = pd.DataFrame(seq, index=idx)
bars = Bars.from_dataframe(df)
cp = ChartPatternLibrary(fractal_k=1)

closes = [b.close for b in bars.bars]
n = len(closes)
center_idx = int(min(range(len(closes)), key=lambda i: closes[i]))
center_min = closes[center_idx]
left_max = max(closes[:center_idx]) if center_idx > 0 else None
right_max = max(closes[center_idx + 1 :]) if center_idx < n - 1 else None
handle_region = closes[int(n * 0.75) :]

print('n, center_idx, center_min, left_max, right_max')
print(n, center_idx, center_min, left_max, right_max)
print('handle_region:', handle_region)
print('detected names:', [d.name for d in cp.detect(bars)])
for d in cp.detect(bars):
    print('PAT:', d.name, d.direction, d.strength, d.meta)
