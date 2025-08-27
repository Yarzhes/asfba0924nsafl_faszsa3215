from ultra_signals.opt.stop_resolver import resolve_stop

SETTINGS = {
    'auto_stop_opt': {
        'enabled': True,
        'mode': 'atr',
        'output_path': 'tests/data/stop_table_test.yaml'
    },
    'risk': {'adaptive_exits': {'atr_mult_stop': 1.2}}
}

# create a small table in-memory
import yaml, os
from pathlib import Path
Path('tests/data').mkdir(parents=True, exist_ok=True)
Path('tests/data/stop_table_test.yaml').write_text(yaml.safe_dump({
    'BTCUSDT': {'5m': {'trend': {'mode':'atr','value':1.4}}},
    '*': {'*': {'mixed': {'mode':'atr','value':1.0}}}
}, sort_keys=False))

def test_resolver_atr_distance():
    d = resolve_stop('BTCUSDT','5m','trend', atr=100.0, price=50000.0, settings=SETTINGS)
    assert abs(d - 140.0) < 1e-6


def test_resolver_fallback():
    d = resolve_stop('ETHUSDT','5m','mean_revert', atr=50.0, price=2000.0, settings=SETTINGS)
    # falls back to global mixed 1.0*atr
    assert abs(d - 50.0) < 1e-6
