import time
from ultra_signals.execution.pricing import build_exec_plan
from ultra_signals.execution.brackets import build_brackets, update_brackets

BASE_SETTINGS = {
    'execution': {
        'maker_first': True,
        'k1_ticks': 1,
        'taker_fallback_ms': 500,
        'taker_offset_ticks': 1,
        'max_spread_pct': 0.06,
        'max_chase_bps': 8,
        'atr_pct_limit': 0.97,
        'max_slip_bps': 12,
        'price_anchor': 'mid',
        'brackets': {
            'enabled': True,
            'stop_atr_mult': 1.4,
            'tp_atr_mults': [1.8,2.6,3.5],
            'tp_scales': [0.5,0.3,0.2],
            'break_even': {'enabled': True,'be_trigger_atr':1.2,'be_lock_ticks':2},
            'trailing': {'enabled': True,'arm_atr':2.0,'trail_atr_mult':1.0}
        }
    }
}

def test_pricing_maker_then_taker_fallback():
    book = {'bid':100.0,'ask':100.1}
    plan = build_exec_plan('BTCUSDT','LONG',book,tick_size=0.1,atr=1.0,atr_pct=0.5,regime='trend',settings=BASE_SETTINGS,now_ms=int(time.time()*1000))
    assert plan is not None
    assert plan.post_only is True
    assert plan.taker_fallback_after_ms == 500
    assert plan.taker_price is not None


def test_price_fences_block_chase():
    # Force chase beyond max_chase_bps by giving large k1_ticks
    settings = BASE_SETTINGS.copy()
    settings['execution'] = dict(BASE_SETTINGS['execution'])
    settings['execution']['k1_ticks'] = 500  # unrealistic big
    book = {'bid':100.0,'ask':100.1}
    plan = build_exec_plan('BTCUSDT','LONG',book,tick_size=0.1,atr=1.0,atr_pct=0.5,regime='trend',settings=settings,now_ms=int(time.time()*1000))
    assert plan is not None
    assert plan.fence_reason == 'chase'


def test_brackets_build_reduce_only():
    b = build_brackets(entry_px=100.0, side='LONG', atr=1.0, size=1.0, execution_cfg=BASE_SETTINGS['execution'])
    assert b is not None
    assert b.stop.kind == 'SL'
    assert all(tp.kind=='TP' for tp in b.tps)
    assert abs(sum(tp.size for tp in b.tps) - 1.0) < 1e-6  # scale normalization


def test_be_and_trailing_progression():
    b = build_brackets(entry_px=100.0, side='LONG', atr=1.0, size=1.0, execution_cfg=BASE_SETTINGS['execution'])
    assert b is not None
    original_stop = b.stop.price
    # Move price to trigger BE (>= 1.2 * ATR move)
    changed = update_brackets(b, current_price=101.3, entry_px=100.0, side='LONG', atr=1.0, tick_size=0.1)
    assert changed is True
    assert b.stop.price >= 100.0  # moved to BE or slightly above
    # Move more to arm trailing (>=2.0 * ATR)
    changed2 = update_brackets(b, current_price=102.5, entry_px=100.0, side='LONG', atr=1.0, tick_size=0.1)
    assert changed2 is True
    assert b.stop.price > original_stop

