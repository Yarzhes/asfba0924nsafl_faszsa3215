import time
from ultra_signals.econ.service import EconEventService, _stable_event_id
from ultra_signals.core.custom_types import EconomicEvent, EconEventClass, EconSeverity, EconEventStatus


def test_surprise_and_pre_alert_and_live_done_calls():
    calls = []
    def cb(ev, phase, threshold=None):
        calls.append((ev.id, phase, threshold))
    now_ms = int(time.time()*1000)
    ev = EconomicEvent(
        id=_stable_event_id('test','1', now_ms + 20*60*1000, 'Test CPI'),
        source='test',
        raw_id='1',
        cls=EconEventClass.CPI,
        title='Test CPI',
        severity=EconSeverity.HIGH,
        ts_start=now_ms + 20*60*1000,
        ts_end=now_ms + 21*60*1000,
        status=EconEventStatus.SCHEDULED,
        risk_pre_min=30,
        risk_post_min=30,
        expected='3.0',
        actual=None
    )
    svc = EconEventService({'refresh_min':0,'telegram_callback':cb,'alert_minutes':[30,10]})
    svc.events[ev.id] = ev
    # First refresh triggers 30m pre alert
    svc.refresh(now_ms)
    assert any(ph=='pre' and thr==30 for _,ph,thr in calls)
    # Move close to 9m before -> trigger 10m alert
    svc.refresh(ev.ts_start - 9*60*1000)
    assert any(ph=='pre' and thr==10 for _,ph,thr in calls)
    # Move to live: provide actual
    ev.actual = '3.5'
    svc.refresh(ev.ts_start + 2*1000)  # just after start
    assert any(ph=='live' for _,ph,_ in calls)
    # Surprise should be computed
    assert ev.surprise_score is not None
    # After end
    svc.refresh(ev.ts_end + 60*1000)
    assert any(ph=='done' for _,ph,_ in calls)
