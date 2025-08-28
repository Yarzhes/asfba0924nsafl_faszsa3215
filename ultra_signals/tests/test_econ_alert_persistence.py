import time, os, json
from ultra_signals.econ.service import EconEventService, _stable_event_id
from ultra_signals.core.custom_types import EconomicEvent, EconEventClass, EconSeverity, EconEventStatus

def test_alert_persistence_and_severity_thresholds(tmp_path):
    calls=[]
    def cb(ev, phase, threshold=None):
        calls.append((ev.id, phase, threshold))
    now_ms = int(time.time()*1000)
    ev = EconomicEvent(
        id=_stable_event_id('test','2', now_ms + 40*60*1000, 'Test FOMC'),
        source='test', raw_id='2', cls=EconEventClass.FOMC, title='Test FOMC', severity=EconSeverity.HIGH,
        ts_start=now_ms + 40*60*1000, ts_end=now_ms + 41*60*1000, status=EconEventStatus.SCHEDULED, risk_pre_min=60, risk_post_min=60
    )
    persist = tmp_path / 'econ_persist.json'
    svc1 = EconEventService({'refresh_min':0,'telegram_callback':cb,'persist_path':str(persist), 'alert_minutes':[15], 'alert_minutes_severity':{'high':[45,15]}})
    svc1.events[ev.id] = ev
    # At 40m before start -> triggers 45m not yet (still within). Move to 44m remaining => triggers 45
    svc1.refresh(ev.ts_start - 44*60*1000)
    assert any(thr==45 for _,ph,thr in calls if ph=='pre')
    # 20m remaining -> should not fire 15 yet
    svc1.refresh(ev.ts_start - 20*60*1000)
    # 14m remaining -> fire 15
    svc1.refresh(ev.ts_start - 14*60*1000)
    assert any(thr==15 for _,ph,thr in calls if ph=='pre')
    # Persist
    svc1._persist()
    # Load new service instance; alert_state should prevent duplicate 45 & 15 when refreshing again at same times
    calls.clear()
    svc2 = EconEventService({'refresh_min':0,'telegram_callback':cb,'persist_path':str(persist), 'alert_minutes':[15], 'alert_minutes_severity':{'high':[45,15]}})
    # Bring event times forward for new instance by copying
    # Already persisted; service load should populate events
    # Fire at 10m remaining -> no re-firing of 15 if previously fired unless new threshold
    svc2.refresh(ev.ts_start - 10*60*1000)
    assert not any(thr==15 for _,ph,thr in calls if ph=='pre')
