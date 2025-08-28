import time
from ultra_signals.econ.service import EconEventService
from ultra_signals.core.custom_types import EconomicEvent, EconEventClass, EconSeverity, EconEventStatus

def make_ev(cls, sev, start_offset_min, dur_min=5, pre=30, post=30):
    now_ms = int(time.time()*1000)
    ts_start = now_ms + start_offset_min*60*1000
    return EconomicEvent(
        id=f"{cls}-{sev}-{start_offset_min}",
        source='test',
        raw_id=None,
        cls=cls,
        title=f"{cls.value} {sev}",
        severity=sev,
        ts_start=ts_start,
        ts_end=ts_start + dur_min*60*1000,
        status=EconEventStatus.SCHEDULED,
        risk_pre_min=pre,
        risk_post_min=post
    )


def test_status_transition_live_done():
    now_ms = int(time.time()*1000)
    svc = EconEventService({'refresh_min':0})
    # Inject one event directly
    ev = make_ev(EconEventClass.CPI, EconSeverity.MED, start_offset_min=0, dur_min=1)
    ev.ts_start = now_ms - 1000  # started 1s ago
    svc.events[ev.id] = ev
    svc._advance_status(now_ms)
    assert ev.status == EconEventStatus.LIVE
    # advance beyond end
    svc._advance_status(now_ms + 2*60*1000)
    assert ev.status == EconEventStatus.DONE


def test_size_mult_min_across_events():
    svc = EconEventService({'refresh_min':0})
    # Two overlapping events different severities
    ev_high = make_ev(EconEventClass.FOMC, EconSeverity.HIGH, start_offset_min=5)
    ev_med = make_ev(EconEventClass.CPI, EconSeverity.MED, start_offset_min=5)
    svc.events[ev_high.id] = ev_high
    svc.events[ev_med.id] = ev_med
    now_ms = ev_high.ts_start - 10*60*1000  # inside pre (risk_pre=30)
    feats = svc.build_features(now_ms)
    # severity policy: high -> size_mult 0.0, med -> 0.5 => min == 0.0
    assert feats.allowed_size_mult_econ == 0.0
