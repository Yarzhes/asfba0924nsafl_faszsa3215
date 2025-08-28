import pytest

from ultra_signals.market.book_rebuilder import BookRebuilder, RebuildError
from ultra_signals.dc.tick_replayer import TickReplayer


def test_snapshot_and_delta_basic():
    br = BookRebuilder(max_levels=5)
    snap = {
        "bids": [(100.0, 1.0), (99.5, 2.0)],
        "asks": [(100.5, 1.5), (101.0, 3.0)],
        "seq": 10,
    }
    br.load_snapshot(snap)
    assert br.best_bid() == 100.0
    assert br.best_ask() == 100.5
    # apply delta: remove top ask
    delta = {"seq": 11, "asks": [(100.5, 0.0), (101.5, 2.0)]}
    br.apply_delta(delta)
    assert br.best_ask() == 101.0 or br.best_ask() == 101.0


def test_snapshot_qa_fails_on_negative_spread():
    br = BookRebuilder()
    # ask below bid -> should raise
    snap = {"bids": [(100.0, 1.0)], "asks": [(99.0, 1.0)], "seq": 1}
    with pytest.raises(RebuildError):
        br.load_snapshot(snap)


def test_tick_replayer_basic_fill():
    tr = TickReplayer()
    # initial snapshot
    tr.add_event({"ts": 1000, "type": "snapshot", "data": {"bids": [(99.0, 2.0)], "asks": [(101.0, 1.0), (102.0, 5.0)], "seq": 1}})
    # incoming buy trade that should consume top ask
    tr.add_event({"ts": 1001, "type": "trade", "side": "buy", "size": 0.6, "price": 101.0})
    # incoming larger buy that sweeps into deeper level
    tr.add_event({"ts": 1002, "type": "trade", "side": "buy", "size": 2.0, "price": 101.5})
    fills = tr.replay()
    assert len(fills) == 2
    assert fills[0]["filled_qty"] == pytest.approx(0.6)
    # second fill will consume remaining ask at 101.0 (0.4) and 1.6 at 102.0
    assert fills[1]["filled_qty"] == pytest.approx(2.0)
