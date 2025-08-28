from ultra_signals.lowlat.ringbuffer import RingBuffer


def test_ringbuffer_basic():
    rb = RingBuffer(4)
    assert len(rb) == 0
    assert rb.push(1)
    assert rb.push(2)
    assert len(rb) == 2
    assert rb.pop() == 1
    assert rb.pop() == 2
    assert rb.pop() is None


def test_ringbuffer_overflow():
    rb = RingBuffer(2)
    assert rb.push('a')
    assert rb.push('b')
    assert not rb.push('c')
    assert len(rb) == 2
    assert rb.pop() == 'a'
    assert rb.pop() == 'b'
