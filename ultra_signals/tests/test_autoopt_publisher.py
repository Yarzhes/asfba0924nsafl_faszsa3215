from pathlib import Path
from ultra_signals.autoopt.publisher import publish_champion, fingerprint

def test_publisher_basic(tmp_path: Path):
    res = publish_champion(tmp_path,'BTCUSDT','5m','trend',{'a':1},{}, {'score':1})
    assert Path(res['path']).exists()
    date_str = __import__('time').strftime('%Y%m%d')
    idx = tmp_path / 'profiles' / 'auto' / f"auto_BTCUSDT_5m_trend_{date_str}_v{res['version']}.yaml"
    assert idx.exists()
    # fingerprint stable
    assert fingerprint({'x':1}) == fingerprint({'x':1})
