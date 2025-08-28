from typing import Dict, Any
from .feature_view import FeatureView

_buffers: Dict[str, list] = {}
_fv = FeatureView()


def dc_post_bar_hook(symbol: str, timeframe: str, bar_row, feature_store) -> None:
    """Hook to run after FeatureStore on_bar: gather recent DC-like events (if any)
    and produce FeatureView features stored into feature cache under 'dc' key.

    For now, this is lightweight: read any buffered DC events in feature_store if present
    (e.g., from samplers). If none, keep previous.
    """
    try:
        cache = feature_store._feature_cache.setdefault(symbol, {}).setdefault(timeframe, {})
        ts = bar_row.index[0]
        # try to find any dc event buffer on the feature_store (signal engines may add them)
        ext_buf = getattr(feature_store, '_dc_event_buffer', None)
        events = []
        if ext_buf and isinstance(ext_buf, dict):
            events = ext_buf.get(symbol) or []
        # fallback: local in-process buffer
        if not events:
            events = _buffers.get(symbol, [])

        feats = _fv.features_from_events(events, now_ts=None)
        # store under a known key
        bucket = cache.setdefault(ts, {})
        bucket['dc'] = feats
    except Exception:
        pass


def register_with_store(store):
    try:
        store.register_post_bar_hook(dc_post_bar_hook)
        # expose simple atr provider
        def _atr_provider(sym, tf, lookback=14):
            """ATR provider callable exposed on FeatureStore for samplers.

            Accepts (symbol, timeframe, lookback) and returns ATR in price units
            or None if not ready.
            """
            return store.compute_atr(sym, tf, lookback)
        store.atr_provider = _atr_provider
    except Exception:
        pass
