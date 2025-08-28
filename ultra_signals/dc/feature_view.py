from typing import List, Dict, Any, Optional
import time
import json
from pathlib import Path

try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # type: ignore


class FeatureView:
    """Convert recent events into model-ready features.

    Minimal implementation: compute dc_state, dc_age_events, dc_age_time_s,
    os_range_pct, dc_event_rate (events/sec), dead_zone_flag, early_turn_flag
    """

    def __init__(self, window_events: int = 50):
        self.window = window_events

    def features_from_events(self, events: List[Dict[str, Any]], now_ts: Optional[float] = None) -> Dict[str, Any]:
        if now_ts is None:
            now_ts = time.time()
        if not events:
            return {}
        last_dc = None
        dc_count = 0
        last_ts = None
        os_ranges = []
        for e in reversed(events[-self.window:]):
            etype = e.get("type")
            last_ts = e.get("timestamp", last_ts)
            if etype in ("DC_UP", "DC_DOWN") and last_dc is None:
                last_dc = e
            if etype in ("DC_UP", "DC_DOWN"):
                dc_count += 1
            if e.get("os_range") is not None:
                os_ranges.append(e.get("os_range"))

        dc_state = last_dc.get("direction") if last_dc else None
        dc_age_time_s = (now_ts - last_dc.get("timestamp")) if last_dc and last_dc.get("timestamp") else None
        dc_age_events = 0
        # count events since last DC
        for e in reversed(events):
            if last_dc and e.get("event_id") == last_dc.get("event_id"):
                break
            dc_age_events += 1

        duration_s = (now_ts - events[0].get("timestamp")) if events and events[0].get("timestamp") else None
        event_rate = (len(events) / duration_s) if duration_s and duration_s > 0 else None

        os_mean = (sum(os_ranges) / len(os_ranges)) if os_ranges else None
        os_max = max(os_ranges) if os_ranges else None

        # simple flags
        dead_zone_flag = os_max is None or (os_max is not None and os_max < 1e-6)
        early_turn_flag = (os_mean is not None and os_mean < 0.0005 and dc_state is not None)

        return {
            "dc_state": dc_state,
            "dc_age_time_s": dc_age_time_s,
            "dc_age_events": dc_age_events,
            "os_mean": os_mean,
            "os_max": os_max,
            "dc_event_rate_e_per_s": event_rate,
            "dead_zone_flag": dead_zone_flag,
            "early_turn_flag": early_turn_flag,
        }

    def write_manifest(self, path: str, version: str = "v1") -> None:
        """Write a tiny JSON manifest describing the feature view fields and version."""
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m = {
            "version": version,
            "features": [
                "dc_state",
                "dc_age_time_s",
                "dc_age_events",
                "os_mean",
                "os_max",
                "dc_event_rate_e_per_s",
                "dead_zone_flag",
                "early_turn_flag",
            ],
            "generated_at": int(time.time())
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2)

    # ----- helpers to map features into meta-scorer input names -----
    def feature_names(self, prefix: str = 'dc') -> List[str]:
        """Return the canonical list of feature names for the meta-scorer.

        Names are prefixed with the provided namespace (dot-separated).
        """
        base = [
            "dc_state",
            "dc_age_time_s",
            "dc_age_events",
            "os_mean",
            "os_max",
            "dc_event_rate_e_per_s",
            "dead_zone_flag",
            "early_turn_flag",
        ]
        return [f"{prefix}.{n}" for n in base]

    def mapped_features(self, feats: Dict[str, Any], prefix: str = 'dc') -> Dict[str, float]:
        """Map internal FeatureView feature dict into numeric meta-scorer inputs.

        Conversions:
          - dc_state ("UP"/"DOWN") -> 1 / -1 / 0
          - booleans -> 1.0/0.0
          - None -> 0.0
        Returns dict keyed by feature_names() order.
        """
        out: Dict[str, float] = {}
        def tof(v):
            if v is None:
                return 0.0
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            try:
                return float(v)
            except Exception:
                return 0.0

        # dc_state mapping
        s = feats.get('dc_state')
        if isinstance(s, str):
            s_u = s.upper()
            if s_u in ('UP', 'LONG'):
                dc_state_val = 1.0
            elif s_u in ('DOWN', 'SHORT'):
                dc_state_val = -1.0
            else:
                dc_state_val = 0.0
        else:
            dc_state_val = 0.0

        mapping = {
            f"{prefix}.dc_state": dc_state_val,
            f"{prefix}.dc_age_time_s": tof(feats.get('dc_age_time_s')),
            f"{prefix}.dc_age_events": tof(feats.get('dc_age_events')),
            f"{prefix}.os_mean": tof(feats.get('os_mean')),
            f"{prefix}.os_max": tof(feats.get('os_max')),
            f"{prefix}.dc_event_rate_e_per_s": tof(feats.get('dc_event_rate_e_per_s')),
            f"{prefix}.dead_zone_flag": tof(feats.get('dead_zone_flag')),
            f"{prefix}.early_turn_flag": tof(feats.get('early_turn_flag')),
        }
        out.update(mapping)
        return out

    def register_with_meta(self, settings: Optional[Dict[str, Any]] = None, bundle_path: Optional[str] = None, prefix: str = 'dc') -> Dict[str, Any]:
        """Convenience helper to register this FeatureView's feature names with the meta-scorer.

        Behavior:
          - If `bundle_path` is provided and points to a joblib bundle that is a dict,
            inject/update the key 'feature_names' in the bundle and save it back (requires joblib).
          - If `settings` is provided, update settings['meta_scorer']['input_features'] to the generated list.

        Returns a dict describing actions taken.
        """
        names = self.feature_names(prefix=prefix)
        out = {'names': names, 'settings_updated': False, 'bundle_updated': False, 'errors': []}

        # Update settings in-memory
        if settings is not None:
            try:
                ms = settings.setdefault('meta_scorer', {})
                ms['input_features'] = list(names)
                out['settings_updated'] = True
            except Exception as e:
                out['errors'].append(f'settings_update:{e}')

        # Update joblib bundle if requested
        if bundle_path:
            try:
                p = Path(bundle_path)
                if not p.exists():
                    out['errors'].append('bundle_missing')
                elif joblib is None:
                    out['errors'].append('joblib_missing')
                else:
                    mdl = joblib.load(str(p))
                    if isinstance(mdl, dict):
                        mdl['feature_names'] = list(names)
                        joblib.dump(mdl, str(p))
                        out['bundle_updated'] = True
                    else:
                        out['errors'].append('bundle_not_dict')
            except Exception as e:
                out['errors'].append(f'bundle_update:{e}')

        return out
