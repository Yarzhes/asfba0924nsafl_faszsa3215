from ultra_signals.core.config import load_settings

s = load_settings("wf_config.yaml")

def safe_model_dump(obj):
    if hasattr(obj, "model_dump"):   # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):         # Pydantic v1
        return obj.dict()
    return obj

# Try attribute-style first
print("\n=== Attribute view ===")
try:
    wf = getattr(s, "walkforward", None)
    print("has walkforward:", wf is not None)
    if wf is not None:
        for k in ("analysis_start_date","analysis_end_date","train_days","test_days","purge_days"):
            print(f"{k:>20}:", getattr(wf, k, None))
        w = getattr(wf, "window", None)
        if w is not None:
            for k in ("train_period","test_period","advance_by"):
                print(f"window.{k:>13}:", getattr(w, k, None))
        dr = getattr(wf, "data_rules", None)
        if dr is not None:
            for k in ("embargo_period","purge_period"):
                print(f"data_rules.{k:>13}:", getattr(dr, k, None))
except Exception as e:
    print("Attr view error:", e)

# Full dump as dict for sanity
print("\n=== model_dump / dict ===")
dump = safe_model_dump(s)
print("keys at root:", list(dump.keys()))
wf_dump = dump.get("walkforward")
print("walkforward present:", wf_dump is not None)
if wf_dump:
    for k in ("analysis_start_date","analysis_end_date","train_days","test_days","purge_days"):
        print(f"{k:>20}:", wf_dump.get(k))
    win = wf_dump.get("window") or {}
    print("window.train_period :", (win or {}).get("train_period"))
    print("window.test_period  :", (win or {}).get("test_period"))
    print("window.advance_by   :", (win or {}).get("advance_by"))
    dr = wf_dump.get("data_rules") or {}
    print("data_rules.embargo_period:", dr.get("embargo_period"))
    print("data_rules.purge_period  :", dr.get("purge_period"))
