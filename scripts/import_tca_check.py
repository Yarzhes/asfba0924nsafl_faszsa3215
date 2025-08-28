import traceback
try:
    import ultra_signals.tca.tca_engine as te
    print('OK')
    print('TCAEngine attrs:', [a for a in dir(te.TCAEngine) if not a.startswith('_')])
except Exception:
    traceback.print_exc()
