def confluence_htf_agrees(signal, feature_store, settings) -> bool:
    """
    Require the higher timeframe regime to agree with the signal direction.
    Mapping: settings.confluence.map (e.g., '15m' -> '1h')
    """
    cmap = settings.get("confluence", {}).get("map", {})
    htf = cmap.get(signal.timeframe)
    if not htf:
        return True  # no mapping → allow

    htf_regime = feature_store.get_regime(signal.symbol, htf)  # e.g., "trend" or "mr" or "mixed"
    if htf_regime is None:
        return True  # unknown → don't block

    # Simple rule: if HTF = trend up → only LONG; if trend down → only SHORT; if mixed → allow both
    if htf_regime == "trend_up" and signal.decision == "SHORT":
        return False
    if htf_regime == "trend_down" and signal.decision == "LONG":
        return False
    return True
