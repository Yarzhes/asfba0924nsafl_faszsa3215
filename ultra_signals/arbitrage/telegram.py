from __future__ import annotations
from .models import ArbitrageFeatureSet

def format_arb_telegram_line(fs: ArbitrageFeatureSet) -> str:
    if not fs.executable_spreads:
        return f"Arb: No spread {fs.symbol}"
    # choose best by largest raw spread
    best = max(fs.executable_spreads, key=lambda e: e.raw_spread_bps)
    # pick a standard bucket (25k) if present
    exec25 = best.exec_spread_bps_by_notional.get('25000') or next(iter(best.exec_spread_bps_by_notional.values()))
    parts = [
        f"Arb: Perp spread {best.raw_spread_bps:.2f} bps (exec {exec25:.2f} bps @ $25k) {best.venue_short.upper()}>{best.venue_long.upper()}"
    ]
    if fs.funding:
        # naive funding diff: max - min current
        cur_rates = [f.current_rate_bps for f in fs.funding if f.current_rate_bps is not None]
        if len(cur_rates) >= 2:
            diff = max(cur_rates) - min(cur_rates)
            parts.append(f"Funding diff {diff:.2f} bps/8h")
    if fs.geo_premium:
        parts.append(f"Geo prem {fs.geo_premium.region_a} {fs.geo_premium.premium_bps:.2f} bps vs {fs.geo_premium.region_b}")
    return ' | '.join(parts)
