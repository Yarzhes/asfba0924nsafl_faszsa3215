"""Sizing package (Sprint 32+)

Provides the advanced position sizer plus backward-compatible helpers
expected by earlier sprint tests (e.g. ``apply_volatility_scaling``).

We duplicate the small ``apply_volatility_scaling`` helper here so tests
importing ``from ultra_signals.engine.sizing import apply_volatility_scaling``
resolve against the package (directory) rather than the sibling module
``sizing.py`` which would otherwise shadow the package name.
"""

from .advanced_sizer import AdvancedSizer  # re-export

from typing import Dict

def apply_volatility_scaling(base_risk: float, atr_percentile: float, cfg: Dict) -> float:
	"""Scale ``base_risk`` up/down based on ATR percentile using config keys.

	Mirrors the implementation in ``engine/sizing.py`` so legacy tests pass
	regardless of import resolution order (package vs module name clash).
	"""
	try:
		low_pct = cfg.get("low_vol_pct", 30)
		high_pct = cfg.get("high_vol_pct", 70)
		if atr_percentile < low_pct:
			return base_risk * cfg.get("low_vol_boost", 1.0)
		if atr_percentile > high_pct:
			return base_risk * cfg.get("high_vol_cut", 1.0)
		return base_risk
	except Exception:  # pragma: no cover defensive
		return base_risk

__all__ = ["AdvancedSizer", "apply_volatility_scaling"]
