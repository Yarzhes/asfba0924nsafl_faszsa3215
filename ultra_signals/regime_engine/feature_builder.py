"""Feature assembly utilities for Sprint 43 Meta-Regime Engine.

Transforms a `FeatureVector` (rich nested object) into a flat numeric dict
ready for clustering / modeling:
- Selects and namespaces features from price/vol, sentiment, whales, macro, positioning.
- Applies safe NaN handling (replace None/NaN with 0.0 or configured neutral).
- Maintains an ordered column list for matrix assembly.

Design:
- `FeatureAssembler` maintains rolling window (deque) of last N flat rows.
- Provides `matrix()` returning (rows x cols) numpy array standardized by an
  internal `StandardScaler` (fit on current buffer) for unsupervised models.
- Avoids permanent state coupling (stateless transform function + light state).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
from collections import deque
import numpy as np
import math

from ultra_signals.core.custom_types import FeatureVector

_NUMERIC_SENTIMENT_KEYS = [
    # Expected keys inside sentiment snapshot (if embedded later)
    'sent_score_s','sent_z_s','fg_index','funding_z'
]

_WHALE_KEYS = [
    'whale_net_inflow_usd_s','whale_net_inflow_usd_m','whale_net_inflow_usd_l',
    'whale_inflow_z_s','whale_inflow_z_m','block_trade_notional_5m',
    'block_trade_notional_p99_z','sweep_sell_flag','sweep_buy_flag',
    'opt_call_put_volratio_z','opt_oi_delta_1h_z','opt_skew_shift_z',
    'composite_pressure_score'
]

_MACRO_KEYS = [
    'btc_spy_corr_30m','btc_spy_corr_4h','btc_spy_corr_1d','btc_spy_corr_1w',
    'btc_dxy_corr_4h','btc_spy_corr_trend_1d','btc_spy_corr_z_1d',
    'btc_vix_proxy','btc_vix_proxy_z','realized_vol_24h',
    'risk_on_prob','risk_off_prob','liquidity_squeeze_prob',
    'carry_unwind_flag','dxy_surge_flag','oil_price_shock_z','gold_safehaven_flow_z'
]

_PRICE_VOL_KEYS = [
    # Trend / momentum / volatility subset
    'trend.adx','trend.ema_short','trend.ema_medium','trend.ema_long',
    'momentum.rsi','volatility.atr','volatility.atr_percentile','volatility.bb_kc_ratio'
]

_POSITIONING_KEYS: List[str] = []  # placeholder for funding/oi/basis derived metrics

@dataclass
class FeatureAssembler:
    max_rows: int = 5000
    buffer: deque = field(default_factory=lambda: deque(maxlen=5000))
    columns: List[str] = field(default_factory=list)

    def _flatten(self, fv: FeatureVector) -> Dict[str, float]:
        row: Dict[str, float] = {}
        # Price/Trend
        if fv.trend:
            row['trend.adx'] = fv.trend.adx or 0.0
            row['trend.ema_short'] = fv.trend.ema_short or 0.0
            row['trend.ema_medium'] = fv.trend.ema_medium or 0.0
            row['trend.ema_long'] = fv.trend.ema_long or 0.0
        if fv.momentum:
            row['momentum.rsi'] = fv.momentum.rsi or 0.0
        if fv.volatility:
            row['volatility.atr'] = fv.volatility.atr or 0.0
            row['volatility.atr_percentile'] = fv.volatility.atr_percentile or 0.0
        if fv.alpha_v2:
            row['volatility.bb_kc_ratio'] = fv.alpha_v2.bb_kc_ratio or 0.0
        # Macro
        if fv.macro:
            m = fv.macro
            for k in _MACRO_KEYS:
                val = getattr(m, k, None)
                if isinstance(val, (int,float)) and not math.isnan(val):
                    row[f'macro.{k}'] = float(val)
                else:
                    row[f'macro.{k}'] = 0.0
        # Whales
        if fv.whales:
            w = fv.whales
            for k in _WHALE_KEYS:
                val = getattr(w, k, None)
                if isinstance(val, (int,float)) and not math.isnan(val):
                    row[f'whale.{k}'] = float(val)
                else:
                    row[f'whale.{k}'] = 0.0
        # Regime legacy features
        if fv.regime:
            r = fv.regime
            row['regime.confidence'] = r.confidence or 0.0
            row['regime.vol_state'] = {'crush':0,'normal':0.5,'expansion':1}.get(r.vol_state.value,0.5)
        # Positioning placeholder (extend when available)
        # Ensure deterministic column order accumulation
        if not self.columns:
            self.columns = sorted(row.keys())
        else:
            for c in row.keys():
                if c not in self.columns:
                    self.columns.append(c)
        return row

    def push(self, fv: FeatureVector) -> Dict[str, float]:
        flat = self._flatten(fv)
        self.buffer.append(flat)
        return flat

    def matrix(self) -> Tuple[np.ndarray, List[str]]:
        if not self.buffer:
            return np.zeros((0,len(self.columns))), list(self.columns)
        cols = self.columns
        data = np.array([[row.get(c,0.0) for c in cols] for row in self.buffer], dtype=float)
        # Standard scale (mean 0, std 1) defensive
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        stds[stds==0] = 1.0
        data = (data - means) / stds
        return data, cols

__all__ = ["FeatureAssembler"]
