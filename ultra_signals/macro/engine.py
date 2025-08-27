"""Cross-Asset Correlation & Macro Regime Engine (Sprint 42)

Provides:
- Rolling correlations for BTC/ETH vs equities, FX (DXY), commodities, rates.
- Synthetic BTC-VIX proxy (realized + optional Deribit implied vol blend).
- Macro regime classification and carry-unwind detection.
- Z-scored macro shocks (DXY, Oil, Gold).

This is a first-pass minimal implementation; sophisticated modeling (clustering,
probabilistic calibration) can be layered without changing public contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import time

import pandas as pd
import numpy as np
from loguru import logger

from ultra_signals.core.custom_types import MacroFeatures
from ultra_signals.macro.collectors import fetch_deribit_iv

@dataclass
class MacroState:
    last_refresh_ts: float = 0.0
    price_frames: Dict[str, pd.DataFrame] = None  # raw external assets
    btc_history: Dict[str, pd.DataFrame] = None   # timeframe -> df for BTC
    eth_history: Dict[str, pd.DataFrame] = None
    corr_cache: Dict[str, float] = None
    z_cache: Dict[str, float] = None
    corr_history: Dict[str, List[float]] = None  # for slope estimation
    carry_confirm: int = 0

    def __post_init__(self):
        self.price_frames = self.price_frames or {}
        self.btc_history = self.btc_history or {}
        self.eth_history = self.eth_history or {}
        self.corr_cache = self.corr_cache or {}
        self.z_cache = self.z_cache or {}
        self.corr_history = self.corr_history or {}

class MacroEngine:
    def __init__(self, settings: Dict):
        self.settings = settings or {}
        self.state = MacroState()

    # Public contract --------------------------------------------------
    def compute_features(self, now_ts: int, btc_df: pd.DataFrame, eth_df: Optional[pd.DataFrame], externals: Dict[str, pd.DataFrame]) -> MacroFeatures:
        """Return MacroFeatures from latest snapshots.

        Args:
            now_ts: epoch ms for alignment
            btc_df: OHLCV DataFrame (index datetime) for BTC primary tf
            eth_df: optional ETH OHLCV
            externals: mapping external_symbol -> OHLCV DataFrame
        """
        if btc_df is None or btc_df.empty:
            return MacroFeatures()
        self.state.price_frames.update(externals or {})
        self.state.btc_history["primary"] = btc_df
        if eth_df is not None:
            self.state.eth_history["primary"] = eth_df

        cfg = (self.settings or {}).get("cross_asset", {}) or {}
        win_defs = cfg.get("correlation_windows") or []
        # Build canonical close series dict
        closes: Dict[str, pd.Series] = {}
        try:
            closes['BTC'] = btc_df['close'].astype(float).rename('BTC')
        except Exception:
            pass
        if eth_df is not None and not eth_df.empty:
            try:
                closes['ETH'] = eth_df['close'].astype(float).rename('ETH')
            except Exception:
                pass
        for k, df in self.state.price_frames.items():
            if df is None or df.empty:
                continue
            if 'close' in df.columns:
                try:
                    closes[k] = df['close'].astype(float).rename(k)
                except Exception:
                    continue
        # Align
        all_df = pd.concat([s for s in closes.values() if s is not None], axis=1, join='inner').dropna(how='any')
        feats: Dict[str, float] = {}
        if not all_df.empty and 'BTC' in all_df.columns:
            for win in win_defs:
                label = win.get('label') if isinstance(win, dict) else getattr(win, 'label', None)
                bars = win.get('bars') if isinstance(win, dict) else getattr(win, 'bars', 0)
                if not label or not bars or bars < 5:
                    continue
                sub = all_df.tail(bars)
                if len(sub) < 5:
                    continue
                # Example correlations
                def _corr(target):
                    if target in sub.columns:
                        try:
                            return float(sub['BTC'].pct_change().corr(sub[target].pct_change()))
                        except Exception:
                            return None
                    return None
                # Core asset lists
                asset_map = {
                    'spy': 'SPY', 'qqq': 'QQQ', 'dxy': 'DX-Y.NYB', 'gold': 'GC=F', 'oil': 'CL=F', 'vix': '^VIX', 'us10y': '^TNX'
                }
                for key, sym in asset_map.items():
                    if sym not in sub.columns:
                        continue
                    # Pearson
                    pear = _corr(sym)
                    if pear is not None:
                        feats[f"btc_{key}_corr_{label}"] = pear
                    # Spearman
                    try:
                        spear = float(sub['BTC'].pct_change().corr(sub[sym].pct_change(), method='spearman'))
                        feats[f"btc_{key}_spr_{label}"] = spear
                    except Exception:
                        pass
                # ETH vs Gold weekly/daily
                if 'GC=F' in sub.columns and 'ETH' in sub.columns and label in ('1w','1d'):
                    try:
                        feats[f"eth_gold_corr_{label}"] = float(sub['ETH'].pct_change().corr(sub['GC=F'].pct_change()))
                    except Exception:
                        pass
                # Keep history for slope -> using only btc_spy 1d as example
                if label == '1d' and 'btc_spy_corr_1d' in feats:
                    hist = self.state.corr_history.setdefault('btc_spy_corr_1d', [])
                    hist.append(feats['btc_spy_corr_1d'] or 0.0)
                    if len(hist) > 50:
                        self.state.corr_history['btc_spy_corr_1d'] = hist[-50:]
                    if len(hist) >= 5:
                        y = pd.Series(hist[-10:])  # last up to 10 points
                        x = np.arange(len(y))
                        try:
                            slope = float(np.polyfit(x, y, 1)[0])
                            feats['btc_spy_corr_trend_1d'] = slope
                            # z-score of latest corr
                            feats['btc_spy_corr_z_1d'] = self._z(y.values, y.values[-1])
                        except Exception:
                            pass
        # Vol proxies -------------------------------------------------
        feats.update(self._compute_vol_proxies(all_df))
        # Commodity / FX shock z-scores
        self._compute_macro_shocks(all_df, feats)
        # Macro regime heuristic --------------------------------------
        self._compute_macro_regime(feats)
        # Carry unwind detection --------------------------------------
        self._compute_carry_unwind(all_df, feats)
        # Macro extreme flag
        self._compute_extreme_flag(feats)
        feats['last_refresh_ts'] = int(now_ts)
        return MacroFeatures(**feats)

    # Helpers ----------------------------------------------------------
    def _z(self, arr: np.ndarray, value: float) -> Optional[float]:
        if arr is None or len(arr) < 10:
            return None
        m = float(np.nanmean(arr))
        s = float(np.nanstd(arr))
        if s <= 1e-12:
            return None
        try:
            return (float(value) - m) / s
        except Exception:
            return None

    async def _get_deribit_iv(self, cfg: Dict) -> Optional[float]:  # pragma: no cover (network)
        if not cfg.get('use_deribit_iv'):
            return None
        try:
            indices = ['btc_usd']
            ivs = await fetch_deribit_iv(indices)
            return ivs.get('btc_usd')
        except Exception:
            return None

    def _compute_vol_proxies(self, df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if df is None or df.empty or 'BTC' not in df.columns:
            return out
        btc = df['BTC']
        rv = float(np.sqrt(288) * btc.pct_change().tail(288).std()) if len(btc) >= 300 else float(btc.pct_change().std() or 0.0)
        out['realized_vol_24h'] = rv
        # Synthetic BTC-VIX: realized_vol * 100 placeholder (scale) until IV blend
        out['btc_vix_proxy'] = rv * 100.0
        # Attempt blend with Deribit IV (async call executed synchronously via loop if enabled)
        try:
            cfg = (self.settings or {}).get('cross_asset', {}) or {}
            if cfg.get('use_deribit_iv'):
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # schedule task and wait briefly
                    iv = None
                else:
                    iv = loop.run_until_complete(self._get_deribit_iv(cfg))
                if iv is not None:
                    w = float(cfg.get('deribit_iv_weight', 0.5))
                    out['btc_vix_proxy'] = (1 - w) * out['btc_vix_proxy'] + w * float(iv) * 100.0
                    out['btc_vix_iv'] = float(iv) * 100.0
        except Exception:
            pass
        # Maintain z history
        hist = self.state.corr_history.setdefault('btc_vix_proxy', [])
        hist.append(out['btc_vix_proxy'])
        if len(hist) > 400:
            self.state.corr_history['btc_vix_proxy'] = hist[-400:]
        if len(hist) >= 30:
            out['btc_vix_proxy_z'] = self._z(np.array(hist[-120:]), hist[-1])
        return out

    def _compute_macro_regime(self, feats: Dict[str, float]):
        # Simple heuristic combining correlations & vol
        corr = feats.get('btc_spy_corr_1d')
        vix_z = feats.get('btc_vix_proxy_z')
        dxy_corr = feats.get('btc_dxy_corr_1d') if 'btc_dxy_corr_1d' in feats else feats.get('btc_dxy_corr_4h')
        risk_on_score = 0.0
        risk_off_score = 0.0
        if corr is not None:
            risk_on_score += max(0.0, corr)
            risk_off_score += max(0.0, -corr)
        if vix_z is not None:
            risk_on_score += max(0.0, -vix_z)
            risk_off_score += max(0.0, vix_z)
        if dxy_corr is not None and dxy_corr < 0:
            risk_on_score += (-dxy_corr)
        if dxy_corr is not None and dxy_corr > 0:
            risk_off_score += dxy_corr
        # Normalize
        tot = risk_on_score + risk_off_score + 1e-9
        feats['risk_on_prob'] = risk_on_score / tot
        feats['risk_off_prob'] = risk_off_score / tot
        if feats['risk_on_prob'] > 0.55 and corr and corr > 0:
            feats['macro_risk_regime'] = 'risk_on'
        elif feats['risk_off_prob'] > 0.55:
            feats['macro_risk_regime'] = 'risk_off'
        else:
            feats['macro_risk_regime'] = 'neutral'

    def _compute_macro_shocks(self, df: pd.DataFrame, feats: Dict[str, float]):
        if df is None or df.empty:
            return
        for sym, key in [('CL=F','oil'), ('GC=F','gold'), ('DX-Y.NYB','dxy')]:
            if sym not in df.columns:
                continue
            series = df[sym].dropna()
            if len(series) < 50:
                continue
            z = self._z(series.tail(200).values, series.iloc[-1]) if len(series) >= 60 else None
            if z is None:
                continue
            if key == 'oil':
                feats['oil_price_shock_z'] = z
            elif key == 'gold':
                feats['gold_safehaven_flow_z'] = z
            elif key == 'dxy':
                feats.setdefault('dxy_z', z)

    def _compute_extreme_flag(self, feats: Dict[str, float]):
        # Basic extreme condition aggregator
        c = 0
        if feats.get('btc_vix_proxy_z') and abs(feats['btc_vix_proxy_z']) >= 2.0:
            c += 1
        if feats.get('oil_price_shock_z') and abs(feats['oil_price_shock_z']) >= 1.5:
            c += 1
        if feats.get('gold_safehaven_flow_z') and feats['gold_safehaven_flow_z'] >= 1.5:
            c += 1
        if feats.get('dxy_z') and feats['dxy_z'] >= 1.25:
            c += 1
        if c >= 2:
            feats['macro_extreme_flag'] = 1
        elif c == 1:
            feats['macro_extreme_flag'] = 0  # single anomaly not extreme
        else:
            feats['macro_extreme_flag'] = 0

    def _compute_carry_unwind(self, df: pd.DataFrame, feats: Dict[str, float]):
        cfg = (self.settings or {}).get('cross_asset', {}) or {}
        rule = cfg.get('carry_unwind', {}) or {}
        # Need DXY (DX-Y.NYB) and 10Y (^TNX)
        if df is None or df.empty:
            return
        needed = [c for c in ['DX-Y.NYB', '^TNX'] if c in df.columns]
        if len(needed) < 2 or 'BTC' not in df.columns:
            return
        # Use last N rows
        tail = df.tail(50)
        def _z(series: pd.Series):
            if len(series) < 10:
                return None
            return self._z(series.values[-30:], series.values[-1])
        dxy_z = _z(tail['DX-Y.NYB'])
        tnx_z = _z(tail['^TNX'])
        btc_ret = float(tail['BTC'].pct_change().tail(10).sum()) if len(tail) >= 10 else None
        if dxy_z is None or tnx_z is None or btc_ret is None:
            return
        if dxy_z >= rule.get('dxy_z_thr', 1.0) and tnx_z >= rule.get('us10y_z_thr', 1.0) and btc_ret <= rule.get('btc_return_thr', -0.01):
            self.state.carry_confirm += 1
        else:
            self.state.carry_confirm = 0
        if self.state.carry_confirm >= rule.get('confirm_bars', 2):
            feats['carry_unwind_flag'] = 1
        # Expose helper flags
        feats['dxy_surge_flag'] = 1 if dxy_z is not None and dxy_z >= rule.get('dxy_z_thr', 1.0) else 0

