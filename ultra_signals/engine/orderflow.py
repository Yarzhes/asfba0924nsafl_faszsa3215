from __future__ import annotations
"""
Order Flow Analyzer (Sprint 14)
--------------------------------
Light-weight, in-process order flow intelligence module.

Data Expectations (for mock/backtest environments):
- trades: list[dict] with keys: 'price','qty','side' (side in {'buy','sell'} or boolean is_buyer_maker)
- liquidations: list[dict] with keys: 'side' ('long'/'short' liquidated), 'notional'
- orderbook: dict with keys: 'bids','asks' where each is list[[price, qty], ...] top 5-10 levels

We stay intentionally defensive: all methods catch exceptions and return None/False/{}.
The RealSignalEngine will treat missing metrics as neutral (no boost / no penalty).
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math

@dataclass
class OrderFlowSnapshot:
    cvd: Optional[float] = None
    cvd_chg: Optional[float] = None
    liq_long_notional: Optional[float] = None
    liq_short_notional: Optional[float] = None
    liq_side_dominant: Optional[str] = None  # 'long' or 'short'
    liq_impulse: Optional[float] = None      # relative spike factor
    sweep_side: Optional[str] = None         # 'bid','ask' or None
    sweep_flag: bool = False
    meta: Dict[str, Any] = None

class OrderFlowAnalyzer:
    @staticmethod
    def compute_cvd(trades: List[Dict[str, Any]], prev_cvd: float | None = None) -> Tuple[Optional[float], Optional[float]]:
        """Compute cumulative volume delta. Returns (cvd, delta_change).
        Buy volume contributes +qty, sell volume -qty.
        If prev_cvd is given, delta_change = cvd - prev_cvd.
        """
        if not trades:
            return prev_cvd, None
        cvd = 0.0 if prev_cvd is None else float(prev_cvd)
        try:
            for t in trades:
                qty = float(t.get('qty') or t.get('quantity') or 0.0)
                side = t.get('side')
                if side is None and 'is_buyer_maker' in t:
                    # Binance aggTrade style: is_buyer_maker True means the buyer is the maker => aggressive seller -> sell volume
                    side = 'sell' if t['is_buyer_maker'] else 'buy'
                if str(side).lower() in ('buy','long'):
                    cvd += qty
                elif str(side).lower() in ('sell','short'):
                    cvd -= qty
            chg = None if prev_cvd is None else cvd - float(prev_cvd)
            return cvd, chg
        except Exception:
            return prev_cvd, None

    @staticmethod
    def detect_liquidation_clusters(liqs: List[Dict[str, Any]], notional_threshold: float, window: int = 50) -> Dict[str, Any]:
        """Aggregate liquidation notional; flag side dominance if threshold exceeded.
        Returns dict with keys: long_notional, short_notional, dominant, impulse.
        """
        out = {"long_notional": 0.0, "short_notional": 0.0, "dominant": None, "impulse": 0.0}
        if not liqs:
            return out
        try:
            for ev in liqs[-window:]:
                side = str(ev.get('side','')).lower()  # side liquidated
                notional = float(ev.get('notional') or ev.get('usd') or 0.0)
                if side in ('long','buy'):
                    out['long_notional'] += notional
                elif side in ('short','sell'):
                    out['short_notional'] += notional
            if out['long_notional'] > out['short_notional'] and out['long_notional'] >= notional_threshold:
                out['dominant'] = 'long'
            elif out['short_notional'] > out['long_notional'] and out['short_notional'] >= notional_threshold:
                out['dominant'] = 'short'
            total = out['long_notional'] + out['short_notional']
            # impulse = how large vs threshold (capped at 3x for stability)
            if total > 0:
                out['impulse'] = min(3.0, (max(out['long_notional'], out['short_notional']) / max(1.0, notional_threshold)))
            return out
        except Exception:
            return out

    @staticmethod
    def detect_liquidity_sweep(orderbook: Dict[str, Any], imbalance_threshold: float = 2.5) -> Dict[str, Any]:
        """Detect a potential liquidity sweep / stop hunt by comparing aggregated bid/ask size.
        Returns: {'sweep_side': 'bid'/'ask'/None, 'imbalance': float}
        If bids >> asks => possible ask-side sweep (price ran stops above); we treat as short opportunity.
        If asks >> bids => possible bid-side sweep (stop run below); treat as long opportunity.
        """
        result = {"sweep_side": None, "imbalance": 0.0}
        if not orderbook:
            return result
        try:
            bids = orderbook.get('bids') or []
            asks = orderbook.get('asks') or []
            def _agg(levels):
                total = 0.0
                for lv in levels[:10]:
                    try:
                        total += float(lv[1])
                    except Exception:
                        continue
                return total
            bid_qty = _agg(bids)
            ask_qty = _agg(asks)
            if bid_qty <= 0 or ask_qty <= 0:
                return result
            if bid_qty / ask_qty >= imbalance_threshold:
                result['sweep_side'] = 'ask'  # price likely swept above (ask side liquidity taken)
                result['imbalance'] = bid_qty / ask_qty
            elif ask_qty / bid_qty >= imbalance_threshold:
                result['sweep_side'] = 'bid'
                result['imbalance'] = ask_qty / bid_qty
            return result
        except Exception:
            return result

    @staticmethod
    def build_snapshot(trades: List[Dict[str, Any]] | None,
                       liquidations: List[Dict[str, Any]] | None,
                       orderbook: Dict[str, Any] | None,
                       settings: Dict[str, Any],
                       prev_cvd: float | None = None) -> OrderFlowSnapshot:
        cfg = (settings.get('orderflow') or {}) if isinstance(settings, dict) else {}
        liq_thr = float(cfg.get('liquidation_threshold', 1_000_000))
        cvd = None
        cvd_chg = None
        if trades:
            cvd, cvd_chg = OrderFlowAnalyzer.compute_cvd(trades, prev_cvd)
        liq_info = OrderFlowAnalyzer.detect_liquidation_clusters(liquidations or [], liq_thr) if liquidations else {"long_notional":0.0,"short_notional":0.0,"dominant":None,"impulse":0.0}
        sweep = OrderFlowAnalyzer.detect_liquidity_sweep(orderbook or {}) if orderbook else {"sweep_side":None,"imbalance":0.0}
        return OrderFlowSnapshot(
            cvd=cvd,
            cvd_chg=cvd_chg,
            liq_long_notional=liq_info.get('long_notional'),
            liq_short_notional=liq_info.get('short_notional'),
            liq_side_dominant=liq_info.get('dominant'),
            liq_impulse=liq_info.get('impulse'),
            sweep_side=sweep.get('sweep_side'),
            sweep_flag=bool(sweep.get('sweep_side')),
            meta={"imbalance": sweep.get('imbalance')}
        )


# -----------------------------
# Confidence Modulation Helper
# -----------------------------
def apply_orderflow_modulation(direction: str,
                               confidence: float,
                               snapshot: OrderFlowSnapshot | None,
                               cfg: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """Apply Sprint 14 orderflow confidence adjustments.

    Rules (mirrors logic previously in real_engine):
      - CVD directional confirmation adds cvd_weight
      - Liquidation dominance contrarian reversal (dominant shorts -> LONG boost, dominant longs -> SHORT boost)
        scaled by impulse/2 capped at liq_weight
      - Liquidity sweep alignment adds sweep_weight
      - Conflicting sweep halves confidence
      - Total boost capped at +30%

    Returns: (new_confidence, detail_dict)
    """
    if snapshot is None or direction not in ("LONG", "SHORT"):
        return confidence, {"boost_applied": 0.0}
    try:
        cvd_w = float(cfg.get("cvd_weight", 0.4))
        liq_w = float(cfg.get("liquidation_weight", 0.3))
        sweep_w = float(cfg.get("liquidity_sweep_weight", 0.3))
    except Exception:
        cvd_w, liq_w, sweep_w = 0.4, 0.3, 0.3

    modifiers: list[float] = []
    new_conf = float(confidence)

    # CVD confirmation
    try:
        if snapshot.cvd_chg is not None:
            if direction == "LONG" and snapshot.cvd_chg > 0:
                modifiers.append(cvd_w)
            elif direction == "SHORT" and snapshot.cvd_chg < 0:
                modifiers.append(cvd_w)
    except Exception:
        pass

    # Liquidation contrarian reversal boost
    try:
        if snapshot.liq_side_dominant and snapshot.liq_impulse is not None:
            if direction == "LONG" and snapshot.liq_side_dominant == "short":
                modifiers.append(liq_w * min(1.0, snapshot.liq_impulse / 2.0))
            elif direction == "SHORT" and snapshot.liq_side_dominant == "long":
                modifiers.append(liq_w * min(1.0, snapshot.liq_impulse / 2.0))
    except Exception:
        pass

    # Liquidity sweep alignment / conflict
    sweep_conflict = False
    try:
        if snapshot.sweep_side:
            if direction == "SHORT" and snapshot.sweep_side == "ask":
                modifiers.append(sweep_w)
            elif direction == "LONG" and snapshot.sweep_side == "bid":
                modifiers.append(sweep_w)
            else:
                # conflicting sweep
                sweep_conflict = True
    except Exception:
        pass

    boost_applied = 0.0
    if modifiers:
        boost_applied = min(0.30, sum(modifiers))
        new_conf = min(1.0, new_conf * (1.0 + boost_applied))

    if sweep_conflict:
        new_conf *= 0.5

    detail = {
        "cvd": snapshot.cvd,
        "cvd_chg": snapshot.cvd_chg,
        "liq_long": snapshot.liq_long_notional,
        "liq_short": snapshot.liq_short_notional,
        "liq_dom": snapshot.liq_side_dominant,
        "liq_impulse": snapshot.liq_impulse,
        "sweep_side": snapshot.sweep_side,
        "sweep_flag": snapshot.sweep_flag,
        "boost_applied": round(boost_applied, 4),
        "conflict_halved": sweep_conflict
    }
    return new_conf, detail

__all__ = [
    'OrderFlowAnalyzer', 'OrderFlowSnapshot', 'apply_orderflow_modulation'
]
