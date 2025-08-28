from __future__ import annotations
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Any
import numpy as np
import time


class VPINEngine:
    """Simple volume-synced VPIN engine.

    Usage:
      engine = VPINEngine(V_bucket=250_000, K_buckets=50)
      for trade in trades: engine.ingest_trade(trade)
      when bucket completes engine.finalize_bucket() will be called internally.
    """

    def __init__(self, V_bucket: float = 250_000.0, K_buckets: int = 50, classifier: str = 'aggressor'):
        self.V_bucket = float(V_bucket)
        self.K_buckets = int(K_buckets)
        self.classifier = classifier
        self.buckets: Deque[Dict[str, Any]] = deque(maxlen=max(2, K_buckets))
        # Keep history of vpin (rolling mean across last K buckets)
        self.vpin_history: List[float] = []
        # current forming bucket
        self.current: Dict[str, Any] = {
            'buy_notional': 0.0,
            'sell_notional': 0.0,
            'notional': 0.0,
            'start_ts': None,
            'end_ts': None,
            'trade_count': 0,
            'class_errors': 0,
        }

    def _classify_trade(self, trade: Tuple[int, float, float, Optional[bool]], book_top: Optional[Dict[str, float]] = None) -> Optional[str]:
        # trade: (ts, price, qty, is_buyer_maker)
        ts, price, qty, is_buyer_maker = trade
        if isinstance(is_buyer_maker, bool):
            # is_buyer_maker == True => taker is seller (aggressor sell)
            return 'SELL' if is_buyer_maker else 'BUY'
        # fallback: use tick-rule against mid from book_top
        try:
            if book_top and 'bid' in book_top and 'ask' in book_top and book_top['bid'] and book_top['ask']:
                mid = (float(book_top['bid']) + float(book_top['ask'])) / 2.0
                if price >= mid:
                    return 'BUY'
                return 'SELL'
        except Exception:
            pass
        return None

    def ingest_trade(self, trade: Tuple[int, float, float, Optional[bool]], book_top: Optional[Dict[str, float]] = None):
        ts, price, qty, is_buyer_maker = trade
        notional = float(price) * float(qty)
        side = self._classify_trade(trade, book_top)
        if side is None:
            self.current['class_errors'] += 1
        if self.current['start_ts'] is None:
            self.current['start_ts'] = int(ts)
        self.current['end_ts'] = int(ts)
        self.current['trade_count'] += 1
        self.current['notional'] += notional
        if side == 'BUY':
            self.current['buy_notional'] += notional
        elif side == 'SELL':
            self.current['sell_notional'] += notional
        # finalize if bucket filled
        if self.current['notional'] >= self.V_bucket:
            self._finalize_bucket()

    def _finalize_bucket(self):
        b = dict(self.current)
        # compute imbalance metric for bucket
        V = max(self.V_bucket, 1e-9)
        imbalance = abs(b.get('buy_notional', 0.0) - b.get('sell_notional', 0.0)) / V
        b['imbalance'] = float(np.clip(imbalance, 0.0, 1.0))
        # store bucket
        self.buckets.append(b)
        # recompute vpin as mean of last K bucket imbalances
        imbalances = [bb.get('imbalance', 0.0) for bb in list(self.buckets)[-self.K_buckets:]]
        vpin = float(np.mean(imbalances)) if imbalances else 0.0
        self.vpin_history.append(vpin)
        # reset current
        self.current = {
            'buy_notional': 0.0,
            'sell_notional': 0.0,
            'notional': 0.0,
            'start_ts': None,
            'end_ts': None,
            'trade_count': 0,
            'class_errors': 0,
        }

    def finalize_partial(self) -> Dict[str, Any]:
        """Return incremental estimate for current forming bucket without finalizing it."""
        cur = self.current
        fill_ratio = min(1.0, (cur.get('notional', 0.0) / max(self.V_bucket, 1e-9)))
        est = {
            'bucket_fill_ratio': float(fill_ratio),
            'bucket_time_sec': None,
            'class_error_est': int(cur.get('class_errors', 0)),
        }
        if cur.get('start_ts') is not None and cur.get('end_ts') is not None:
            est['bucket_time_sec'] = float((cur['end_ts'] - cur['start_ts']) / 1000.0)
        return est

    def get_latest_vpin(self) -> Dict[str, Any]:
        if not self.vpin_history:
            return {'vpin': 0.0, 'vpin_z': 0.0, 'vpin_pctl': 0.0}
        vpin = float(self.vpin_history[-1])
        hist = np.array(self.vpin_history, dtype=float)
        if len(hist) < 3:
            z = 0.0
        else:
            mu = float(np.mean(hist))
            sd = float(np.std(hist, ddof=0)) if float(np.std(hist, ddof=0))>0 else 0.0
            z = float((vpin - mu) / (sd if sd>0 else 1.0))
        # percentile rank
        pctl = float((hist < vpin).sum() / max(1, len(hist)))
        return {'vpin': vpin, 'vpin_z': z, 'vpin_pctl': pctl}

    def get_buckets_summary(self) -> Dict[str, Any]:
        return {
            'total_buckets': len(self.buckets),
            'vpin_history_len': len(self.vpin_history),
        }


def fuse_toxicity(vpin_z: float, ofi_z: Optional[float], spread_z: Optional[float], micro_vol_z: Optional[float], depth_imb_z: Optional[float], liq_burst_z: Optional[float], weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted fusion into a 0..1 tox_score (clipped).
    Inputs are assumed to be z-scores (can be None)."""
    if weights is None:
        weights = {'vpin_z': 0.5, 'ofi_z': 0.2, 'spread_z': 0.1, 'micro_vol_z': 0.1, 'depth_imb_z': 0.05, 'liq_burst_z': 0.05}
    acc = 0.0
    total_w = 0.0
    def use(w, v):
        nonlocal acc, total_w
        if v is None:
            return
        acc += w * float(v)
        total_w += w
    use(weights.get('vpin_z',0.0), vpin_z)
    use(weights.get('ofi_z',0.0), ofi_z)
    use(weights.get('spread_z',0.0), spread_z)
    use(weights.get('micro_vol_z',0.0), micro_vol_z)
    use(weights.get('depth_imb_z',0.0), depth_imb_z)
    use(weights.get('liq_burst_z',0.0), liq_burst_z)
    if total_w <= 0:
        return 0.0
    # normalize by total weight, then map z -> [0,1] via sigmoid-ish transform
    score_raw = acc / total_w
    # simple squash: logistic centered at 0, steepness tuned
    score = 1.0 / (1.0 + np.exp(-0.6 * score_raw))
    return float(np.clip(score, 0.0, 1.0))
