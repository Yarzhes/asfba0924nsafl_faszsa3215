import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import hashlib
import json


def _tf_to_ms(tf: str) -> int:
    """Convert timeframe like '1m','5m','1h' to milliseconds (coarse)."""
    try:
        if tf.endswith('ms'):
            return int(tf[:-2])
        if tf.endswith('s'):
            return int(tf[:-1]) * 1000
        if tf.endswith('m'):
            return int(tf[:-1]) * 60_000
        if tf.endswith('h'):
            return int(tf[:-1]) * 3_600_000
        if tf.endswith('d'):
            return int(tf[:-1]) * 86_400_000
    except Exception:  # pragma: no cover
        return 0
    return 0

# ultra_signals/backtest/data_adapter.py

class DataAdapter:
    # Default routing map (can be extended). Maps canonical symbol -> venue specific.
    DEFAULT_SYMBOL_ROUTING: Dict[str, str] = {
        # Perpetual futures (example mappings; adjust for real venue naming conventions)
        "BTCUSDT": "BTCUSDT",
        "ETHUSDT": "ETHUSDT",
        "SOLUSDT": "SOLUSDT",
        "BNBUSDT": "BNBUSDT",
        "XRPUSDT": "XRPUSDT",
        "ADAUSDT": "ADAUSDT",
        "DOGEUSDT": "DOGEUSDT",
        "TRXUSDT": "TRXUSDT",
        "AVAXUSDT": "AVAXUSDT",
        "LINKUSDT": "LINKUSDT",
        "DOTUSDT": "DOTUSDT",
        "MATICUSDT": "MATICUSDT",
        "LTCUSDT": "LTCUSDT",
        "BCHUSDT": "BCHUSDT",
        "SHIBUSDT": "SHIBUSDT",
        "NEARUSDT": "NEARUSDT",
        "ATOMUSDT": "ATOMUSDT",
        "XLMUSDT": "XLMUSDT",
        "APTUSDT": "APTUSDT",
        "ARBUSDT": "ARBUSDT",
        "OPUSDT": "OPUSDT",
        "FILUSDT": "FILUSDT",
        "SUIUSDT": "SUIUSDT",
        "INJUSDT": "INJUSDT",
        "AAVEUSDT": "AAVEUSDT",
        "UNIUSDT": "UNIUSDT",
        "RUNEUSDT": "RUNEUSDT",
        "ETCUSDT": "ETCUSDT",
        "TIAUSDT": "TIAUSDT",
        "TONUSDT": "TONUSDT",
    }

    def __init__(self, config: Any):
        """
        Accepts either a plain dict or a Pydantic BaseModel (v1 or v2).
        Normalizes to a dict so downstream code can use .get(...)
        """
        # Pydantic v2
        if hasattr(config, "model_dump"):
            cfg = config.model_dump()
        # Pydantic v1
        elif hasattr(config, "dict"):
            cfg = config.dict()
        # already a dict
        elif isinstance(config, dict):
            cfg = config
        else:
            # last-resort best effort (keeps old behavior from breaking)
            try:
                cfg = dict(config)
            except Exception:
                cfg = {}

        self.config = cfg.get("data", {}) or {}
        # (optional) keep a reference to other top-level sections if you use them
        self.features_cfg = cfg.get("features", {}) or {}
        self.runtime_cfg = cfg.get("runtime", {}) or {}
        self.backtest_cfg = cfg.get("backtest", {}) or {}
        self.batch_cfg = cfg.get("batch_run", {}) or {}

        # Cache root (backtest.data.cache_path preferred)
        try:
            self.cache_root = Path(((self.backtest_cfg.get("data") or {}).get("cache_path")) or ".cache/data")
        except Exception:
            self.cache_root = Path(".cache/data")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.reuse_cache = bool(self.batch_cfg.get("reuse_cache", False))
        # Allow user override of routing map via config: runtime.symbol_routing or data.symbol_routing
        self.symbol_routing: Dict[str, str] = {
            **self.DEFAULT_SYMBOL_ROUTING,
            **(self.runtime_cfg.get("symbol_routing", {}) or {}),
            **(self.config.get("symbol_routing", {}) or {}),
        }

    # ---------------- Symbol Routing -----------------
    def route_symbol(self, symbol: str) -> str:
        """Return venue-specific routed symbol (falls back to original)."""
        if not symbol:
            return symbol
        routed = self.symbol_routing.get(symbol.upper())
        if routed:
            return routed
        # Heuristic: convert BTCUSDT -> BTC/USDT if mapping missing and slash variant exists in directory
        if symbol.endswith("USDT"):
            slash_form = symbol[:-4] + "/USDT"
            if slash_form in self.symbol_routing.values():
                return slash_form
        return symbol

    # ---------------- Cache Helpers ------------------
    def _cache_file(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Path:
        """Return deterministic cache file path for symbol/timeframe/date-range."""
        key = f"{symbol}|{timeframe}|{start_date}|{end_date}".encode()
        h = hashlib.md5(key).hexdigest()[:16]
        subdir = self.cache_root / symbol / timeframe
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{h}.parquet"

    def _load_from_local_cache(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        path = self._cache_file(symbol, timeframe, start_date, end_date)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            # Basic schema sanity
            if df.empty or not {"open","high","low","close"}.issubset(df.columns):
                return None
            logger.info(f"[cache] hit {path}")
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return None

    def _store_to_local_cache(self, symbol: str, timeframe: str, start_date: str, end_date: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        try:
            path = self._cache_file(symbol, timeframe, start_date, end_date)
            # Store with index (timestamp) retained
            df.to_parquet(path)
            meta = {"rows": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max())}
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info(f"[cache] stored {path} rows={len(df)}")
        except Exception as e:
            logger.warning(f"[cache] store failed: {e}")


    def load_ohlcv(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Loads OHLCV data for a given symbol and timeframe.
        Supports loading from CSV, Parquet, or exchange (via cache).
        """
        provider = self.config.get("provider", "csv")
        routed_symbol = self.route_symbol(symbol)
        logger.info(f"Loading OHLCV for {symbol} -> {routed_symbol} ({timeframe}) from {provider} between {start_date} and {end_date}")

        # Local cache layer (independent of provider); only if reuse_cache flag set
        if self.reuse_cache:
            cached = self._load_from_local_cache(symbol, timeframe, start_date, end_date)
            if cached is not None:
                return cached

        if provider in ["csv", "parquet"]:
            df = self._load_from_file(routed_symbol, timeframe, start_date, end_date, provider)
            if df is not None and not df.empty:
                # --- DQ Pipeline (non-intrusive) ---
                try:
                    from ultra_signals.dq import normalizer as _dq_norm, validators as _dq_val, gap_filler as _dq_gap
                    settings_like = {"data_quality": self.config.get("data_quality", {})}
                    # convert to canonical schema with ts column (ms)
                    work = df.reset_index().rename(columns={df.index.name or 'timestamp': 'timestamp'})
                    work['ts'] = pd.to_datetime(work['timestamp']).astype('int64') // 1_000_000
                    work = work[['ts','open','high','low','close','volume']]
                    tf_ms = _tf_to_ms(timeframe)
                    # validate
                    rep = _dq_val.validate_ohlcv_df(work, tf_ms, settings_like, symbol, 'BACKTEST')
                    if not rep.ok:
                        logger.warning(f"dq.validation_failed symbol={symbol} errors={rep.errors}")
                    # gap heal if needed
                    if rep.warnings and any('low_coverage' in w for w in rep.warnings):
                        def _fetcher(symbol: str, ts: int):
                            return None  # placeholder offline backfill disabled
                        healed, greport = _dq_gap.heal_gaps_ohlcv(work, symbol, tf_ms, _fetcher, settings_like)
                        work = healed
                    # restore index
                    work['timestamp'] = pd.to_datetime(work['ts'], unit='ms')
                    df = work.set_index('timestamp')[['open','high','low','close','volume']]
                except Exception as e:  # pragma: no cover
                    logger.debug(f"dq.pipeline_skip reason={e}")
            if self.reuse_cache and df is not None and not df.empty:
                self._store_to_local_cache(symbol, timeframe, start_date, end_date, df)
            return df
        elif provider == "exchange":
            df = self._load_from_cache(routed_symbol, timeframe, start_date, end_date)
            if self.reuse_cache and df is not None and not df.empty:
                self._store_to_local_cache(symbol, timeframe, start_date, end_date, df)
            return df
        else:
            logger.error(f"Unsupported data provider: {provider}")
            return None

    def _load_from_file(self, symbol: str, timeframe: str, start_date: str, end_date: str, file_type: str) -> Optional[pd.DataFrame]:
        """
        Robust CSV/Parquet loader:
        - Detect timestamp column (timestamp/time/open_time/date/datetime) unless configured.
        - Parse ISO or epoch (s/ms), normalize to naive UTC.
        - Slice to [start, end) (exclusive end).
        - Log file coverage when slice is empty.
        - ✅ NEW: For Parquet, if no timestamp column is found, accept a DatetimeIndex.
        """
        base_path = Path(self.config.get("base_path", "data"))
        file_path = base_path / f"{symbol}_{timeframe}.{file_type}"

        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return None

        try:
            if file_type == "parquet":
                df = pd.read_parquet(file_path)
                ts_col = self.config.get("timestamp_col")

                # Try common timestamp column names first
                if not ts_col:
                    lowers = {c.lower(): c for c in df.columns}
                    for cand in ("timestamp", "time", "open_time", "date", "datetime"):
                        if cand in lowers:
                            ts_col = lowers[cand]
                            break

                # ✅ Accept DatetimeIndex as timestamp if no column found
                if not ts_col:
                    if isinstance(df.index, pd.DatetimeIndex):
                        idx = df.index
                        # Make UTC-naive milliseconds if tz-aware
                        if idx.tz is not None:
                            idx = idx.tz_convert("UTC").tz_localize(None)
                        df = df.copy()
                        df.insert(0, "timestamp", idx)
                        df = df.reset_index(drop=True)
                        ts_col = "timestamp"
                    else:
                        raise ValueError(f"Cannot find timestamp column in {list(df.columns)}")
            else:
                # CSV — sniff timestamp column
                sniff = pd.read_csv(file_path, nrows=5)
                ts_col = self.config.get("timestamp_col")
                if not ts_col:
                    lowers = {c.lower(): c for c in sniff.columns}
                    for cand in ("timestamp", "time", "open_time", "date", "datetime"):
                        if cand in lowers:
                            ts_col = lowers[cand]
                            break
                if not ts_col:
                    raise ValueError(f"Cannot find timestamp column in {list(sniff.columns)}")

                epoch = self.config.get("epoch")  # "ms" | "s" | None
                if epoch in ("ms", "s"):
                    df = pd.read_csv(file_path, converters={ts_col: lambda v: pd.to_datetime(int(v), unit=epoch, utc=True)})
                    df[ts_col] = df[ts_col].dt.tz_convert("UTC").dt.tz_localize(None)
                else:
                    try:
                        df = pd.read_csv(file_path, parse_dates=[ts_col])
                    except Exception:
                        probe = pd.read_csv(file_path, usecols=[ts_col]).iloc[0, 0]
                        unit = "ms" if int(probe) > 10_000_000_000 else "s"
                        df = pd.read_csv(file_path, converters={ts_col: lambda x: pd.to_datetime(int(x), unit=unit, utc=True)})
                        df[ts_col] = df[ts_col].dt.tz_convert("UTC").dt.tz_localize(None)
                    else:
                        if getattr(df[ts_col].dt, "tz", None) is not None:
                            df[ts_col] = df[ts_col].dt.tz_convert("UTC").dt.tz_localize(None)

            # sort + index
            df = df.sort_values(ts_col).set_index(ts_col)

            if df.empty:
                logger.warning(f"{file_path.name} is empty.")
                return df

            coverage_min, coverage_max = df.index.min(), df.index.max()
            start = pd.to_datetime(start_date) if start_date else coverage_min
            end   = pd.to_datetime(end_date)   if end_date   else coverage_max
            view = df.loc[(df.index >= start) & (df.index < end)]

            if view.empty:
                logger.warning(f"{file_path.name} has no rows in [{start} .. {end}); file coverage: {coverage_min} .. {coverage_max}")
            else:
                logger.success(f"Loaded {len(view)} rows from {file_path} within [{start} .. {end})")
            return view

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            return None

    def _load_from_cache(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Simulates loading data from an on-disk cache populated from an exchange."""
        logger.info("Simulating load from exchange cache.")
        return self._load_from_file(symbol, timeframe, start_date, end_date, "parquet")  # Assume cache is parquet

    def load_trades(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads trade data or generates synthetic trades if none are available.
        """
        ohlcv = self.load_ohlcv(symbol, "1m", start_date, end_date)  # Use 1m for trade synthesis
        if ohlcv is None or ohlcv.empty:
            logger.warning(f"Cannot generate synthetic trades for {symbol}; OHLCV data is missing.")
            return pd.DataFrame()

        # Simple synthesis: create four trades per bar (O, H, L, C)
        trades = []
        for index, row in ohlcv.iterrows():
            trades.append({"timestamp": index, "price": row["open"],  "volume": row["volume"] / 4})
            trades.append({"timestamp": index, "price": row["high"],  "volume": row["volume"] / 4})
            trades.append({"timestamp": index, "price": row["low"],   "volume": row["volume"] / 4})
            trades.append({"timestamp": index, "price": row["close"], "volume": row["volume"] / 4})

        trade_df = pd.DataFrame(trades)
        trade_df["timestamp"] = pd.to_datetime(trade_df["timestamp"])
        trade_df = trade_df.set_index("timestamp").sort_index()

        logger.info(f"Generated {len(trade_df)} synthetic trades for {symbol}.")
        return trade_df
