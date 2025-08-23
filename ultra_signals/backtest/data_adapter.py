import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

class DataAdapter:
    """Handles loading market data for backtesting from various sources."""

    def __init__(self, config: Dict):
        self.config = config.get("data", {})
        self.cache_path = Path(self.config.get("cache_path", ".cache/data"))
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def load_ohlcv(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Loads OHLCV data for a given symbol and timeframe.
        Supports loading from CSV, Parquet, or exchange (via cache).
        """
        provider = self.config.get("provider", "csv")
        logger.info(f"Loading OHLCV for {symbol} ({timeframe}) from {provider} between {start_date} and {end_date}")

        if provider in ["csv", "parquet"]:
            return self._load_from_file(symbol, timeframe, start_date, end_date, provider)
        elif provider == "exchange":
            return self._load_from_cache(symbol, timeframe, start_date, end_date)
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
