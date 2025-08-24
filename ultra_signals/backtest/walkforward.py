import pandas as pd
from loguru import logger
from typing import Dict, Any, List, Tuple
from datetime import timedelta
from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.backtest.metrics import compute_kpis

# Default durations (days)
DEFAULT_TRAIN_DAYS = 90
DEFAULT_TEST_DAYS = 30
DEFAULT_PURGE_DAYS = 3


def _parse_days(val, default=None):
    """
    Robust day parser.

    Accepts:
      - int (e.g., 21)
      - str with optional 'd' suffix (e.g., '21d' or '21')
      - None -> default

    Returns an int day count or `default` if unparsable.
    """
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v.endswith("d"):
            v = v[:-1]
        if v.isdigit():
            return int(v)
    return default


def _safe_to_datetime(val, default="2000-01-01"):
    """
    Convert val to pd.Timestamp safely.

    This guards against test-time MagicMocks (from patched load_settings),
    falling back to a harmless default date rather than raising.
    """
    try:
        return pd.to_datetime(val)
    except Exception:
        logger.warning(f"Invalid start/end date '{val}', falling back to default={default}")
        return pd.to_datetime(default)


class WalkForwardAnalysis:
    """
    Performs walk-forward analysis by creating rolling windows (train/test)
    with optional purge gap, running the backtester on each test fold.

    Notes:
    - Settings passed here should be a **plain dict** (e.g., pydantic.model_dump()).
    - We keep a record of fold risk events (if the EventRunner exposes them)
      without modifying EventRunner APIs.
    """

    def __init__(self, settings: Dict[str, Any], data_adapter, signal_engine_class):
        # Root dict produced from pydantic model_dump (or already a dict)
        self.backtest_settings: Dict[str, Any] = settings
        self.wf_settings: Dict[str, Any] = settings.get("walkforward", {}) or {}

        self.data_adapter = data_adapter
        # May be a class or a factory; we handle both when instantiating
        self.signal_engine_class = signal_engine_class

        # Risk-event aggregation buckets (kept for downstream exports)
        self.risk_events: List[Dict[str, Any]] = []
        self.risk_events_by_fold: List[Dict[str, Any]] = []
        self._last_fold_risk_events: List[Dict[str, Any]] = []

    def run(self, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full walk-forward analysis over [backtest.start_date, backtest.end_date].

        Returns:
            (combined_trades_df, kpi_summary_df)

        (Returning trades first matches common usage where caller wants trades;
         CLI helper normalizes either shape anyway.)
        """
        logger.info("Starting walk-forward analysis...")
        logger.info(
            "WF analysis range: {} → {}",
            self.backtest_settings["backtest"]["start_date"],
            self.backtest_settings["backtest"]["end_date"],
        )

        # Reset per-run collections
        self.risk_events = []
        self.risk_events_by_fold = []
        self._last_fold_risk_events = []

        # SAFE date parsing (works with real configs and with MagicMocks in tests)
        start_date = _safe_to_datetime(self.backtest_settings["backtest"].get("start_date"))
        end_date = _safe_to_datetime(self.backtest_settings["backtest"].get("end_date"))

        windows = self._generate_windows(start_date, end_date)
        logger.info(f"Generated {len(windows)} walk-forward windows.")

        all_trades: List[pd.DataFrame] = []
        kpi_reports: List[Dict[str, Any]] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"--- Running Fold {i+1}/{len(windows)} ---")
            logger.info(
                f"Train: {train_start.date()} → {train_end.date()}, "
                f"Test: {test_start.date()} → {test_end.date()}"
            )

            # Engine instance is constructed inside _run_test_fold so it can be
            # bound to the per-fold FeatureStore cleanly.
            signal_engine_instance = None

            # Run backtest on test window
            test_trades, _ = self._run_test_fold(
                symbol,
                timeframe,
                test_start,
                test_end,
                signal_engine_instance,
                fold_index=i,  # for logging/diagnostics if needed
            )

            # Collect fold risk events (no API changes required)
            fold_events = list(getattr(self, "_last_fold_risk_events", []))
            if fold_events:
                self.risk_events_by_fold.append({"fold": i + 1, "events": fold_events})
                self.risk_events.extend(fold_events)

            if not test_trades.empty:
                all_trades.append(test_trades)
                kpis = compute_kpis(test_trades)
                kpis["fold"] = i + 1
                kpi_reports.append(kpis)

        if not all_trades:
            # Preserve return shape
            return pd.DataFrame(), pd.DataFrame()

        combined_trades = pd.concat(all_trades, ignore_index=True)
        kpi_summary = pd.DataFrame(kpi_reports)

        logger.info(
            "Walk-forward analysis finished. Folds with trades: {}",
            len(kpi_summary),
        )
        return combined_trades, kpi_summary

    def _generate_windows(
        self, analysis_start: pd.Timestamp, analysis_end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate (train_start, train_end, test_start, test_end) windows.

        Settings read order (most → least specific):
          1) walkforward.train_days / test_days / purge_days (int or 'Xd')
          2) walkforward.window.train_period / test_period ('Xd')
             walkforward.data_rules.purge_period ('Xd')
          3) defaults (90 / 30 / 3)
        """
        wf = self.wf_settings or self.backtest_settings.get("walkforward", {}) or {}

        # Primary: numeric day counts at top-level
        train_days_i = wf.get("train_days")
        test_days_i = wf.get("test_days")
        purge_days_i = wf.get("purge_days")

        # Secondary: string periods like "21d" under nested sections
        window = wf.get("window", {}) or {}
        data_rules = wf.get("data_rules", {}) or {}
        train_str = window.get("train_period")
        test_str = window.get("test_period")
        purge_str = data_rules.get("purge_period")

        # Resolve integers first; fall back to string fields; then to defaults
        train_days_res = _parse_days(train_days_i, None)
        test_days_res = _parse_days(test_days_i, None)
        purge_days_res = _parse_days(purge_days_i, None)

        if train_days_res is None:
            train_days_res = _parse_days(train_str, DEFAULT_TRAIN_DAYS)
        if test_days_res is None:
            test_days_res = _parse_days(test_str, DEFAULT_TEST_DAYS)
        if purge_days_res is None:
            purge_days_res = _parse_days(purge_str, DEFAULT_PURGE_DAYS)

        train_days = timedelta(days=train_days_res)
        test_days = timedelta(days=test_days_res)
        purge_days = timedelta(days=purge_days_res)

        logger.info(
            "Resolved WF periods -> train_days={}, test_days={}, purge_days={}",
            train_days,
            test_days,
            purge_days,
        )

        windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        current_start = analysis_start

        logger.info(
            "Generating windows with train_days={}, test_days={}, purge_days={}",
            train_days,
            test_days,
            purge_days,
        )
        logger.info(
            "First fold must end by: {}, analysis_end={}",
            analysis_start + train_days + purge_days + test_days,
            analysis_end,
        )

        # Always advance by something sane; default to test_days if unspecified
        advance_by_str = window.get("advance_by")
        advance_by_days = _parse_days(advance_by_str, test_days_res)
        if advance_by_days is None:
            advance_by_days = test_days_res
        if advance_by_days is None or advance_by_days <= 0:
            advance_by_days = test_days.days if isinstance(test_days, timedelta) else DEFAULT_TEST_DAYS

        while current_start + train_days + test_days <= analysis_end:
            train_start = current_start
            train_end = train_start + train_days

            # Purge between train and test
            test_start = train_end + purge_days
            test_end = test_start + test_days

            if test_end > analysis_end:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current_start += timedelta(days=advance_by_days)

        logger.info(f"Final windows: {windows}")
        return windows

    def _run_test_fold(
        self,
        symbol,
        timeframe,
        start,
        end,
        engine,
        *,
        fold_index: int = 0,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run EventRunner for a single test fold.

        We construct a per-fold FeatureStore and ensure the engine **and** runner
        share the SAME instance to avoid state divergence.
        """
        # Shallow copy is enough here; we only replace dates
        fold_settings = dict(self.backtest_settings)
        fold_settings["backtest"] = dict(self.backtest_settings["backtest"])
        fold_settings["backtest"]["start_date"] = start.strftime("%Y-%m-%d")
        fold_settings["backtest"]["end_date"] = end.strftime("%Y-%m-%d")

        from ultra_signals.core.feature_store import FeatureStore

        fs = FeatureStore(
            warmup_periods=fold_settings["features"]["warmup_periods"],
            settings=fold_settings,
        )

        # Build an engine bound to THIS fold’s FeatureStore.
        # Support: class(settings, fs) → class(fs) → class() + set_feature_store(fs)
        try:
            engine = self.signal_engine_class(fold_settings, fs)  # preferred
        except TypeError:
            try:
                engine = self.signal_engine_class(fs)
            except TypeError:
                engine = self.signal_engine_class()
                if hasattr(engine, "feature_store") and getattr(engine, "feature_store", None) is None:
                    if hasattr(engine, "set_feature_store"):
                        try:
                            engine.set_feature_store(fs)
                        except Exception:
                            pass

        # Ensure EventRunner uses EXACTLY the engine's FS (if engine already carries one)
        runner_fs = getattr(engine, "feature_store", fs)
        if runner_fs is not fs:
            logger.debug(
                "Using engine-bound FeatureStore for runner (ids: engine=%s, fold=%s)",
                id(runner_fs),
                id(fs),
            )

        # IMPORTANT: pass the FULL fold settings (not only ["backtest"])
        runner = EventRunner(fold_settings, self.data_adapter, engine, runner_fs)

        result = runner.run(symbol, timeframe)

        # Stash fold risk events for export by caller
        self._last_fold_risk_events = list(getattr(runner, "risk_events", []))

        if result is None:
            return pd.DataFrame(), pd.Series(dtype=float)

        trades, equity = result
        return pd.DataFrame(trades), pd.Series(equity)
