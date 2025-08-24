import pandas as pd
from loguru import logger
from typing import Dict, Any, List, Tuple
from datetime import timedelta
from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.backtest.metrics import compute_kpis

DEFAULT_TRAIN_DAYS = 90
DEFAULT_TEST_DAYS = 30
DEFAULT_PURGE_DAYS = 3


def _parse_days(val, default=None):
    """
    Accepts:
      - int (e.g., 21)
      - str with 'd' suffix (e.g., '21d')
      - None -> default
    Returns int days.
    """
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v.endswith("d") and v[:-1].isdigit():
            return int(v[:-1])
        if v.isdigit():
            return int(v)
    return default


class WalkForwardAnalysis:
    """
    Performs a walk-forward analysis by creating rolling time windows
    for training and testing.
    """

    def __init__(self, settings: Dict[str, Any], data_adapter, signal_engine_class):
        # settings is a dict (from pydantic model_dump)
        self.backtest_settings: Dict[str, Any] = settings
        self.wf_settings: Dict[str, Any] = settings.get("walkforward", {}) or {}
        self.data_adapter = data_adapter
        self.signal_engine_class = signal_engine_class
        # --- step 1: store risk events without changing any APIs ---
        self.risk_events = []
        self.risk_events_by_fold: List[Dict[str, Any]] = []
        self._last_fold_risk_events = []
        # -----------------------------------------------------------

    def run(self, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the full walk-forward analysis.
        Returns:
            (kpi_summary_df, combined_trades_df)
        """
        logger.info("Starting walk-forward analysis...")
        logger.info(f"WF analysis range: {self.backtest_settings['backtest']['start_date']} → {self.backtest_settings['backtest']['end_date']}")

        # --- ADDED: reset per-run risk aggregation bucket so repeated runs are clean ---
        self.risk_events = []
        self.risk_events_by_fold = []
        self._last_fold_risk_events = []
        # --------------------------------------------------------------------------------

        start_date = pd.to_datetime(self.backtest_settings["backtest"]["start_date"])
        end_date = pd.to_datetime(self.backtest_settings["backtest"]["end_date"])

        windows = self._generate_windows(start_date, end_date)
        logger.info(f"Generated {len(windows)} walk-forward windows.")

        all_trades: List[pd.DataFrame] = []
        kpi_reports: List[Dict[str, Any]] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"--- Running Fold {i+1}/{len(windows)} ---")
            logger.info(
                f"Train: {train_start.date()} to {train_end.date()}, "
                f"Test: {test_start.date()} to {test_end.date()}"
            )

            # (placeholder for training/tuning on the train window)
            # Engine is created in _run_test_fold so it can be bound to that fold’s FeatureStore.
            signal_engine_instance = None

            # Run backtest on test window
            test_trades, _ = self._run_test_fold(
                symbol, timeframe, test_start, test_end, signal_engine_instance, fold_index=i  # ADDED: pass fold index
            )

            # --- step 1: collect fold's risk events (no API changes) ---
            fold_events = list(getattr(self, "_last_fold_risk_events", []))
            if fold_events:
                self.risk_events_by_fold.append({"fold": i + 1, "events": fold_events})
                self.risk_events.extend(fold_events)
            # -----------------------------------------------------------

            if not test_trades.empty:
                all_trades.append(test_trades)
                kpis = compute_kpis(test_trades)
                kpis["fold"] = i + 1
                kpi_reports.append(kpis)

        if not all_trades:
            return pd.DataFrame(), pd.DataFrame()

        combined_trades = pd.concat(all_trades, ignore_index=True)
        kpi_summary = pd.DataFrame(kpi_reports)

        logger.info(
            f"Walk-forward analysis finished. Found {len(kpi_summary)} folds with trades."
        )
        return combined_trades, kpi_summary

    def _generate_windows(
        self, analysis_start: pd.Timestamp, analysis_end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generates training and testing windows with purging and advance-by."""

        # Read from dict-based settings
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
            f"Resolved WF periods -> train_days={train_days}, "
            f"test_days={test_days}, purge_days={purge_days}"
        )

        windows: List[
            Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]
        ] = []
        current_start = analysis_start

        logger.info(
            f"Generating windows with train_days={train_days}, "
            f"test_days={test_days}, purge_days={purge_days}"
        )
        
        logger.info(f"First fold must end by: {analysis_start + train_days + purge_days + test_days}, analysis_end={analysis_end}")

        while current_start + train_days + test_days <= analysis_end:
            train_start = current_start
            train_end = train_start + train_days

            # Purge between train and test
            test_start = train_end + purge_days
            test_end = test_start + test_days

            if test_end > analysis_end:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Advance window (use window.advance_by if present; else advance by test_days)
            advance_by_str = window.get("advance_by")
            advance_by_days = _parse_days(advance_by_str, test_days_res)
            current_start += timedelta(days=advance_by_days)

            logger.info(f"Final windows: {windows}")
        return windows

    def _run_test_fold(
        self, symbol, timeframe, start, end, engine, *, fold_index: int = 0  # ADDED: fold_index (kw-only)
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Runs the event runner for a single test fold."""
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

        # Build a fresh engine that is bound to THIS fold’s FeatureStore when possible.
        # We support three forms seamlessly:
        #   1) class(settings, fs)
        #   2) factory(fs)
        #   3) factory()  (and we only set fs if engine has none)
        try:
            engine = self.signal_engine_class(fold_settings, fs)  # try (settings, fs)
        except TypeError:
            try:
                engine = self.signal_engine_class(fs)              # try (fs)
            except TypeError:
                engine = self.signal_engine_class()                # fallback zero-arg
                # only set fs if engine doesn't already carry one
                if hasattr(engine, "feature_store"):
                    if getattr(engine, "feature_store", None) is None:
                        if hasattr(engine, "set_feature_store"):
                            try:
                                engine.set_feature_store(fs)
                            except Exception:
                                pass

        # ---- FIX: ensure EventRunner uses the SAME FeatureStore object as the engine ----
        runner_fs = getattr(engine, "feature_store", fs)
        if runner_fs is not fs:
            logger.debug(
                "Using engine-bound FeatureStore for runner (ids: engine=%s, fold=%s)",
                id(runner_fs), id(fs)
            )
        # ---------------------------------------------------------------------------------

        # IMPORTANT: pass the FULL fold settings, not only ["backtest"]
        runner = EventRunner(fold_settings, self.data_adapter, engine, runner_fs)

        result = runner.run(symbol, timeframe)

        # --- step 1: stash fold risk events for the caller to aggregate ---
        self._last_fold_risk_events = list(getattr(runner, "risk_events", []))
        # ------------------------------------------------------------------

        if result is None:
            return pd.DataFrame(), pd.Series(dtype=float)

        trades, equity = result
        return pd.DataFrame(trades), pd.Series(equity)
