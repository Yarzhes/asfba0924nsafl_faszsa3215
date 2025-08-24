import argparse
from datetime import datetime
from typing import Any, List
from loguru import logger
from ultra_signals.core.config import load_settings

def setup_logging(log_level: str):
    """Sets up basic logging."""
    logger.add(
        "backtest.log",
        level=log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {message}",
        rotation="10 MB",
        mode="w",  # Overwrite log file on each run
    )
    logger.info(f"Logging level set to {log_level.upper()}")

from ultra_signals.backtest.data_adapter import DataAdapter
from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.backtest.event_runner import MockSignalEngine
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.backtest.walkforward import WalkForwardAnalysis
from ultra_signals.backtest.reporting import ReportGenerator
from ultra_signals.calibration import calibrate
from ultra_signals.backtest.metrics import compute_kpis, calculate_brier_score
import pandas as pd
import matplotlib.pyplot as plt


def handle_run(args: argparse.Namespace, settings: Any) -> None:
    """Entrypoint for the 'run' command."""
    logger.info("Command: Run Backtest")

    # --- NEW: echo settings & exit early ---
    if getattr(args, "echo", False):
        try:
            # pydantic v2 has model_dump_json
            print(settings.model_dump_json(indent=2))
        except Exception:
            import json
            print(json.dumps(settings.model_dump(), indent=2, default=str))
        return
    # ---------------------------------------

    # DataAdapter expects a dict-like config with .get
    adapter = DataAdapter(settings.model_dump())

    raw_warmup = getattr(settings.features, "warmup_periods", 100)
    try:
        warmup = int(raw_warmup)
    except Exception:
        warmup = 100
    if warmup <= 1:
        warmup = 2

    # --- SINGLE FEATURESTORE FIX: create exactly one instance and share it everywhere ---
    feature_store = FeatureStore(warmup_periods=warmup, settings=settings.model_dump())
    signal_engine = RealSignalEngine(settings.model_dump(), feature_store)
    runner = EventRunner(settings.model_dump(), adapter, signal_engine, feature_store)

    # Guard + helpful debug so the "two stores" bug can't happen silently.
    logger.debug(f"[backtest_cli] FeatureStore(shared) id={id(feature_store)}")
    logger.debug(f"[backtest_cli] Engine.FeatureStore id={id(signal_engine.feature_store)}")
    logger.debug(f"[backtest_cli] Runner.FeatureStore id={id(runner.feature_store)}")
    assert signal_engine.feature_store is runner.feature_store is feature_store, (
        "Two different FeatureStore instances detected! "
        f"engine_store_id={id(signal_engine.feature_store)} "
        f"runner_store_id={id(runner.feature_store)} "
        f"shared_store_id={id(feature_store)}"
    )
    # -------------------------------------------------------------------------------

    # For now, we run on the first symbol specified in runtime config
    symbol = settings.runtime.symbols[0]
    timeframe = settings.runtime.primary_timeframe

    trades, equity = runner.run(symbol, timeframe)

    if trades:
        trades_df = pd.DataFrame(trades)
        kpis = compute_kpis(trades_df)

        report_settings = settings.reports.model_dump()
        if getattr(args, "output_dir", None):
            report_settings["output_dir"] = args.output_dir

        reporter = ReportGenerator(report_settings)
        reporter.generate_report(kpis, equity, trades_df)  # Pass raw equity list
        logger.success(f"Backtest finished. Report generated in {report_settings['output_dir']}.")
    else:
        logger.warning("Backtest finished with no trades.")
        # Return non-zero exit code if no trades
        exit(1)


def _extract_trades(result):
    """Accepts DataFrame OR (trades_df, ...) OR [df, df, ...] and returns a list of dfs."""
    import pandas as pd
    if result is None:
        return []
    if isinstance(result, pd.DataFrame):
        return [result]
    if isinstance(result, (list, tuple)):
        dfs = []
        for item in result:
            if item is None:
                continue
            if hasattr(item, "empty"):  # looks like a DataFrame
                dfs.append(item)
        return dfs
    return []


def handle_wf(args, settings):
    """Walk-forward analysis entrypoint."""
    from ultra_signals.backtest.data_adapter import DataAdapter
    from ultra_signals.core.feature_store import FeatureStore
    from ultra_signals.engine.real_engine import RealSignalEngine
    from ultra_signals.backtest.walkforward import WalkForwardAnalysis
    from loguru import logger
    import pandas as pd
    from pathlib import Path

    logger.info("Command: Walk-Forward Analysis")

    # --------- OPTION B FIX: normalize settings to a plain dict everywhere ---------
    if hasattr(settings, "model_dump"):
        settings_dict = settings.model_dump()
    elif hasattr(settings, "dict"):
        settings_dict = settings.dict()
    else:
        settings_dict = settings  # already a dict
    # ------------------------------------------------------------------------------

    # pick a warmup size safely (use the same key the backtest uses)
    warmup_guess = (
        settings_dict.get("features", {}).get("warmup_periods", 100)
    )
    try:
        warmup_guess = int(warmup_guess)
    except Exception:
        warmup_guess = 100
    warmup_guess = max(warmup_guess, 2)

    # Build one FeatureStore and reuse it for every engine produced by the factory.
    fs = FeatureStore(warmup_periods=warmup_guess, settings=settings_dict)

    # Factory that returns engines which all share the SAME FeatureStore instance.
    def engine_factory():
        try:
            return RealSignalEngine(settings_dict, feature_store=fs)   # preferred kw
        except TypeError:
            try:
                return RealSignalEngine(settings_dict, fs)             # positional fs
            except TypeError:
                eng = RealSignalEngine(settings_dict)                  # no fs in ctor
                if hasattr(eng, "set_feature_store"):
                    try:
                        eng.set_feature_store(fs)
                    except Exception:
                        pass
                return eng

    # Data adapter (dict config)
    adapter = DataAdapter(settings_dict)

    # WalkForwardAnalysis expects dict-like settings
    wf = WalkForwardAnalysis(settings_dict, adapter, engine_factory)

    # Pull symbols/timeframe safely from dict
    symbols = []
    try:
        rt = settings_dict.get("runtime", {})
        symbols = list(rt.get("symbols", [])) or []
    except Exception:
        symbols = []
    if not symbols:
        symbols = ["BTCUSDT"]  # safe default

    timeframe = (settings_dict.get("runtime", {}) or {}).get("primary_timeframe") or "5m"  # <- default to 5m

    # --- NORMALIZE WF OUTPUTS (handles DataFrame, (trades, equity), or nested lists) ---
    def extract_trades(x):
        """Return a list of trade DataFrames from x."""
        if x is None:
            return []
        if isinstance(x, list):
            out = []
            for item in x:
                out.extend(extract_trades(item))
            return out
        if isinstance(x, tuple):
            # Expect (trades_df, kpi_df, *rest). Keep the first element if it's a DF.
            first = x[0] if len(x) > 0 else None
            return [first] if (first is not None and hasattr(first, "empty")) else []
        # Assume it's already a DataFrame
        return [x] if hasattr(x, "empty") else []

    # Run WF per symbol and collect normalized trades
    all_trades = []
    for sym in symbols:
        result = wf.run(sym, timeframe)  # may be DataFrame or (trades_df, kpi_df)
        for df in extract_trades(result):
            if df is not None and hasattr(df, "empty") and not df.empty:
                all_trades.append(df)

    if not all_trades:
        logger.warning("Walk-forward analysis generated no results.")
        return

    trades = pd.concat(all_trades, ignore_index=True)
    logger.info(f"Walk-forward analysis finished. Found {len(trades)} rows of trades.")

    # Outputs
    from pathlib import Path
    out_dir = Path(getattr(args, "output_dir", "reports/wf"))
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "walk_forward_trades.csv"
    trades.to_csv(trades_path, index=False)

    pred_cols = [c for c in ["raw_score", "calibrated_score", "outcome"] if c in trades.columns]
    if pred_cols:
        preds_path = out_dir / "walk_forward_predictions.csv"
        trades[pred_cols].to_csv(preds_path, index=False)

    # ------------------- ADDED: risk events export if available -------------------
    try:
        # Expect WalkForwardAnalysis to have 'risk_events_by_fold' if instrumented.
        buckets = getattr(wf, "risk_events_by_fold", [])
        events_flat = []
        for bucket in buckets or []:
            fold = bucket.get("fold")
            for ev in bucket.get("events", []) or []:
                if isinstance(ev, dict):
                    row = dict(ev)
                elif hasattr(ev, "__dict__"):
                    row = dict(ev.__dict__)
                else:
                    try:
                        row = dict(vars(ev))
                    except Exception:
                        row = {}
                # ensure fold label on each row
                row["fold"] = row.get("fold", fold)
                events_flat.append(row)

        if events_flat:
            risk_path = out_dir / "risk_events.csv"
            write_risk_events_csv(risk_path, events_flat)
            logger.success(f"Risk events written to {risk_path}")

            # Optional summary of VETO reasons
            summary_pairs = summarize_veto_reasons(events_flat, top_n=50)
            if summary_pairs:
                import csv as _csv  # local alias to avoid any confusion
                sum_path = out_dir / "risk_events_summary.csv"
                with sum_path.open("w", newline="", encoding="utf-8") as f:
                    w = _csv.writer(f)
                    w.writerow(["reason", "count"])
                    for reason, count in summary_pairs:
                        w.writerow([reason, count])
                logger.success(f"Risk events summary written to {sum_path}")
        else:
            logger.info("No risk events collected (wf.risk_events_by_fold empty or missing).")
    except Exception as e:
        logger.warning(f"Skipping risk events export due to: {e}")
    # ------------------------------------------------------------------------------

    logger.success(f"WF outputs written to {out_dir}")


def handle_cal(args: argparse.Namespace, settings: Any) -> None:
    """Entrypoint for the 'cal' (calibration) command."""
    logger.info("Command: Calibration")
    
    # 1. Load prediction data (this would typically come from a WF run)
    # As a placeholder, we create dummy data.
    predictions = pd.read_csv("walk_forward_predictions.csv")  # Assumes this file exists
    
    # 2. Fit model
    model = calibrate.fit_calibration_model(
        predictions['raw_score'],
        predictions['outcome'],
        method=args.method
    )
    
    # 3. Save model
    model_path = "calibration_model.joblib"
    from pathlib import Path
    calibrate.save_model(model, Path(model_path))
    logger.success(f"Calibration model saved to {model_path}")

    # 4. Evaluate calibration
    raw_brier = calculate_brier_score(predictions['outcome'], predictions['raw_score'])
    
    try:
        calibrated_preds = calibrate.apply_calibration(model, predictions['raw_score'])
        calibrated_brier = calculate_brier_score(predictions['outcome'], calibrated_preds)
        logger.info(f"Raw Brier Score: {raw_brier:.4f}")
        logger.info(f"Calibrated Brier Score: {calibrated_brier:.4f}")

        if calibrated_brier < raw_brier:
            logger.success("Calibration improved the Brier score.")
        else:
            logger.warning("Calibration did not improve the Brier score.")

        # 5. Generate and save reliability plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        calibrate.plot_reliability_diagram(ax, predictions['outcome'], predictions['raw_score'], calibrated_preds)
        plot_path = "reports/reliability_plot.png"
        Path("reports").mkdir(exist_ok=True)
        fig.savefig(plot_path)
        logger.success(f"Reliability plot saved to {plot_path}")
    except Exception as e:
        logger.warning(f"Skipping reliability plot due to: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Creates the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ultra-Signals Backtesting & Analysis Framework",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available sub-commands")

    # Parent parser for common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config", type=str, default="settings.yaml", help="Path to the root configuration file."
    )
    common_parser.add_argument(
        "--log-level",
        type=str,
        default=None,  # Set default to None to detect if it was user-provided
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    # --- 'run' sub-command ---
    parser_run = subparsers.add_parser(
        "run",
        help="Execute a single backtest over a specified period.",
        parents=[common_parser],
    )
    parser_run.add_argument("--symbol", type=str, help="Symbol to backtest (e.g., BTCUSDT). Overrides config.")
    parser_run.add_argument("--interval", type=str, help="Candle interval (e.g., 5m). Overrides config.")
    parser_run.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD). Overrides config.")
    parser_run.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD). Overrides config.")
    parser_run.add_argument("--output-dir", type=str, help="Directory to save backtest report.")
    # NEW: add --echo flag
    parser_run.add_argument(
        "--echo",
        action="store_true",
        help="Print the resolved settings (JSON) and exit without running."
    )
    parser_run.set_defaults(func=handle_run)

    # --- 'wf' sub-command ---
    parser_wf = subparsers.add_parser(
        "wf",
        help="Execute a rolling walk-forward analysis.",
        parents=[common_parser],
    )
    parser_wf.add_argument("--output-dir", type=str, help="Directory to save walk-forward report.")
    parser_wf.set_defaults(func=handle_wf)

    # --- 'cal' sub-command ---
    parser_cal = subparsers.add_parser(
        "cal",
        help="Fit a calibration model to prediction data.",
        parents=[common_parser],
    )
    parser_cal.add_argument(
        "--method",
        type=str,
        default="isotonic",
        choices=["isotonic", "platt"],
        help="Calibration method to use.",
    )
    parser_cal.set_defaults(func=handle_cal)

    return parser


def main(argv: List[str] = None) -> None:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    settings = load_settings(args.config)

    # Determine log level with correct precedence: CLI > config > default
    if args.log_level:
        log_level = args.log_level
    elif settings.logging and settings.logging.level:
        log_level = settings.logging.level
    else:
        log_level = "INFO"

    setup_logging(log_level)

    if hasattr(args, "func"):
        args.func(args, settings)
    else:
        parser.print_help()


# ------------------------------
# STEP 2 ADDITIONS (helpers only)
# ------------------------------
from pathlib import Path
from collections import Counter
import csv
import json
from typing import Iterable, Mapping, Union

def write_risk_events_csv(out_path: Union[str, Path], rows: Iterable[Any]) -> None:
    """
    Write RiskEvents (or dict-like rows) to CSV with stable columns.
    Expected keys: fold, ts, symbol, reason, action, detail.
    Extra keys are ignored. Dataclass/objects are supported via __dict__.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["fold", "ts", "symbol", "reason", "action", "detail"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows or []:
            if row is None:
                continue
            # accept dataclass/obj/dict
            if isinstance(row, Mapping):
                d = dict(row)
            elif hasattr(row, "__dict__"):
                d = dict(row.__dict__)
            else:
                try:
                    d = dict(vars(row))
                except Exception:
                    continue

            detail = d.get("detail")
            if isinstance(detail, (dict, list)):
                try:
                    detail = json.dumps(detail, ensure_ascii=False)
                except Exception:
                    detail = str(detail)

            writer.writerow({
                "fold": d.get("fold"),
                "ts": d.get("ts"),
                "symbol": d.get("symbol"),
                "reason": d.get("reason"),
                "action": d.get("action"),
                "detail": detail,
            })

def summarize_veto_reasons(rows: Iterable[Any], top_n: int = 10):
    """
    Return a list of (reason, count) for events with action == 'VETO' (case-insensitive).
    """
    counts = Counter()
    for row in rows or []:
        if row is None:
            continue
        if isinstance(row, Mapping):
            d = row
        elif hasattr(row, "__dict__"):
            d = row.__dict__
        else:
            try:
                d = vars(row)
            except Exception:
                continue

        action = str(d.get("action", "")).upper()
        if action == "VETO":
            reason = d.get("reason")
            if reason:
                counts[str(reason)] += 1

    return counts.most_common(top_n)


if __name__ == "__main__":
    main()
