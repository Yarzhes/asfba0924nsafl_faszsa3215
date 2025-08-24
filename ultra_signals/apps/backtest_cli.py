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

    feature_store = FeatureStore(warmup_periods=warmup, settings=settings.model_dump())
    signal_engine = RealSignalEngine(settings.model_dump(), feature_store)
    runner = EventRunner(settings.model_dump(), adapter, signal_engine, feature_store)
    
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

    # pick a warmup size safely
    warmup_guess = (
        getattr(getattr(settings, "features", object()), "warmup_bars", None)
        or getattr(getattr(settings, "runtime", object()), "warmup_bars", None)
        or 1600
    )
    fs = FeatureStore(warmup_periods=int(warmup_guess), settings=settings)

    # Build a zero-arg factory that returns a fresh RealSignalEngine instance.
    # Try common constructor signatures so it works with your version.
    def engine_factory():
        try:
            return RealSignalEngine(settings, feature_store=fs)     # preferred kw
        except TypeError:
            try:
                return RealSignalEngine(settings, fs)               # positional fs
            except TypeError:
                eng = RealSignalEngine(settings)                    # no fs in ctor
                if hasattr(eng, "set_feature_store"):
                    try:
                        eng.set_feature_store(fs)
                    except Exception:
                        pass
                return eng

    # Data adapter
    adapter = DataAdapter(settings)

    # WalkForwardAnalysis wants dict-like settings
    wf_settings = (
        settings.model_dump() if hasattr(settings, "model_dump")
        else (settings.dict() if hasattr(settings, "dict") else settings)
    )
    wf = WalkForwardAnalysis(wf_settings, adapter, engine_factory)

    # Pull symbols/timeframe from settings safely
    try:
        symbols = list(settings.runtime.symbols) if getattr(settings, "runtime", None) else []
    except Exception:
        symbols = []
    if not symbols:
        symbols = ["BTCUSDT"]  # safe default

    timeframe = getattr(getattr(settings, "runtime", object()), "primary_timeframe", None) or "1h"

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
            # Expect (trades_df, equity_df, *rest). Keep the first element if it's a DF.
            return [x[0]] if len(x) > 0 else []
        # Assume it's already a DataFrame
        return [x]

    # Run WF per symbol and collect normalized trades
    all_trades = []
    for sym in symbols:
        result = wf.run(sym, timeframe)  # WF expects (symbol, timeframe)
        for d in extract_trades(result):
            if d is not None and hasattr(d, "empty") and not d.empty:
                all_trades.append(d)

    if not all_trades:
        logger.warning("Walk-forward analysis generated no results.")
        return

    trades = pd.concat(all_trades, ignore_index=True)
    logger.info(f"Walk-forward analysis finished. Found {len(trades)} rows of trades.")

    # Outputs
    out_dir = Path(getattr(args, "output_dir", "reports/wf"))
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "walk_forward_trades.csv"
    trades.to_csv(trades_path, index=False)

    pred_cols = [c for c in ["raw_score", "calibrated_score", "outcome"] if c in trades.columns]
    if pred_cols:
        preds_path = out_dir / "walk_forward_predictions.csv"
        trades[pred_cols].to_csv(preds_path, index=False)

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

if __name__ == "__main__":
    main()
