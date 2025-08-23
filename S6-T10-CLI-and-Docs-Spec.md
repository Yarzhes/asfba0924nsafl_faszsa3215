# S6-T10: Backtester CLI and Documentation Specification

This document outlines the command-line interface (CLI) for the new backtesting script and the plan for its documentation.

## 1. CLI Specification: `backtest_cli.py`

The backtester will be executed via a new script located at `ultra_signals/apps/backtest_cli.py`. It will support both simple backtests and walk-forward analysis through a clean, sub-command-based interface.

### 1.1. Command Synopsis

The base command will use the Python module runner flag (`-m`) for consistency with the existing `realtime_runner`.

```bash
python -m ultra_signals.apps.backtest_cli [SUBCOMMAND] [OPTIONS]
```

### 1.2. Sub-commands

#### 1.2.1. `run-backtest`

Runs a single backtest over a specified historical period using a single configuration file.

*   **Usage:**
    ```bash
    python -m ultra_signals.apps.backtest_cli run-backtest --config <path_to_config.yaml> --output-dir <path_to_results>
    ```

#### 1.2.2. `run-walk-forward`

Runs a full walk-forward analysis, which consists of multiple sequential backtesting windows.

*   **Usage:**
    ```bash
    python -m ultra_signals.apps.backtest_cli run-walk-forward --config <path_to_wf_config.yaml> --output-dir <path_to_results>
    ```

### 1.3. Arguments and Options

*   **`--config`** (Required)
    *   **Type:** `string` (File Path)
    *   **Description:** Path to the configuration file. For `run-backtest`, this is a standard strategy/backtest configuration. For `run-walk-forward`, this points to a specific walk-forward configuration file (e.g., `walk_forward_config.yaml`).

*   **`--output-dir`** (Optional)
    *   **Type:** `string` (Directory Path)
    *   **Description:** Path to the directory where all output artifacts (reports, charts, logs) will be saved.
    *   **Default:** `backtest_results/<strategy_name>_<timestamp>/`

*   **`--log-level`** (Optional)
    *   **Type:** `string`
    *   **Description:** Sets the logging verbosity.
    *   **Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`
    *   **Default:** `INFO`

*   **`--no-cache`** (Optional)
    *   **Type:** `boolean` (Flag)
    *   **Description:** If present, forces the backtester to re-fetch and re-process all historical data, ignoring any cached data.
    *   **Default:** `false` (uses cache)

## 2. Documentation Plan

The documentation will be added to the main project `README.md` to ensure high visibility for all users. It will follow the existing structure with clear, copy-pasteable examples.

### 2.1. New `README.md` Section (DRAFT)

---

## How to Run the Backtester

The backtesting framework allows you to test your trading strategies on historical data. You can run a single backtest for a specific period or a full walk-forward analysis to assess the strategy's robustness over time.

### 1. Running a Simple Backtest

A simple backtest runs your strategy with a single configuration file over a defined date range.

**Command:**
```bash
python -m ultra_signals.apps.backtest_cli run-backtest \
  --config /path/to/your/strategy.yaml \
  --output-dir backtest_results/my_first_test
```

*   `--config`: Points to your strategy configuration file, which defines the assets, features, and risk parameters.
*   `--output-dir`: (Optional) Specifies where to save the results. If omitted, a timestamped directory will be created automatically.

### 2. Running a Walk-Forward Analysis

A walk-forward analysis simulates how a strategy would have been re-optimized and traded over a long historical period, providing a more realistic performance estimate.

**Command:**
```bash
python -m ultra_signals.apps.backtest_cli run-walk-forward \
  --config ultra_signals/backtest/walk_forward_config.yaml \
  --output-dir backtest_results/walk_forward_test
```

*   `--config`: Points to the walk-forward configuration file that defines the backtesting windows (e.g., training length, testing length) and the strategy configuration to use.

### 3. Expected Output Artifacts

After running a backtest, the specified `--output-dir` will contain the following artifacts:

*   **`report.md` / `report.html`**: A detailed performance summary, including key metrics like Sharpe Ratio, Profit Factor, Drawdown, and Win Rate.
*   **`equity_curve.png`**: A chart showing the growth of the strategy's equity over time.
*   **`trades.csv`**: A list of all trades executed during the backtest, including entry/exit times, prices, and P&L.
*   **`backtest.log`**: A log file containing detailed information about the backtest run, useful for debugging.
*   **`config_snapshot.yaml`**: A copy of the exact configuration file used for the run, ensuring reproducibility.