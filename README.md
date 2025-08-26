# Ultra-Signals: High-Frequency Trading Signal Generation and Backtesting

**Ultra-Signals** is a professional-grade, Python-based framework for developing, testing, and deploying high-frequency trading strategies in cryptocurrency markets. It features a real-time signal generation engine and a comprehensive backtesting suite to ensure strategies are robust and reliable.

The project emphasizes performance, modularity, and clean design, utilizing modern Python libraries like `asyncio`, `Pydantic`, `loguru`, and `scikit-learn`.

---
## Key Features

*   **Real-time Signal Engine**: Connects to live exchange data (Binance USDⓈ-M Futures) to compute features and generate trading signals.
*   **Advanced Feature Library**: Includes a wide range of technical indicators, from standard trend and momentum oscillators to advanced order book and derivatives-based features.
*   **Comprehensive Backtesting Suite (Sprint 6)**: A powerful command-line interface for strategy validation.
    *   **Single Run Backtester**: Test strategies over specific historical periods.
    *   **Walk-Forward Analysis**: Mitigate overfitting by simulating rolling optimization periods with data purging and embargoes.
    *   **Confidence & Auto-Calibration (Sprint 19)**: Isotonic / Platt probability calibration plus walk-forward Bayesian parameter optimization (Optuna) with composite objective & holdout guard.
    *   **Detailed Reporting**: Automatically generates performance reports, equity curves, and trade logs.
*   **Flexible Configuration**: All aspects of the system are controlled via a `settings.yaml` file, with support for environment variable overrides for secrets and CI/CD integration.

---
## Getting Started

### Prerequisites

*   Python 3.11+
*   `venv` for environment management
*   A C++ compiler (required for some dependencies). On Windows, this can be installed with Visual Studio Build Tools.

### 1. Set Up the Environment

A `Makefile` is provided for convenience. For manual setup on Windows, use the commands below.

**Using `make`:**
```bash
make setup
```

**Manual Setup (Windows `cmd` or `PowerShell`):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure the Application

Copy the `.env.example` file to `.env` and `settings.yaml.example` to `settings.yaml`.

*   **`.env`**: Add your API keys and secrets here.
*   **`settings.yaml`**: This is the main configuration file for all modules, including the real-time engine and backtester.

---
## How to Run

### Real-time Engine

To start the live signal generator:
```bash
python -m ultra_signals.apps.realtime_runner --config settings.yaml
```

### Unit Tests

To verify the system's integrity, run the full test suite:
```bash
pytest -q
```

---
## Backtesting Framework (CLI)

The backtesting CLI is the primary tool for strategy validation.

### 1. Standard Backtest

Run a strategy over a fixed historical period.
```bash
# The start/end dates in the command override those in settings.yaml
python -m ultra_signals.apps.backtest_cli run --config settings.yaml --start 2024-01-01 --end 2024-06-30
```

### 2. Walk-Forward Analysis

Perform a robust, rolling backtest to assess out-of-sample performance.
```bash
python -m ultra_signals.apps.backtest_cli wf --config settings.yaml
```

### 3. Confidence Calibration

Fit a calibration model to align prediction scores with true probabilities.
```bash
# This command requires walk-forward analysis to be run first to generate predictions
python -m ultra_signals.apps.backtest_cli cal --config settings.yaml --method isotonic
```

### 4. Auto-Calibration (Sprint 19)

Run Bayesian optimization over strategy knobs using walk-forward evaluation.

1. Create / edit `cal_config.yaml` (see template in repo) to define search space & objective weights.
2. Run optimization (example):
```bash
python -m ultra_signals.apps.backtest_cli cal \
    --config cal_config.yaml \
    --optimize \
    --trials 50 \
    --study-name demo_btc_5m \
    --output-dir reports/cal/demo \
    --holdout-start 2023-06-02 \
    --holdout-end   2023-07-31
```
3. Inspect `reports/cal/demo/` for:
     - `leaderboard.csv` (all trials)
     - `best_params.yaml` (flattened tuned keys)
     - `tuned_params.yaml` + `settings_autotuned.yaml` (full settings snapshot)
     - `holdout_result.yaml` (PROMOTED / REJECTED)
    - `opt_history.(png|html)` & `param_importances.(png|html)` if visualization libs available
4. If holdout status is PROMOTED, adopt `settings_autotuned.yaml` for further backtests.

Composite score balances PF, Winrate, Sharpe, Drawdown (penalized), Stability (1 - PF stdev), with penalties for insufficient trades and overfit gap. Grade distribution (A+/A/B/C/D) is collected; you can optionally add a weight like `grade_good_poor_ratio: 0.05` under `objective.weights` to encourage more high-quality signals.

Parallel optimization: use `--parallel N` (threads) or `--parallel N --parallel-mode process` (multi-process via SQLite study DB) for faster searches when trials are expensive.

### 5. Output Artifacts

All backtesting commands save their results to the `reports/` directory. This includes:
*   `summary.txt`: Key performance metrics.
*   `equity_curve.png`: A plot of portfolio value over time.
*   `trades.csv`: A detailed log of all simulated trades.

---

## Project Structure

```
ultra-signals/
├── apps/               # Application entry points (realtime, backtest CLI)
├── backtest/           # Backtesting-specific modules (runner, metrics, etc.)
├── calibration/        # Confidence + parameter auto-calibration modules (Sprint 19)
├── core/               # Core components (config, types, utils)
├── data/               # Data acquisition and providers
├── engine/             # Signal generation logic (scoring, risk)
├── features/           # Feature computation modules
├── reports/            # Default output directory for backtest artifacts
├── transport/          # Notification services (e.g., Telegram)
├── tests/              # Unit and integration tests
├── .env.example        # Example environment variables
├── settings.yaml       # Main configuration file
└── requirements.txt    # Project dependencies
```

---
## Change Log (Excerpt)

* Sprint 19: Added walk-forward Bayesian optimization (Optuna) CLI via `cal --optimize`, composite objective, persistence of tuned settings & holdout validation.