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
    *   **Confidence Calibration**: Use Isotonic Regression or Platt Scaling to align model probabilities with true event frequencies.
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

### 4. Output Artifacts

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
├── calibration/        # Confidence calibration models
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