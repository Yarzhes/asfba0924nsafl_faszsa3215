"""
Lightweight Signal Backtesting

This module provides a lightweight backtesting framework for evaluating
signal performance using triple-barrier method and standardized metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import time

from .labeling import label_trades, TripleBarrierResult, calculate_outcome_statistics
from .calibration import calibrate_ensemble_scores, evaluate_calibration


@dataclass
class BacktestSignal:
    """Signal record for backtesting."""
    timestamp: pd.Timestamp
    symbol: str
    timeframe: str
    decision: str  # 'LONG', 'SHORT', 'FLAT'
    confidence: float
    entry_price: float
    atr: float
    regime: str
    features: Dict[str, Any]


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    symbol: str
    timeframe: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_signals: int
    long_signals: int
    short_signals: int
    flat_signals: int
    win_rate: float
    avg_return: float
    avg_horizon: float
    sharpe_ratio: float
    max_drawdown: float
    expectancy: float
    precision_top_decile: float
    calibration_error: float
    regime_performance: Dict[str, Dict[str, float]]
    signal_details: List[Dict[str, Any]]


def run_signal_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    primary_tf: str = "5m",
    settings: Optional[Dict] = None
) -> Dict[str, BacktestResult]:
    """
    Run signal backtest on multiple symbols.
    
    Args:
        symbols: List of symbols to backtest
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        primary_tf: Primary timeframe for signals
        settings: Application settings
        
    Returns:
        Dictionary mapping symbols to backtest results
    """
    if settings is None:
        settings = {}
    
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Running backtest for {symbol}")
            result = _backtest_single_symbol(symbol, start_date, end_date, primary_tf, settings)
            results[symbol] = result
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            continue
    
    return results


def _backtest_single_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    primary_tf: str,
    settings: Dict
) -> BacktestResult:
    """Backtest a single symbol."""
    # Load price data (placeholder - would integrate with actual data source)
    prices = _load_price_data(symbol, start_date, end_date, primary_tf)
    
    if len(prices) < 100:
        raise ValueError(f"Insufficient data for {symbol}: {len(prices)} bars")
    
    # Generate signals (placeholder - would integrate with actual signal engine)
    signals = _generate_signals(symbol, prices, primary_tf, settings)
    
    # Evaluate signals using triple-barrier method
    outcomes = _evaluate_signals(signals, prices, settings)
    
    # Calculate performance metrics
    metrics = _calculate_performance_metrics(signals, outcomes, settings)
    
    # Create result object
    result = BacktestResult(
        symbol=symbol,
        timeframe=primary_tf,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        total_signals=len(signals),
        long_signals=sum(1 for s in signals if s.decision == "LONG"),
        short_signals=sum(1 for s in signals if s.decision == "SHORT"),
        flat_signals=sum(1 for s in signals if s.decision == "FLAT"),
        win_rate=metrics['win_rate'],
        avg_return=metrics['avg_return'],
        avg_horizon=metrics['avg_horizon'],
        sharpe_ratio=metrics['sharpe_ratio'],
        max_drawdown=metrics['max_drawdown'],
        expectancy=metrics['expectancy'],
        precision_top_decile=metrics['precision_top_decile'],
        calibration_error=metrics['calibration_error'],
        regime_performance=metrics['regime_performance'],
        signal_details=metrics['signal_details']
    )
    
    return result


def _load_price_data(symbol: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
    """Load price data for backtesting (placeholder implementation)."""
    # This would integrate with actual data source
    # For now, generate synthetic data for testing
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    # Generate synthetic OHLCV data
    periods = pd.date_range(start=start_ts, end=end_ts, freq=timeframe)
    
    # Simple random walk with trend
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.0001, 0.02, len(periods))  # Small positive drift
    
    # Add some trend
    trend = np.linspace(0, 0.1, len(periods))
    returns += trend * 0.001
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV DataFrame
    data = []
    for i, (ts, price) in enumerate(zip(periods, prices)):
        # Generate realistic OHLC from close
        volatility = 0.01
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.lognormal(10, 1)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def _generate_signals(symbol: str, prices: pd.DataFrame, timeframe: str, settings: Dict) -> List[BacktestSignal]:
    """Generate signals for backtesting (placeholder implementation)."""
    signals = []
    
    # Simple signal generation for testing
    # In practice, this would use the actual signal engine
    
    for i in range(100, len(prices) - 20):  # Skip warmup, leave room for evaluation
        price = prices.iloc[i]['close']
        
        # Simple momentum-based signal
        returns_5 = (price / prices.iloc[i-5]['close']) - 1
        returns_20 = (price / prices.iloc[i-20]['close']) - 1
        
        # Calculate ATR
        high_low = prices.iloc[i-14:i+1]['high'] - prices.iloc[i-14:i+1]['low']
        high_close = np.abs(prices.iloc[i-14:i+1]['high'] - prices.iloc[i-14:i+1]['close'].shift(1))
        low_close = np.abs(prices.iloc[i-14:i+1]['low'] - prices.iloc[i-14:i+1]['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.mean()
        
        # Simple signal logic
        if returns_5 > 0.02 and returns_20 > 0.05:
            decision = "LONG"
            confidence = min(0.9, 0.5 + abs(returns_5) * 10)
        elif returns_5 < -0.02 and returns_20 < -0.05:
            decision = "SHORT"
            confidence = min(0.9, 0.5 + abs(returns_5) * 10)
        else:
            decision = "FLAT"
            confidence = 0.5
        
        # Simple regime detection
        if abs(returns_20) > 0.1:
            regime = "trend"
        elif abs(returns_5) < 0.01:
            regime = "chop"
        else:
            regime = "mixed"
        
        signal = BacktestSignal(
            timestamp=prices.index[i],
            symbol=symbol,
            timeframe=timeframe,
            decision=decision,
            confidence=confidence,
            entry_price=price,
            atr=atr,
            regime=regime,
            features={
                'returns_5': returns_5,
                'returns_20': returns_20,
                'atr': atr
            }
        )
        
        signals.append(signal)
    
    return signals


def _evaluate_signals(
    signals: List[BacktestSignal],
    prices: pd.DataFrame,
    settings: Dict
) -> List[TripleBarrierResult]:
    """Evaluate signals using triple-barrier method."""
    outcomes = []
    
    for signal in signals:
        if signal.decision == "FLAT":
            # Skip flat signals
            continue
        
        # Find entry index
        entry_idx = prices.index.get_loc(signal.timestamp)
        
        # Get price series
        price_series = prices['close']
        
        # Label trade using triple-barrier method
        pt_mult = 2.0  # 2:1 risk-reward
        sl_mult = 1.0  # 1 ATR stop loss
        max_horizon = 20  # 20 bars max
        
        outcome = label_trades(
            prices=price_series,
            entry_idx=entry_idx,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            max_horizon_bars=max_horizon
        )
        
        outcomes.append(outcome)
    
    return outcomes


def _calculate_performance_metrics(
    signals: List[BacktestSignal],
    outcomes: List[TripleBarrierResult],
    settings: Dict
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    if not outcomes:
        return {
            'win_rate': 0.0,
            'avg_return': 0.0,
            'avg_horizon': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'expectancy': 0.0,
            'precision_top_decile': 0.0,
            'calibration_error': 0.0,
            'regime_performance': {},
            'signal_details': []
        }
    
    # Basic statistics
    stats = calculate_outcome_statistics(outcomes)
    
    # Calculate returns for Sharpe ratio
    returns = [outcome.return_pct for outcome in outcomes]
    
    # Sharpe ratio (assuming 0% risk-free rate)
    if len(returns) > 1:
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
    else:
        sharpe_ratio = 0.0
    
    # Maximum drawdown
    cumulative_returns = np.cumprod([1 + r for r in returns])
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Expectancy
    expectancy = np.mean(returns)
    
    # Precision at top decile
    confidences = [s.confidence for s in signals if s.decision != "FLAT"]
    if len(confidences) >= 10:
        threshold = np.percentile(confidences, 90)
        top_signals = [i for i, conf in enumerate(confidences) if conf >= threshold]
        top_outcomes = [outcomes[i] for i in top_signals if i < len(outcomes)]
        if top_outcomes:
            precision_top_decile = np.mean([o.outcome == 1 for o in top_outcomes])
        else:
            precision_top_decile = 0.0
    else:
        precision_top_decile = 0.0
    
    # Calibration error
    if len(confidences) >= 10 and len(outcomes) >= 10:
        binary_outcomes = [1 if o.outcome == 1 else 0 for o in outcomes]
        calibration_metrics = evaluate_calibration(
            np.array(confidences[:len(binary_outcomes)]),
            np.array(binary_outcomes)
        )
        calibration_error = calibration_metrics['calibration_error']
    else:
        calibration_error = 0.0
    
    # Regime performance
    regime_performance = {}
    for regime in set(s.regime for s in signals):
        regime_signals = [i for i, s in enumerate(signals) if s.regime == regime and s.decision != "FLAT"]
        regime_outcomes = [outcomes[i] for i in regime_signals if i < len(outcomes)]
        if regime_outcomes:
            regime_stats = calculate_outcome_statistics(regime_outcomes)
            regime_performance[regime] = regime_stats
    
    # Signal details
    signal_details = []
    for i, signal in enumerate(signals):
        if signal.decision != "FLAT" and i < len(outcomes):
            detail = {
                'timestamp': signal.timestamp,
                'decision': signal.decision,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'regime': signal.regime,
                'outcome': outcomes[i].outcome,
                'return_pct': outcomes[i].return_pct,
                'horizon_bars': outcomes[i].horizon_bars,
                'hit_barrier': outcomes[i].hit_barrier
            }
            signal_details.append(detail)
    
    return {
        'win_rate': stats['win_rate'],
        'avg_return': stats['avg_return'],
        'avg_horizon': stats['avg_horizon'],
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'expectancy': expectancy,
        'precision_top_decile': precision_top_decile,
        'calibration_error': calibration_error,
        'regime_performance': regime_performance,
        'signal_details': signal_details
    }


def generate_backtest_report(results: Dict[str, BacktestResult]) -> str:
    """Generate a comprehensive backtest report."""
    report = []
    report.append("=" * 80)
    report.append("ULTRA SIGNALS BACKTEST REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    total_signals = sum(r.total_signals for r in results.values())
    total_long = sum(r.long_signals for r in results.values())
    total_short = sum(r.short_signals for r in results.values())
    avg_win_rate = np.mean([r.win_rate for r in results.values()])
    avg_sharpe = np.mean([r.sharpe_ratio for r in results.values()])
    
    report.append(f"SUMMARY:")
    report.append(f"  Total Symbols: {len(results)}")
    report.append(f"  Total Signals: {total_signals}")
    report.append(f"  Long Signals: {total_long}")
    report.append(f"  Short Signals: {total_short}")
    report.append(f"  Average Win Rate: {avg_win_rate:.2%}")
    report.append(f"  Average Sharpe Ratio: {avg_sharpe:.3f}")
    report.append("")
    
    # Per-symbol results
    report.append("PER-SYMBOL RESULTS:")
    report.append("-" * 80)
    report.append(f"{'Symbol':<12} {'Signals':<8} {'Win Rate':<10} {'Sharpe':<8} {'Max DD':<8} {'Expectancy':<10}")
    report.append("-" * 80)
    
    for symbol, result in results.items():
        report.append(
            f"{symbol:<12} {result.total_signals:<8} {result.win_rate:<10.2%} "
            f"{result.sharpe_ratio:<8.3f} {result.max_drawdown:<8.2%} {result.expectancy:<10.2%}"
        )
    
    report.append("")
    
    # Regime performance
    report.append("REGIME PERFORMANCE:")
    report.append("-" * 40)
    
    all_regimes = set()
    for result in results.values():
        all_regimes.update(result.regime_performance.keys())
    
    for regime in sorted(all_regimes):
        regime_results = []
        for result in results.values():
            if regime in result.regime_performance:
                regime_results.append(result.regime_performance[regime])
        
        if regime_results:
            avg_win_rate = np.mean([r['win_rate'] for r in regime_results])
            avg_return = np.mean([r['avg_return'] for r in regime_results])
            report.append(f"{regime:<15} Win Rate: {avg_win_rate:.2%}, Avg Return: {avg_return:.2%}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def save_backtest_results(results: Dict[str, BacktestResult], filename: str):
    """Save backtest results to file."""
    import json
    from datetime import datetime
    
    # Convert results to serializable format
    serializable_results = {}
    for symbol, result in results.items():
        serializable_results[symbol] = {
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'total_signals': result.total_signals,
            'long_signals': result.long_signals,
            'short_signals': result.short_signals,
            'flat_signals': result.flat_signals,
            'win_rate': result.win_rate,
            'avg_return': result.avg_return,
            'avg_horizon': result.avg_horizon,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'expectancy': result.expectancy,
            'precision_top_decile': result.precision_top_decile,
            'calibration_error': result.calibration_error,
            'regime_performance': result.regime_performance,
            'signal_details': result.signal_details
        }
    
    # Add metadata
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'Ultra Signals Backtest Results'
        },
        'results': serializable_results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Backtest results saved to {filename}")



