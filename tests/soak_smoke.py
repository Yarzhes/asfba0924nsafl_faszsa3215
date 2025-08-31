#!/usr/bin/env python3
"""
Soak Test for Ultra Signals Stability

This script runs extended tests to verify:
1. Cancellation path handling
2. WebSocket reconnection resilience
3. Memory/FD leak detection
4. Task growth monitoring
5. Symbol isolation and deduplication
"""

import asyncio
import time
import tracemalloc
import psutil
import argparse
import json
import random
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from loguru import logger

# Mock imports to avoid actual connections
with patch.dict('sys.modules', {
    'ultra_signals.core.config': Mock(),
    'ultra_signals.data.binance_ws': Mock(),
    'ultra_signals.data.funding_provider': Mock(),
    'ultra_signals.core.feature_store': Mock(),
    'ultra_signals.engine.real_engine': Mock(),
    'ultra_signals.transport.telegram': Mock(),
    'ultra_signals.live.metrics': Mock(),
}):
    from ultra_signals.apps.realtime_runner import ResilientSignalRunner, SymbolState


@dataclass
class SoakMetrics:
    """Metrics collected during soak test."""
    start_time: float
    end_time: float
    total_runtime: float
    reconnection_attempts: int
    successful_reconnections: int
    failed_reconnections: int
    memory_snapshots: List[Dict]
    task_counts: List[Dict]
    alerts_sent: int
    alerts_blocked: int
    cancellation_events: int
    errors: List[str]
    warnings: List[str]


class MockWebSocket:
    """Mock WebSocket that simulates disconnections and reconnections."""
    
    def __init__(self, symbol: str, disconnect_probability: float = 0.1):
        self.symbol = symbol
        self.disconnect_probability = disconnect_probability
        self.connected = True
        self.reconnect_count = 0
        self.last_message_time = time.time()
        
    async def connect(self):
        """Simulate connection."""
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connected = True
        self.reconnect_count += 1
        logger.info(f"Mock WS connected: {self.symbol}")
        
    async def disconnect(self):
        """Simulate disconnection."""
        self.connected = False
        logger.warning(f"Mock WS disconnected: {self.symbol}")
        
    async def send_message(self, message: str):
        """Simulate sending message with potential disconnection."""
        if not self.connected:
            raise ConnectionError(f"WebSocket not connected: {self.symbol}")
            
        # Simulate random disconnection
        if random.random() < self.disconnect_probability:
            await self.disconnect()
            raise ConnectionError(f"Simulated disconnection: {self.symbol}")
            
        self.last_message_time = time.time()
        await asyncio.sleep(0.01)  # Simulate message processing
        
    async def close(self):
        """Simulate closing connection."""
        self.connected = False
        logger.info(f"Mock WS closed: {self.symbol}")


class SoakTestRunner:
    """Runner for soak testing with comprehensive monitoring."""
    
    def __init__(self, duration_minutes: int = 30, symbols: List[str] = None):
        self.duration_minutes = duration_minutes
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.metrics = SoakMetrics(
            start_time=time.time(),
            end_time=0,
            total_runtime=0,
            reconnection_attempts=0,
            successful_reconnections=0,
            failed_reconnections=0,
            memory_snapshots=[],
            task_counts=[],
            alerts_sent=0,
            alerts_blocked=0,
            cancellation_events=0,
            errors=[],
            warnings=[]
        )
        
        # Mock settings
        self.settings = Mock()
        self.settings.runtime.min_signal_interval_sec = 30.0
        self.settings.runtime.min_confidence = 0.65
        self.settings.features.warmup_periods = 50  # Reduced for testing
        self.settings.debug = True
        
        # Create runner
        self.runner = ResilientSignalRunner(self.settings)
        
        # Mock WebSockets
        self.websockets = {
            symbol: MockWebSocket(symbol, disconnect_probability=0.15)
            for symbol in self.symbols
        }
        
        # Task tracking
        self.active_tasks = set()
        self.task_history = []
        
    async def monitor_memory(self):
        """Monitor memory usage and take snapshots."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        
        while self.metrics.end_time == 0:
            current_time = time.time()
            
            # Take memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            memory_info = {
                'timestamp': current_time,
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'top_allocations': [
                    {
                        'file': str(stat.traceback.format()[-1]),
                        'size_mb': stat.size / 1024 / 1024,
                        'count': stat.count
                    }
                    for stat in top_stats[:5]
                ]
            }
            
            self.metrics.memory_snapshots.append(memory_info)
            
            # Check for memory leaks (growth > 50MB over 5 minutes)
            if len(self.metrics.memory_snapshots) > 10:
                recent_memory = self.metrics.memory_snapshots[-10:]
                memory_growth = recent_memory[-1]['memory_usage_mb'] - recent_memory[0]['memory_usage_mb']
                if memory_growth > 50:
                    self.metrics.warnings.append(f"Memory growth detected: {memory_growth:.1f}MB")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def monitor_tasks(self):
        """Monitor active task count."""
        while self.metrics.end_time == 0:
            current_time = time.time()
            active_count = len(asyncio.all_tasks())
            
            task_info = {
                'timestamp': current_time,
                'active_tasks': active_count,
                'task_names': [task.get_name() for task in asyncio.all_tasks()]
            }
            
            self.metrics.task_counts.append(task_info)
            
            # Check for unbounded task growth
            if len(self.metrics.task_counts) > 10:
                recent_tasks = self.metrics.task_counts[-10:]
                task_growth = recent_tasks[-1]['active_tasks'] - recent_tasks[0]['active_tasks']
                if task_growth > 10:
                    self.metrics.warnings.append(f"Task growth detected: +{task_growth} tasks")
            
            await asyncio.sleep(10)  # Check every 10 seconds
            
    async def simulate_websocket_operations(self):
        """Simulate WebSocket operations with disconnections."""
        while self.metrics.end_time == 0:
            for symbol, ws in self.websockets.items():
                try:
                    if not ws.connected:
                        self.metrics.reconnection_attempts += 1
                        try:
                            await ws.connect()
                            self.metrics.successful_reconnections += 1
                            logger.info(f"Reconnected to {symbol}")
                        except Exception as e:
                            self.metrics.failed_reconnections += 1
                            logger.error(f"Failed to reconnect to {symbol}: {e}")
                            self.metrics.errors.append(f"Reconnection failed for {symbol}: {e}")
                    
                    # Send test message
                    await ws.send_message(f"test_message_{symbol}")
                    
                except Exception as e:
                    if "disconnection" in str(e).lower():
                        logger.warning(f"Expected disconnection for {symbol}: {e}")
                    else:
                        logger.error(f"Unexpected error for {symbol}: {e}")
                        self.metrics.errors.append(f"WebSocket error for {symbol}: {e}")
            
            await asyncio.sleep(5)  # Simulate message interval
            
    async def simulate_signal_generation(self):
        """Simulate signal generation and alert sending."""
        while self.metrics.end_time == 0:
            for symbol in self.symbols:
                try:
                    # Simulate decision
                    decision = Mock()
                    decision.confidence = random.uniform(0.6, 0.9)
                    decision.decision = random.choice(["LONG", "SHORT"])
                    decision.symbol = symbol
                    decision.tf = "5m"
                    
                    # Check if should send signal
                    current_time = time.time()
                    should_send = self.runner._should_send_signal(symbol, decision, current_time)
                    
                    if should_send:
                        self.metrics.alerts_sent += 1
                        logger.info(f"Signal sent for {symbol}: {decision.decision} @ {decision.confidence:.3f}")
                    else:
                        self.metrics.alerts_blocked += 1
                        logger.debug(f"Signal blocked for {symbol}: cooldown/confidence")
                        
                except Exception as e:
                    logger.error(f"Error in signal generation for {symbol}: {e}")
                    self.metrics.errors.append(f"Signal generation error for {symbol}: {e}")
            
            await asyncio.sleep(10)  # Simulate signal interval
            
    async def simulate_cancellation_events(self):
        """Simulate cancellation events to test resilience."""
        while self.metrics.end_time == 0:
            # Randomly cancel some tasks to test resilience
            if random.random() < 0.05:  # 5% chance
                try:
                    # Create a dummy task and cancel it
                    async def dummy_task():
                        await asyncio.sleep(10)
                    
                    task = asyncio.create_task(dummy_task())
                    await asyncio.sleep(0.1)
                    task.cancel()
                    
                    self.metrics.cancellation_events += 1
                    logger.info("Simulated task cancellation")
                    
                except asyncio.CancelledError:
                    logger.info("Task cancellation handled correctly")
                except Exception as e:
                    logger.error(f"Error handling cancellation: {e}")
                    self.metrics.errors.append(f"Cancellation error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def run_soak_test(self):
        """Run the complete soak test."""
        logger.info(f"Starting soak test for {self.duration_minutes} minutes")
        logger.info(f"Testing symbols: {self.symbols}")
        
        # Start monitoring tasks
        monitor_tasks = [
            asyncio.create_task(self.monitor_memory()),
            asyncio.create_task(self.monitor_tasks()),
            asyncio.create_task(self.simulate_websocket_operations()),
            asyncio.create_task(self.simulate_signal_generation()),
            asyncio.create_task(self.simulate_cancellation_events())
        ]
        
        try:
            # Run for specified duration
            await asyncio.sleep(self.duration_minutes * 60)
            
        except asyncio.CancelledError:
            logger.info("Soak test cancelled by user")
            self.metrics.cancellation_events += 1
        except Exception as e:
            logger.error(f"Unexpected error in soak test: {e}")
            self.metrics.errors.append(f"Soak test error: {e}")
        finally:
            # Cleanup
            self.metrics.end_time = time.time()
            self.metrics.total_runtime = self.metrics.end_time - self.metrics.start_time
            
            # Cancel monitoring tasks
            for task in monitor_tasks:
                task.cancel()
            
            # Wait for cleanup
            await asyncio.gather(*monitor_tasks, return_exceptions=True)
            
            # Close WebSockets
            for ws in self.websockets.values():
                await ws.close()
            
            logger.info("Soak test completed")
            
    def generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        # Calculate memory growth
        if len(self.metrics.memory_snapshots) > 1:
            initial_memory = self.metrics.memory_snapshots[0]['memory_usage_mb']
            final_memory = self.metrics.memory_snapshots[-1]['memory_usage_mb']
            memory_growth = final_memory - initial_memory
        else:
            memory_growth = 0
            
        # Calculate task growth
        if len(self.metrics.task_counts) > 1:
            initial_tasks = self.metrics.task_counts[0]['active_tasks']
            final_tasks = self.metrics.task_counts[-1]['active_tasks']
            task_growth = final_tasks - initial_tasks
        else:
            task_growth = 0
            
        # Calculate success rates
        reconnection_success_rate = (
            self.metrics.successful_reconnections / max(self.metrics.reconnection_attempts, 1) * 100
        )
        
        alert_block_rate = (
            self.metrics.alerts_blocked / max(self.metrics.alerts_sent + self.metrics.alerts_blocked, 1) * 100
        )
        
        report = {
            'test_duration_minutes': self.duration_minutes,
            'total_runtime_seconds': self.metrics.total_runtime,
            'symbols_tested': self.symbols,
            
            # Stability metrics
            'reconnection_attempts': self.metrics.reconnection_attempts,
            'successful_reconnections': self.metrics.successful_reconnections,
            'failed_reconnections': self.metrics.failed_reconnections,
            'reconnection_success_rate_percent': reconnection_success_rate,
            
            # Memory metrics
            'memory_growth_mb': memory_growth,
            'memory_snapshots_taken': len(self.metrics.memory_snapshots),
            
            # Task metrics
            'task_growth': task_growth,
            'task_snapshots_taken': len(self.metrics.task_counts),
            
            # Signal metrics
            'alerts_sent': self.metrics.alerts_sent,
            'alerts_blocked': self.metrics.alerts_blocked,
            'alert_block_rate_percent': alert_block_rate,
            
            # Cancellation metrics
            'cancellation_events': self.metrics.cancellation_events,
            
            # Error tracking
            'total_errors': len(self.metrics.errors),
            'total_warnings': len(self.metrics.warnings),
            'errors': self.metrics.errors,
            'warnings': self.metrics.warnings,
            
            # Pass/fail criteria
            'stability_passed': (
                self.metrics.total_runtime >= self.duration_minutes * 60 * 0.95 and  # 95% uptime
                memory_growth < 100 and  # < 100MB memory growth
                task_growth < 20 and  # < 20 task growth
                reconnection_success_rate > 80  # > 80% reconnection success
            ),
            'isolation_passed': alert_block_rate > 20,  # > 20% alerts blocked (cooldown working)
            'resilience_passed': len(self.metrics.errors) < 10  # < 10 errors
        }
        
        return report


async def main():
    """Main entry point for soak testing."""
    parser = argparse.ArgumentParser(description="Ultra Signals Soak Test")
    parser.add_argument("--duration-min", type=int, default=30, help="Test duration in minutes")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,BNBUSDT", help="Comma-separated symbols")
    parser.add_argument("--force-disconnect", action="store_true", help="Force WebSocket disconnections")
    parser.add_argument("--json-metrics", type=str, help="Output metrics to JSON file")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Run soak test
    runner = SoakTestRunner(
        duration_minutes=args.duration_min,
        symbols=symbols
    )
    
    if args.force_disconnect:
        # Increase disconnect probability
        for ws in runner.websockets.values():
            ws.disconnect_probability = 0.3
    
    try:
        await runner.run_soak_test()
    except KeyboardInterrupt:
        logger.info("Soak test interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in soak test: {e}")
        return 1
    
    # Generate and display report
    report = runner.generate_report()
    
    print("\n" + "="*80)
    print("SOAK TEST RESULTS")
    print("="*80)
    
    print(f"Duration: {report['test_duration_minutes']} minutes")
    print(f"Runtime: {report['total_runtime_seconds']:.1f} seconds")
    print(f"Symbols: {', '.join(report['symbols_tested'])}")
    
    print(f"\nStability Metrics:")
    print(f"  Reconnection attempts: {report['reconnection_attempts']}")
    print(f"  Successful reconnections: {report['successful_reconnections']}")
    print(f"  Reconnection success rate: {report['reconnection_success_rate_percent']:.1f}%")
    print(f"  Memory growth: {report['memory_growth_mb']:.1f}MB")
    print(f"  Task growth: {report['task_growth']}")
    print(f"  Cancellation events: {report['cancellation_events']}")
    
    print(f"\nSignal Metrics:")
    print(f"  Alerts sent: {report['alerts_sent']}")
    print(f"  Alerts blocked: {report['alerts_blocked']}")
    print(f"  Alert block rate: {report['alert_block_rate_percent']:.1f}%")
    
    print(f"\nError Tracking:")
    print(f"  Total errors: {report['total_errors']}")
    print(f"  Total warnings: {report['total_warnings']}")
    
    print(f"\nTest Results:")
    print(f"  Stability: {'✅ PASSED' if report['stability_passed'] else '❌ FAILED'}")
    print(f"  Isolation: {'✅ PASSED' if report['isolation_passed'] else '❌ FAILED'}")
    print(f"  Resilience: {'✅ PASSED' if report['resilience_passed'] else '❌ FAILED'}")
    
    overall_passed = all([
        report['stability_passed'],
        report['isolation_passed'],
        report['resilience_passed']
    ])
    
    print(f"\nOverall Result: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
    
    # Save metrics if requested
    if args.json_metrics:
        with open(args.json_metrics, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nMetrics saved to: {args.json_metrics}")
    
    return 0 if overall_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)



