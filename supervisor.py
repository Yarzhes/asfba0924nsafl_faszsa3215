#!/usr/bin/env python3
"""
Supervisor script for Ultra Signals Trading Helper

This script provides robust error handling and automatic restart capabilities:
1. Captures all terminal output
2. Saves last 100 lines when errors occur
3. Implements exponential backoff for restarts
4. Provides detailed error reporting
5. Maintains a restart counter and history

Usage:
    python supervisor.py [--config settings.yaml] [--max-restarts 10] [--backoff-base 60]
"""

import asyncio
import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import List, Optional
import json

class OutputCapture:
    """Captures and manages terminal output with ring buffer for last N lines."""
    
    def __init__(self, max_lines: int = 100):
        self.max_lines = max_lines
        self.lines = deque(maxlen=max_lines)
        self.start_time = time.time()
        self.total_lines = 0
        
    def add_line(self, line: str):
        """Add a line to the capture buffer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_line = f"[{timestamp}] {line.rstrip()}"
        self.lines.append(formatted_line)
        self.total_lines += 1
        
    def get_last_lines(self, n: int = None) -> List[str]:
        """Get the last N lines (defaults to all captured lines)."""
        if n is None:
            n = len(self.lines)
        return list(self.lines)[-n:]
        
    def save_to_file(self, filename: str, lines: int = None):
        """Save captured lines to a file."""
        if lines is None:
            lines = len(self.lines)
            
        output_lines = self.get_last_lines(lines)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Ultra Signals Output Capture\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total lines captured: {self.total_lines}\n")
            f.write(f"# Lines shown: {len(output_lines)}\n")
            f.write(f"# Runtime: {time.time() - self.start_time:.1f} seconds\n")
            f.write(f"# {'='*50}\n\n")
            
            for line in output_lines:
                f.write(f"{line}\n")
                
        print(f"📄 Saved {len(output_lines)} lines to {filename}")

class Supervisor:
    """Main supervisor class for managing the trading application."""
    
    def __init__(self, config: str = "settings.yaml", max_restarts: int = 10, 
                 backoff_base: int = 60, backoff_max: int = 3600):
        self.config = config
        self.max_restarts = max_restarts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.restart_count = 0
        self.start_time = time.time()
        self.output_capture = OutputCapture(max_lines=100)
        self.shutdown_requested = False
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Restart history
        self.restart_history = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown_requested = True
        
    def _calculate_backoff(self, attempt: int) -> int:
        """Calculate exponential backoff time."""
        backoff = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        return int(backoff)
        
    def _save_error_report(self, error: Exception, process_output: List[str]):
        """Save detailed error report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = self.logs_dir / f"error_{timestamp}.txt"
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"# Ultra Signals Error Report\n")
            f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Restart Count: {self.restart_count}/{self.max_restarts}\n")
            f.write(f"# Total Runtime: {time.time() - self.start_time:.1f} seconds\n")
            f.write(f"# {'='*50}\n\n")
            
            f.write(f"## Error Details\n")
            f.write(f"Error Type: {type(error).__name__}\n")
            f.write(f"Error Message: {str(error)}\n")
            f.write(f"Error Args: {error.args}\n\n")
            
            f.write(f"## Last 100 Lines of Output\n")
            f.write(f"{'='*50}\n")
            for line in process_output:
                f.write(f"{line}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"## Restart History\n")
            for i, restart in enumerate(self.restart_history):
                f.write(f"{i+1}. {restart['timestamp']} - {restart['reason']}\n")
                
        print(f"📋 Error report saved to {error_file}")
        
    def _save_restart_history(self):
        """Save restart history to JSON file."""
        history_file = self.logs_dir / "restart_history.json"
        
        history_data = {
            "supervisor_start": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_restarts": self.restart_count,
            "max_restarts": self.max_restarts,
            "restarts": self.restart_history
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, default=str)
            
    async def _run_process(self) -> subprocess.Popen:
        """Run the trading application process."""
        # Build command
        cmd = [
            sys.executable, "-m", "ultra_signals.apps.realtime_runner"
        ]
        
        # Add environment variables
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        # Remove canary-specific environment variable
        # env['CANARY_DURATION_SEC'] = '3600'  # 1 hour runs
        
        print(f"🚀 Starting process: {' '.join(cmd)}")
        print(f"📊 Max restarts: {self.max_restarts}, Backoff base: {self.backoff_base}s")
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        return process
        
    async def _monitor_process(self, process: subprocess.Popen):
        """Monitor the running process and capture output."""
        print(f"👀 Monitoring process PID: {process.pid}")
        
        try:
            while process.poll() is None and not self.shutdown_requested:
                # Read output line by line
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    self.output_capture.add_line(line)
                    print(f"[{process.pid}] {line}")
                else:
                    # No output, check if process is still alive
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Error monitoring process: {e}")
            
        finally:
            # Get remaining output
            remaining_output, _ = process.communicate()
            if remaining_output:
                for line in remaining_output.splitlines():
                    if line.strip():
                        self.output_capture.add_line(line)
                        print(f"[{process.pid}] {line}")
                        
    async def run(self):
        """Main supervisor loop."""
        print("🎯 Ultra Signals Supervisor Starting...")
        print(f"📁 Logs directory: {self.logs_dir.absolute()}")
        print(f"⚙️  Config: {self.config}")
        
        while self.restart_count < self.max_restarts and not self.shutdown_requested:
            try:
                # Start the process
                process = await self._run_process()
                
                # Monitor the process
                await self._monitor_process(process)
                
                # Check exit code
                exit_code = process.returncode
                
                if exit_code == 0:
                    print("✅ Process completed successfully")
                    break
                else:
                    print(f"⚠️  Process exited with code {exit_code}")
                    
            except Exception as e:
                print(f"❌ Supervisor error: {e}")
                exit_code = -1
                
            # Handle restart logic
            if not self.shutdown_requested:
                self.restart_count += 1
                
                if self.restart_count > self.max_restarts:
                    print(f"🛑 Max restarts ({self.max_restarts}) exceeded. Stopping.")
                    break
                    
                # Save error information
                last_lines = self.output_capture.get_last_lines(100)
                self._save_error_report(Exception(f"Process exit code: {exit_code}"), last_lines)
                
                # Record restart
                restart_info = {
                    "timestamp": datetime.now().isoformat(),
                    "restart_number": self.restart_count,
                    "reason": f"Exit code: {exit_code}",
                    "uptime_seconds": time.time() - self.start_time
                }
                self.restart_history.append(restart_info)
                
                # Calculate backoff
                backoff_time = self._calculate_backoff(self.restart_count - 1)
                print(f"🔄 Restarting in {backoff_time}s (attempt {self.restart_count}/{self.max_restarts})")
                
                # Save restart history
                self._save_restart_history()
                
                # Wait before restart
                await asyncio.sleep(backoff_time)
                
        # Final cleanup
        print("🏁 Supervisor shutdown complete")
        print(f"📈 Total restarts: {self.restart_count}")
        print(f"⏱️  Total runtime: {time.time() - self.start_time:.1f}s")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ultra Signals Supervisor")
    parser.add_argument("--config", default="settings.yaml", help="Configuration file")
    parser.add_argument("--max-restarts", type=int, default=10, help="Maximum restart attempts")
    parser.add_argument("--backoff-base", type=int, default=60, help="Base backoff time in seconds")
    parser.add_argument("--backoff-max", type=int, default=3600, help="Maximum backoff time in seconds")
    
    args = parser.parse_args()
    
    # Create and run supervisor
    supervisor = Supervisor(
        config=args.config,
        max_restarts=args.max_restarts,
        backoff_base=args.backoff_base,
        backoff_max=args.backoff_max
    )
    
    try:
        asyncio.run(supervisor.run())
    except KeyboardInterrupt:
        print("\n🛑 Supervisor interrupted by user")
    except Exception as e:
        print(f"❌ Fatal supervisor error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
