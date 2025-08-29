#!/usr/bin/env python3
"""
Capture Signal Generation Test
============================
This script runs the real-time system for a short time to capture
actual signal generation in the logs.
"""

import subprocess
import sys
import time
import threading
from datetime import datetime

def run_signal_capture():
    """Run the signal generator and capture output."""
    print(f"🚀 Starting signal capture test at {datetime.now()}")
    print("=" * 60)
    
    # Start the process
    process = subprocess.Popen([
        sys.executable, '-m', 'ultra_signals.apps.realtime_runner'
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Set a timer to kill it after 90 seconds
    def kill_after_timeout():
        time.sleep(90)
        try:
            process.terminate()
            print("\n⏰ Stopped after 90 seconds")
        except:
            pass
    
    timer = threading.Thread(target=kill_after_timeout, daemon=True)
    timer.start()
    
    signal_count = 0
    feature_compute_count = 0
    
    try:
        for line in process.stdout:
            line = line.strip()
            print(line)
            
            # Count interesting events
            if "PASSED with" in line:
                feature_compute_count += 1
                
            if any(keyword in line.lower() for keyword in ["signal", "long", "short", "entry", "telegram"]):
                signal_count += 1
                print(f"🎯 SIGNAL EVENT DETECTED: {line}")
                
            if process.poll() is not None:
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        process.terminate()
    
    print("\n" + "=" * 60)
    print(f"📊 CAPTURE SUMMARY:")
    print(f"   Feature computations: {feature_compute_count}")
    print(f"   Signal events: {signal_count}")
    print(f"   Test completed at: {datetime.now()}")

if __name__ == "__main__":
    run_signal_capture()
