import subprocess
import time
import signal
import sys
from datetime import datetime

class SignalCapture:
    def __init__(self):
        self.process = None
        self.signal_count = 0
        self.feature_count = 0
        
    def run_extended_capture(self, duration_minutes=15):
        """Run capture for extended period to catch the transition from warmup to signals"""
        
        print(f"🚀 Starting EXTENDED signal capture test for {duration_minutes} minutes...")
        print(f"⏱️  This will capture the moment signals start generating!")
        print("=" * 80)
        
        start_time = time.time()
        timeout = duration_minutes * 60  # Convert to seconds
        
        try:
            # Start the realtime runner
            self.process = subprocess.Popen(
                ['python', 'ultra_signals/apps/realtime_runner.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"📡 Real-time runner started (PID: {self.process.pid})")
            print("🔍 Monitoring for warmup completion and signal generation...")
            print("=" * 60)
            
            while time.time() - start_time < timeout:
                try:
                    # Read output with timeout
                    line = self.process.stdout.readline()
                    if line:
                        line = line.strip()
                        print(line)
                        
                        # Track important events
                        if "feature computation" in line.lower():
                            self.feature_count += 1
                            
                        if "signal" in line.lower() and ("buy" in line.lower() or "sell" in line.lower()):
                            self.signal_count += 1
                            print(f"🎯 SIGNAL DETECTED! Total signals: {self.signal_count}")
                            
                        if "warmup check: PASSED" in line:
                            print(f"✅ WARMUP COMPLETE: {line}")
                            
                        if "ohlcv_len=2" in line:
                            print(f"📈 DATA READY: {line}")
                            
                    # Check if process ended
                    if self.process.poll() is not None:
                        print("⚠️ Process ended unexpectedly")
                        break
                        
                except UnicodeDecodeError:
                    continue
                    
        except KeyboardInterrupt:
            print("\n⏸️ Capture interrupted by user")
            
        finally:
            print(f"\n⏰ Extended capture completed after {(time.time() - start_time)/60:.1f} minutes")
            self.cleanup()
            
        # Print summary
        print("=" * 80)
        print("📊 EXTENDED CAPTURE SUMMARY:")
        print(f"   Feature computations: {self.feature_count}")
        print(f"   Signal events: {self.signal_count}")
        print(f"   Duration: {(time.time() - start_time)/60:.1f} minutes")
        print(f"   Test completed at: {datetime.now()}")
        
        if self.signal_count > 0:
            print("🎉 SUCCESS: Real-time signals are being generated!")
        elif self.feature_count > 0:
            print("⚡ PROGRESS: Feature computation active, signals should follow")
        else:
            print("⏳ WARMUP: System still in warmup phase, needs more time")
            
    def cleanup(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()

if __name__ == "__main__":
    capture = SignalCapture()
    
    def signal_handler(sig, frame):
        print("\n🛑 Stopping capture...")
        capture.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    capture.run_extended_capture(15)  # Run for 15 minutes
