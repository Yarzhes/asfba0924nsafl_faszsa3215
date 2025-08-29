"""
30-minute canary test with maximum debug output to prove signal generation
"""
import subprocess
import time
import threading
import os
from datetime import datetime, timedelta

def run_canary_test():
    print("=" * 60)
    print("🚀 STARTING 30-MINUTE CANARY TEST")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"Expected end time: {datetime.now() + timedelta(minutes=30)}")
    print()
    print("SETTINGS SUMMARY:")
    print("- Entry threshold: 0.001 (ultra-low)")
    print("- Scoring weights: trend=1.0, momentum=1.0") 
    print("- Warmup periods: 5 (25 minutes)")
    print("- Sniper mode: disabled MTF, min_confidence=0.1")
    print("- All filters: disabled")
    print()
    print("WATCHING FOR:")
    print("✅ [DEBUG] warmup check messages")
    print("✅ [DEBUG] scoring messages") 
    print("✅ [DEBUG] signal generation messages")
    print("🎯 ACTUAL SIGNALS (LONG/SHORT)")
    print()
    print("=" * 60)
    
    # Start the realtime runner
    process = subprocess.Popen([
        'python', '-m', 'ultra_signals.apps.realtime_runner'
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, 
    cwd=r"C:\Users\Almir\Projects\Trading Helper")
    
    start_time = time.time()
    max_duration = 30 * 60  # 30 minutes
    signal_count = 0
    debug_count = 0
    kline_count = 0
    
    try:
        while True:
            # Check if we've exceeded 30 minutes
            if time.time() - start_time > max_duration:
                print(f"\n⏰ 30 minutes elapsed - stopping test")
                break
            
            # Read output with timeout
            try:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        print(f"\n❌ Process died unexpectedly")
                        break
                    time.sleep(0.1)
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                # Print with timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {line}")
                
                # Count important events
                if "Processing closed kline" in line:
                    kline_count += 1
                    
                if "[DEBUG]" in line:
                    debug_count += 1
                    
                if any(signal in line for signal in ["LONG", "SHORT"]) and "signal generated" in line:
                    signal_count += 1
                    print(f"🎉🎉🎉 SIGNAL #{signal_count} DETECTED! 🎉🎉🎉")
                    
            except Exception as e:
                print(f"Error reading output: {e}")
                break
                
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
    finally:
        print(f"\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"⏱️  Duration: {(time.time() - start_time)/60:.1f} minutes")
        print(f"📈 Klines processed: {kline_count}")
        print(f"🔍 Debug messages: {debug_count}")
        print(f"🎯 Signals generated: {signal_count}")
        print()
        
        if signal_count > 0:
            print("✅ SUCCESS! Signals were generated!")
            print("🚀 Your trading bot is working correctly!")
        elif debug_count > 0:
            print("⚠️  No signals, but debug output detected.")
            print("   This means scoring is happening but conditions aren't met.")
        elif kline_count > 0:
            print("⚠️  Klines processed but no scoring detected.")
            print("   This means warmup period hasn't been reached yet.")
        else:
            print("❌ No klines processed - connection or configuration issue.")
        
        print("=" * 60)
        
        # Terminate the process
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

if __name__ == "__main__":
    run_canary_test()
