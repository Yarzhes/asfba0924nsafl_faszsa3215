#!/usr/bin/env python3
"""
Simple debug script to check why signals aren't being generated
"""
import os
import sys
# Add current directory to path to find ultra_signals module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from loguru import logger

def simple_debug():
    """Simple debug to check the scoring values"""
    print("=== Simple Signal Debug ===")
    
    # Let's check if we can find recent log output that shows component scores
    print("Looking for recent terminal output or log files...")
    
    # Check if there are any recent log files with scoring info
    import glob
    import re
    
    log_files = glob.glob("*.log")
    print(f"Found {len(log_files)} log files")
    
    # Look for the most recent backtest log
    recent_logs = sorted([f for f in log_files if f.startswith("backtest.")], reverse=True)[:3]
    
    for log_file in recent_logs:
        print(f"\n--- Checking {log_file} ---")
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Look for component scores
                score_matches = re.findall(r'Component scores.*?{[^}]+}', content)
                if score_matches:
                    print(f"Found {len(score_matches)} component score entries")
                    for i, match in enumerate(score_matches[:3]):  # Show first 3
                        print(f"  {i+1}: {match}")
                else:
                    print("No component scores found")
                
                # Look for signal generation
                signal_matches = re.findall(r'Signal.*generated.*', content, re.IGNORECASE)
                print(f"Found {len(signal_matches)} signal generation entries")
                
                # Look for risk filter blocks
                filter_matches = re.findall(r'blocked by risk filter.*', content, re.IGNORECASE)
                print(f"Found {len(filter_matches)} risk filter blocks")
                
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    # Also check the live state database for any signals
    print(f"\n--- Checking live_state.db ---")
    try:
        import sqlite3
        conn = sqlite3.connect('live_state.db')
        cursor = conn.cursor()
        
        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in live_state.db: {[t[0] for t in tables]}")
        
        # Check if there's a signals or trades table
        for table_name in ['signals', 'trades', 'signal_history']:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name[0] if isinstance(table_name, tuple) else table_name}")
                count = cursor.fetchone()[0]
                print(f"Records in {table_name}: {count}")
                
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table_name[0] if isinstance(table_name, tuple) else table_name} ORDER BY rowid DESC LIMIT 3")
                    recent = cursor.fetchall()
                    print(f"Recent records: {recent}")
            except Exception as e:
                # Table doesn't exist, skip
                pass
        
        conn.close()
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    simple_debug()
