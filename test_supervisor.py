#!/usr/bin/env python3
"""
Test script for the supervisor - simulates a program that crashes after a few seconds.
"""

import time
import sys
import random

def main():
    """Simulate a program that might crash."""
    print("Test program starting...")
    print(f"Arguments: {sys.argv}")
    
    # Simulate some work
    for i in range(5):
        print(f"Working... {i+1}/5")
        time.sleep(1)
    
    # Simulate different exit scenarios
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        
        if scenario == "crash":
            print("Simulating crash...")
            raise Exception("Test crash exception")
            
        elif scenario == "exit_error":
            print("Exiting with error code 1")
            sys.exit(1)
            
        elif scenario == "exit_success":
            print("Exiting successfully")
            sys.exit(0)
            
        elif scenario == "random":
            # Randomly crash or succeed
            if random.random() < 0.7:  # 70% chance of crash
                print("Random crash!")
                raise Exception("Random test crash")
            else:
                print("Random success!")
                sys.exit(0)
    
    # Default: exit successfully
    print("Test completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main()
