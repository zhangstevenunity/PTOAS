#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ptoas_binary = os.path.join(script_dir, "ptoas")
    
    if not os.path.exists(ptoas_binary):
        sys.stderr.write(f"Error: ptoas binary not found at {ptoas_binary}\n")
        sys.exit(1)
    
    # Execute the binary with all arguments
    sys.exit(subprocess.call([ptoas_binary] + sys.argv[1:]))

if __name__ == "__main__":
    main()

