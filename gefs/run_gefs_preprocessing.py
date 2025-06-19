#!/usr/bin/env python3
"""
Run GEFS preprocessing for the first 5 ensemble members
"""
import subprocess
import sys
import time

# Configuration
DATE = "20241112"
RUN = "00"
BUCKET = "gik-gefs-aws-tf"
MEMBERS = ["gep01", "gep02", "gep03", "gep04", "gep05"]

# First update the credentials path in the preprocessing script
print("Updating credentials path in preprocessing script...")
with open("gefs_index_preprocessing.py", "r") as f:
    content = f.read()

# Replace the hardcoded path
content = content.replace('"/home/runner/workspace/coiled-data.json"', '"/tmp/coiled-data.json"')

with open("gefs_index_preprocessing_fixed.py", "w") as f:
    f.write(content)

print("Credentials path updated.")

# Process each member
for i, member in enumerate(MEMBERS, 1):
    print(f"\n{'='*80}")
    print(f"Processing {member} ({i}/{len(MEMBERS)})")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    cmd = [
        "python", "gefs_index_preprocessing_fixed.py",
        "--date", DATE,
        "--run", RUN,
        "--member", member,
        "--bucket", BUCKET
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"✅ {member} completed successfully in {elapsed:.1f} seconds")
            
            # Show last few lines of output
            output_lines = result.stdout.strip().split('\n')
            print("Last output lines:")
            for line in output_lines[-5:]:
                print(f"  {line}")
        else:
            print(f"❌ {member} failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Exception processing {member}: {e}")
    
    # Add a small delay between members
    if i < len(MEMBERS):
        print(f"\nWaiting 5 seconds before next member...")
        time.sleep(5)

print(f"\n{'='*80}")
print("All members processed!")
print(f"{'='*80}")