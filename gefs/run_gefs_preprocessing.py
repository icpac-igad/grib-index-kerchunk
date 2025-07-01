#!/usr/bin/env python3
"""
Run GEFS preprocessing for ensemble members 6-30
"""
import subprocess
import sys
import time

# Configuration
DATE = "20241112"
RUN = "00"
BUCKET = "gik-gefs-aws-tf"

# Generate member list from gep06 to gep30
MEMBERS = [f"gep{i:02d}" for i in range(6, 31)]

print(f"Will process {len(MEMBERS)} ensemble members: {', '.join(MEMBERS[:5])}...{MEMBERS[-1]}")

# Use the fixed preprocessing script from earlier run
start_total = time.time()

# Process each member
for i, member in enumerate(MEMBERS, 1):
    print(f"\n{'='*80}")
    print(f"Processing {member} ({i}/{len(MEMBERS)}) - Member {int(member[3:])} of 30")
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
                
            # Show progress
            total_elapsed = time.time() - start_total
            avg_time = total_elapsed / i
            remaining = (len(MEMBERS) - i) * avg_time
            print(f"\nProgress: {i}/{len(MEMBERS)} completed")
            print(f"Average time per member: {avg_time:.1f} seconds")
            print(f"Estimated time remaining: {remaining/60:.1f} minutes")
            
        else:
            print(f"❌ {member} failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr[:1000])  # Show first 1000 chars of error
            
    except Exception as e:
        print(f"❌ Exception processing {member}: {e}")
    
    # Add a small delay between members
    if i < len(MEMBERS):
        print(f"\nWaiting 5 seconds before next member...")
        time.sleep(5)

total_time = time.time() - start_total
print(f"\n{'='*80}")
print(f"All {len(MEMBERS)} members processed!")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Average time per member: {total_time/len(MEMBERS):.1f} seconds")
print(f"{'='*80}")