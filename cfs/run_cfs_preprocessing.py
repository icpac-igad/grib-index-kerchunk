#!/usr/bin/env python3
"""
Run CFS preprocessing for Climate Forecast System data
Adapted from GEFS preprocessing, focused on single runs without ensemble members
"""
import subprocess
import sys
import time

# Configuration
DATE = "20250801"
RUN = "00"
BUCKET = "gik-fmrc"

print(f"CFS preprocessing for date: {DATE}, run: {RUN}")
print(f"Target: s3://noaa-cfs-pds/cfs.{DATE}/{RUN}/6hrly_grib_01/")
print(f"Storage bucket: {BUCKET}")

# CFS doesn't use ensemble members like GEFS
# We process a single run for the specified date
start_total = time.time()

print(f"\n{'='*80}")
print(f"Processing CFS data for {DATE} {RUN}Z")
print(f"{'='*80}")

start_time = time.time()

cmd = [
    "python", "cfs_index_processing.py",
    "--date", DATE,
    "--run", RUN,
    "--bucket", BUCKET
]

print(f"Running command: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        elapsed = time.time() - start_time
        print(f"✅ CFS preprocessing completed successfully in {elapsed:.1f} seconds")
        
        # Show last few lines of output
        output_lines = result.stdout.strip().split('\n')
        print("Last output lines:")
        for line in output_lines[-10:]:
            print(f"  {line}")
            
    else:
        print(f"❌ CFS preprocessing failed with return code {result.returncode}")
        print("Error output:")
        print(result.stderr[:2000])  # Show first 2000 chars of error
        print("\nStdout output:")
        print(result.stdout[:1000])  # Show first 1000 chars of stdout
        
except Exception as e:
    print(f"❌ Exception during CFS preprocessing: {e}")

total_time = time.time() - start_total
print(f"\n{'='*80}")
print(f"CFS preprocessing completed!")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"{'='*80}")