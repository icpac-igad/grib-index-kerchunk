#!/usr/bin/env python3
"""
Test gribberish with subprocess isolation to avoid panic crashes.
Tests each step in a separate subprocess.
"""

import pandas as pd
import numpy as np
import json
import time
import subprocess
import sys
from pathlib import Path
import fsspec

PARQUET_FILE = Path("ecmwf_three_stage_20251126_00z/stage3_ens_01_final.parquet")


def read_parquet_refs(parquet_path):
    df = pd.read_parquet(parquet_path)
    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            try:
                value = json.loads(value)
            except:
                pass
        zstore[key] = value
    return zstore


def find_tp_chunks(zstore):
    import re
    chunks = []
    for key in zstore.keys():
        if '/tp/' in key and key.endswith('/0.0.0'):
            match = re.match(r'step_(\d+)/', key)
            if match:
                step = int(match.group(1))
                chunks.append((step, key))
    return sorted(chunks, key=lambda x: x[0])


def test_single_step(step, chunk_key, url, offset, length):
    """Test a single step in subprocess."""
    code = f'''
import gribberish
import fsspec
import numpy as np

url = "{url}"
offset = {offset}
length = {length}

if not url.endswith('.grib2'):
    url = url + '.grib2'
s3_path = url if url.startswith('s3://') else f's3://{{url}}'

fs = fsspec.filesystem('s3', anon=True)
with fs.open(s3_path, 'rb') as f:
    f.seek(offset)
    grib_bytes = f.read(length)

import time
t0 = time.time()
flat_array = gribberish.parse_grib_array(grib_bytes, 0)
elapsed = (time.time() - t0) * 1000
array_2d = flat_array.reshape((721, 1440))
print(f"OK {{elapsed:.1f}}ms max={{np.nanmax(array_2d):.4f}}")
'''
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.returncode == 0, result.stdout.strip(), result.stderr


def main():
    print("=" * 70)
    print("GRIBBERISH ISOLATED TEST - ALL 85 TP TIMESTEPS")
    print("=" * 70)

    zstore = read_parquet_refs(PARQUET_FILE)
    tp_chunks = find_tp_chunks(zstore)
    print(f"Found {len(tp_chunks)} TP timesteps\n")

    success_count = 0
    fail_count = 0
    fail_steps = []

    for i, (step, chunk_key) in enumerate(tp_chunks):
        ref = zstore[chunk_key]
        url, offset, length = ref[0], ref[1], ref[2]

        success, stdout, stderr = test_single_step(step, chunk_key, url, offset, length)

        if success:
            print(f"  [{i+1:2d}/85] Step {step:3d}h: {stdout}")
            success_count += 1
        else:
            print(f"  [{i+1:2d}/85] Step {step:3d}h: FAIL (CCSDS panic)")
            fail_count += 1
            fail_steps.append(step)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  SUCCESS: {success_count}/85 ({100*success_count/85:.0f}%)")
    print(f"  FAILED:  {fail_count}/85 ({100*fail_count/85:.0f}%)")
    if fail_steps:
        print(f"  Failed steps: {fail_steps}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
