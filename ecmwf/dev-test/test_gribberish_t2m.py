#!/usr/bin/env python3
"""
Test gribberish with ECMWF 2m temperature (t2m) variable.
Testing if the panic issue is specific to TP or affects all variables.
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import gribberish
import fsspec

# Config
PARQUET_FILE = Path("ecmwf_three_stage_20251126_00z/stage3_ens_01_final.parquet")
GRID_SHAPE = (721, 1440)  # ECMWF 0.25° grid


def read_parquet_refs(parquet_path):
    """Read parquet and extract zstore references."""
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


def find_variable_chunks(zstore, variable):
    """Find variable chunks sorted by step."""
    import re
    chunks = []
    for key in zstore.keys():
        if f'/{variable}/' in key and key.endswith('/0.0.0'):
            match = re.match(r'step_(\d+)/', key)
            if match:
                step = int(match.group(1))
                chunks.append((step, key))
    return sorted(chunks, key=lambda x: x[0])


def fetch_grib_bytes(zstore, chunk_key, fs):
    """Fetch GRIB bytes from S3."""
    ref = zstore[chunk_key]
    url, offset, length = ref[0], ref[1], ref[2]

    if not url.endswith('.grib2'):
        url = url + '.grib2'

    s3_path = url if url.startswith('s3://') else f's3://{url}'

    with fs.open(s3_path, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def main():
    print("=" * 70)
    print("GRIBBERISH ECMWF VARIABLE COMPATIBILITY TEST")
    print("=" * 70)

    # Read parquet
    print(f"\n1. Reading parquet: {PARQUET_FILE}")
    zstore = read_parquet_refs(PARQUET_FILE)
    print(f"   References: {len(zstore)}")

    # List available variables
    print("\n2. Finding available variables...")
    variables = set()
    for key in zstore.keys():
        if '/0.0.0' in key:
            parts = key.split('/')
            if len(parts) >= 2:
                variables.add(parts[1])

    print(f"   Variables found: {sorted(variables)}")

    # Create S3 filesystem
    fs = fsspec.filesystem('s3', anon=True)

    # Test each variable
    test_variables = ['t2m', '2t', 'tp', 'msl', 'u10', 'v10', 'sp']

    print("\n3. Testing gribberish on each variable (first 5 + last 5 timesteps)...")
    print("=" * 70)

    for var in test_variables:
        chunks = find_variable_chunks(zstore, var)
        if not chunks:
            print(f"\n   {var}: NOT FOUND in parquet")
            continue

        print(f"\n   {var}: Found {len(chunks)} timesteps")

        # Test first 5 and last 5 timesteps
        test_indices = list(range(min(5, len(chunks)))) + list(range(max(0, len(chunks)-5), len(chunks)))
        test_indices = sorted(set(test_indices))

        success_count = 0
        fail_count = 0
        total_time = 0

        for idx in test_indices:
            step, chunk_key = chunks[idx]

            try:
                grib_bytes, length = fetch_grib_bytes(zstore, chunk_key, fs)

                t0 = time.time()
                flat_array = gribberish.parse_grib_array(grib_bytes, 0)
                elapsed = (time.time() - t0) * 1000
                total_time += elapsed

                array_2d = flat_array.reshape(GRID_SHAPE)

                print(f"      Step {step:3d}h: OK  {elapsed:6.1f}ms  shape={array_2d.shape}  "
                      f"min={np.nanmin(array_2d):10.4f}  max={np.nanmax(array_2d):10.4f}  "
                      f"bytes={length}")
                success_count += 1

            except Exception as e:
                print(f"      Step {step:3d}h: FAIL - {type(e).__name__}: {str(e)[:60]}")
                fail_count += 1

        print(f"      Summary: {success_count} OK, {fail_count} FAIL")
        if success_count > 0:
            print(f"      Avg decode time: {total_time/success_count:.1f}ms")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
