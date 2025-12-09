#!/usr/bin/env python3
"""
Test gribberish with ALL 85 TP timesteps to find which ones panic.
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
GRID_SHAPE = (721, 1440)


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


def fetch_grib_bytes(zstore, chunk_key, fs):
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
    print("GRIBBERISH - TEST ALL 85 TP TIMESTEPS")
    print("=" * 70)

    zstore = read_parquet_refs(PARQUET_FILE)
    tp_chunks = find_tp_chunks(zstore)
    print(f"Found {len(tp_chunks)} TP timesteps")

    fs = fsspec.filesystem('s3', anon=True)

    print("\nTesting each timestep...")
    success = 0
    total_time = 0

    for i, (step, chunk_key) in enumerate(tp_chunks):
        try:
            grib_bytes, length = fetch_grib_bytes(zstore, chunk_key, fs)

            t0 = time.time()
            flat_array = gribberish.parse_grib_array(grib_bytes, 0)
            elapsed = (time.time() - t0) * 1000
            total_time += elapsed

            array_2d = flat_array.reshape(GRID_SHAPE)

            print(f"  [{i+1:2d}/85] Step {step:3d}h: OK  {elapsed:6.1f}ms  "
                  f"max={np.nanmax(array_2d):.4f}  bytes={length}")
            success += 1

        except Exception as e:
            print(f"  [{i+1:2d}/85] Step {step:3d}h: FAIL - {type(e).__name__}")
            # This will crash if it's a Rust panic

    print(f"\n{'='*70}")
    print(f"RESULT: {success}/85 timesteps successful")
    print(f"Total decode time: {total_time:.0f}ms")
    print(f"Average: {total_time/max(1,success):.1f}ms per chunk")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
