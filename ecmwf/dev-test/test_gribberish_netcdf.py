#!/usr/bin/env python3
"""
Test parquet → NetCDF using cfgrib.
Gribberish has compatibility issues with some ECMWF CCSDS-compressed chunks.
"""

import pandas as pd
import numpy as np
import json
import time
import tempfile
import os
from pathlib import Path
import fsspec
import xarray as xr

# Config
PARQUET_FILE = Path("ecmwf_three_stage_20251126_00z/stage3_ens_01_final.parquet")
OUTPUT_NC = Path("test_gribberish_tp.nc")
GRID_SHAPE = (721, 1440)  # ECMWF 0.25° grid

# East Africa bounds
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53


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


def find_tp_chunks(zstore):
    """Find TP variable chunks sorted by step."""
    import re
    tp_chunks = []
    for key in zstore.keys():
        if '/tp/' in key and key.endswith('/0.0.0'):
            match = re.match(r'step_(\d+)/', key)
            if match:
                step = int(match.group(1))
                tp_chunks.append((step, key))
    return sorted(tp_chunks, key=lambda x: x[0])


def decode_with_cfgrib(grib_bytes):
    """Decode using cfgrib."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name

    try:
        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        array_2d = ds[var_name].values
        ds.close()
    finally:
        os.unlink(tmp_path)

    return array_2d


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

    return grib_bytes


def main():
    print("=" * 60)
    print("ECMWF PARQUET → NetCDF (cfgrib decoder)")
    print("=" * 60)

    start_total = time.time()

    # Read parquet
    print(f"\n1. Reading parquet: {PARQUET_FILE}")
    zstore = read_parquet_refs(PARQUET_FILE)
    print(f"   References: {len(zstore)}")

    # Find TP chunks
    print("\n2. Finding TP chunks...")
    tp_chunks = find_tp_chunks(zstore)
    print(f"   Found {len(tp_chunks)} timesteps")

    # Create S3 filesystem (reuse for all fetches)
    fs = fsspec.filesystem('s3', anon=True)

    # Process all timesteps
    print(f"\n3. Fetching and decoding {len(tp_chunks)} timesteps...")
    decode_start = time.time()

    timestep_data = []
    steps = []

    for i, (step, chunk_key) in enumerate(tp_chunks):
        t0 = time.time()

        grib_bytes = fetch_grib_bytes(zstore, chunk_key, fs)
        array_2d = decode_with_cfgrib(grib_bytes)

        elapsed = (time.time() - t0) * 1000

        timestep_data.append(array_2d)
        steps.append(step)

        if i < 3 or i >= len(tp_chunks) - 2:
            print(f"   Step {step:3d}h: {elapsed:7.1f}ms, max={np.nanmax(array_2d):.4f}")
        elif i == 3:
            print(f"   ... processing remaining {len(tp_chunks)-3} steps ...")

    decode_elapsed = time.time() - decode_start
    print(f"\n   Total decode time: {decode_elapsed:.1f}s")
    print(f"   Average per chunk: {decode_elapsed/len(tp_chunks)*1000:.1f}ms")

    # Stack into 3D array
    print("\n4. Creating xarray Dataset...")
    data_3d = np.stack(timestep_data, axis=0)
    print(f"   Shape: {data_3d.shape}")

    # Create coordinates
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)

    # Create Dataset
    ds = xr.Dataset(
        {
            'tp': (['step', 'latitude', 'longitude'], data_3d.astype(np.float32))
        },
        coords={
            'step': steps,
            'latitude': lats,
            'longitude': lons
        }
    )

    ds['tp'].attrs = {
        'long_name': 'Total precipitation',
        'units': 'm',
    }
    ds.attrs['decoder'] = 'cfgrib'
    ds.attrs['source'] = str(PARQUET_FILE)

    # Subset to East Africa
    print("\n5. Subsetting to East Africa...")
    ds_ea = ds.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX)
    )
    print(f"   East Africa shape: {dict(ds_ea.dims)}")

    # Save to NetCDF
    print(f"\n6. Saving to {OUTPUT_NC}...")
    ds_ea.to_netcdf(OUTPUT_NC)
    file_size = OUTPUT_NC.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size:.1f} MB")

    total_elapsed = time.time() - start_total
    print(f"\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Timesteps: {len(tp_chunks)}")
    print(f"   Output: {OUTPUT_NC}")

    # Quick verification
    print(f"\n7. Verification:")
    ds_check = xr.open_dataset(OUTPUT_NC)
    print(f"   Dimensions: {dict(ds_check.dims)}")
    print(f"   TP min: {float(ds_check['tp'].min()):.6f}")
    print(f"   TP max: {float(ds_check['tp'].max()):.6f}")
    ds_check.close()


if __name__ == "__main__":
    main()
