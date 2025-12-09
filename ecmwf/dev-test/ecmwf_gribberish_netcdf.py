#!/usr/bin/env python3
"""
ECMWF Parquet → NetCDF using gribberish (Rust) with cfgrib fallback.

Performance comparison:
  - gribberish: ~25ms per chunk (99% of chunks)
  - cfgrib: ~2000ms per chunk (1% of chunks that fail gribberish)

For 85 timesteps:
  - Pure cfgrib: ~170s
  - Gribberish + fallback: ~3s (50x faster)
"""

import pandas as pd
import numpy as np
import json
import time
import tempfile
import os
import subprocess
import sys
from pathlib import Path
import fsspec
import xarray as xr

# Config
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


def decode_with_gribberish_subprocess(grib_bytes):
    """
    Decode GRIB using gribberish in a subprocess to catch Rust panics.
    Returns (array, success) tuple.
    """
    # Write GRIB to temp file, decode in subprocess
    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp_in:
        tmp_in.write(grib_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + '.npy'

    code = f'''
import gribberish
import numpy as np

with open("{tmp_in_path}", "rb") as f:
    grib_bytes = f.read()

flat_array = gribberish.parse_grib_array(grib_bytes, 0)
array_2d = flat_array.reshape((721, 1440)).astype(np.float32)
np.save("{tmp_out_path}", array_2d)
'''

    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            timeout=30
        )

        if result.returncode == 0 and os.path.exists(tmp_out_path):
            array = np.load(tmp_out_path)
            os.unlink(tmp_out_path)
            os.unlink(tmp_in_path)
            return array, True
    except Exception:
        pass

    # Cleanup
    if os.path.exists(tmp_in_path):
        os.unlink(tmp_in_path)
    if os.path.exists(tmp_out_path):
        os.unlink(tmp_out_path)

    return None, False


def decode_with_gribberish_direct(grib_bytes):
    """Try direct gribberish decode (faster but may panic)."""
    try:
        import gribberish
        flat_array = gribberish.parse_grib_array(grib_bytes, 0)
        return flat_array.reshape(GRID_SHAPE), True
    except Exception:
        return None, False


def decode_with_cfgrib(grib_bytes):
    """Fallback: decode using cfgrib (slow but reliable)."""
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


def process_parquet_to_netcdf(parquet_file, output_nc, variable='tp', use_subprocess=False):
    """
    Convert parquet to NetCDF using gribberish with cfgrib fallback.

    Args:
        parquet_file: Path to parquet file
        output_nc: Output NetCDF path
        variable: Variable to extract (default: 'tp')
        use_subprocess: Use subprocess isolation for gribberish (safer but slower)
    """
    print("=" * 60)
    print("ECMWF PARQUET → NetCDF (gribberish + cfgrib fallback)")
    print("=" * 60)

    start_total = time.time()

    # Read parquet
    print(f"\n1. Reading parquet: {parquet_file}")
    zstore = read_parquet_refs(parquet_file)
    print(f"   References: {len(zstore)}")

    # Find variable chunks
    print(f"\n2. Finding {variable} chunks...")
    chunks = find_variable_chunks(zstore, variable)
    print(f"   Found {len(chunks)} timesteps")

    # Create S3 filesystem (reuse for all fetches)
    fs = fsspec.filesystem('s3', anon=True)

    # Process all timesteps
    print(f"\n3. Fetching and decoding {len(chunks)} timesteps...")
    decode_start = time.time()

    timestep_data = []
    steps = []
    gribberish_count = 0
    cfgrib_count = 0
    gribberish_time = 0
    cfgrib_time = 0

    decode_func = decode_with_gribberish_subprocess if use_subprocess else decode_with_gribberish_direct

    for i, (step, chunk_key) in enumerate(chunks):
        t0 = time.time()

        grib_bytes = fetch_grib_bytes(zstore, chunk_key, fs)

        # Try gribberish first
        array_2d, success = decode_func(grib_bytes)

        if success:
            elapsed = (time.time() - t0) * 1000
            gribberish_count += 1
            gribberish_time += elapsed
            decoder = 'gribberish'
        else:
            # Fallback to cfgrib
            t1 = time.time()
            array_2d = decode_with_cfgrib(grib_bytes)
            elapsed = (time.time() - t0) * 1000
            cfgrib_count += 1
            cfgrib_time += (time.time() - t1) * 1000
            decoder = 'cfgrib'

        timestep_data.append(array_2d)
        steps.append(step)

        if i < 3 or i >= len(chunks) - 2:
            print(f"   Step {step:3d}h: {elapsed:7.1f}ms [{decoder:10s}] max={np.nanmax(array_2d):.4f}")
        elif i == 3:
            print(f"   ... processing remaining {len(chunks)-3} steps ...")

    decode_elapsed = time.time() - decode_start

    print(f"\n   Decode summary:")
    print(f"     gribberish: {gribberish_count} chunks, {gribberish_time:.0f}ms total, {gribberish_time/max(1,gribberish_count):.1f}ms avg")
    print(f"     cfgrib:     {cfgrib_count} chunks, {cfgrib_time:.0f}ms total, {cfgrib_time/max(1,cfgrib_count):.1f}ms avg")
    print(f"     Total: {decode_elapsed:.1f}s")

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
            variable: (['step', 'latitude', 'longitude'], data_3d.astype(np.float32))
        },
        coords={
            'step': steps,
            'latitude': lats,
            'longitude': lons
        }
    )

    ds[variable].attrs = {
        'long_name': f'{variable} (gribberish decoded)',
        'units': 'm' if variable == 'tp' else 'unknown',
    }
    ds.attrs['gribberish_chunks'] = gribberish_count
    ds.attrs['cfgrib_chunks'] = cfgrib_count
    ds.attrs['source'] = str(parquet_file)

    # Subset to East Africa
    print("\n5. Subsetting to East Africa...")
    ds_ea = ds.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX)
    )
    print(f"   East Africa shape: {dict(ds_ea.sizes)}")

    # Save to NetCDF
    print(f"\n6. Saving to {output_nc}...")
    ds_ea.to_netcdf(output_nc)
    file_size = Path(output_nc).stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size:.1f} MB")

    total_elapsed = time.time() - start_total
    print(f"\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   gribberish: {gribberish_count}/{len(chunks)} ({100*gribberish_count/len(chunks):.0f}%)")
    print(f"   cfgrib fallback: {cfgrib_count}/{len(chunks)} ({100*cfgrib_count/len(chunks):.0f}%)")
    print(f"   Output: {output_nc}")

    # Verification
    print(f"\n7. Verification:")
    ds_check = xr.open_dataset(output_nc)
    print(f"   Dimensions: {dict(ds_check.sizes)}")
    print(f"   {variable} min: {float(ds_check[variable].min()):.6f}")
    print(f"   {variable} max: {float(ds_check[variable].max()):.6f}")
    ds_check.close()

    return {
        'total_time': total_elapsed,
        'gribberish_count': gribberish_count,
        'cfgrib_count': cfgrib_count,
        'output_file': str(output_nc)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert ECMWF parquet to NetCDF using gribberish')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input parquet file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output NetCDF file')
    parser.add_argument('--variable', '-v', type=str, default='tp', help='Variable to extract')
    parser.add_argument('--subprocess', action='store_true', help='Use subprocess isolation (safer)')

    args = parser.parse_args()

    process_parquet_to_netcdf(
        parquet_file=Path(args.input),
        output_nc=Path(args.output),
        variable=args.variable,
        use_subprocess=args.subprocess
    )
