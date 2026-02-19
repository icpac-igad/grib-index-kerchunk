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
import re
from datetime import datetime, timedelta
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
    chunks = []
    for key in zstore.keys():
        if f'/{variable}/' in key and key.endswith('/0.0.0'):
            match = re.match(r'step_(\d+)/', key)
            if match:
                step = int(match.group(1))
                chunks.append((step, key))
    return sorted(chunks, key=lambda x: x[0])


def extract_model_run_time(zstore):
    """
    Extract model run datetime from S3 URL in parquet references.

    URL format: s3://ecmwf-forecasts/YYYYMMDD/HHz/ifs/0p25/enfo/...
    Returns: datetime object for model initialization time
    """
    # Find any chunk reference to extract the URL
    for key in zstore.keys():
        if key.endswith('/0.0.0'):
            ref = zstore[key]
            if isinstance(ref, list) and len(ref) >= 1:
                url = ref[0]
                # Parse date and hour from URL
                match = re.search(r'/(\d{8})/(\d{2})z/', url)
                if match:
                    date_str = match.group(1)  # YYYYMMDD
                    hour = int(match.group(2))  # HH
                    model_run = datetime(
                        year=int(date_str[:4]),
                        month=int(date_str[4:6]),
                        day=int(date_str[6:8]),
                        hour=hour
                    )
                    return model_run
    return None


def compute_valid_times(model_run_time, steps):
    """
    Compute valid times from model run time and forecast steps.

    Args:
        model_run_time: datetime of model initialization
        steps: list of forecast step hours

    Returns:
        list of datetime objects for valid times
    """
    if model_run_time is None:
        return None

    valid_times = [model_run_time + timedelta(hours=step) for step in steps]
    return valid_times


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

    # Extract model run time
    model_run_time = extract_model_run_time(zstore)
    if model_run_time:
        print(f"   Model run: {model_run_time.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        print("   WARNING: Could not extract model run time from URL")

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

    # Compute valid times
    valid_times = compute_valid_times(model_run_time, steps)
    if valid_times:
        print(f"   Valid time range: {valid_times[0].strftime('%Y-%m-%d %H:%M')} to {valid_times[-1].strftime('%Y-%m-%d %H:%M')}")

    # Create coordinates
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)

    # Build coordinates dict
    coords = {
        'latitude': lats,
        'longitude': lons
    }

    # Add time coordinate (primary dimension)
    if valid_times:
        # Use valid_time as primary time dimension
        time_values = np.array(valid_times, dtype='datetime64[ns]')
        coords['time'] = time_values
        time_dim = 'time'

        # Also add step as a coordinate variable (not dimension)
        step_timedelta = [np.timedelta64(s, 'h') for s in steps]
    else:
        # Fallback to step if no datetime available
        coords['step'] = steps
        time_dim = 'step'

    # Create Dataset
    ds = xr.Dataset(
        {
            variable: ([time_dim, 'latitude', 'longitude'], data_3d.astype(np.float32))
        },
        coords=coords
    )

    # Add step as auxiliary coordinate if using time dimension
    if valid_times:
        ds['step'] = (time_dim, steps)
        ds['step'].attrs = {
            'long_name': 'forecast step',
            'units': 'hours'
        }

    # Variable attributes
    var_attrs = {
        'long_name': 'Total precipitation' if variable == 'tp' else variable,
        'units': 'm' if variable == 'tp' else 'unknown',
    }
    if variable == 'tp':
        var_attrs['standard_name'] = 'precipitation_amount'
    ds[variable].attrs = var_attrs

    # Global attributes
    ds.attrs['Conventions'] = 'CF-1.8'
    ds.attrs['institution'] = 'ECMWF'
    ds.attrs['source'] = 'IFS Ensemble Forecast'
    ds.attrs['gribberish_chunks'] = gribberish_count
    ds.attrs['cfgrib_chunks'] = cfgrib_count
    ds.attrs['parquet_source'] = str(parquet_file)

    if model_run_time:
        ds.attrs['model_run_time'] = model_run_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        ds.attrs['forecast_reference_time'] = model_run_time.strftime('%Y-%m-%dT%H:%M:%SZ')

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
    ds_check = xr.open_dataset(output_nc, decode_timedelta=False)
    print(f"   Dimensions: {dict(ds_check.sizes)}")
    print(f"   Coordinates: {list(ds_check.coords)}")
    if 'time' in ds_check.dims:
        print(f"   Time range: {pd.Timestamp(ds_check.time.values[0])} to {pd.Timestamp(ds_check.time.values[-1])}")
    if 'step' in ds_check:
        print(f"   Step range: {int(ds_check.step.values[0])}h to {int(ds_check.step.values[-1])}h")
    print(f"   {variable} min: {float(ds_check[variable].min()):.6f}")
    print(f"   {variable} max: {float(ds_check[variable].max()):.6f}")
    if model_run_time:
        print(f"   Model run: {ds_check.attrs.get('model_run_time', 'N/A')}")
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
