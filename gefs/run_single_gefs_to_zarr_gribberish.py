#!/usr/bin/env python3
"""
GEFS Single Ensemble Member Processing Script using Gribberish

This script processes a single GEFS ensemble member from parquet to zarr format
using gribberish (Rust-based GRIB decoder) for ~80x faster decoding compared to cfgrib.

Based on the gribberish vs cfgrib analysis:
- Gribberish: ~25ms per chunk (direct byte buffer decoding)
- cfgrib: ~2000ms per chunk (temp file + eccodes)

Usage:
    python run_single_gefs_to_zarr_gribberish.py <date> <run> <member> [options]

Example:
    python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep01 --region east_africa --variables t2m,tp,u10,v10

Features:
    - Uses gribberish for fast GRIB decoding (~80x faster than cfgrib)
    - Falls back to cfgrib for chunks that fail gribberish decoding
    - East Africa regional subsetting
    - Variable filtering and validation
    - Enhanced compression and chunking
"""

import os
import sys
import argparse
import json
import warnings
import time
import tempfile
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import re

import numpy as np
import pandas as pd
import xarray as xr
import fsspec

# Try to import gribberish
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available, will use cfgrib only")

# Suppress warnings
warnings.filterwarnings('ignore')

# GEFS grid specification (0.25 degree global)
GEFS_GRID_SHAPE = (721, 1440)  # lat x lon
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# Regional definitions
REGIONS = {
    'global': {
        'lat_min': -90.0, 'lat_max': 90.0,
        'lon_min': 0.0, 'lon_max': 359.75,
        'description': 'Global coverage'
    },
    'east_africa': {
        'lat_min': -12.0, 'lat_max': 23.0,
        'lon_min': 21.0, 'lon_max': 53.0,
        'description': 'East Africa region (Kenya, Tanzania, Uganda, Ethiopia, etc.)'
    }
}

# Variable metadata for CF-compliant output
VARIABLE_METADATA = {
    'sp': {'long_name': 'Surface pressure', 'units': 'Pa', 'standard_name': 'surface_air_pressure'},
    't2m': {'long_name': '2 metre temperature', 'units': 'K', 'standard_name': 'air_temperature'},
    'u10': {'long_name': '10 metre U wind component', 'units': 'm s**-1', 'standard_name': 'eastward_wind'},
    'v10': {'long_name': '10 metre V wind component', 'units': 'm s**-1', 'standard_name': 'northward_wind'},
    'tp': {'long_name': 'Total precipitation', 'units': 'm', 'standard_name': 'precipitation_amount'},
    'cape': {'long_name': 'Convective available potential energy', 'units': 'J kg**-1'},
    'pwat': {'long_name': 'Precipitable water', 'units': 'kg m**-2'},
    'mslet': {'long_name': 'Mean sea level pressure (Eta model reduction)', 'units': 'Pa'},
}


def read_parquet_refs(parquet_path):
    """Read parquet file and extract zstore references."""
    df = pd.read_parquet(parquet_path)
    print(f"  Parquet file loaded: {len(df)} rows")

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

    print(f"  Loaded {len(zstore)} entries")
    return zstore


def discover_variables(zstore):
    """Discover available variables and their chunks from zstore."""
    variables = {}

    for key in zstore.keys():
        # Match pattern like: sp/instant/surface/0.0.0 or t2m/instant/heightAboveGround/0.0.0
        if key.endswith('/0.0.0'):
            parts = key.split('/')
            if len(parts) >= 3:
                var_name = parts[0]
                if var_name not in variables:
                    variables[var_name] = {
                        'chunks': [],
                        'path_prefix': '/'.join(parts[:-1])
                    }

    # Find all chunks for each variable
    for var_name, var_info in variables.items():
        prefix = var_info['path_prefix']
        chunk_pattern = re.compile(rf'^{re.escape(prefix)}/(\d+)\.0\.0$')

        for key in zstore.keys():
            match = chunk_pattern.match(key)
            if match:
                step_idx = int(match.group(1))
                var_info['chunks'].append((step_idx, key))

        var_info['chunks'].sort(key=lambda x: x[0])

    return variables


def fetch_grib_bytes(zstore, chunk_key, fs):
    """Fetch GRIB bytes from S3 using the reference."""
    ref = zstore[chunk_key]

    if isinstance(ref, list) and len(ref) >= 3:
        url, offset, length = ref[0], ref[1], ref[2]
    else:
        raise ValueError(f"Invalid reference format for {chunk_key}: {ref}")

    # The URL from GEFS parquet is already complete (e.g., s3://noaa-gefs-pds/gefs.../gep01.t00z.pgrb2s.0p25.f000)
    # No need to add .grib2 extension for GEFS data
    s3_path = url

    with fs.open(s3_path, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_with_gribberish(grib_bytes, grid_shape=GEFS_GRID_SHAPE):
    """Decode GRIB bytes using gribberish (fast path)."""
    if not GRIBBERISH_AVAILABLE:
        raise RuntimeError("gribberish not available")

    flat_array = gribberish.parse_grib_array(grib_bytes, 0)
    array_2d = flat_array.reshape(grid_shape)
    return array_2d


def decode_with_cfgrib(grib_bytes):
    """Decode GRIB bytes using cfgrib (fallback path)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name

    try:
        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        array_2d = ds[var_name].values.copy()
        ds.close()
    finally:
        os.unlink(tmp_path)

    return array_2d


def decode_grib_hybrid(grib_bytes, grid_shape=GEFS_GRID_SHAPE):
    """
    Decode GRIB with gribberish, fallback to cfgrib on failure.

    Returns:
        tuple: (array_2d, decoder_used)
    """
    if GRIBBERISH_AVAILABLE:
        try:
            array_2d = decode_with_gribberish(grib_bytes, grid_shape)
            return array_2d, 'gribberish'
        except Exception as e:
            # Gribberish failed, fall back to cfgrib
            pass

    # Fallback to cfgrib
    array_2d = decode_with_cfgrib(grib_bytes)
    return array_2d, 'cfgrib'


def get_step_hours(zstore, var_info):
    """Extract step hours from zstore metadata."""
    # Try to read step coordinate metadata
    prefix = var_info['path_prefix']
    step_zarray_key = f"{prefix}/step/.zarray"
    step_zattrs_key = f"{prefix}/step/.zattrs"

    # Default: use chunk indices as step hours (for GEFS this is typically 3-hourly)
    chunk_indices = [idx for idx, _ in var_info['chunks']]

    # GEFS typical forecast hours: 0, 3, 6, 9, ... up to 240 or 384
    # The parquet indices should map to forecast hours
    return chunk_indices


def process_variable(zstore, var_name, var_info, fs, grid_shape=GEFS_GRID_SHAPE):
    """Process a single variable, fetching and decoding all timesteps."""
    print(f"\n  Processing variable: {var_name}")
    print(f"    Chunks: {len(var_info['chunks'])}")

    timestep_data = []
    steps = []
    decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}
    total_decode_time = 0

    for i, (step_idx, chunk_key) in enumerate(var_info['chunks']):
        t0 = time.time()

        try:
            # Fetch GRIB bytes
            grib_bytes, byte_length = fetch_grib_bytes(zstore, chunk_key, fs)

            # Decode using hybrid approach
            array_2d, decoder = decode_grib_hybrid(grib_bytes, grid_shape)

            elapsed = (time.time() - t0) * 1000
            total_decode_time += elapsed
            decode_stats[decoder] += 1

            timestep_data.append(array_2d)
            steps.append(step_idx)

            # Progress logging
            if i < 3 or i >= len(var_info['chunks']) - 2:
                print(f"      Step {step_idx:3d}: {elapsed:7.1f}ms [{decoder}] "
                      f"max={np.nanmax(array_2d):.4f}")
            elif i == 3:
                print(f"      ... processing {len(var_info['chunks']) - 5} more steps ...")

        except Exception as e:
            print(f"      Step {step_idx:3d}: FAILED - {type(e).__name__}: {str(e)[:50]}")
            decode_stats['failed'] += 1
            # Fill with NaN for failed chunks
            timestep_data.append(np.full(grid_shape, np.nan, dtype=np.float32))
            steps.append(step_idx)

    # Stack into 3D array (time, lat, lon)
    data_3d = np.stack(timestep_data, axis=0).astype(np.float32)

    print(f"    Total decode time: {total_decode_time/1000:.1f}s")
    print(f"    Average per chunk: {total_decode_time/len(var_info['chunks']):.1f}ms")
    print(f"    Decoder stats: gribberish={decode_stats['gribberish']}, "
          f"cfgrib={decode_stats['cfgrib']}, failed={decode_stats['failed']}")

    return data_3d, steps, decode_stats


def subset_to_region(data_3d, steps, region='global'):
    """Subset data to specified region."""
    if region == 'global':
        return data_3d, GEFS_LATS, GEFS_LONS

    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}")

    region_info = REGIONS[region]
    lat_min, lat_max = region_info['lat_min'], region_info['lat_max']
    lon_min, lon_max = region_info['lon_min'], region_info['lon_max']

    # Find indices for subsetting
    # Note: GEFS latitudes go from 90 to -90 (decreasing)
    lat_mask = (GEFS_LATS >= lat_min) & (GEFS_LATS <= lat_max)
    lon_mask = (GEFS_LONS >= lon_min) & (GEFS_LONS <= lon_max)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    # Subset
    data_subset = data_3d[:, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    lats_subset = GEFS_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons_subset = GEFS_LONS[lon_indices[0]:lon_indices[-1]+1]

    orig_size = data_3d.shape[1] * data_3d.shape[2]
    new_size = data_subset.shape[1] * data_subset.shape[2]
    reduction = (1 - new_size / orig_size) * 100

    print(f"    Region subset: {data_3d.shape} -> {data_subset.shape} ({reduction:.1f}% reduction)")

    return data_subset, lats_subset, lons_subset


def create_zarr_dataset(variables_data, steps, lats, lons, member_name, region):
    """Create xarray Dataset from processed variables."""
    data_vars = {}

    for var_name, data_3d in variables_data.items():
        # Get metadata
        meta = VARIABLE_METADATA.get(var_name, {'long_name': var_name, 'units': 'unknown'})

        data_vars[var_name] = xr.DataArray(
            data=data_3d,
            dims=['step', 'latitude', 'longitude'],
            attrs=meta
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'step': steps,
            'latitude': lats,
            'longitude': lons
        }
    )

    # Add global attributes
    ds.attrs.update({
        'title': 'GEFS Ensemble Forecast Data',
        'institution': 'NOAA/NCEP',
        'source': 'GEFS (Global Ensemble Forecast System)',
        'member': member_name,
        'region': region,
        'decoder': 'gribberish (hybrid with cfgrib fallback)',
        'processing_version': '1.0-gribberish',
        'created_by': 'run_single_gefs_to_zarr_gribberish.py'
    })

    # Coordinate attributes
    ds['latitude'].attrs = {
        'units': 'degrees_north',
        'standard_name': 'latitude',
        'long_name': 'latitude'
    }
    ds['longitude'].attrs = {
        'units': 'degrees_east',
        'standard_name': 'longitude',
        'long_name': 'longitude'
    }
    ds['step'].attrs = {
        'units': 'hours',
        'long_name': 'forecast step',
        'standard_name': 'forecast_period'
    }

    return ds


def save_zarr(ds, output_path):
    """Save dataset to zarr with compression."""
    # Use zarr_format=2 for compatibility with numcodecs compressors
    ds.to_zarr(output_path, mode='w', consolidated=True, zarr_format=2)

    # Get file size
    import subprocess
    result = subprocess.run(['du', '-sh', output_path], capture_output=True, text=True)
    if result.returncode == 0:
        size = result.stdout.split()[0]
        print(f"    Saved: {output_path} ({size})")
    else:
        print(f"    Saved: {output_path}")


def main(date_str: str, run_str: str, member: str, region: str = 'global',
         variables: str = None, parquet_dir: str = None):
    """
    Process a single GEFS ensemble member from parquet to zarr using gribberish.
    """
    print("=" * 70)
    print("GEFS Processing with Gribberish (Fast GRIB Decoder)")
    print("=" * 70)
    print(f"  Member: {member}")
    print(f"  Date: {date_str}, Run: {run_str}")
    print(f"  Region: {region}")
    print(f"  Gribberish available: {GRIBBERISH_AVAILABLE}")

    start_time = time.time()

    # Set up parquet directory
    if parquet_dir is None:
        parquet_dir = f"{date_str}_{run_str}"

    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"  Error: Parquet directory {parquet_path} does not exist.")
        return False

    # Find parquet file
    parquet_file = parquet_path / f"{member}.par"
    if not parquet_file.exists():
        print(f"  Error: Parquet file {parquet_file} not found.")
        return False

    print(f"\n1. Reading parquet: {parquet_file}")
    zstore = read_parquet_refs(str(parquet_file))

    print(f"\n2. Discovering variables...")
    all_variables = discover_variables(zstore)
    print(f"  Found {len(all_variables)} variables: {list(all_variables.keys())}")

    # Filter variables if requested
    if variables:
        requested = [v.strip() for v in variables.split(',')]
        filtered_vars = {k: v for k, v in all_variables.items() if k in requested}
        if not filtered_vars:
            print(f"  Warning: None of requested variables {requested} found")
            print(f"  Available: {list(all_variables.keys())}")
            return False
        all_variables = filtered_vars
        print(f"  Filtered to: {list(all_variables.keys())}")

    # Create S3 filesystem
    fs = fsspec.filesystem('s3', anon=True)

    print(f"\n3. Processing {len(all_variables)} variables...")
    variables_data = {}
    all_steps = None
    total_decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}

    for var_name, var_info in all_variables.items():
        data_3d, steps, decode_stats = process_variable(zstore, var_name, var_info, fs)

        # Subset to region
        data_subset, lats, lons = subset_to_region(data_3d, steps, region)

        variables_data[var_name] = data_subset
        if all_steps is None:
            all_steps = steps

        # Accumulate stats
        for key in total_decode_stats:
            total_decode_stats[key] += decode_stats[key]

        # Free memory
        del data_3d
        gc.collect()

    print(f"\n4. Creating xarray Dataset...")
    ds = create_zarr_dataset(variables_data, all_steps, lats, lons, member, region)
    print(f"  Dataset: {dict(ds.sizes)}")
    print(f"  Variables: {list(ds.data_vars)}")

    print(f"\n5. Saving to zarr...")
    output_dir = Path(f"./zarr_stores/{date_str}_{run_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    region_suffix = f"_{region}" if region != 'global' else '_global'
    output_path = output_dir / f"{member}_gribberish{region_suffix}.zarr"
    save_zarr(ds, str(output_path))

    # Summary
    elapsed = time.time() - start_time
    total_chunks = sum(total_decode_stats.values())

    print(f"\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Chunks processed: {total_chunks}")
    print(f"  Decoder usage: gribberish={total_decode_stats['gribberish']} "
          f"({100*total_decode_stats['gribberish']/max(1,total_chunks):.1f}%), "
          f"cfgrib={total_decode_stats['cfgrib']}, failed={total_decode_stats['failed']}")
    print(f"  Output: {output_path}")

    # Quick verification
    print(f"\n6. Verification...")
    try:
        ds_check = xr.open_dataset(str(output_path), engine='zarr')
        print(f"  Dimensions: {dict(ds_check.sizes)}")
        for var in list(ds_check.data_vars)[:3]:
            print(f"  {var}: min={float(ds_check[var].min()):.4f}, max={float(ds_check[var].max()):.4f}")
        ds_check.close()
        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process GEFS ensemble member to zarr using gribberish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process gep01 with global coverage
  python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep01

  # Process with East Africa subsetting
  python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep01 --region east_africa

  # Process specific variables only
  python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep01 --variables t2m,tp,u10,v10

  # Combine regional subsetting with variable filtering
  python run_single_gefs_to_zarr_gribberish.py 20250918 00 gep01 --region east_africa --variables t2m,tp
        """
    )

    parser.add_argument('date', help='Date in YYYYMMDD format (e.g., 20250918)')
    parser.add_argument('run', help='Run hour in HH format (e.g., 00)')
    parser.add_argument('member', help='Ensemble member (e.g., gep01)')
    parser.add_argument('--region', choices=list(REGIONS.keys()), default='east_africa',
                       help='Region to subset (default: east_africa)')
    parser.add_argument('--variables', type=str,
                       help='Comma-separated list of variables to process (default: all available)')
    parser.add_argument('--parquet_dir', type=str,
                       help='Directory containing GEFS parquet files (defaults to {date}_{run})')

    args = parser.parse_args()
    success = main(args.date, args.run, args.member, args.region, args.variables, args.parquet_dir)

    if not success:
        sys.exit(1)
