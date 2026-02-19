#!/usr/bin/env python3
"""
ECMWF IFS Ensemble Member Processing Script using Gribberish

This script processes a single ECMWF IFS ensemble member from parquet to zarr format
using gribberish (Rust-based GRIB decoder) for fast decoding.

Based on the GEFS processing script but adapted for ECMWF parquet structure.

Key differences from GEFS:
- ECMWF uses lon range -180 to 180 (GEFS uses 0-360)
- Different parquet key structure: varname/type/level/varname/X.X.X.X
- First chunk (0.0.0.0) may be base64 encoded inline
- u700/v700 are at isobaric levels with 5D chunks
- cape has a legacy format structure

Usage:
    python run_single_ecmwf_to_zarr_gribberish.py <member> [options]

Example:
    python run_single_ecmwf_to_zarr_gribberish.py control --region east_africa
    python run_single_ecmwf_to_zarr_gribberish.py ens_01 --variables t2m,tp,u700,v700
    python run_single_ecmwf_to_zarr_gribberish.py ens_25 --parquet_dir ./test_ecmwf_three_stage_prebuilt_output

Required cGAN variables (14 total):
    cape, cp, mcc, sp, ssr, t2m, tciw, tclw, tcrw, tcw, tcwv, tp, u700, v700

Available alternatives in ECMWF:
    - cape -> mucape (Most Unstable CAPE)
    - mcc -> tcc (Total Cloud Cover)
    - cp -> not available (convective precip)
    - tciw, tclw, tcrw -> tcw (Total Column Water as proxy)
"""

import os
import sys
import argparse
import json
import warnings
import time
import tempfile
import gc
import base64
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

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

# ECMWF IFS grid specification (0.25 degree global)
ECMWF_GRID_SHAPE = (721, 1440)  # lat x lon
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# Regional definitions
REGIONS = {
    'global': {
        'lat_min': -90.0, 'lat_max': 90.0,
        'lon_min': -180.0, 'lon_max': 179.75,
        'description': 'Global coverage'
    },
    'east_africa': {
        'lat_min': -12.0, 'lat_max': 23.0,
        'lon_min': 21.0, 'lon_max': 53.0,
        'description': 'East Africa region (Kenya, Tanzania, Uganda, Ethiopia, etc.)'
    }
}

# Variable metadata for CF-compliant output (including cGAN required variables)
VARIABLE_METADATA = {
    'cape': {'long_name': 'Convective Available Potential Energy', 'units': 'J kg**-1', 'standard_name': 'atmosphere_convective_available_potential_energy'},
    'mucape': {'long_name': 'Most Unstable CAPE', 'units': 'J kg**-1'},
    'cp': {'long_name': 'Convective Precipitation', 'units': 'm', 'standard_name': 'convective_precipitation_amount'},
    'mcc': {'long_name': 'Medium Cloud Cover', 'units': '1'},
    'tcc': {'long_name': 'Total Cloud Cover', 'units': '1'},
    'sp': {'long_name': 'Surface Pressure', 'units': 'Pa', 'standard_name': 'surface_air_pressure'},
    'ssr': {'long_name': 'Surface Net Solar Radiation', 'units': 'J m**-2'},
    'ssrd': {'long_name': 'Surface Solar Radiation Downwards', 'units': 'J m**-2'},
    't2m': {'long_name': '2 metre Temperature', 'units': 'K', 'standard_name': 'air_temperature'},
    'tciw': {'long_name': 'Total Column Ice Water', 'units': 'kg m**-2'},
    'tclw': {'long_name': 'Total Column Liquid Water', 'units': 'kg m**-2'},
    'tcrw': {'long_name': 'Total Column Rain Water', 'units': 'kg m**-2'},
    'tcw': {'long_name': 'Total Column Water', 'units': 'kg m**-2'},
    'tcwv': {'long_name': 'Total Column Water Vapour', 'units': 'kg m**-2', 'standard_name': 'atmosphere_mass_content_of_water_vapor'},
    'tp': {'long_name': 'Total Precipitation', 'units': 'm', 'standard_name': 'precipitation_amount'},
    'u700': {'long_name': 'U-wind at 700 hPa', 'units': 'm s**-1', 'standard_name': 'eastward_wind'},
    'v700': {'long_name': 'V-wind at 700 hPa', 'units': 'm s**-1', 'standard_name': 'northward_wind'},
    'u10': {'long_name': '10 metre U wind component', 'units': 'm s**-1', 'standard_name': 'eastward_wind'},
    'v10': {'long_name': '10 metre V wind component', 'units': 'm s**-1', 'standard_name': 'northward_wind'},
    'msl': {'long_name': 'Mean Sea Level Pressure', 'units': 'Pa', 'standard_name': 'air_pressure_at_mean_sea_level'},
    'd2m': {'long_name': '2 metre Dewpoint Temperature', 'units': 'K'},
}

# ECMWF variable path mappings for step_XXX format
# Format: output_var -> (parquet_var, level_type)
# The step_XXX format is: step_XXX/varname/level/member/0.0.0
# where level is 'sfc' (surface), 'pl' (pressure level), or 'sol' (soil)
ECMWF_VARIABLE_PATHS = {
    # Surface/single level variables (level='sfc')
    't2m': ('2t', 'sfc'),      # 2m temperature
    'sp': ('sp', 'sfc'),       # Surface pressure
    'tp': ('tp', 'sfc'),       # Total precipitation
    'ssr': ('ssr', 'sfc'),     # Surface net solar radiation
    'ssrd': ('ssrd', 'sfc'),   # Surface solar radiation downwards
    'tcw': ('tcw', 'sfc'),     # Total column water
    'tcwv': ('tcwv', 'sfc'),   # Total column water vapour
    'tcc': ('tcc', 'sfc'),     # Total cloud cover
    'mucape': ('mucape', 'sfc'),  # Most unstable CAPE
    'u10': ('10u', 'sfc'),     # 10m U wind
    'v10': ('10v', 'sfc'),     # 10m V wind
    'msl': ('msl', 'sfc'),     # Mean sea level pressure
    'd2m': ('2d', 'sfc'),      # 2m dewpoint temperature
    # Pressure level variables (level='pl')
    'u700': ('u', 'pl'),       # U wind at pressure levels
    'v700': ('v', 'pl'),       # V wind at pressure levels
}

# cGAN required variables with available alternatives
CGAN_VARIABLES = {
    'cape': ['mucape'],  # Use mucape as alternative
    'cp': [],  # Not available
    'mcc': ['tcc'],  # Use tcc as alternative
    'sp': ['sp'],
    'ssr': ['ssr', 'ssrd'],
    't2m': ['t2m'],
    'tciw': ['tcw'],  # Use tcw as proxy
    'tclw': ['tcw'],  # Use tcw as proxy
    'tcrw': ['tcw'],  # Use tcw as proxy
    'tcw': ['tcw'],
    'tcwv': ['tcwv'],
    'tp': ['tp'],
    'u700': ['u700'],
    'v700': ['v700'],
}


def read_parquet_refs(parquet_path):
    """Read parquet file and extract zstore references."""
    df = pd.read_parquet(parquet_path)
    print(f"  Parquet file loaded: {len(df)} rows")

    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        # Keep bytes as-is for base64 encoded data
        if isinstance(value, bytes):
            # Check if it's a JSON string
            try:
                decoded = value.decode('utf-8')
                if decoded.startswith('[') or decoded.startswith('{'):
                    value = json.loads(decoded)
                elif decoded.startswith('base64:'):
                    value = value  # Keep as bytes
                else:
                    value = decoded
            except:
                value = value  # Keep as bytes

        elif isinstance(value, str):
            if value.startswith('[') or value.startswith('{'):
                try:
                    value = json.loads(value)
                except:
                    pass

        zstore[key] = value

    print(f"  Loaded {len(zstore)} entries")
    return zstore


def discover_ecmwf_variables(zstore):
    """Discover available ECMWF variables and their chunk paths from zstore.

    Handles the step_XXX format: step_XXX/varname/level/member/0.0.0
    where level is 'sfc' (surface), 'pl' (pressure level), or 'sol' (soil).

    Returns a dict: {varname/level: var_info}
    """
    variables = {}

    for key in zstore.keys():
        # Match step_XXX format: step_XXX/varname/level/member/0.0.0
        match = re.match(r'^step_(\d+)/([^/]+)/([^/]+)/([^/]+)/0\.0\.0$', key)
        if match:
            step_str, var_name, level, member = match.groups()
            step = int(step_str)
            full_key = f'{var_name}/{level}'

            if full_key not in variables:
                variables[full_key] = {
                    'var_name': var_name,
                    'chunks': [],
                    'level_type': level,
                    'member': member,
                    'is_isobaric': level == 'pl'
                }

            variables[full_key]['chunks'].append((step, key))

    # Sort chunks by step number
    for var_info in variables.values():
        var_info['chunks'].sort(key=lambda x: x[0])

    return variables


def get_chunk_key_for_step(var_info, step_idx, pressure_idx=None):
    """Generate chunk key for a given step (and pressure level for isobaric vars)."""
    prefix = var_info['path_prefix']

    if var_info['is_isobaric'] and pressure_idx is not None:
        # 5D: time.step.pressure.lat.lon -> 0.{step}.{pressure}.0.0
        return f"{prefix}/0.{step_idx}.{pressure_idx}.0.0"
    else:
        # 4D: time.step.lat.lon -> 0.{step}.0.0
        return f"{prefix}/0.{step_idx}.0.0"


def fetch_grib_bytes(zstore, chunk_key, fs):
    """Fetch GRIB bytes from S3 or decode from base64."""
    if chunk_key not in zstore:
        raise KeyError(f"Chunk key not found: {chunk_key}")

    ref = zstore[chunk_key]

    # Handle base64 encoded inline data
    if isinstance(ref, bytes):
        if ref.startswith(b'base64:'):
            grib_bytes = base64.b64decode(ref[7:])
            return grib_bytes, len(grib_bytes)
        else:
            # Raw bytes
            return ref, len(ref)

    # Handle string base64
    if isinstance(ref, str) and ref.startswith('base64:'):
        grib_bytes = base64.b64decode(ref[7:])
        return grib_bytes, len(grib_bytes)

    # Handle S3 reference [url, offset, length]
    if isinstance(ref, list) and len(ref) >= 3:
        url, offset, length = ref[0], ref[1], ref[2]

        # Add .grib2 extension if missing (ECMWF S3 files require it)
        if not url.endswith('.grib2'):
            url = url + '.grib2'

        with fs.open(url, 'rb') as f:
            f.seek(offset)
            grib_bytes = f.read(length)

        return grib_bytes, length

    raise ValueError(f"Unknown reference format for {chunk_key}: {type(ref)}")


def decode_with_gribberish_subprocess(grib_bytes, grid_shape=ECMWF_GRID_SHAPE):
    """
    Decode GRIB using gribberish in a subprocess to safely catch Rust panics.
    Returns (array, success) tuple.
    """
    import subprocess
    import sys

    # Write GRIB to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp_in:
        tmp_in.write(grib_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + '.npy'

    # Subprocess code to decode GRIB
    code = f'''
import gribberish
import numpy as np

with open("{tmp_in_path}", "rb") as f:
    grib_bytes = f.read()

flat_array = gribberish.parse_grib_array(grib_bytes, 0)
array_2d = flat_array.reshape({grid_shape}).astype(np.float32)
np.save("{tmp_out_path}", array_2d)
'''

    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            timeout=60
        )

        if result.returncode == 0 and os.path.exists(tmp_out_path):
            array = np.load(tmp_out_path)
            os.unlink(tmp_out_path)
            os.unlink(tmp_in_path)
            return array, True
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # Cleanup on failure
    if os.path.exists(tmp_in_path):
        os.unlink(tmp_in_path)
    if os.path.exists(tmp_out_path):
        os.unlink(tmp_out_path)

    return None, False


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


def decode_grib_hybrid(grib_bytes, grid_shape=ECMWF_GRID_SHAPE):
    """
    Decode GRIB with gribberish (subprocess-isolated), fallback to cfgrib on failure.

    Uses subprocess isolation to safely handle Rust panics in gribberish.
    When gribberish fails (panic or error), falls back to cfgrib.

    Returns:
        tuple: (array_2d, decoder_used)
    """
    # Try gribberish in subprocess (safe from panics)
    array_2d, success = decode_with_gribberish_subprocess(grib_bytes, grid_shape)
    if success:
        return array_2d, 'gribberish'

    # Fallback to cfgrib
    array_2d = decode_with_cfgrib(grib_bytes)
    return array_2d, 'cfgrib'


def process_surface_variable(zstore, var_name, var_info, fs, grid_shape=ECMWF_GRID_SHAPE):
    """Process a surface/single-level variable."""
    print(f"\n  Processing surface variable: {var_name}")
    print(f"    Source var: {var_info['var_name']}/{var_info['level_type']}")
    print(f"    Chunks: {len(var_info['chunks'])}")

    timestep_data = []
    steps = []
    decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}
    total_decode_time = 0

    for i, (step_idx, chunk_key) in enumerate(var_info['chunks']):
        t0 = time.time()

        try:
            grib_bytes, byte_length = fetch_grib_bytes(zstore, chunk_key, fs)
            array_2d, decoder = decode_grib_hybrid(grib_bytes, grid_shape)

            elapsed = (time.time() - t0) * 1000
            total_decode_time += elapsed
            decode_stats[decoder] += 1

            timestep_data.append(array_2d)
            steps.append(step_idx)

            if i < 3 or i >= len(var_info['chunks']) - 2:
                print(f"      Step {step_idx:3d}: {elapsed:7.1f}ms [{decoder}] "
                      f"shape={array_2d.shape} max={np.nanmax(array_2d):.4f}")
            elif i == 3:
                print(f"      ... processing {len(var_info['chunks']) - 5} more steps ...")

        except Exception as e:
            print(f"      Step {step_idx:3d}: FAILED - {type(e).__name__}: {str(e)[:80]}")
            decode_stats['failed'] += 1
            timestep_data.append(np.full(grid_shape, np.nan, dtype=np.float32))
            steps.append(step_idx)

    # Stack into 3D array (time, lat, lon)
    data_3d = np.stack(timestep_data, axis=0).astype(np.float32)

    print(f"    Total decode time: {total_decode_time/1000:.1f}s")
    print(f"    Average per chunk: {total_decode_time/max(1, len(var_info['chunks'])):.1f}ms")
    print(f"    Decoder stats: gribberish={decode_stats['gribberish']}, "
          f"cfgrib={decode_stats['cfgrib']}, failed={decode_stats['failed']}")

    return data_3d, steps, decode_stats


def process_isobaric_variable(zstore, var_name, var_info, pressure_idx, fs, grid_shape=ECMWF_GRID_SHAPE):
    """Process an isobaric (pressure level) variable for a specific pressure level."""
    print(f"\n  Processing isobaric variable: {var_name} at pressure index {pressure_idx}")
    print(f"    Path: {var_info['path_prefix']}")

    timestep_data = []
    steps = []
    decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}
    total_decode_time = 0

    # Find chunks for this pressure level
    # Chunk format: 0.{step}.{pressure}.0.0
    relevant_chunks = []
    for chunk_id, chunk_key in var_info['chunks']:
        parts = chunk_id.split('.')
        if len(parts) >= 3:
            chunk_pressure_idx = int(parts[2])
            if chunk_pressure_idx == pressure_idx:
                step_idx = int(parts[1])
                relevant_chunks.append((step_idx, chunk_key))

    relevant_chunks.sort(key=lambda x: x[0])
    print(f"    Found {len(relevant_chunks)} chunks for pressure level {pressure_idx}")

    for i, (step_idx, chunk_key) in enumerate(relevant_chunks):
        t0 = time.time()

        try:
            grib_bytes, byte_length = fetch_grib_bytes(zstore, chunk_key, fs)
            array_2d, decoder = decode_grib_hybrid(grib_bytes, grid_shape)

            elapsed = (time.time() - t0) * 1000
            total_decode_time += elapsed
            decode_stats[decoder] += 1

            timestep_data.append(array_2d)
            steps.append(step_idx)

            if i < 3 or i >= len(relevant_chunks) - 2:
                print(f"      Step {step_idx:3d}: {elapsed:7.1f}ms [{decoder}] "
                      f"shape={array_2d.shape} max={np.nanmax(array_2d):.4f}")
            elif i == 3:
                print(f"      ... processing {len(relevant_chunks) - 5} more steps ...")

        except Exception as e:
            print(f"      Step {step_idx:3d}: FAILED - {type(e).__name__}: {str(e)[:80]}")
            decode_stats['failed'] += 1
            timestep_data.append(np.full(grid_shape, np.nan, dtype=np.float32))
            steps.append(step_idx)

    if not timestep_data:
        return None, [], decode_stats

    # Stack into 3D array (time, lat, lon)
    data_3d = np.stack(timestep_data, axis=0).astype(np.float32)

    print(f"    Total decode time: {total_decode_time/1000:.1f}s")
    print(f"    Average per chunk: {total_decode_time/max(1, len(relevant_chunks)):.1f}ms")

    return data_3d, steps, decode_stats


def subset_to_region(data_3d, region='global'):
    """Subset data to specified region."""
    if region == 'global':
        return data_3d, ECMWF_LATS.copy(), ECMWF_LONS.copy()

    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}")

    region_info = REGIONS[region]
    lat_min, lat_max = region_info['lat_min'], region_info['lat_max']
    lon_min, lon_max = region_info['lon_min'], region_info['lon_max']

    # Find indices for subsetting
    # Note: ECMWF latitudes go from 90 to -90 (decreasing)
    lat_mask = (ECMWF_LATS >= lat_min) & (ECMWF_LATS <= lat_max)
    lon_mask = (ECMWF_LONS >= lon_min) & (ECMWF_LONS <= lon_max)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    # Subset
    data_subset = data_3d[:, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    lats_subset = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons_subset = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]

    orig_size = data_3d.shape[1] * data_3d.shape[2]
    new_size = data_subset.shape[1] * data_subset.shape[2]
    reduction = (1 - new_size / orig_size) * 100

    print(f"    Region subset: {data_3d.shape} -> {data_subset.shape} ({reduction:.1f}% reduction)")

    return data_subset, lats_subset, lons_subset


def create_zarr_dataset(variables_data, steps_dict, lats, lons, member_name, region):
    """Create xarray Dataset from processed variables."""
    # Find common steps across all variables
    all_step_sets = [set(steps_dict[var]) for var in variables_data.keys() if steps_dict.get(var)]

    if all_step_sets:
        common_steps = sorted(set.intersection(*all_step_sets))
        if not common_steps:
            # Use union if no common steps
            common_steps = sorted(set.union(*all_step_sets))
    else:
        common_steps = [0]

    print(f"    Creating dataset with {len(common_steps)} steps")

    data_vars = {}
    for var_name, data_3d in variables_data.items():
        if data_3d is None:
            continue

        var_steps = steps_dict.get(var_name, list(range(data_3d.shape[0])))
        meta = VARIABLE_METADATA.get(var_name, {'long_name': var_name, 'units': 'unknown'})

        # Align steps if needed
        if set(var_steps) == set(common_steps):
            aligned_data = data_3d
        else:
            aligned_data = np.full((len(common_steps), data_3d.shape[1], data_3d.shape[2]),
                                   np.nan, dtype=np.float32)
            step_to_idx = {s: i for i, s in enumerate(common_steps)}
            for orig_idx, step in enumerate(var_steps):
                if step in step_to_idx:
                    aligned_data[step_to_idx[step]] = data_3d[orig_idx]
            print(f"    Aligned {var_name}: {len(var_steps)} steps -> {len(common_steps)} steps")

        data_vars[var_name] = xr.DataArray(
            data=aligned_data,
            dims=['step', 'latitude', 'longitude'],
            attrs=meta
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'step': common_steps,
            'latitude': lats,
            'longitude': lons
        }
    )

    # Add global attributes
    ds.attrs.update({
        'title': 'ECMWF IFS Ensemble Forecast Data',
        'institution': 'ECMWF',
        'source': 'IFS (Integrated Forecasting System)',
        'member': member_name,
        'region': region,
        'decoder': 'gribberish (hybrid with cfgrib fallback)',
        'processing_version': '1.0-gribberish',
        'created_by': 'run_single_ecmwf_to_zarr_gribberish.py'
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
    ds.to_zarr(output_path, mode='w', consolidated=True, zarr_format=2)

    # Get file size
    import subprocess
    result = subprocess.run(['du', '-sh', output_path], capture_output=True, text=True)
    if result.returncode == 0:
        size = result.stdout.split()[0]
        print(f"    Saved: {output_path} ({size})")
    else:
        print(f"    Saved: {output_path}")


def main(member: str, region: str = 'global', variables: str = None,
         parquet_dir: str = None, output_dir: str = None):
    """
    Process a single ECMWF IFS ensemble member from parquet to zarr using gribberish.
    """
    print("=" * 70)
    print("ECMWF IFS Processing with Gribberish (Fast GRIB Decoder)")
    print("=" * 70)
    print(f"  Member: {member}")
    print(f"  Region: {region}")
    print(f"  Gribberish available: {GRIBBERISH_AVAILABLE}")

    start_time = time.time()

    # Set up parquet directory
    if parquet_dir is None:
        parquet_dir = "./test_ecmwf_three_stage_prebuilt_output"

    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"  Error: Parquet directory {parquet_path} does not exist.")
        return False

    # Find parquet file based on member
    if member == 'control':
        parquet_file = parquet_path / "stage3_control_final.parquet"
    else:
        # Handle ens_XX format
        if member.startswith('ens_'):
            ens_num = member.replace('ens_', '')
        else:
            ens_num = member.zfill(2)
        parquet_file = parquet_path / f"stage3_ens_{ens_num}_final.parquet"

    if not parquet_file.exists():
        print(f"  Error: Parquet file {parquet_file} not found.")
        # List available files
        available = list(parquet_path.glob("stage3_*.parquet"))
        print(f"  Available files: {[f.name for f in available[:10]]}")
        return False

    print(f"\n1. Reading parquet: {parquet_file}")
    zstore = read_parquet_refs(str(parquet_file))

    print(f"\n2. Discovering variables...")
    all_variables = discover_ecmwf_variables(zstore)
    # Show unique base variable names
    unique_vars = sorted(set(v['var_name'] for v in all_variables.values()))
    print(f"  Found {len(all_variables)} variable paths from {len(unique_vars)} unique variables")
    print(f"  Variables: {unique_vars}")

    # Determine which variables to process
    if variables:
        requested = [v.strip() for v in variables.split(',')]
    else:
        # Default: process cGAN-relevant variables that are available
        requested = ['t2m', 'tp', 'sp', 'ssr', 'tcw', 'tcwv', 'tcc', 'mucape', 'u700', 'v700']

    # Map requested variables to actual parquet variables using the step_XXX format
    # ECMWF_VARIABLE_PATHS maps: output_var -> (parquet_var, level_type)
    vars_to_process = []
    for req_var in requested:
        if req_var in ECMWF_VARIABLE_PATHS:
            parquet_var, level_type = ECMWF_VARIABLE_PATHS[req_var]
            full_key = f'{parquet_var}/{level_type}'
            if full_key in all_variables:
                vars_to_process.append((req_var, full_key))
                print(f"  Mapped {req_var} -> {full_key} ({len(all_variables[full_key]['chunks'])} chunks)")
            else:
                print(f"  Warning: Variable {req_var} ({full_key}) not found in parquet")
                # Show available paths for this variable
                matching = [k for k in all_variables.keys() if k.startswith(f'{parquet_var}/')]
                if matching:
                    print(f"    Available paths: {matching}")
        else:
            # Try to find a matching path directly
            matching = [k for k in all_variables.keys() if k.startswith(f'{req_var}/')]
            if matching:
                # Prefer 'sfc' (surface) if available
                sfc_match = [m for m in matching if m.endswith('/sfc')]
                chosen = sfc_match[0] if sfc_match else matching[0]
                vars_to_process.append((req_var, chosen))
                print(f"  Direct match {req_var} -> {chosen} ({len(all_variables[chosen]['chunks'])} chunks)")
            else:
                print(f"  Warning: Variable {req_var} not found")

    print(f"  Will process: {[v[0] for v in vars_to_process]}")

    # Create S3 filesystem
    fs = fsspec.filesystem('s3', anon=True)

    # Set up output path early for incremental saving
    if output_dir is None:
        output_dir = Path("./zarr_stores/ecmwf")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_suffix = f"_{region}" if region != 'global' else '_global'
    output_path = output_dir / f"{member}_gribberish{region_suffix}.zarr"

    print(f"\n3. Processing {len(vars_to_process)} variables (incremental save to {output_path})...")
    total_decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}
    lats, lons = None, None
    common_steps = None
    processed_vars = []

    for var_idx, (output_var, full_path_key) in enumerate(vars_to_process):
        var_info = all_variables[full_path_key]

        # In the step_XXX format, both surface and pressure level variables
        # are processed the same way (each chunk is a single 2D field)
        data_3d, steps, decode_stats = process_surface_variable(
            zstore, output_var, var_info, fs
        )

        if data_3d is not None and len(steps) > 0:
            # Subset to region
            data_subset, lats, lons = subset_to_region(data_3d, region)

            # Track common steps (use first variable's steps as reference)
            if common_steps is None:
                common_steps = steps

            # Create single-variable dataset and save incrementally
            meta = VARIABLE_METADATA.get(output_var, {'long_name': output_var, 'units': 'unknown'})
            var_ds = xr.Dataset(
                {
                    output_var: xr.DataArray(
                        data=data_subset.astype(np.float32),
                        dims=['step', 'latitude', 'longitude'],
                        attrs=meta
                    )
                },
                coords={
                    'step': steps,
                    'latitude': lats,
                    'longitude': lons
                }
            )

            # Save to zarr (first variable creates, others append)
            if var_idx == 0:
                # First variable - create new zarr store with coordinates
                var_ds['latitude'].attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
                var_ds['longitude'].attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}
                var_ds['step'].attrs = {'units': 'hours', 'long_name': 'forecast step'}
                var_ds.attrs = {
                    'title': 'ECMWF IFS Ensemble Forecast Data',
                    'institution': 'ECMWF',
                    'member': member,
                    'region': region,
                    'decoder': 'gribberish (subprocess-isolated with cfgrib fallback)',
                }
                var_ds.to_zarr(str(output_path), mode='w', consolidated=False)
            else:
                # Append variable to existing zarr store
                var_ds.to_zarr(str(output_path), mode='a', consolidated=False)

            processed_vars.append(output_var)
            print(f"    Saved {output_var} to zarr ({var_idx + 1}/{len(vars_to_process)})")

            # Free memory immediately after saving
            del data_subset
            del var_ds

        # Accumulate stats
        for key in total_decode_stats:
            total_decode_stats[key] += decode_stats[key]

        # Free memory
        del data_3d
        gc.collect()

    if not processed_vars:
        print("  Error: No variables were successfully processed.")
        return False

    # Consolidate zarr metadata at the end
    import zarr
    zarr.consolidate_metadata(str(output_path))
    print(f"\n4. Consolidated zarr metadata")

    # Summary
    elapsed = time.time() - start_time
    total_chunks = sum(total_decode_stats.values())

    print(f"\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Chunks processed: {total_chunks}")
    if total_chunks > 0:
        print(f"  Decoder usage: gribberish={total_decode_stats['gribberish']} "
              f"({100*total_decode_stats['gribberish']/total_chunks:.1f}%), "
              f"cfgrib={total_decode_stats['cfgrib']}, failed={total_decode_stats['failed']}")
    print(f"  Output: {output_path}")

    # Quick verification
    print(f"\n5. Verification...")
    try:
        ds_check = xr.open_dataset(str(output_path), engine='zarr')
        print(f"  Dimensions: {dict(ds_check.sizes)}")
        for var in list(ds_check.data_vars)[:5]:
            print(f"  {var}: min={float(ds_check[var].min()):.4f}, max={float(ds_check[var].max()):.4f}")
        ds_check.close()
        return True
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ECMWF IFS ensemble member to zarr using gribberish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process control member with East Africa subsetting
  python run_single_ecmwf_to_zarr_gribberish.py control --region east_africa

  # Process ensemble member 01
  python run_single_ecmwf_to_zarr_gribberish.py ens_01

  # Process specific variables only
  python run_single_ecmwf_to_zarr_gribberish.py ens_25 --variables t2m,tp,u700,v700

  # Process with custom parquet directory
  python run_single_ecmwf_to_zarr_gribberish.py control --parquet_dir ./my_parquet_files

Available members:
  - control: Control run
  - ens_01 to ens_50: Ensemble members 01-50

cGAN required variables (with alternatives):
  cape (-> mucape), cp (N/A), mcc (-> tcc), sp, ssr, t2m,
  tciw/tclw/tcrw (-> tcw), tcw, tcwv, tp, u700, v700
        """
    )

    parser.add_argument('member', help='Ensemble member (control, ens_01, ens_02, ..., ens_50)')
    parser.add_argument('--region', choices=list(REGIONS.keys()), default='global',
                       help='Region to subset (default: global)')
    parser.add_argument('--variables', type=str,
                       help='Comma-separated list of variables to process (default: cGAN variables)')
    parser.add_argument('--parquet_dir', type=str,
                       help='Directory containing ECMWF parquet files (default: ./test_ecmwf_three_stage_prebuilt_output)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for zarr files (default: ./zarr_stores/ecmwf)')

    args = parser.parse_args()
    success = main(args.member, args.region, args.variables, args.parquet_dir, args.output_dir)

    if not success:
        sys.exit(1)
