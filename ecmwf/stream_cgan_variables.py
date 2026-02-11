#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "pyarrow",
#     "zarr<3",
#     "netcdf4",
#     "gribberish",
#     "cfgrib",
#     "eccodes",
#     "python-dotenv",
# ]
# ///
"""
ECMWF Data Streaming for cGAN Inference Variables
===================================================

This script streams the 13 variables required for cGAN inference from
ECMWF ensemble data processed through the GIK three-stage pipeline.

Variables extracted:
- tp:    Total Precipitation (accum/surface)
- t2m:   2-meter Temperature (instant/heightAboveGround)
- sp:    Surface Pressure (instant/surface)
- tcw:   Total Cloud Water (instant/entireAtmosphere)
- tcwv:  Total Column Water Vapour (instant/entireAtmosphere)
- tcc:   Total Cloud Cover (instant/entireAtmosphere) - used for mcc
- ssr:   Surface Solar Radiation (accum/surface)
- ssrd:  Surface Solar Radiation Downwards (accum/surface)
- u:     U-wind at 700 hPa (instant/isobaricInhPa)
- v:     V-wind at 700 hPa (instant/isobaricInhPa)
- cape:  CAPE (instant/entireAtmosphere) - via mucape
- sf:    Snowfall (accum/surface) - for convective precipitation estimation
- ro:    Runoff (accum/surface) - additional moisture indicator

Output: NetCDF file with ensemble mean and standard deviation for each variable

Author: ICPAC GIK Team
Date: 2026-02-04
"""

import os
import sys
import time
import json
import warnings
import tempfile
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import fsspec

warnings.filterwarnings('ignore')
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import gribberish for fast decoding
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available, will use cfgrib")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# GCS Configuration (loaded from .env or environment variables)
GCS_BUCKET = os.environ.get('GCS_BUCKET', '')
GCS_PARQUET_PREFIX = os.environ.get('GCS_PARQUET_PREFIX', '')
GCS_SERVICE_ACCOUNT_FILE = os.environ.get('GCS_SERVICE_ACCOUNT_FILE', '')

# Directory containing parquet files (local fallback)
PARQUET_DIR = Path("ecmwf_three_stage_20260203_00z")

# Target timesteps (36-60 hours in 3-hour intervals)
TARGET_STEPS = [36, 39, 42, 45, 48, 51, 54, 57, 60]

# cGAN variables to extract (ECMWF variable name -> output name)
# Key format: step_{hour:03d}/{var}/sfc/control/0.0.0
CGAN_SURFACE_VARS = {
    'tp': 'tp',           # Total Precipitation
    '2t': 't2m',          # 2-meter Temperature (ECMWF uses 2t)
    'sp': 'sp',           # Surface Pressure
    'ssr': 'ssr',         # Surface Solar Radiation
    'ssrd': 'ssrd',       # Surface Solar Radiation Downwards
    'sf': 'sf',           # Snowfall (for cp estimation)
    'ro': 'ro',           # Runoff
    'tcw': 'tcw',         # Total Cloud Water
    'tcwv': 'tcwv',       # Total Column Water Vapour
    'tcc': 'tcc',         # Total Cloud Cover (for mcc)
}

CGAN_ATMOS_VARS = {
    # These are also at surface level in ECMWF
}

CGAN_PRESSURE_VARS = {
    'u': 'u700',          # U-wind at 700 hPa
    'v': 'v700',          # V-wind at 700 hPa
}

# Pressure level for wind extraction
TARGET_PRESSURE_LEVEL = 700  # hPa

# ECMWF grid specification
ECMWF_GRID_SHAPE = (721, 1440)
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# ICPAC region subset
LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

# Parallel S3 fetches
MAX_PARALLEL_FETCHES = 8

# Output configuration
OUTPUT_DIR = Path("cgan_output")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_gcs_filesystem():
    """Create a GCS filesystem using service account from .env or ADC."""
    import gcsfs
    sa_file = GCS_SERVICE_ACCOUNT_FILE
    if sa_file and os.path.exists(sa_file):
        return gcsfs.GCSFileSystem(token=sa_file)
    return gcsfs.GCSFileSystem()


def list_gcs_parquets(gcs_path: str, max_members: int = None) -> List[str]:
    """List stage3 parquet files on GCS. Returns list of gs:// paths."""
    gcs_fs = get_gcs_filesystem()
    # Strip gs:// prefix for gcsfs
    bucket_path = gcs_path.replace('gs://', '')
    files = sorted(gcs_fs.glob(f"{bucket_path}/stage3_*_final.parquet"))
    if max_members:
        files = files[:max_members]
    return [f"gs://{f}" for f in files]


def get_icpac_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get ICPAC region subset indices and coordinates."""
    lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
    lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lats = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]

    return lat_indices, lon_indices, lats, lons


LAT_INDICES, LON_INDICES, ICPAC_LATS, ICPAC_LONS = get_icpac_indices()


def read_parquet_refs(parquet_path: str) -> Dict:
    """Read parquet file and extract zstore references.

    Supports local paths and gs:// GCS paths.
    """
    if parquet_path.startswith('gs://'):
        gcs_fs = get_gcs_filesystem()
        df = pd.read_parquet(parquet_path, filesystem=gcs_fs)
    else:
        df = pd.read_parquet(parquet_path)

    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        if isinstance(value, bytes):
            try:
                decoded = value.decode('utf-8')
                if decoded.startswith('[') or decoded.startswith('{'):
                    value = json.loads(decoded)
                else:
                    value = decoded
            except:
                pass
        elif isinstance(value, str):
            if value.startswith('[') or value.startswith('{'):
                try:
                    value = json.loads(value)
                except:
                    pass

        zstore[key] = value

    return zstore


def discover_variable_chunks(zstore: Dict, var_name: str, step_hours: List[int],
                             member_name: str = 'control') -> List[Tuple[int, str, List]]:
    """
    Discover chunks for a variable at specific timesteps.

    Returns list of (step_hour, key, reference) tuples.
    Key format: step_{hour:03d}/{var}/sfc/{member}/0.0.0
    """
    chunks = []

    for step in step_hours:
        # Try different key patterns - ECMWF uses 0.0.0 format
        patterns = [
            f'step_{step:03d}/{var_name}/sfc/{member_name}/0.0.0',
            f'step_{step:03d}/{var_name}/sfc/0.0.0',
            f'step_{step:03d}/{var_name}/surface/{member_name}/0.0.0',
        ]

        for pattern in patterns:
            if pattern in zstore:
                ref = zstore[pattern]
                if isinstance(ref, list) and len(ref) >= 3:
                    chunks.append((step, pattern, ref))
                    break

    return chunks


def discover_pressure_level_chunks(zstore: Dict, var_name: str, step_hours: List[int],
                                    member_name: str = 'control',
                                    pressure_level: int = 700) -> List[Tuple[int, str, List]]:
    """
    Discover chunks for pressure-level variables.
    Key format: step_{hour:03d}/{var}/pl/{member}/0.0.0
    """
    chunks = []

    for step in step_hours:
        # ECMWF uses 'pl' for pressure level, 0.0.0 format
        patterns = [
            f'step_{step:03d}/{var_name}/pl/{member_name}/0.0.0',
            f'step_{step:03d}/{var_name}/pl/0.0.0',
        ]

        for pattern in patterns:
            if pattern in zstore:
                ref = zstore[pattern]
                if isinstance(ref, list) and len(ref) >= 3:
                    chunks.append((step, pattern, ref))
                    break

    return chunks


def fetch_grib_bytes(ref: List, fs) -> Tuple[bytes, int]:
    """Fetch GRIB bytes from S3 using the reference."""
    url, offset, length = ref[0], ref[1], ref[2]

    if not url.endswith('.grib2'):
        url = url + '.grib2'

    with fs.open(url, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_grib_bytes(grib_bytes: bytes, grid_shape=ECMWF_GRID_SHAPE) -> np.ndarray:
    """Decode GRIB bytes to 2D array."""
    if GRIBBERISH_AVAILABLE:
        try:
            flat_array = gribberish.parse_grib_array(grib_bytes, 0)
            return flat_array.reshape(grid_shape)
        except:
            pass

    # Fallback to cfgrib
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


def subset_to_icpac(data: np.ndarray) -> np.ndarray:
    """Subset global data to ICPAC region."""
    return data[LAT_INDICES[0]:LAT_INDICES[-1]+1,
                LON_INDICES[0]:LON_INDICES[-1]+1]


# ==============================================================================
# VARIABLE STREAMING
# ==============================================================================

def _fetch_decode_one_chunk(step, var_name, ref, fs):
    """Fetch and decode a single GRIB chunk from S3. Thread-safe."""
    try:
        grib_bytes, _ = fetch_grib_bytes(ref, fs)
        array_2d = decode_grib_bytes(grib_bytes)
        subset = subset_to_icpac(array_2d)
        return step, subset.astype(np.float32), None
    except Exception as e:
        return step, None, str(e)


def stream_variable_for_member(
    parquet_path: str,
    var_name: str,
    step_hours: List[int],
    fs,
    member_name: str = 'control',
    is_pressure_level: bool = False,
    pressure_level: int = 700
) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Stream a single variable for one member across timesteps.

    Returns (data_array, valid_steps) where data_array is (n_steps, lat, lon).
    S3 fetches for the 9 timesteps run in parallel using threads.
    """
    zstore = read_parquet_refs(parquet_path)

    # Discover chunks
    if is_pressure_level:
        chunks = discover_pressure_level_chunks(zstore, var_name, step_hours, member_name, pressure_level)
    else:
        chunks = discover_variable_chunks(zstore, var_name, step_hours, member_name)

    if not chunks:
        return None, []

    # Fetch and decode — parallel S3 fetches across timesteps
    n_lats = len(ICPAC_LATS)
    n_lons = len(ICPAC_LONS)
    data = np.full((len(step_hours), n_lats, n_lons), np.nan, dtype=np.float32)
    valid_steps = []

    with ThreadPoolExecutor(max_workers=min(len(chunks), MAX_PARALLEL_FETCHES)) as pool:
        futures = {
            pool.submit(_fetch_decode_one_chunk, step, var_name, ref, fs): step
            for step, key, ref in chunks
        }
        for future in as_completed(futures):
            step, subset, error = future.result()
            if error:
                print(f"    Warning: Failed to decode {var_name} at step {step}: {error}")
            elif subset is not None:
                step_idx = step_hours.index(step)
                data[step_idx] = subset
                valid_steps.append(step)

    return data, valid_steps


def _stream_one_member(args):
    """Worker: stream one variable for one member. Each thread gets its own S3 session."""
    parquet_path, var_name, step_hours, member_key, is_pressure_level, pressure_level = args
    fs = fsspec.filesystem('s3', anon=True)
    return stream_variable_for_member(
        parquet_path, var_name, step_hours, fs,
        member_name=member_key,
        is_pressure_level=is_pressure_level,
        pressure_level=pressure_level
    )


def stream_all_members_for_variable(
    parquet_dir,
    var_name: str,
    step_hours: List[int],
    output_name: str,
    is_pressure_level: bool = False,
    pressure_level: int = 700,
    max_members: int = 51,
    parallel_members: int = MAX_PARALLEL_FETCHES,
    gcs_parquet_path: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """
    Stream a variable across all ensemble members in parallel.

    Uses ThreadPoolExecutor to process `parallel_members` members concurrently.
    Each member's S3 fetches also run in parallel threads.

    Supports local parquet_dir or GCS via gcs_parquet_path.

    Returns (ensemble_mean, ensemble_std, valid_steps).
    """
    print(f"\n  Streaming {var_name} -> {output_name}...")

    # Find all member parquet files — GCS or local
    if gcs_parquet_path:
        parquet_paths = list_gcs_parquets(gcs_parquet_path, max_members)
    else:
        parquet_paths = [str(pf) for pf in sorted(parquet_dir.glob("stage3_*.parquet"))]
        if max_members:
            parquet_paths = parquet_paths[:max_members]

    n_members = len(parquet_paths)
    n_steps = len(step_hours)
    n_lats = len(ICPAC_LATS)
    n_lons = len(ICPAC_LONS)

    print(f"    Members: {n_members}, Steps: {step_hours}, Parallel: {parallel_members}")

    # Accumulate data for ensemble statistics
    all_data = np.full((n_members, n_steps, n_lats, n_lons), np.nan, dtype=np.float32)
    valid_steps = []

    # Build args for each member
    member_args = []
    for pf_path in parquet_paths:
        # Extract member name from filename (works for both local and GCS paths)
        stem = os.path.basename(pf_path).replace('.parquet', '')
        raw_member = stem.replace('stage3_', '').replace('_final', '')
        member_key = raw_member.replace('_', '')
        member_args.append((
            pf_path, var_name, step_hours, member_key,
            is_pressure_level, pressure_level
        ))

    # Parallel member processing
    completed = 0
    with ThreadPoolExecutor(max_workers=parallel_members) as pool:
        future_to_idx = {
            pool.submit(_stream_one_member, args): m_idx
            for m_idx, args in enumerate(member_args)
        }
        for future in as_completed(future_to_idx):
            m_idx = future_to_idx[future]
            try:
                data, steps = future.result()
                if data is not None:
                    all_data[m_idx] = data
                    if not valid_steps:
                        valid_steps = steps
            except Exception as e:
                print(f"    Warning: Member {m_idx} failed: {e}")

            completed += 1
            if completed % 10 == 0:
                print(f"    Processed {completed}/{n_members} members")

    # Calculate ensemble statistics
    valid_mask = ~np.all(np.isnan(all_data), axis=0)

    if not np.any(valid_mask):
        print(f"    Warning: No valid data for {var_name}")
        return None, None, []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ensemble_mean = np.nanmean(all_data, axis=0)
        ensemble_std = np.nanstd(all_data, axis=0)

    print(f"    Completed: mean range [{np.nanmin(ensemble_mean):.4f}, {np.nanmax(ensemble_mean):.4f}]")

    gc.collect()
    return ensemble_mean, ensemble_std, valid_steps


# ==============================================================================
# NETCDF OUTPUT
# ==============================================================================

def create_cgan_netcdf(
    output_path: Path,
    data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    valid_steps: List[int],
    model_date: datetime,
    run_hour: int
):
    """
    Create NetCDF file with cGAN input variables.

    data_dict: {var_name: (ensemble_mean, ensemble_std)}
    """
    print(f"\n  Creating NetCDF: {output_path}")

    n_steps = len(valid_steps)
    n_lats = len(ICPAC_LATS)
    n_lons = len(ICPAC_LONS)

    # Create time coordinates
    base_time = model_date + timedelta(hours=run_hour)
    valid_times = [base_time + timedelta(hours=h) for h in valid_steps]

    # Create dataset
    coords = {
        'time': [base_time],
        'valid_time': valid_times,
        'latitude': ICPAC_LATS,
        'longitude': ICPAC_LONS,
    }

    data_vars = {}
    for var_name, (mean_data, std_data) in data_dict.items():
        if mean_data is None:
            continue

        # Add ensemble mean
        data_vars[f'{var_name}_ensemble_mean'] = xr.DataArray(
            mean_data[np.newaxis, :, :, :],  # Add time dimension
            dims=['time', 'valid_time', 'latitude', 'longitude'],
            attrs={'long_name': f'{var_name} ensemble mean', 'units': 'varies'}
        )

        # Add ensemble std
        data_vars[f'{var_name}_ensemble_standard_deviation'] = xr.DataArray(
            std_data[np.newaxis, :, :, :],
            dims=['time', 'valid_time', 'latitude', 'longitude'],
            attrs={'long_name': f'{var_name} ensemble standard deviation', 'units': 'varies'}
        )

    ds = xr.Dataset(data_vars, coords=coords)

    # Add global attributes
    ds.attrs['title'] = 'ECMWF Ensemble Data for cGAN Inference'
    ds.attrs['institution'] = 'ICPAC'
    ds.attrs['source'] = 'ECMWF IFS Ensemble'
    ds.attrs['model_date'] = model_date.strftime('%Y-%m-%d')
    ds.attrs['model_run'] = f'{run_hour:02d}Z'
    ds.attrs['forecast_hours'] = str(valid_steps)
    ds.attrs['n_ensemble_members'] = 51
    ds.attrs['history'] = f'Created {datetime.now().isoformat()}'

    # Save to NetCDF
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32'
        }

    ds.to_netcdf(output_path, encoding=encoding)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Saved: {output_path} ({file_size_mb:.2f} MB)")

    return ds


# ==============================================================================
# MAIN
# ==============================================================================

def main(
    parquet_dir: Path = PARQUET_DIR,
    target_steps: List[int] = TARGET_STEPS,
    max_members: int = 51,
    output_dir: Path = OUTPUT_DIR,
    parallel_members: int = MAX_PARALLEL_FETCHES,
    gcs_parquet_path: str = None,
    variables: List[str] = None
):
    """Main streaming routine.

    If *variables* is given (list of output-variable names such as
    ``['tp']``), only those variables will be streamed.
    """
    print("="*70)
    print("ECMWF Data Streaming for cGAN Inference")
    print("="*70)
    if gcs_parquet_path:
        print(f"Parquet Source: {gcs_parquet_path} (GCS)")
    else:
        print(f"Parquet Source: {parquet_dir} (local)")
    print(f"Target Steps: {target_steps}")
    print(f"Max Members: {max_members}")
    print(f"Parallel Members: {parallel_members}")
    print(f"ICPAC Region: {len(ICPAC_LATS)} x {len(ICPAC_LONS)} grid points")
    print(f"Gribberish Available: {GRIBBERISH_AVAILABLE}")
    print("="*70)

    start_time = time.time()

    # Validate source
    if gcs_parquet_path:
        test_files = list_gcs_parquets(gcs_parquet_path, max_members=1)
        if not test_files:
            print(f"\nError: No parquet files found at {gcs_parquet_path}")
            return False
    else:
        if not parquet_dir.exists():
            print(f"\nError: Parquet directory {parquet_dir} not found!")
            return False

    # Extract date from path name (works for both local dir and GCS path)
    source_str = gcs_parquet_path if gcs_parquet_path else str(parquet_dir)
    match = re.search(r'(\d{8})_(\d{2})z', source_str)
    if match:
        model_date = datetime.strptime(match.group(1), '%Y%m%d')
        run_hour = int(match.group(2))
    else:
        model_date = datetime.now()
        run_hour = 0

    print(f"\nModel Date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")

    # Stream each variable
    data_dict = {}
    stream_kwargs = dict(
        max_members=max_members,
        parallel_members=parallel_members,
        gcs_parquet_path=gcs_parquet_path,
    )

    # Optionally filter variable dicts when --variables is supplied
    var_set = set(variables) if variables else None

    surface_vars = {k: v for k, v in CGAN_SURFACE_VARS.items()
                    if var_set is None or v in var_set}
    atmos_vars   = {k: v for k, v in CGAN_ATMOS_VARS.items()
                    if var_set is None or v in var_set}
    pressure_vars = {k: v for k, v in CGAN_PRESSURE_VARS.items()
                     if var_set is None or v in var_set}

    if var_set:
        print(f"Variable filter: {sorted(var_set)}")

    # Surface variables
    print("\n[Phase 1] Streaming Surface Variables")
    print("-"*50)
    for ecmwf_var, output_var in surface_vars.items():
        mean, std, steps = stream_all_members_for_variable(
            parquet_dir, ecmwf_var, target_steps, output_var,
            **stream_kwargs
        )
        if mean is not None:
            data_dict[output_var] = (mean, std)

    # Atmospheric variables
    print("\n[Phase 2] Streaming Atmospheric Variables")
    print("-"*50)
    for ecmwf_var, output_var in atmos_vars.items():
        mean, std, steps = stream_all_members_for_variable(
            parquet_dir, ecmwf_var, target_steps, output_var,
            **stream_kwargs
        )
        if mean is not None:
            data_dict[output_var] = (mean, std)

    # Pressure level variables (700 hPa winds)
    print("\n[Phase 3] Streaming Pressure Level Variables (700 hPa)")
    print("-"*50)
    for ecmwf_var, output_var in pressure_vars.items():
        mean, std, steps = stream_all_members_for_variable(
            parquet_dir, ecmwf_var, target_steps, output_var,
            is_pressure_level=True,
            pressure_level=TARGET_PRESSURE_LEVEL,
            **stream_kwargs
        )
        if mean is not None:
            data_dict[output_var] = (mean, std)

    # Create output
    print("\n[Phase 4] Creating Output NetCDF")
    print("-"*50)

    output_dir.mkdir(exist_ok=True)
    date_str = model_date.strftime('%Y%m%d')
    output_file = output_dir / f"IFS_{date_str}_{run_hour:02d}Z_cgan.nc"

    ds = create_cgan_netcdf(
        output_file,
        data_dict,
        target_steps,
        model_date,
        run_hour
    )

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("STREAMING COMPLETE!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Model Date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")
    print(f"  Variables Extracted: {len(data_dict)}")
    print(f"  Timesteps: {target_steps}")
    print(f"  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nOutput File: {output_file}")

    # Print variable info
    print(f"\nVariables in output:")
    for var_name in sorted(data_dict.keys()):
        mean, std = data_dict[var_name]
        print(f"  {var_name}: mean=[{np.nanmin(mean):.4g}, {np.nanmax(mean):.4g}], "
              f"std=[{np.nanmin(std):.4g}, {np.nanmax(std):.4g}]")

    return True


if __name__ == "__main__":
    import argparse

    # Build default GCS path from .env if bucket and prefix are configured
    default_gcs_path = None
    if GCS_BUCKET and GCS_PARQUET_PREFIX:
        default_gcs_path = f"gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}"

    parser = argparse.ArgumentParser(description='Stream ECMWF data for cGAN inference')
    parser.add_argument('--parquet-dir', type=str, default=str(PARQUET_DIR),
                        help='Local directory containing stage3 parquet files')
    parser.add_argument('--gcs-parquet-path', type=str, default=None,
                        help='GCS path to parquet files (e.g. gs://bucket/prefix/20260207_00z). '
                             'Overrides --parquet-dir. Uses service account from .env')
    parser.add_argument('--date', type=str, default=None,
                        help='Date to stream (YYYYMMDD). Builds GCS path from .env bucket/prefix '
                             'if --gcs-parquet-path is not set')
    parser.add_argument('--run', type=str, default='00',
                        help='Model run hour (default: 00)')
    parser.add_argument('--steps', type=str, default=','.join(map(str, TARGET_STEPS)),
                        help='Comma-separated forecast hours (default: 36-60)')
    parser.add_argument('--max-members', type=int, default=51,
                        help='Maximum number of ensemble members')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory for NetCDF file')
    parser.add_argument('--parallel-fetches', type=int, default=MAX_PARALLEL_FETCHES,
                        help=f'Number of concurrent member streams (default: {MAX_PARALLEL_FETCHES})')
    parser.add_argument('--variables', type=str, default=None,
                        help='Comma-separated output variable names to stream '
                             '(e.g. "tp" or "tp,t2m"). Default: all variables')

    args = parser.parse_args()

    steps = [int(s.strip()) for s in args.steps.split(',')]

    # Parse variable filter
    var_filter = None
    if args.variables:
        var_filter = [v.strip() for v in args.variables.split(',')]

    # Resolve GCS path: explicit > --date + .env > None (local)
    gcs_path = args.gcs_parquet_path
    if not gcs_path and args.date and default_gcs_path:
        gcs_path = f"{default_gcs_path}/{args.date}_{args.run}z"

    success = main(
        parquet_dir=Path(args.parquet_dir),
        target_steps=steps,
        max_members=args.max_members,
        output_dir=Path(args.output_dir),
        parallel_members=args.parallel_fetches,
        gcs_parquet_path=gcs_path,
        variables=var_filter
    )

    if success:
        print("\nData streaming completed successfully!")
    else:
        print("\nData streaming failed.")
        sys.exit(1)
