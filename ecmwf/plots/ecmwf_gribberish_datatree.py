#!/usr/bin/env python3
"""
ECMWF IFS Ensemble Processing with Gribberish and Zarr DataTree

This script:
1. Reads parquet files to get byte offset/length references
2. Uses gribberish for fast GRIB decoding (~80x faster than cfgrib)
3. Builds an xarray DataTree with all ensemble members
4. Stores data in zarr format for efficient access
5. Calculates empirical probabilities and creates plots

Adapted from GEFS version for ECMWF IFS parquet structure.

Key differences from GEFS:
- ECMWF uses lon range -180 to 180 (GEFS uses 0-360)
- ECMWF parquet has step_XXX format: step_XXX/varname/level/member/0.0.0
- Single ensemble member per parquet file (vs GEFS which has one parquet per member)

Usage:
    python ecmwf_gribberish_datatree.py [options]

Example:
    python ecmwf_gribberish_datatree.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --variable tp --timesteps 0-24 --members 5
"""

import os
import sys
import argparse
import json
import warnings
import time
import gc
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataTree
import zarr
import fsspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Try to import gribberish
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available, will use cfgrib fallback")

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for ECMWF data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ECMWF IFS grid specification (0.25 degree global)
ECMWF_GRID_SHAPE = (721, 1440)  # lat x lon
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# East Africa region
EA_LAT_MIN, EA_LAT_MAX = -12, 23
EA_LON_MIN, EA_LON_MAX = 21, 53

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# Variable metadata
VARIABLE_METADATA = {
    'tp': {
        'long_name': 'Total precipitation',
        'units': 'm',
        'standard_name': 'precipitation_amount'
    },
    'tp_24h': {
        'long_name': '24-hour accumulated precipitation',
        'units': 'mm',
        'standard_name': 'precipitation_amount'
    },
    't2m': {
        'long_name': '2 metre temperature',
        'units': 'K',
        'standard_name': 'air_temperature'
    },
    'sp': {
        'long_name': 'Surface pressure',
        'units': 'Pa',
        'standard_name': 'surface_air_pressure'
    },
    'tcwv': {
        'long_name': 'Total column water vapour',
        'units': 'kg m**-2',
        'standard_name': 'atmosphere_mass_content_of_water_vapor'
    },
    'probability': {
        'long_name': 'Exceedance probability',
        'units': '%',
    }
}

# Default precipitation thresholds in mm/day for probability calculations
DEFAULT_PRECIP_THRESHOLDS_MM = [20, 40, 60, 80, 100]

# ECMWF variable name mappings (output_var -> parquet_var)
ECMWF_VAR_NAMES = {
    't2m': '2t',
    'u10': '10u',
    'v10': '10v',
    'd2m': '2d',
    # Most variables use same name
}


def get_parquet_var_name(var_name: str) -> str:
    """Get the parquet variable name from the output variable name."""
    return ECMWF_VAR_NAMES.get(var_name, var_name)


def convert_cumulative_to_24h_precip(data_3d: np.ndarray, timesteps: List[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Convert cumulative precipitation to 24-hour accumulated precipitation.

    ECMWF tp is cumulative from T+0, so to get 24h accumulation:
    - For T+24: tp(T+24) - tp(T+0)
    - For T+48: tp(T+48) - tp(T+24)
    - etc.

    Also converts from meters to mm (multiply by 1000).

    Args:
        data_3d: Cumulative precipitation array (time, lat, lon) in meters
        timesteps: List of forecast hours (e.g., [0, 3, 6, 9, 12, ..., 72])

    Returns:
        Tuple of (24h_precip_array, 24h_valid_times)
        - 24h_precip_array: 24-hour accumulation in mm (n_24h_periods, lat, lon)
        - 24h_valid_times: List of ending forecast hours for each 24h period
    """
    # Find pairs of timesteps that are 24 hours apart
    timestep_to_idx = {ts: i for i, ts in enumerate(timesteps)}

    periods_24h = []
    valid_hours = []

    for end_hour in timesteps:
        start_hour = end_hour - 24
        if start_hour >= 0 and start_hour in timestep_to_idx:
            start_idx = timestep_to_idx[start_hour]
            end_idx = timestep_to_idx[end_hour]

            # Calculate 24h accumulation: end - start, convert m to mm
            precip_24h = (data_3d[end_idx] - data_3d[start_idx]) * 1000.0

            # Ensure non-negative (numerical precision issues)
            precip_24h = np.maximum(precip_24h, 0.0)

            periods_24h.append(precip_24h)
            valid_hours.append(end_hour)

    if not periods_24h:
        # If no 24h periods available, fall back to using raw cumulative values converted to mm
        # This happens when timesteps don't span 24h
        print("    Warning: No 24h periods found, using cumulative values converted to mm")
        return data_3d * 1000.0, timesteps

    return np.stack(periods_24h, axis=0).astype(np.float32), valid_hours


class GribberishDecoder:
    """Fast GRIB decoder using gribberish with S3 byte fetching."""

    def __init__(self, grid_shape: Tuple[int, int] = ECMWF_GRID_SHAPE):
        self.grid_shape = grid_shape
        self.fs = fsspec.filesystem('s3', anon=True)
        self.decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}
        self.total_decode_time = 0

    def fetch_grib_bytes(self, url: str, offset: int, length: int) -> bytes:
        """Fetch GRIB bytes from S3."""
        # Add .grib2 extension if missing (ECMWF S3 files require it)
        if not url.endswith('.grib2'):
            url = url + '.grib2'

        with self.fs.open(url, 'rb') as f:
            f.seek(offset)
            return f.read(length)

    def decode_with_gribberish_subprocess(self, grib_bytes: bytes) -> Tuple[Optional[np.ndarray], bool]:
        """
        Decode GRIB using gribberish in a subprocess to safely catch Rust panics.
        """
        import subprocess
        import sys

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
array_2d = flat_array.reshape({self.grid_shape}).astype(np.float32)
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

        if os.path.exists(tmp_in_path):
            os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)

        return None, False

    def decode_with_cfgrib(self, grib_bytes: bytes) -> np.ndarray:
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

    def decode(self, grib_bytes: bytes) -> Tuple[np.ndarray, str]:
        """Decode GRIB bytes using gribberish with cfgrib fallback."""
        t0 = time.time()

        if GRIBBERISH_AVAILABLE:
            array, success = self.decode_with_gribberish_subprocess(grib_bytes)
            if success:
                self.total_decode_time += (time.time() - t0) * 1000
                self.decode_stats['gribberish'] += 1
                return array, 'gribberish'

        # Fallback to cfgrib
        try:
            array = self.decode_with_cfgrib(grib_bytes)
            self.total_decode_time += (time.time() - t0) * 1000
            self.decode_stats['cfgrib'] += 1
            return array, 'cfgrib'
        except Exception as e:
            self.decode_stats['failed'] += 1
            raise e

    def decode_from_reference(self, ref: List) -> Tuple[Optional[np.ndarray], str]:
        """Decode GRIB data from a zstore reference [url, offset, length]."""
        if not isinstance(ref, list) or len(ref) < 3:
            return None, 'invalid'

        url, offset, length = ref[0], ref[1], ref[2]

        try:
            grib_bytes = self.fetch_grib_bytes(url, offset, length)
            return self.decode(grib_bytes)
        except Exception as e:
            self.decode_stats['failed'] += 1
            return None, f'failed: {e}'

    def get_stats(self) -> Dict:
        """Return decoding statistics."""
        total = self.decode_stats['gribberish'] + self.decode_stats['cfgrib']
        return {
            'decoded': total,
            'gribberish': self.decode_stats['gribberish'],
            'cfgrib': self.decode_stats['cfgrib'],
            'failed': self.decode_stats['failed'],
            'total_time_ms': self.total_decode_time,
            'avg_time_ms': self.total_decode_time / max(1, total)
        }


def read_parquet_to_zstore(parquet_path: str) -> dict:
    """Read parquet file and extract zstore references."""
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
                value = value  # Keep as bytes

        elif isinstance(value, str):
            if value.startswith('[') or value.startswith('{'):
                try:
                    value = json.loads(value)
                except:
                    pass

        zstore[key] = value

    return zstore


def find_variable_chunks_step_format(zstore: dict, var_name: str, level: str = 'sfc') -> List[Tuple[int, str, List]]:
    """
    Find all chunks for a variable in the step_XXX format.

    ECMWF format: step_XXX/varname/level/member/0.0.0
    where level is 'sfc' (surface), 'pl' (pressure level), or 'sol' (soil).

    Returns list of (step_hours, key, reference) tuples.
    """
    chunks = []
    parquet_var = get_parquet_var_name(var_name)

    # Pattern: step_XXX/varname/level/member/0.0.0
    pattern = re.compile(rf'^step_(\d+)/{re.escape(parquet_var)}/{re.escape(level)}/[^/]+/0\.0\.0$')

    for key, value in zstore.items():
        match = pattern.match(key)
        if match and isinstance(value, list) and len(value) >= 3:
            step_hours = int(match.group(1))
            chunks.append((step_hours, key, value))

    chunks.sort(key=lambda x: x[0])
    return chunks


def find_variable_chunks_traditional(zstore: dict, var_name: str, var_path: str = None) -> List[Tuple[int, str, List]]:
    """
    Find all chunks for a variable in the traditional format.

    Traditional format: varname/type/surface/varname/0.X.0.0
    Example: tp/accum/surface/tp/0.1.0.0

    Returns list of (step_index, key, reference) tuples.
    """
    chunks = []
    parquet_var = get_parquet_var_name(var_name)

    if var_path is None:
        # Common path patterns for different variable types
        paths_to_try = [
            f'{parquet_var}/accum/surface/{parquet_var}',  # Accumulated vars like tp
            f'{parquet_var}/instant/surface/{parquet_var}',  # Instant vars
            f'{parquet_var}/avg/surface/{parquet_var}',  # Averaged vars
        ]
    else:
        paths_to_try = [var_path]

    for var_path in paths_to_try:
        # Pattern: varpath/0.X.0.0
        pattern = re.compile(rf'^{re.escape(var_path)}/0\.(\d+)\.0\.0$')

        for key, value in zstore.items():
            match = pattern.match(key)
            if match and isinstance(value, list) and len(value) >= 3:
                step_idx = int(match.group(1))
                chunks.append((step_idx, key, value))

        if chunks:
            break

    chunks.sort(key=lambda x: x[0])
    return chunks


def get_ea_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get East Africa subset indices and coordinates."""
    lat_mask = (ECMWF_LATS >= EA_LAT_MIN) & (ECMWF_LATS <= EA_LAT_MAX)
    lon_mask = (ECMWF_LONS >= EA_LON_MIN) & (ECMWF_LONS <= EA_LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lats = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]

    return lat_indices, lon_indices, lats, lons


def load_member_data(parquet_file: Path, decoder: GribberishDecoder,
                     var_name: str, timesteps: List[int],
                     lat_indices: np.ndarray, lon_indices: np.ndarray,
                     use_step_format: bool = True) -> Tuple[Optional[np.ndarray], str]:
    """
    Load variable data for a single ensemble member.

    Returns 3D array (time, lat, lon) for East Africa region.
    """
    member = parquet_file.stem
    # Extract member number from filename like stage3_ens_48_final
    match = re.search(r'(control|ens_?\d+)', member)
    member_label = match.group(1) if match else member

    try:
        zstore = read_parquet_to_zstore(str(parquet_file))

        # Find chunks using step_XXX format (preferred for ECMWF)
        if use_step_format:
            chunks = find_variable_chunks_step_format(zstore, var_name, level='sfc')
        else:
            chunks = find_variable_chunks_traditional(zstore, var_name)

        if not chunks:
            print(f"    {member_label}: No {var_name} chunks found")
            return None, member_label

        # Create chunk lookup by step
        chunk_lookup = {step: ref for step, key, ref in chunks}

        # Load data for requested timesteps
        n_lats = lat_indices[-1] - lat_indices[0] + 1
        n_lons = lon_indices[-1] - lon_indices[0] + 1
        data_3d = np.full((len(timesteps), n_lats, n_lons), np.nan, dtype=np.float32)

        for i, ts in enumerate(timesteps):
            if ts in chunk_lookup:
                ref = chunk_lookup[ts]
                data_2d, decoder_used = decoder.decode_from_reference(ref)

                if data_2d is not None:
                    # Subset to East Africa
                    data_3d[i] = data_2d[lat_indices[0]:lat_indices[-1]+1,
                                         lon_indices[0]:lon_indices[-1]+1]

        return data_3d, member_label

    except Exception as e:
        print(f"    {member_label}: Error - {type(e).__name__}: {str(e)[:50]}")
        return None, member_label


def build_ensemble_datatree(parquet_dir: Path, var_name: str, timesteps: List[int],
                           max_members: int = None,
                           model_date: datetime = None,
                           run_hour: int = 0,
                           convert_precip_to_24h: bool = True) -> Tuple[DataTree, Dict]:
    """
    Build xarray DataTree with all ensemble members using gribberish decoding.

    For precipitation (tp), converts cumulative values to 24-hour accumulations in mm
    when convert_precip_to_24h=True. This allows using meaningful thresholds like
    20, 40, 60, 80, 100 mm/day.

    Structure:
        /
        ├── ens_01/
        │   └── {var_name} (time, latitude, longitude)
        ├── ens_02/
        │   └── {var_name} (time, latitude, longitude)
        ...
        ├── control/
        │   └── {var_name} (time, latitude, longitude)
        └── ensemble_stats/
            ├── mean
            ├── std
            └── probability_X (for various thresholds)
    """
    print("\n" + "="*70)
    print("Building ECMWF Ensemble DataTree with Gribberish")
    print("="*70)

    # Get parquet files - ECMWF has one file per member
    parquet_files = sorted(parquet_dir.glob("stage3_ens_*_final.parquet"))
    control_file = parquet_dir / "stage3_control_final.parquet"
    if control_file.exists():
        parquet_files.insert(0, control_file)

    if max_members:
        parquet_files = parquet_files[:max_members]

    print(f"  Found {len(parquet_files)} parquet files")
    print(f"  Variable: {var_name}")
    print(f"  Timesteps to load: {timesteps[0]} to {timesteps[-1]} ({len(timesteps)} total)")

    # For precipitation, explain the conversion
    is_precip = var_name == 'tp'
    if is_precip and convert_precip_to_24h:
        print(f"  Precipitation mode: Converting cumulative tp to 24-hour accumulation (mm/day)")

    # Get East Africa indices
    lat_indices, lon_indices, lats, lons = get_ea_indices()
    print(f"  East Africa region: {len(lats)} x {len(lons)} grid points")

    # Create decoder
    decoder = GribberishDecoder()

    # Calculate valid times
    if model_date is None:
        model_date = datetime.now()

    base_time = model_date + timedelta(hours=run_hour)

    # Load raw data for all members first
    print(f"\n  Loading ensemble member data...")
    raw_member_data = {}

    for pf in parquet_files:
        t0 = time.time()
        data, member_label = load_member_data(pf, decoder, var_name, timesteps,
                                              lat_indices, lon_indices)
        elapsed = time.time() - t0

        if data is not None:
            raw_member_data[member_label] = data
            print(f"    {member_label}: OK ({elapsed:.1f}s, max_cumulative={np.nanmax(data):.4f}m)")
        else:
            print(f"    {member_label}: FAILED")

    # Print decoder stats
    stats = decoder.get_stats()
    print(f"\n  Decoder stats: {stats['gribberish']} gribberish, "
          f"{stats['cfgrib']} cfgrib, {stats['failed']} failed, "
          f"avg {stats['avg_time_ms']:.1f}ms/chunk")

    # Convert precipitation to 24h accumulation if requested
    member_datasets = {}
    output_var_name = var_name
    final_timesteps = timesteps
    final_valid_times = [base_time + timedelta(hours=ts) for ts in timesteps]

    if is_precip and convert_precip_to_24h and raw_member_data:
        print(f"\n  Converting to 24-hour precipitation accumulation...")
        output_var_name = 'tp_24h'

        # Convert each member's data
        converted_member_data = {}
        for member_label, data in raw_member_data.items():
            data_24h, hours_24h = convert_cumulative_to_24h_precip(data, timesteps)
            converted_member_data[member_label] = data_24h

            if member_label == list(raw_member_data.keys())[0]:
                # Use first member to determine output timesteps
                final_timesteps = hours_24h
                final_valid_times = [base_time + timedelta(hours=ts) for ts in hours_24h]
                print(f"    24h periods ending at: T+{hours_24h}")

        raw_member_data = converted_member_data
        print(f"    Max 24h precip across all members: {max(np.nanmax(d) for d in raw_member_data.values()):.1f} mm")

    # Now create datasets
    meta = VARIABLE_METADATA.get(output_var_name, {'long_name': output_var_name, 'units': 'unknown'})

    for member_label, data in raw_member_data.items():
        ds = xr.Dataset(
            data_vars={
                output_var_name: (['time', 'latitude', 'longitude'], data, meta)
            },
            coords={
                'time': final_valid_times,
                'latitude': lats,
                'longitude': lons,
                'forecast_hour': ('time', final_timesteps)
            },
            attrs={
                'member': member_label,
                'model_date': model_date.strftime('%Y-%m-%d'),
                'run_hour': run_hour
            }
        )
        member_datasets[member_label] = ds

    # Build DataTree
    print(f"\n  Building DataTree structure...")
    tree_dict = {f"/{member}": ds for member, ds in member_datasets.items()}

    # Calculate ensemble statistics
    if len(member_datasets) > 0:
        print(f"  Calculating ensemble statistics...")

        # Stack all members
        all_data = np.stack([ds[output_var_name].values for ds in member_datasets.values()], axis=0)
        n_members = len(member_datasets)

        # Calculate statistics
        ensemble_mean = np.nanmean(all_data, axis=0)
        ensemble_std = np.nanstd(all_data, axis=0)
        ensemble_max = np.nanmax(all_data, axis=0)
        ensemble_min = np.nanmin(all_data, axis=0)

        # Create stats dataset
        stats_ds = xr.Dataset(
            data_vars={
                'mean': (['time', 'latitude', 'longitude'], ensemble_mean,
                        {'long_name': f'Ensemble mean {output_var_name}', 'units': meta.get('units', 'unknown')}),
                'std': (['time', 'latitude', 'longitude'], ensemble_std,
                       {'long_name': 'Ensemble standard deviation', 'units': meta.get('units', 'unknown')}),
                'max': (['time', 'latitude', 'longitude'], ensemble_max,
                       {'long_name': 'Ensemble maximum', 'units': meta.get('units', 'unknown')}),
                'min': (['time', 'latitude', 'longitude'], ensemble_min,
                       {'long_name': 'Ensemble minimum', 'units': meta.get('units', 'unknown')}),
            },
            coords={
                'time': final_valid_times,
                'latitude': lats,
                'longitude': lons,
                'forecast_hour': ('time', final_timesteps)
            },
            attrs={
                'n_members': n_members,
                'members': list(member_datasets.keys())
            }
        )

        tree_dict['/ensemble_stats'] = stats_ds

    # Create DataTree
    dt = DataTree.from_dict(tree_dict)

    # Add root attributes
    dt.attrs = {
        'title': 'ECMWF IFS Ensemble Forecast Data',
        'institution': 'ECMWF',
        'source': 'IFS (Integrated Forecasting System)',
        'decoder': 'gribberish (subprocess-isolated with cfgrib fallback)',
        'variable': output_var_name,
        'original_variable': var_name,
        'model_date': model_date.strftime('%Y-%m-%d'),
        'run_hour': run_hour,
        'n_members': len(member_datasets),
        'region': 'East Africa',
        'created': datetime.now().isoformat()
    }

    print(f"\n  DataTree created with {len(member_datasets)} members")
    print(f"  Output variable: {output_var_name}")

    return dt, {'lats': lats, 'lons': lons, 'n_members': len(member_datasets), 'var_name': output_var_name}


def calculate_exceedance_probabilities(dt: DataTree, var_name: str, thresholds: List[float]) -> xr.Dataset:
    """
    Calculate exceedance probabilities from ensemble DataTree.

    Returns Dataset with probability fields for each threshold.
    """
    print(f"\n  Calculating exceedance probabilities for thresholds: {thresholds}")

    # Get member names (exclude ensemble_stats and probabilities)
    members = sorted([name for name in dt.children.keys()
                     if name not in ['ensemble_stats', 'probabilities']])
    n_members = len(members)

    if n_members == 0:
        return None

    # Stack all member data
    all_data = np.stack([dt[member].ds[var_name].values for member in members], axis=0)

    # Get coordinates from first member
    first_ds = dt[members[0]].ds
    times = first_ds['time'].values
    lats = first_ds['latitude'].values
    lons = first_ds['longitude'].values
    forecast_hours = first_ds['forecast_hour'].values

    # Calculate probabilities
    prob_vars = {}

    for threshold in thresholds:
        exceedance_count = np.sum(all_data >= threshold, axis=0)
        probability = (exceedance_count / n_members) * 100

        var_name_clean = f'prob_gt_{str(threshold).replace(".", "p")}'
        prob_vars[var_name_clean] = (
            ['time', 'latitude', 'longitude'],
            probability.astype(np.float32),
            {
                'long_name': f'Probability of {var_name} > {threshold}',
                'units': '%',
                'threshold': threshold,
                'n_members': n_members
            }
        )

        max_prob = np.nanmax(probability)
        print(f"    {threshold}: max probability = {max_prob:.1f}%")

    # Create dataset
    prob_ds = xr.Dataset(
        data_vars=prob_vars,
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
            'forecast_hour': ('time', forecast_hours)
        },
        attrs={
            'n_members': n_members,
            'thresholds': thresholds,
            'variable': var_name
        }
    )

    return prob_ds


def save_datatree_to_zarr(dt: DataTree, output_path: Path) -> None:
    """Save DataTree to zarr store."""
    print(f"\n  Saving DataTree to zarr: {output_path}")

    # Remove existing store if present
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    # Save to zarr
    dt.to_zarr(str(output_path))

    # Get size
    import subprocess
    result = subprocess.run(['du', '-sh', str(output_path)], capture_output=True, text=True)
    if result.returncode == 0:
        size = result.stdout.split()[0]
        print(f"  Saved: {output_path} ({size})")


def plot_ensemble_mean(dt: DataTree, var_name: str, timestep_idx: int, output_dir: Path,
                       model_date: datetime, run_hour: int) -> str:
    """Plot ensemble mean."""
    stats = dt['ensemble_stats'].ds

    # Get data for timestep
    mean_data = stats['mean'].isel(time=timestep_idx).values
    lats = stats['latitude'].values
    lons = stats['longitude'].values

    forecast_hour = int(stats['forecast_hour'].isel(time=timestep_idx).values)
    valid_time = pd.Timestamp(stats['time'].isel(time=timestep_idx).values)

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Determine colormap and levels based on variable
    if var_name == 'tp_24h':
        # 24-hour precipitation already in mm
        vmax = max(10, np.ceil(np.nanmax(mean_data) * 1.1 / 10) * 10)
        levels = np.linspace(0, vmax, 11)
        cmap = 'Blues'
        cbar_label = '24h Precipitation (mm/day)'
        plot_data = mean_data
    elif var_name == 'tp':
        # Cumulative precipitation in meters, convert to mm for display
        mean_data_mm = mean_data * 1000
        vmax = max(5, np.ceil(np.nanmax(mean_data_mm) * 1.1 / 5) * 5)
        levels = np.linspace(0, vmax, 11)
        cmap = 'Blues'
        cbar_label = 'Cumulative Precipitation (mm)'
        plot_data = mean_data_mm
    elif var_name == 't2m':
        # Temperature in K, convert to C for display
        mean_data_c = mean_data - 273.15
        vmin, vmax = np.nanmin(mean_data_c), np.nanmax(mean_data_c)
        levels = np.linspace(vmin, vmax, 11)
        cmap = 'RdYlBu_r'
        cbar_label = 'Temperature (C)'
        plot_data = mean_data_c
    else:
        vmax = max(1, np.ceil(np.nanmax(mean_data) * 1.1))
        levels = np.linspace(0, vmax, 11)
        cmap = 'viridis'
        cbar_label = VARIABLE_METADATA.get(var_name, {}).get('units', 'unknown')
        plot_data = mean_data

    cf = ax.contourf(lons, lats, plot_data, levels=levels, cmap=cmap,
                     transform=ccrs.PlateCarree(), extend='max')

    # Map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)

    # Title
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24
    n_members = stats.attrs.get('n_members', '?')

    plt.title(f'ECMWF IFS Ensemble Mean {var_name.upper()}\n'
              f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
              f'Valid: {valid_time.strftime("%Y-%m-%d %H:%M")} UTC ({eat_hour:02d}:00 EAT)\n'
              f'{n_members} members | Max: {np.nanmax(plot_data):.2f}',
              fontsize=12)

    # Save
    output_file = output_dir / f'ecmwf_ensemble_mean_{var_name}_T{forecast_hour:03d}.png'
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return str(output_file)


def plot_probability_panels(prob_ds: xr.Dataset, var_name: str, timestep_idx: int, output_dir: Path,
                           model_date: datetime, run_hour: int) -> str:
    """Plot probability panels for all thresholds."""
    thresholds = prob_ds.attrs.get('thresholds', [0.001, 0.005, 0.01, 0.02])
    n_thresholds = len(thresholds)

    # Get coordinates
    lats = prob_ds['latitude'].values
    lons = prob_ds['longitude'].values
    forecast_hour = int(prob_ds['forecast_hour'].isel(time=timestep_idx).values)
    valid_time = pd.Timestamp(prob_ds['time'].isel(time=timestep_idx).values)

    # Create figure
    n_cols = min(3, n_thresholds)
    n_rows = (n_thresholds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    if n_thresholds == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    # Probability colors
    prob_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    prob_colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933',
                   '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']

    im = None
    for idx, threshold in enumerate(thresholds):
        ax = axes_flat[idx]

        var_name_clean = f'prob_gt_{str(threshold).replace(".", "p")}'
        prob_data = prob_ds[var_name_clean].isel(time=timestep_idx).values

        # Plot
        im = ax.contourf(lons, lats, prob_data, levels=prob_levels,
                        colors=prob_colors, transform=ccrs.PlateCarree())

        # 50% contour
        ax.contour(lons, lats, prob_data, levels=[50], colors='black',
                  linewidths=1.5, transform=ccrs.PlateCarree())

        # Map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

        # Title - format threshold nicely
        if var_name == 'tp_24h':
            threshold_display = f'{threshold:.0f}mm/day'
        elif var_name == 'tp':
            threshold_display = f'{threshold*1000:.1f}mm'
        else:
            threshold_display = f'{threshold}'

        max_prob = np.nanmax(prob_data)
        ax.set_title(f'P(precip > {threshold_display})\nMax: {max_prob:.0f}%', fontsize=11)

    # Hide unused
    for idx in range(n_thresholds, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)

    # Title
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24
    n_members = prob_ds.attrs.get('n_members', '?')

    fig.suptitle(f'ECMWF IFS Exceedance Probabilities (Gribberish + Zarr DataTree)\n'
                 f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
                 f'Valid: {valid_time.strftime("%Y-%m-%d %H:%M")} UTC ({eat_hour:02d}:00 EAT) | {n_members} members',
                 fontsize=13, y=0.98)

    # Save
    output_file = output_dir / f'ecmwf_probability_panels_{var_name}_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return str(output_file)


def plot_member_comparison(dt: DataTree, var_name: str, timestep_idx: int, output_dir: Path,
                          model_date: datetime, run_hour: int) -> str:
    """Plot all ensemble members side by side."""
    # Filter to only ensemble members (exclude ensemble_stats and probabilities)
    members = sorted([name for name in dt.children.keys()
                     if name not in ['ensemble_stats', 'probabilities']])
    n_members = len(members)

    if n_members == 0:
        return None

    # Get first member for coordinates
    first_ds = dt[members[0]].ds
    lats = first_ds['latitude'].values
    lons = first_ds['longitude'].values
    forecast_hour = int(first_ds['forecast_hour'].isel(time=timestep_idx).values)

    # Calculate grid
    n_cols = min(6, n_members)
    n_rows = (n_members + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes_flat = axes.flatten() if n_members > 1 else [axes]

    # Get global max for consistent colorbar
    all_data = [dt[m].ds[var_name].isel(time=timestep_idx).values for m in members]

    # Handle variable-specific scaling
    if var_name == 'tp_24h':
        # Already in mm/day
        cbar_label = '24h Precipitation (mm/day)'
        cmap = 'Blues'
    elif var_name == 'tp':
        all_data = [d * 1000 for d in all_data]  # Convert to mm
        cbar_label = 'Precipitation (mm)'
        cmap = 'Blues'
    elif var_name == 't2m':
        all_data = [d - 273.15 for d in all_data]  # Convert to C
        cbar_label = 'Temperature (C)'
        cmap = 'RdYlBu_r'
    else:
        cbar_label = VARIABLE_METADATA.get(var_name, {}).get('units', 'unknown')
        cmap = 'viridis'

    all_max = max(np.nanmax(d) for d in all_data)
    all_min = min(np.nanmin(d) for d in all_data)

    if var_name in ['tp', 'tp_24h']:
        vmax = max(10, np.ceil(all_max * 1.1 / 10) * 10)
        levels = np.linspace(0, vmax, 11)
    else:
        levels = np.linspace(all_min, all_max, 11)

    im = None
    for idx, (member, data) in enumerate(zip(members, all_data)):
        ax = axes_flat[idx]

        im = ax.contourf(lons, lats, data, levels=levels, cmap=cmap,
                        transform=ccrs.PlateCarree(), extend='max')

        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2)
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

        data_max = np.nanmax(data)
        ax.set_title(f'{member}\nMax: {data_max:.2f}', fontsize=9)

    # Hide unused
    for idx in range(n_members, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # Title
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    fig.suptitle(f'ECMWF IFS Ensemble Members - {var_name.upper()} (Gribberish Decoder)\n'
                 f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}Z - T+{forecast_hour}h '
                 f'({utc_hour:02d}:00 UTC / {eat_hour:02d}:00 EAT)',
                 fontsize=13, y=0.99)

    # Save
    output_file = output_dir / f'ecmwf_member_comparison_{var_name}_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return str(output_file)


def main(parquet_dir: str, variable: str = "tp", timesteps_str: str = "0-72",
         max_members: int = None, plot_timestep: int = 24,
         thresholds_str: str = None,
         save_zarr_flag: bool = True,
         convert_precip_to_24h: bool = True):
    """Main processing function.

    For precipitation (tp):
    - Converts cumulative values to 24-hour accumulation in mm
    - Uses thresholds in mm/day: 20, 40, 60, 80, 100 by default
    - Requires timesteps spanning at least 24 hours for 24h accumulation
    """
    print("=" * 70)
    print("ECMWF IFS Ensemble Processing with Gribberish and Zarr DataTree")
    print("=" * 70)

    if not GRIBBERISH_AVAILABLE:
        print("\nWARNING: gribberish is not installed, will use cfgrib fallback")
        print("Install with: pip install gribberish")

    start_time = time.time()

    # Parse inputs
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"\nERROR: Directory {parquet_path} not found")
        return False

    # Parse timesteps (e.g., "0-72" or "0,6,12,24,48")
    if '-' in timesteps_str:
        parts = timesteps_str.split('-')
        start, end = int(parts[0]), int(parts[1])
        # ECMWF uses 3-hourly timesteps
        timesteps = list(range(start, end + 1, 3))
    else:
        timesteps = [int(t) for t in timesteps_str.split(',')]

    # Parse thresholds - use sensible defaults for precipitation
    is_precip = variable == 'tp'
    if thresholds_str is None:
        if is_precip and convert_precip_to_24h:
            # Default thresholds for 24h precipitation in mm/day
            thresholds = DEFAULT_PRECIP_THRESHOLDS_MM  # [20, 40, 60, 80, 100]
        elif is_precip:
            # Thresholds for cumulative precipitation in meters
            thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
        else:
            # Generic thresholds
            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    else:
        thresholds = [float(t) for t in thresholds_str.split(',')]

    # Try to extract date from directory or files
    model_date = datetime.now()
    run_hour = 0

    # Check parquet files for date info
    sample_files = list(parquet_path.glob("stage3_*.parquet"))
    if sample_files:
        # Try to read date from parquet metadata or filename pattern
        try:
            zstore = read_parquet_to_zstore(str(sample_files[0]))
            # Look for S3 URL to extract date
            for key, value in zstore.items():
                if isinstance(value, list) and len(value) >= 1:
                    url = value[0]
                    if 's3://' in str(url):
                        # Extract date from URL like: s3://ecmwf-forecasts/20251127/00z/...
                        match = re.search(r'/(\d{8})/(\d{2})z/', str(url))
                        if match:
                            model_date = datetime.strptime(match.group(1), '%Y%m%d')
                            run_hour = int(match.group(2))
                            break
        except:
            pass

    print(f"\nConfiguration:")
    print(f"  Parquet directory: {parquet_path}")
    print(f"  Variable: {variable}")
    print(f"  Model date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")
    print(f"  Timesteps: {timesteps[0]}-{timesteps[-1]} ({len(timesteps)} steps)")
    print(f"  Max members: {max_members or 'all'}")
    print(f"  Plot timestep: {plot_timestep} (T+{plot_timestep}h)")
    if is_precip and convert_precip_to_24h:
        print(f"  Precipitation: Converting to 24h accumulation (mm/day)")
        print(f"  Thresholds (mm/day): {thresholds}")
    else:
        print(f"  Thresholds: {thresholds}")

    # Build DataTree
    dt, meta = build_ensemble_datatree(
        parquet_path, variable, timesteps, max_members, model_date, run_hour,
        convert_precip_to_24h=convert_precip_to_24h
    )

    if meta['n_members'] == 0:
        print("\nERROR: No members loaded successfully!")
        return False

    # Save to zarr
    output_dir = parquet_path

    if save_zarr_flag:
        zarr_path = output_dir / f'ecmwf_ensemble_datatree_{variable}_{model_date.strftime("%Y%m%d")}_{run_hour:02d}z.zarr'
        save_datatree_to_zarr(dt, zarr_path)

    # Use the actual variable name from meta (may be tp_24h for converted precip)
    output_var_name = meta['var_name']

    # Calculate probabilities
    print("\n" + "="*70)
    print("Calculating Exceedance Probabilities")
    print("="*70)

    prob_ds = calculate_exceedance_probabilities(dt, output_var_name, thresholds)

    # Add probabilities to DataTree
    if prob_ds is not None:
        dt['/probabilities'] = DataTree(prob_ds)

    # Create plots
    print("\n" + "="*70)
    print("Creating Plots")
    print("="*70)

    # Get the final timesteps from the datatree (may differ due to 24h conversion)
    first_member = [k for k in dt.children.keys() if k not in ['ensemble_stats', 'probabilities']][0]
    final_forecast_hours = list(dt[first_member].ds['forecast_hour'].values)

    # Find the closest timestep index
    if plot_timestep >= final_forecast_hours[-1]:
        plot_timestep_idx = len(final_forecast_hours) - 1
    else:
        plot_timestep_idx = min(range(len(final_forecast_hours)),
                               key=lambda i: abs(final_forecast_hours[i] - plot_timestep))

    print(f"  Using timestep index {plot_timestep_idx} (T+{final_forecast_hours[plot_timestep_idx]}h)")

    plot_files = []

    # Ensemble mean plot
    print("\n  Creating ensemble mean plot...")
    plot_files.append(plot_ensemble_mean(dt, output_var_name, plot_timestep_idx, output_dir, model_date, run_hour))

    # Probability panels
    if prob_ds is not None:
        print("\n  Creating probability panels...")
        plot_files.append(plot_probability_panels(prob_ds, output_var_name, plot_timestep_idx, output_dir, model_date, run_hour))

    # Member comparison
    print("\n  Creating member comparison plot...")
    plot_files.append(plot_member_comparison(dt, output_var_name, plot_timestep_idx, output_dir, model_date, run_hour))

    # Summary
    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Members processed: {meta['n_members']}")
    print(f"  Timesteps: {len(timesteps)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Plots created: {len([p for p in plot_files if p])}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ECMWF IFS ensemble processing with gribberish and zarr DataTree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process precipitation with default 24h accumulation and thresholds (20,40,60,80,100 mm/day)
  python ecmwf_gribberish_datatree.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --members 50 --timesteps 0-72

  # Process 5 members with specific plot timestep (24h forecast)
  python ecmwf_gribberish_datatree.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --members 5 --timesteps 0-48 --plot_timestep 24

  # Use cumulative precipitation (no 24h conversion)
  python ecmwf_gribberish_datatree.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --no_24h_convert

  # Custom thresholds for 24h precipitation (in mm/day)
  python ecmwf_gribberish_datatree.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --thresholds "10,20,30,50,75,100"

  # Process temperature variable
  python ecmwf_gribberish_datatree.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --variable t2m

Note: For precipitation (tp), the script automatically converts cumulative values
to 24-hour accumulation (mm/day) and uses thresholds of 20, 40, 60, 80, 100 mm/day.
Use --no_24h_convert to disable this behavior.
        """
    )

    parser.add_argument('--parquet_dir', type=str, default='test_ecmwf_three_stage_prebuilt_output',
                       help='Directory containing ECMWF parquet files')
    parser.add_argument('--variable', type=str, default='tp',
                       help='Variable to process (default: tp)')
    parser.add_argument('--timesteps', type=str, default='0-72',
                       help='Timesteps in hours to load (e.g., "0-72" or "0,6,12,24")')
    parser.add_argument('--members', type=int, default=None,
                       help='Max number of members to process')
    parser.add_argument('--plot_timestep', type=int, default=24,
                       help='Forecast hour for plotting (default: 24 = first 24h accumulation period)')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Exceedance thresholds (for tp with 24h conversion: mm/day, default: 20,40,60,80,100)')
    parser.add_argument('--no_zarr', action='store_true',
                       help='Skip saving to zarr')
    parser.add_argument('--no_24h_convert', action='store_true',
                       help='Do not convert precipitation to 24h accumulation (use cumulative values)')

    args = parser.parse_args()

    success = main(
        args.parquet_dir,
        args.variable,
        args.timesteps,
        args.members,
        args.plot_timestep,
        args.thresholds,
        save_zarr_flag=not args.no_zarr,
        convert_precip_to_24h=not args.no_24h_convert
    )

    if not success:
        sys.exit(1)
