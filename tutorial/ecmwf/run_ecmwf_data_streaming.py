#!/usr/bin/env python3
"""
ECMWF Data Streaming V2 - For 20260106 Three-Stage Parquet Files
===================================================================

Modified version to process the full 51-member ensemble from:
ecmwf_three_stage_20260106_00z/

Author: ICPAC GIK Team
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import sys
import warnings
import time
import shutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import zarr
from zarr.codecs import BloscCodec

# Number of parallel S3 fetches (ECMWF S3 is slow, parallelism helps)
MAX_PARALLEL_FETCHES = 8

# Try to import gribberish for fast decoding
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available, will use cfgrib only")

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import geopandas as gp
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/cartopy/geopandas not available for plotting")

warnings.filterwarnings('ignore')
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ==============================================================================
# CONFIGURATION - Modified for 20260106 data
# ==============================================================================

# Directory containing parquet files
PARQUET_DIR = Path("ecmwf_three_stage_20260106_00z")

# GeoJSON boundary file for plotting
BOUNDARY_JSON = Path(__file__).parent / "ea_ghcf_simple.geojson"

# Coverage area for plotting (East Africa)
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53

# 24-hour rainfall thresholds (mm)
THRESHOLDS_24H = [5, 25, 50, 75, 100, 125]

# ECMWF grid specification (0.25 degree global)
ECMWF_GRID_SHAPE = (721, 1440)
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# Output directory for plots
OUTPUT_DIR = Path("output_plots")

# Temporary zarr storage directory
TEMP_ZARR_DIR = Path("temp_zarr_cache")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def read_parquet_refs(parquet_path: str) -> Dict:
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
                value = value
        elif isinstance(value, str):
            if value.startswith('[') or value.startswith('{'):
                try:
                    value = json.loads(value)
                except:
                    pass

        zstore[key] = value

    return zstore


def discover_precipitation_chunks_ecmwf(zstore: Dict, member_name: str) -> List[Tuple[int, str, List]]:
    """
    Discover precipitation chunks for ECMWF data.

    Handles both formats:
    - step_XXX/tp/sfc/control/0.0.0
    - step_XXX/tp/sfc/ensXX/0.0.0 (without underscore)
    """
    chunks = []

    # Try both patterns - with and without underscore
    patterns = [
        re.compile(rf'^step_(\d+)/tp/sfc/{re.escape(member_name)}/0\.0\.0$'),
        re.compile(rf'^step_(\d+)/tp/sfc/{re.escape(member_name.replace("_", ""))}/0\.0\.0$'),
    ]

    for key, value in zstore.items():
        for pattern in patterns:
            match = pattern.match(key)
            if match and isinstance(value, list) and len(value) >= 3:
                step_hours = int(match.group(1))
                chunks.append((step_hours, key, value))
                break

    chunks.sort(key=lambda x: x[0])
    return chunks


def extract_model_date_from_zstore(zstore: Dict) -> Tuple[datetime, int]:
    """Extract model date and run hour from zstore references."""
    model_date = datetime.now()
    run_hour = 0

    for key, value in zstore.items():
        if isinstance(value, list) and len(value) >= 1:
            url = str(value[0])
            if 's3://' in url or 'ecmwf-forecasts' in url:
                match = re.search(r'/(\d{8})/(\d{2})z/', url)
                if match:
                    model_date = datetime.strptime(match.group(1), '%Y%m%d')
                    run_hour = int(match.group(2))
                    break

    return model_date, run_hour


def get_ea_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get East Africa subset indices and coordinates."""
    lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
    lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lats = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]

    return lat_indices, lon_indices, lats, lons


LAT_INDICES, LON_INDICES, EA_LATS, EA_LONS = get_ea_indices()


def fetch_grib_bytes(ref: List, fs) -> Tuple[bytes, int]:
    """Fetch GRIB bytes from S3 using the reference."""
    if not isinstance(ref, list) or len(ref) < 3:
        raise ValueError(f"Invalid reference format: {ref}")

    url, offset, length = ref[0], ref[1], ref[2]

    if not url.endswith('.grib2'):
        url = url + '.grib2'

    with fs.open(url, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_with_gribberish(grib_bytes: bytes, grid_shape=ECMWF_GRID_SHAPE) -> np.ndarray:
    """Decode GRIB bytes using gribberish."""
    if not GRIBBERISH_AVAILABLE:
        raise RuntimeError("gribberish not available")

    flat_array = gribberish.parse_grib_array(grib_bytes, 0)
    array_2d = flat_array.reshape(grid_shape)
    return array_2d


def decode_with_cfgrib(grib_bytes: bytes) -> np.ndarray:
    """Decode GRIB bytes using cfgrib."""
    import tempfile

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


def decode_grib_hybrid(grib_bytes: bytes, grid_shape=ECMWF_GRID_SHAPE) -> Tuple[np.ndarray, str]:
    """Decode GRIB with gribberish, fallback to cfgrib."""
    if GRIBBERISH_AVAILABLE:
        try:
            array_2d = decode_with_gribberish(grib_bytes, grid_shape)
            return array_2d, 'gribberish'
        except Exception:
            pass

    array_2d = decode_with_cfgrib(grib_bytes)
    return array_2d, 'cfgrib'


# ==============================================================================
# ZARR STORAGE
# ==============================================================================

def create_temp_zarr_store(n_members: int, n_timesteps: int, n_lats: int, n_lons: int,
                           date_str: str, run_str: str) -> Tuple[zarr.Group, Path]:
    """Create temporary zarr store."""
    temp_dir = TEMP_ZARR_DIR / f"ecmwf_{date_str}_{run_str}z_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Creating temporary zarr store: {temp_dir}")

    store = zarr.open_group(str(temp_dir), mode='w')

    store.create_dataset(
        'precipitation',
        shape=(n_members, n_timesteps, n_lats, n_lons),
        chunks=(1, n_timesteps, n_lats, n_lons),
        dtype=np.float32,
        fill_value=np.nan,
        compressors=[BloscCodec(cname='lz4', clevel=3)]
    )

    store.create_dataset(
        'timesteps',
        shape=(n_members, n_timesteps),
        dtype=np.int32,
        fill_value=-1
    )

    store.attrs['n_members'] = n_members
    store.attrs['n_timesteps'] = n_timesteps
    store.attrs['n_lats'] = n_lats
    store.attrs['n_lons'] = n_lons

    return store, temp_dir


def cleanup_temp_zarr_store(temp_dir: Path):
    """Remove temporary zarr store."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n  Cleaned up temporary storage: {temp_dir}")


# ==============================================================================
# DATA STREAMING
# ==============================================================================

def fetch_chunk_parallel(args):
    """Fetch a single chunk - helper for parallel execution."""
    idx, step_hours, chunk_key, ref, fs = args
    try:
        grib_bytes, length = fetch_grib_bytes(ref, fs)
        return idx, step_hours, grib_bytes, None
    except Exception as e:
        return idx, step_hours, None, str(e)


def stream_single_member_precipitation_ecmwf(
    parquet_path: str,
    zarr_store: zarr.Group,
    member_idx: int,
    member_name: str,
    fs
) -> Tuple[bool, List[int]]:
    """Stream precipitation data for a single ECMWF ensemble member with parallel S3 fetching."""
    print(f"\n  [{member_idx+1}] Streaming {member_name}...")

    start_time = time.time()

    try:
        zstore = read_parquet_refs(parquet_path)
        chunks = discover_precipitation_chunks_ecmwf(zstore, member_name)

        if not chunks:
            print(f"    No precipitation data found for {member_name}")
            return False, []

        n_chunks = len(chunks)
        print(f"    Found {n_chunks} timesteps")
        print(f"    Fetching with {MAX_PARALLEL_FETCHES} parallel connections...")

        # Prepare fetch arguments
        fetch_args = [
            (i, step_hours, chunk_key, ref, fs)
            for i, (step_hours, chunk_key, ref) in enumerate(chunks)
        ]

        # Parallel fetch all GRIB bytes
        fetch_start = time.time()
        grib_data = {}  # idx -> (step_hours, grib_bytes, error)

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
            futures = {executor.submit(fetch_chunk_parallel, args): args[0] for args in fetch_args}
            for future in as_completed(futures):
                idx, step_hours, grib_bytes, error = future.result()
                grib_data[idx] = (step_hours, grib_bytes, error)

        fetch_elapsed = time.time() - fetch_start
        fetched_count = sum(1 for v in grib_data.values() if v[1] is not None)
        print(f"    Fetched {fetched_count}/{n_chunks} in {fetch_elapsed:.1f}s ({fetch_elapsed/n_chunks:.2f}s avg)")

        # Decode sequentially (gribberish is fast, no need to parallelize)
        decode_start = time.time()
        decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}
        timesteps_loaded = []

        for i in range(n_chunks):
            step_hours, grib_bytes, fetch_error = grib_data[i]

            if fetch_error:
                print(f"      Step {step_hours:3d}h: FETCH FAILED - {fetch_error[:50]}")
                decode_stats['failed'] += 1
                zarr_store['precipitation'][member_idx, i, :, :] = np.nan
                continue

            try:
                array_2d, decoder = decode_grib_hybrid(grib_bytes)

                decode_stats[decoder] += 1
                timesteps_loaded.append(step_hours)

                data_subset = array_2d[LAT_INDICES[0]:LAT_INDICES[-1]+1,
                                       LON_INDICES[0]:LON_INDICES[-1]+1]

                zarr_store['precipitation'][member_idx, i, :, :] = data_subset.astype(np.float32)
                zarr_store['timesteps'][member_idx, i] = step_hours

                if i < 2 or i >= n_chunks - 2:
                    print(f"      Step {step_hours:3d}h: decoded [{decoder}]")
                elif i == 2:
                    print(f"      ... processing {n_chunks - 4} more steps ...")

            except Exception as e:
                print(f"      Step {step_hours:3d}h: DECODE FAILED - {str(e)[:50]}")
                decode_stats['failed'] += 1
                zarr_store['precipitation'][member_idx, i, :, :] = np.nan

        decode_elapsed = time.time() - decode_start
        elapsed = time.time() - start_time
        print(f"    Completed: gribberish={decode_stats['gribberish']}, "
              f"cfgrib={decode_stats['cfgrib']}, "
              f"failed={decode_stats['failed']}, "
              f"time={elapsed:.1f}s (fetch={fetch_elapsed:.1f}s, decode={decode_elapsed:.1f}s)")

        gc.collect()
        return True, timesteps_loaded

    except Exception as e:
        print(f"    Error streaming {member_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, []


# ==============================================================================
# 24-HOUR PRECIPITATION
# ==============================================================================

def convert_cumulative_to_24h_precip(data_3d: np.ndarray, timesteps: List[int]) -> Tuple[np.ndarray, List[int]]:
    """Convert ECMWF cumulative precipitation to 24-hour accumulated."""
    timestep_to_idx = {ts: i for i, ts in enumerate(timesteps)}

    periods_24h = []
    valid_hours = []

    for end_hour in timesteps:
        start_hour = end_hour - 24
        if start_hour >= 0 and start_hour in timestep_to_idx:
            start_idx = timestep_to_idx[start_hour]
            end_idx = timestep_to_idx[end_hour]

            precip_24h = (data_3d[end_idx] - data_3d[start_idx]) * 1000.0
            precip_24h = np.maximum(precip_24h, 0.0)

            periods_24h.append(precip_24h)
            valid_hours.append(end_hour)

    if not periods_24h:
        print("    Warning: No 24h periods found")
        return data_3d * 1000.0, timesteps

    return np.stack(periods_24h, axis=0).astype(np.float32), valid_hours


def calculate_exceedance_probabilities_from_zarr(
    zarr_store: zarr.Group,
    n_members: int,
    thresholds: List[float]
) -> Tuple[Dict, int, int, List[int]]:
    """Calculate probability of exceeding thresholds."""
    print(f"\n  Calculating 24-hour accumulations and probabilities...")

    first_member_data = None
    first_member_timesteps = None

    for member_idx in range(n_members):
        timesteps = zarr_store['timesteps'][member_idx, :]
        valid_timesteps = [int(ts) for ts in timesteps if ts >= 0]

        if len(valid_timesteps) > 0:
            data = zarr_store['precipitation'][member_idx, :len(valid_timesteps), :, :]
            if not np.all(np.isnan(data)):
                first_member_data = data
                first_member_timesteps = valid_timesteps
                break

    if first_member_data is None:
        print("    Error: No valid data found")
        return {}, 0, 0, []

    first_24h, valid_24h_hours = convert_cumulative_to_24h_precip(
        first_member_data, first_member_timesteps
    )
    n_24h_periods = len(valid_24h_hours)
    spatial_shape = first_24h.shape[1:]

    print(f"    24h periods ending at hours: {valid_24h_hours[:5]}...{valid_24h_hours[-3:] if len(valid_24h_hours) > 5 else ''}")
    print(f"    Number of 24h periods: {n_24h_periods}")
    print(f"    Spatial shape: {spatial_shape}")

    exceedance_counts = {}
    for day_idx in range(n_24h_periods):
        exceedance_counts[day_idx] = {}
        for threshold in thresholds:
            exceedance_counts[day_idx][threshold] = np.zeros(spatial_shape, dtype=np.int32)

    valid_members = 0
    for member_idx in range(n_members):
        try:
            timesteps = zarr_store['timesteps'][member_idx, :]
            valid_timesteps = [int(ts) for ts in timesteps if ts >= 0]

            if len(valid_timesteps) == 0:
                continue

            data = zarr_store['precipitation'][member_idx, :len(valid_timesteps), :, :]

            if np.all(np.isnan(data)):
                continue

            data_24h, member_24h_hours = convert_cumulative_to_24h_precip(data, valid_timesteps)

            valid_members += 1

            for day_idx, hour in enumerate(member_24h_hours):
                if hour in valid_24h_hours:
                    target_idx = valid_24h_hours.index(hour)
                    for threshold in thresholds:
                        exceedance_counts[target_idx][threshold] += (data_24h[day_idx] >= threshold).astype(np.int32)

            if (member_idx + 1) % 10 == 0:
                print(f"    Processed {member_idx + 1}/{n_members} members")

            del data, data_24h
            gc.collect()

        except Exception as e:
            print(f"    Warning: Failed to process member {member_idx}: {e}")

    print(f"    Valid members: {valid_members}/{n_members}")

    probabilities = {}
    for day_idx in range(n_24h_periods):
        probabilities[day_idx] = {}
        for threshold in thresholds:
            if valid_members > 0:
                probabilities[day_idx][threshold] = (exceedance_counts[day_idx][threshold] / valid_members) * 100
            else:
                probabilities[day_idx][threshold] = np.zeros(spatial_shape, dtype=np.float32)

    return probabilities, valid_members, n_24h_periods, valid_24h_hours


# ==============================================================================
# PLOTTING
# ==============================================================================

def load_geojson_boundaries(json_file: Path):
    """Load GeoJSON boundaries from file."""
    if not PLOTTING_AVAILABLE:
        return None
    try:
        if json_file.exists():
            gdf = gp.read_file(json_file)
            return gdf
        else:
            print(f"    Warning: Boundary file not found: {json_file}")
            return None
    except Exception as e:
        print(f"    Warning: Could not load boundary file: {e}")
        return None


def create_24h_probability_plot(
    probabilities: Dict,
    lons: np.ndarray,
    lats: np.ndarray,
    n_members: int,
    n_days: int,
    valid_24h_hours: List[int],
    model_date: datetime,
    run_hour: int,
    output_dir: Path = None
) -> Optional[str]:
    """Create multi-panel probability plot."""
    if not PLOTTING_AVAILABLE:
        print("  Plotting not available")
        return None

    # Limit to first 10 days for reasonable plot size
    n_days_to_plot = min(n_days, 10)

    fig, axes = plt.subplots(n_days_to_plot, len(THRESHOLDS_24H),
                            figsize=(4*len(THRESHOLDS_24H), 4*n_days_to_plot),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    if n_days_to_plot == 1:
        axes = axes.reshape(1, -1)
    elif len(THRESHOLDS_24H) == 1:
        axes = axes.reshape(-1, 1)

    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933',
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']

    gdf = load_geojson_boundaries(BOUNDARY_JSON)
    base_datetime = model_date + timedelta(hours=run_hour)

    for day_idx in range(n_days_to_plot):
        ending_hour = valid_24h_hours[day_idx]

        for t_idx, threshold in enumerate(THRESHOLDS_24H):
            ax = axes[day_idx, t_idx]

            prob_data = probabilities[day_idx][threshold]

            cf = ax.contourf(lons, lats, prob_data, levels=levels, colors=colors,
                           transform=ccrs.PlateCarree(), extend='neither')

            ax.contour(lons, lats, prob_data, levels=[50],
                      colors='black', linewidths=1, alpha=0.7,
                      transform=ccrs.PlateCarree())

            if gdf is not None:
                ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                                facecolor="none", edgecolor="black", linewidth=0.8)

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', alpha=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

            start_datetime = base_datetime + timedelta(hours=ending_hour - 24)
            end_datetime = base_datetime + timedelta(hours=ending_hour)
            start_eat = start_datetime + timedelta(hours=EAT_OFFSET)
            end_eat = end_datetime + timedelta(hours=EAT_OFFSET)

            max_prob = np.nanmax(prob_data)

            if t_idx == 0:
                if day_idx == 0:
                    title = f'>{threshold}mm\n'
                else:
                    title = ''
                title += f'{start_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'to {end_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'Max: {max_prob:.0f}%'
                ax.set_title(title, fontsize=9, pad=10)
            else:
                if day_idx == 0:
                    ax.set_title(f'>{threshold}mm\nMax: {max_prob:.0f}%', fontsize=10)
                else:
                    ax.set_title(f'Max: {max_prob:.0f}%', fontsize=10)

            if t_idx == 0:
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                linewidth=0.3, color='gray', alpha=0.3, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 8}
                gl.ylabel_style = {'size': 8}

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)

    model_run_str = f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}:00 UTC'
    model_run_eat = f'{model_date.strftime("%Y-%m-%d")} {(run_hour + EAT_OFFSET) % 24:02d}:00 EAT'

    fig.suptitle(f'ECMWF IFS 24-Hour Rainfall Exceedance Probabilities\n'
                 f'Model Run: {model_run_str} ({model_run_eat})\n'
                 f'Based on {n_members} ensemble members | Coverage: {LAT_MIN}N-{LAT_MAX}N, {LON_MIN}E-{LON_MAX}E',
                 fontsize=14, y=0.98)

    date_str = model_date.strftime('%Y%m%d')
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'ecmwf_24h_probability_{date_str}_{run_hour:02d}z_all_thresholds.png'
    else:
        output_file = f'ecmwf_24h_probability_{date_str}_{run_hour:02d}z_all_thresholds.png'

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {output_file}")
    plt.close()

    return str(output_file)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main processing routine."""
    print("="*70)
    print("ECMWF Data Streaming - 20260106 Full Ensemble")
    print("="*70)
    print(f"Gribberish Available: {GRIBBERISH_AVAILABLE}")
    print(f"Plotting Available: {PLOTTING_AVAILABLE}")
    print(f"Parquet Directory: {PARQUET_DIR}")
    print(f"East Africa Region: {len(EA_LATS)} x {len(EA_LONS)} grid points")
    print("="*70)

    start_time = time.time()
    temp_zarr_path = None

    try:
        if not PARQUET_DIR.exists():
            print(f"\nError: Parquet directory {PARQUET_DIR} not found!")
            return False

        # Find all stage3 parquet files
        parquet_files = sorted(PARQUET_DIR.glob("stage3_ens_*_final.parquet"))
        control_file = PARQUET_DIR / "stage3_control_final.parquet"
        if control_file.exists():
            parquet_files.insert(0, control_file)

        if not parquet_files:
            print(f"\nError: No parquet files found in {PARQUET_DIR}")
            return False

        print(f"\nFound {len(parquet_files)} parquet files")
        for pf in parquet_files[:5]:
            print(f"  - {pf.name}")
        if len(parquet_files) > 5:
            print(f"  ... and {len(parquet_files) - 5} more")

        # Extract member names - use ens01, ens02, ... ens09, ens10, ... format
        member_names = []
        for pf in parquet_files:
            stem = pf.stem
            if 'control' in stem:
                member_names.append('control')
            else:
                match = re.search(r'ens_?(\d+)', stem)
                if match:
                    # Use format with leading zero for single digits: ens01, ens02, etc.
                    member_num = int(match.group(1))
                    member_names.append(f'ens{member_num:02d}')
                else:
                    member_names.append(stem.replace('stage3_', '').replace('_final', ''))

        print(f"\nMember names: {member_names[:5]}...{member_names[-3:] if len(member_names) > 5 else ''}")

        # Extract model date
        print("\n[Step 1] Extracting model information...")
        zstore = read_parquet_refs(str(parquet_files[0]))
        model_date, run_hour = extract_model_date_from_zstore(zstore)
        print(f"  Model date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")

        # Discover timesteps
        first_member_name = member_names[0]
        chunks = discover_precipitation_chunks_ecmwf(zstore, first_member_name)
        n_timesteps = len(chunks)
        print(f"  Timesteps per member: {n_timesteps}")

        if n_timesteps == 0:
            print(f"  Error: No precipitation chunks found")
            return False

        # Create S3 filesystem
        fs = fsspec.filesystem('s3', anon=True)

        # Create temporary zarr store
        print("\n[Step 2] Creating temporary disk storage...")
        n_members = len(parquet_files)
        n_lats = len(EA_LATS)
        n_lons = len(EA_LONS)
        date_str = model_date.strftime('%Y%m%d')

        zarr_store, temp_zarr_path = create_temp_zarr_store(
            n_members, n_timesteps, n_lats, n_lons, date_str, f"{run_hour:02d}"
        )
        print(f"  Storage shape: ({n_members}, {n_timesteps}, {n_lats}, {n_lons})")

        # Stream data
        print("\n[Step 3] Streaming Precipitation Data")
        print("="*70)

        if GRIBBERISH_AVAILABLE:
            print("Using GRIBBERISH for fast data streaming (~80x faster)")
        else:
            print("Using CFGRIB for data streaming")

        successful_members = 0
        for member_idx, (pf, member_name) in enumerate(zip(parquet_files, member_names)):
            success, _ = stream_single_member_precipitation_ecmwf(
                str(pf), zarr_store, member_idx, member_name, fs
            )
            if success:
                successful_members += 1

        print(f"\nSuccessfully loaded {successful_members} ensemble members")

        if successful_members == 0:
            print("Error: No data loaded!")
            return False

        # Calculate probabilities
        print("\n[Step 4] Calculating Exceedance Probabilities")
        print("="*70)

        probabilities, n_valid_members, n_24h_periods, valid_24h_hours = calculate_exceedance_probabilities_from_zarr(
            zarr_store, n_members, THRESHOLDS_24H
        )

        if n_24h_periods == 0:
            print("Error: No 24-hour periods calculated!")
            return False

        print(f"  24h Periods: {n_24h_periods}")
        print(f"  Thresholds: {THRESHOLDS_24H} mm")
        print(f"  Valid Members: {n_valid_members}")

        # Summary statistics
        print("\n  Summary Statistics:")
        for day_idx in range(min(3, n_24h_periods)):
            ending_hour = valid_24h_hours[day_idx]
            print(f"\n    Period ending T+{ending_hour}h:")
            for threshold in THRESHOLDS_24H[:3]:
                max_prob = np.nanmax(probabilities[day_idx][threshold])
                area_50 = np.sum(probabilities[day_idx][threshold] >= 50)
                print(f"      >{threshold:3d}mm: Max={max_prob:5.1f}%, P>=50% at {area_50} points")

        # Create visualization
        print("\n[Step 5] Creating Visualization")
        print("="*70)

        plot_file = create_24h_probability_plot(
            probabilities, EA_LONS, EA_LATS, n_valid_members, n_24h_periods,
            valid_24h_hours, model_date, run_hour, OUTPUT_DIR
        )

        # Summary
        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Model Date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")
        print(f"  Ensemble Members: {n_valid_members}")
        print(f"  24h Periods: {n_24h_periods}")
        print(f"  Thresholds: {THRESHOLDS_24H}")
        print(f"  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

        if plot_file:
            print(f"\nOutput Plot: {plot_file}")

        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if temp_zarr_path and temp_zarr_path.exists():
            cleanup_temp_zarr_store(temp_zarr_path)


if __name__ == "__main__":
    success = main()
    if success:
        print("\nData streaming completed successfully!")
    else:
        print("\nData streaming failed.")
        sys.exit(1)
