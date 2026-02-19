#!/usr/bin/env python3
"""
GEFS Data Streaming V2 - Memory-Efficient with Disk-Based Storage
===================================================================

This script demonstrates how to use the parquet reference files to stream GEFS
ensemble data and create precipitation exceedance probability plots.

Key Features:
- Uses gribberish for fast GRIB decoding (~80x faster than cfgrib)
- Uses disk-based zarr storage to avoid memory issues with large ensembles
- Processes members one at a time, writing to disk immediately
- Creates elaborate 24-hour rainfall exceedance probability plots
- Uses ea_ghcf_simple.geojson for East Africa boundary overlay
- Follows plotting style from run_gefs_24h_accumulation.py

Prerequisites:
    pip install kerchunk zarr xarray pandas numpy fsspec s3fs gribberish
    pip install matplotlib cartopy geopandas

Usage:
    python run_gefs_data_streaming_v2.py

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
import tempfile
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import re

import fsspec
import zarr
from zarr.codecs import BloscCodec

# Try to import gribberish for fast decoding
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available, will use cfgrib only")
    print("Install with: pip install gribberish")

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
    print("Install with: pip install matplotlib cartopy geopandas")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Directory containing parquet files from run_gefs_tutorial.py
PARQUET_DIR = Path("output_parquet")

# Target date and run (should match parquet files)
TARGET_DATE = '20250106'  # YYYYMMDD format
TARGET_RUN = '00'         # Model run time (00, 06, 12, 18)

# Ensemble members to process
ENSEMBLE_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # gep01 to gep30

# GeoJSON boundary file for plotting
BOUNDARY_JSON = Path(__file__).parent.parent.parent / "gefs" / "ea_ghcf_simple.geojson"

# Coverage area for plotting (East Africa)
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53

# 24-hour rainfall thresholds (mm)
THRESHOLDS_24H = [5, 25, 50, 75, 100, 125]

# GEFS timestep is 3 hours, so 24 hours = 8 timesteps
TIMESTEPS_PER_DAY = 8

# GEFS grid specification (0.25 degree global)
GEFS_GRID_SHAPE = (721, 1440)  # lat x lon
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# Output directory for plots
OUTPUT_DIR = Path("output_plots")

# Temporary zarr storage directory (to avoid memory issues)
TEMP_ZARR_DIR = Path("temp_zarr_cache")


# ==============================================================================
# INITIALIZATION
# ==============================================================================

# Parse date for model run information
MODEL_DATE = datetime.strptime(TARGET_DATE, "%Y%m%d")
MODEL_RUN_HOUR = int(TARGET_RUN)

# Calculate East Africa region indices once
lat_mask = (GEFS_LATS >= LAT_MIN) & (GEFS_LATS <= LAT_MAX)
lon_mask = (GEFS_LONS >= LON_MIN) & (GEFS_LONS <= LON_MAX)
LAT_INDICES = np.where(lat_mask)[0]
LON_INDICES = np.where(lon_mask)[0]
EA_LATS = GEFS_LATS[LAT_INDICES[0]:LAT_INDICES[-1]+1]
EA_LONS = GEFS_LONS[LON_INDICES[0]:LON_INDICES[-1]+1]

print("="*70)
print("GEFS Data Streaming V2 - Memory Efficient")
print("="*70)
print(f"Target Date: {TARGET_DATE}")
print(f"Model Run: {TARGET_RUN}Z")
print(f"Ensemble Members: {len(ENSEMBLE_MEMBERS)}")
print(f"Gribberish Available: {GRIBBERISH_AVAILABLE}")
print(f"Plotting Available: {PLOTTING_AVAILABLE}")
print(f"East Africa Region: {len(EA_LATS)} x {len(EA_LONS)} grid points")
print("="*70)


# ==============================================================================
# PARQUET READING UTILITIES
# ==============================================================================

def read_parquet_refs(parquet_path: str) -> Dict:
    """Read parquet file and extract zstore references."""
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


def discover_precipitation_chunks(zstore: Dict) -> Dict:
    """Discover precipitation (tp) chunks in the zstore."""
    # GEFS precipitation path: tp/accum/surface/tp/X.0.0
    tp_prefix = 'tp/accum/surface/tp'

    chunks = []
    chunk_pattern = re.compile(rf'^{re.escape(tp_prefix)}/(\d+)\.0\.0$')

    for key in zstore.keys():
        match = chunk_pattern.match(key)
        if match:
            step_idx = int(match.group(1))
            chunks.append((step_idx, key))

    chunks.sort(key=lambda x: x[0])

    return {
        'path_prefix': tp_prefix,
        'chunks': chunks
    }


# ==============================================================================
# GRIBBERISH DATA STREAMING
# ==============================================================================

def fetch_grib_bytes(zstore: Dict, chunk_key: str, fs) -> Tuple[bytes, int]:
    """Fetch GRIB bytes from S3 using the reference."""
    ref = zstore[chunk_key]

    if isinstance(ref, list) and len(ref) >= 3:
        url, offset, length = ref[0], ref[1], ref[2]
    else:
        raise ValueError(f"Invalid reference format for {chunk_key}: {ref}")

    with fs.open(url, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_with_gribberish(grib_bytes: bytes, grid_shape=GEFS_GRID_SHAPE) -> np.ndarray:
    """Decode GRIB bytes using gribberish (fast path)."""
    if not GRIBBERISH_AVAILABLE:
        raise RuntimeError("gribberish not available")

    flat_array = gribberish.parse_grib_array(grib_bytes, 0)
    array_2d = flat_array.reshape(grid_shape)
    return array_2d


def decode_with_cfgrib(grib_bytes: bytes) -> np.ndarray:
    """Decode GRIB bytes using cfgrib (fallback path)."""
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


def decode_grib_hybrid(grib_bytes: bytes, grid_shape=GEFS_GRID_SHAPE) -> Tuple[np.ndarray, str]:
    """Decode GRIB with gribberish, fallback to cfgrib on failure."""
    if GRIBBERISH_AVAILABLE:
        try:
            array_2d = decode_with_gribberish(grib_bytes, grid_shape)
            return array_2d, 'gribberish'
        except Exception:
            pass

    # Fallback to cfgrib
    array_2d = decode_with_cfgrib(grib_bytes)
    return array_2d, 'cfgrib'


def stream_single_member_precipitation(parquet_path: str, zarr_store: zarr.Group,
                                        member_idx: int) -> bool:
    """
    Stream precipitation data for a single ensemble member and write to zarr store.

    This approach avoids memory issues by writing directly to disk.

    Args:
        parquet_path: Path to the parquet reference file
        zarr_store: Zarr group to write data to
        member_idx: Index of this member in the zarr array

    Returns:
        True if successful, False otherwise
    """
    member_name = Path(parquet_path).stem.split('_')[0]
    print(f"\n  Streaming {member_name}...")

    start_time = time.time()

    try:
        # Read parquet references
        zstore = read_parquet_refs(parquet_path)

        # Discover precipitation chunks
        tp_info = discover_precipitation_chunks(zstore)

        if not tp_info['chunks']:
            print(f"    No precipitation data found")
            return False

        print(f"    Found {len(tp_info['chunks'])} timesteps")

        # Create S3 filesystem
        fs = fsspec.filesystem('s3', anon=True)

        # Stream and decode all timesteps, writing directly to zarr
        decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}

        n_timesteps = len(tp_info['chunks'])

        for i, (step_idx, chunk_key) in enumerate(tp_info['chunks']):
            try:
                # Fetch GRIB bytes
                grib_bytes, _ = fetch_grib_bytes(zstore, chunk_key, fs)

                # Decode using hybrid approach
                array_2d, decoder = decode_grib_hybrid(grib_bytes)

                decode_stats[decoder] += 1

                # Subset to East Africa and write to zarr immediately
                data_subset = array_2d[LAT_INDICES[0]:LAT_INDICES[-1]+1,
                                       LON_INDICES[0]:LON_INDICES[-1]+1]

                zarr_store['precipitation'][member_idx, i, :, :] = data_subset.astype(np.float32)

                # Progress logging
                if i < 2 or i >= n_timesteps - 2:
                    print(f"      Step {step_idx:3d}: decoded [{decoder}]")
                elif i == 2:
                    print(f"      ... processing {n_timesteps - 4} more steps ...")

            except Exception as e:
                print(f"      Step {step_idx:3d}: FAILED - {str(e)[:50]}")
                decode_stats['failed'] += 1
                # Fill with NaN for failed chunks
                zarr_store['precipitation'][member_idx, i, :, :] = np.nan

        elapsed = time.time() - start_time
        print(f"    Completed: gribberish={decode_stats['gribberish']}, "
              f"cfgrib={decode_stats['cfgrib']}, "
              f"failed={decode_stats['failed']}, "
              f"time={elapsed:.1f}s")

        # Force garbage collection to free memory
        gc.collect()

        return True

    except Exception as e:
        print(f"    Error streaming {member_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# ZARR-BASED STORAGE FOR MEMORY EFFICIENCY
# ==============================================================================

def create_temp_zarr_store(n_members: int, n_timesteps: int, n_lats: int, n_lons: int) -> Tuple[zarr.Group, Path]:
    """Create temporary zarr store for holding ensemble data on disk."""
    # Create temporary directory
    temp_dir = TEMP_ZARR_DIR / f"gefs_{TARGET_DATE}_{TARGET_RUN}z_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Creating temporary zarr store: {temp_dir}")

    # Create zarr store
    store = zarr.open_group(str(temp_dir), mode='w')

    # Create arrays with chunking optimized for member-by-member access
    store.create_dataset(
        'precipitation',
        shape=(n_members, n_timesteps, n_lats, n_lons),
        chunks=(1, n_timesteps, n_lats, n_lons),  # One member per chunk
        dtype=np.float32,
        fill_value=np.nan,
        compressors=[BloscCodec(cname='lz4', clevel=3)]
    )

    # Store metadata
    store.attrs['n_members'] = n_members
    store.attrs['n_timesteps'] = n_timesteps
    store.attrs['n_lats'] = n_lats
    store.attrs['n_lons'] = n_lons
    store.attrs['target_date'] = TARGET_DATE
    store.attrs['target_run'] = TARGET_RUN

    return store, temp_dir


def cleanup_temp_zarr_store(temp_dir: Path):
    """Remove temporary zarr store."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"\n  Cleaned up temporary storage: {temp_dir}")


# ==============================================================================
# 24-HOUR ACCUMULATION AND PROBABILITY CALCULATIONS
# ==============================================================================

def calculate_24h_accumulations_from_zarr(zarr_store: zarr.Group, member_idx: int) -> np.ndarray:
    """
    Calculate 24-hour accumulated precipitation from 3-hourly data stored in zarr.

    Parameters:
        zarr_store: Zarr group containing precipitation data
        member_idx: Index of the member to process

    Returns:
        24-hour accumulations with shape (days, lat, lon)
    """
    # Load data for this member (loads from disk, not memory intensive)
    precip_data = zarr_store['precipitation'][member_idx, :, :, :]

    n_timesteps = precip_data.shape[0]

    # Skip timestep 0 (initial condition) and work with forecast timesteps
    forecast_data = precip_data[1:]  # Skip first timestep
    forecast_timesteps = forecast_data.shape[0]

    # Calculate number of complete 24-hour periods
    n_days = forecast_timesteps // TIMESTEPS_PER_DAY

    # Create array to store 24-hour accumulations
    daily_shape = (n_days,) + precip_data.shape[1:]
    daily_accumulations = np.zeros(daily_shape, dtype=np.float32)

    for day in range(n_days):
        start_idx = day * TIMESTEPS_PER_DAY
        end_idx = (day + 1) * TIMESTEPS_PER_DAY

        if end_idx <= forecast_timesteps:
            # Sum precipitation over 24-hour period
            daily_accumulations[day] = np.nansum(forecast_data[start_idx:end_idx], axis=0)

    return daily_accumulations


def calculate_exceedance_probabilities_from_zarr(
    zarr_store: zarr.Group,
    n_members: int,
    thresholds: List[float]
) -> Tuple[Dict, int, int]:
    """
    Calculate probability of exceeding thresholds from zarr-stored ensemble data.

    Returns:
        probabilities: dict with structure {day: {threshold: probability_array}}
        n_members: number of ensemble members
        n_days: number of forecast days
    """
    print(f"\n  Calculating 24-hour accumulations and probabilities...")

    # First pass: determine number of days from first member
    first_accum = calculate_24h_accumulations_from_zarr(zarr_store, 0)
    n_days = first_accum.shape[0]
    spatial_shape = first_accum.shape[1:]

    print(f"    Forecast days: {n_days}")
    print(f"    Spatial shape: {spatial_shape}")

    # Initialize probability accumulators (counting exceeding members)
    exceedance_counts = {}
    for day in range(n_days):
        exceedance_counts[day] = {}
        for threshold in thresholds:
            exceedance_counts[day][threshold] = np.zeros(spatial_shape, dtype=np.int32)

    # Process each member
    valid_members = 0
    for member_idx in range(n_members):
        try:
            # Calculate 24h accumulations for this member
            daily_accum = calculate_24h_accumulations_from_zarr(zarr_store, member_idx)

            # Check if data is valid
            if np.all(np.isnan(daily_accum)):
                continue

            valid_members += 1

            # Count exceedances for each day and threshold
            for day in range(min(n_days, daily_accum.shape[0])):
                for threshold in thresholds:
                    exceedance_counts[day][threshold] += (daily_accum[day] >= threshold).astype(np.int32)

            if (member_idx + 1) % 5 == 0:
                print(f"    Processed {member_idx + 1}/{n_members} members")

            # Force garbage collection
            del daily_accum
            gc.collect()

        except Exception as e:
            print(f"    Warning: Failed to process member {member_idx}: {e}")

    print(f"    Valid members: {valid_members}/{n_members}")

    # Convert counts to probabilities
    probabilities = {}
    for day in range(n_days):
        probabilities[day] = {}
        for threshold in thresholds:
            if valid_members > 0:
                probabilities[day][threshold] = (exceedance_counts[day][threshold] / valid_members) * 100
            else:
                probabilities[day][threshold] = np.zeros(spatial_shape, dtype=np.float32)

    return probabilities, valid_members, n_days


# ==============================================================================
# PLOTTING FUNCTIONS (Following run_gefs_24h_accumulation.py style)
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
    output_dir: Path = None
) -> Optional[str]:
    """
    Create multi-panel plot showing 24h rainfall exceedance probabilities.

    The plot has rows for each 24-hour period and columns for each threshold.
    Follows the style from run_gefs_24h_accumulation.py
    """
    if not PLOTTING_AVAILABLE:
        print("  Plotting not available - skipping visualization")
        return None

    # Create figure
    fig, axes = plt.subplots(n_days, len(THRESHOLDS_24H),
                            figsize=(4*len(THRESHOLDS_24H), 4*n_days),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    # Handle single row or column
    if n_days == 1:
        axes = axes.reshape(1, -1)
    elif len(THRESHOLDS_24H) == 1:
        axes = axes.reshape(-1, 1)

    # Common color levels
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933',
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']

    # Load boundaries
    gdf = load_geojson_boundaries(BOUNDARY_JSON)

    # Calculate base datetime (model run time)
    base_datetime = MODEL_DATE + timedelta(hours=MODEL_RUN_HOUR)

    # Plot each panel
    for day in range(n_days):
        for t_idx, threshold in enumerate(THRESHOLDS_24H):
            ax = axes[day, t_idx]

            # Get probability data
            prob_data = probabilities[day][threshold]

            # Create contour plot
            cf = ax.contourf(lons, lats, prob_data, levels=levels, colors=colors,
                           transform=ccrs.PlateCarree(), extend='neither')

            # Add 50% contour line
            ax.contour(lons, lats, prob_data, levels=[50],
                      colors='black', linewidths=1, alpha=0.7,
                      transform=ccrs.PlateCarree())

            # Add boundaries from GeoJSON
            if gdf is not None:
                ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                                facecolor="none", edgecolor="black", linewidth=0.8)

            # Add coastlines and features (as backup)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', alpha=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.1)

            # Set extent
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

            # Calculate actual forecast dates
            start_datetime = base_datetime + timedelta(hours=day*24)
            end_datetime = base_datetime + timedelta(hours=(day+1)*24)

            # Convert to East Africa Time
            start_eat = start_datetime + timedelta(hours=EAT_OFFSET)
            end_eat = end_datetime + timedelta(hours=EAT_OFFSET)

            # Format title based on column
            max_prob = np.nanmax(prob_data)

            if t_idx == 0:  # First column - show full date information
                if day == 0:  # First row - show threshold header
                    title = f'>{threshold}mm\n'
                else:
                    title = ''

                # Add date range in EAT
                title += f'{start_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'to {end_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'Max: {max_prob:.0f}%'

                ax.set_title(title, fontsize=9, pad=10)
            else:
                # Other columns - just show threshold and max probability
                if day == 0:
                    ax.set_title(f'>{threshold}mm\nMax: {max_prob:.0f}%', fontsize=10)
                else:
                    ax.set_title(f'Max: {max_prob:.0f}%', fontsize=10)

            # Add gridlines to first column
            if t_idx == 0:
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                linewidth=0.3, color='gray', alpha=0.3, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 8}
                gl.ylabel_style = {'size': 8}

    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)

    # Overall title with model run information
    model_run_str = f'{MODEL_DATE.strftime("%Y-%m-%d")} {MODEL_RUN_HOUR:02d}:00 UTC'
    model_run_eat = f'{MODEL_DATE.strftime("%Y-%m-%d")} {(MODEL_RUN_HOUR + EAT_OFFSET) % 24:02d}:00 EAT'

    fig.suptitle(f'GEFS 24-Hour Rainfall Exceedance Probabilities\n'
                 f'Model Run: {model_run_str} ({model_run_eat})\n'
                 f'Based on {n_members} ensemble members | Coverage: {LAT_MIN}N-{LAT_MAX}N, {LON_MIN}E-{LON_MAX}E',
                 fontsize=14, y=0.98)

    # Save figure with date and run info
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'gefs_24h_probability_{TARGET_DATE}_{TARGET_RUN}z_all_thresholds.png'
    else:
        output_file = f'gefs_24h_probability_{TARGET_DATE}_{TARGET_RUN}z_all_thresholds.png'

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {output_file}")
    plt.close()

    return str(output_file)


# ==============================================================================
# MAIN ROUTINE
# ==============================================================================

def main():
    """Main processing routine with memory-efficient zarr-based storage."""
    print("\nStarting GEFS Data Streaming V2 (Memory Efficient)\n")

    start_time = time.time()
    temp_zarr_path = None

    try:
        # Check if parquet directory exists
        if not PARQUET_DIR.exists():
            print(f"Error: Parquet directory {PARQUET_DIR} not found!")
            print(f"Run run_gefs_tutorial.py first to create parquet files.")
            return False

        # Find parquet files
        parquet_files = sorted(PARQUET_DIR.glob(f"gep*_{TARGET_DATE}_{TARGET_RUN}z.parquet"))

        if not parquet_files:
            print(f"Error: No parquet files found in {PARQUET_DIR}")
            print(f"Looking for pattern: gep*_{TARGET_DATE}_{TARGET_RUN}z.parquet")
            return False

        print(f"Found {len(parquet_files)} parquet files")

        # Filter to requested members
        valid_parquet_files = []
        for pf in parquet_files:
            member_name = pf.stem.split('_')[0]
            if member_name in ENSEMBLE_MEMBERS:
                valid_parquet_files.append(pf)

        print(f"Using {len(valid_parquet_files)} ensemble members")

        if len(valid_parquet_files) == 0:
            print("Error: No matching ensemble member files found!")
            return False

        # Determine number of timesteps from first parquet file
        print("\n[Step 1] Discovering data structure...")
        zstore = read_parquet_refs(str(valid_parquet_files[0]))
        tp_info = discover_precipitation_chunks(zstore)
        n_timesteps = len(tp_info['chunks'])
        print(f"  Timesteps per member: {n_timesteps}")

        # Create temporary zarr store
        print("\n[Step 2] Creating temporary disk storage...")
        n_members = len(valid_parquet_files)
        n_lats = len(EA_LATS)
        n_lons = len(EA_LONS)

        zarr_store, temp_zarr_path = create_temp_zarr_store(n_members, n_timesteps, n_lats, n_lons)
        print(f"  Storage shape: ({n_members}, {n_timesteps}, {n_lats}, {n_lons})")

        # Stream data for all ensemble members
        print("\n[Step 3] Streaming Precipitation Data")
        print("="*70)

        if GRIBBERISH_AVAILABLE:
            print("Using GRIBBERISH for fast data streaming (~80x faster)")
        else:
            print("Using CFGRIB for data streaming (gribberish not available)")

        successful_members = 0
        for member_idx, pf in enumerate(valid_parquet_files):
            success = stream_single_member_precipitation(str(pf), zarr_store, member_idx)
            if success:
                successful_members += 1

        print(f"\nSuccessfully loaded {successful_members} ensemble members")

        if successful_members == 0:
            print("Error: No data loaded successfully!")
            return False

        # Calculate exceedance probabilities
        print("\n[Step 4] Calculating Exceedance Probabilities")
        print("="*70)

        probabilities, n_valid_members, n_days = calculate_exceedance_probabilities_from_zarr(
            zarr_store, n_members, THRESHOLDS_24H
        )

        print(f"  Days: {n_days}")
        print(f"  Thresholds: {THRESHOLDS_24H} mm")
        print(f"  Valid Members: {n_valid_members}")

        # Print summary statistics
        print("\n  Summary Statistics:")
        for day in range(min(3, n_days)):  # Show first 3 days
            print(f"\n    Day {day+1} (Hours {day*24}-{(day+1)*24}):")
            for threshold in THRESHOLDS_24H[:3]:  # Show first 3 thresholds
                max_prob = np.nanmax(probabilities[day][threshold])
                area_50 = np.sum(probabilities[day][threshold] >= 50)
                print(f"      >{threshold:3d}mm: Max={max_prob:5.1f}%, P>=50% at {area_50} points")

        # Create visualization
        print("\n[Step 5] Creating Visualization")
        print("="*70)

        plot_file = create_24h_probability_plot(
            probabilities, EA_LONS, EA_LATS, n_valid_members, n_days, OUTPUT_DIR
        )

        # Summary
        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Target Date: {TARGET_DATE} {TARGET_RUN}Z")
        print(f"  Ensemble Members: {n_valid_members}")
        print(f"  Forecast Days: {n_days}")
        print(f"  Thresholds: {THRESHOLDS_24H}")
        print(f"  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

        if plot_file:
            print(f"\nOutput Plot: {plot_file}")

        print("\nNext Steps:")
        print("  1. Examine the probability plot for flood risk assessment")
        print("  2. Adjust THRESHOLDS_24H for different rainfall categories")
        print("  3. Modify LAT/LON bounds for different regions")

        return True

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup temporary storage
        if temp_zarr_path and temp_zarr_path.exists():
            cleanup_temp_zarr_store(temp_zarr_path)


if __name__ == "__main__":
    success = main()
    if success:
        print("\nData streaming completed successfully!")
    else:
        print("\nData streaming failed. Check error messages above.")
        sys.exit(1)
