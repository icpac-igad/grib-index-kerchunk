#!/usr/bin/env python3
"""
GEFS Data Streaming and 24-Hour Precipitation Exceedance Plotting
===================================================================

This script demonstrates how to use the parquet reference files created by
run_gefs_tutorial.py to stream GEFS ensemble data and create precipitation
exceedance probability plots.

The script uses the gribberish library for fast GRIB decoding (~80x faster
than cfgrib) to efficiently stream precipitation data from AWS S3.

Prerequisites:
    pip install kerchunk zarr xarray pandas numpy fsspec s3fs gribberish
    pip install matplotlib cartopy geopandas

Usage:
    python run_gefs_data_streaming.py

Features:
    - Uses gribberish for fast GRIB decoding (Rust-based, ~25ms per chunk)
    - Falls back to cfgrib if gribberish fails
    - Calculates 24-hour rainfall accumulations from ensemble data
    - Computes exceedance probabilities for multiple thresholds
    - Creates multi-panel probability plots with cartopy

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
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import re

import fsspec

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
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/cartopy not available for plotting")
    print("Install with: pip install matplotlib cartopy")

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

print("="*70)
print("GEFS Data Streaming and Precipitation Analysis")
print("="*70)
print(f"Target Date: {TARGET_DATE}")
print(f"Model Run: {TARGET_RUN}Z")
print(f"Ensemble Members: {len(ENSEMBLE_MEMBERS)}")
print(f"Gribberish Available: {GRIBBERISH_AVAILABLE}")
print(f"Plotting Available: {PLOTTING_AVAILABLE}")
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


def stream_precipitation_gribberish(parquet_path: str) -> Optional[np.ndarray]:
    """
    Stream precipitation data from parquet using gribberish for fast decoding.

    Returns:
        numpy array with shape (time, lat, lon) for the regional subset
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
            return None

        print(f"    Found {len(tp_info['chunks'])} timesteps")

        # Create S3 filesystem
        fs = fsspec.filesystem('s3', anon=True)

        # Stream and decode all timesteps
        timestep_data = []
        decode_stats = {'gribberish': 0, 'cfgrib': 0, 'failed': 0}

        for i, (step_idx, chunk_key) in enumerate(tp_info['chunks']):
            try:
                # Fetch GRIB bytes
                grib_bytes, _ = fetch_grib_bytes(zstore, chunk_key, fs)

                # Decode using hybrid approach
                array_2d, decoder = decode_grib_hybrid(grib_bytes)

                decode_stats[decoder] += 1
                timestep_data.append(array_2d)

                # Progress logging
                if i < 2 or i >= len(tp_info['chunks']) - 2:
                    print(f"      Step {step_idx:3d}: decoded [{decoder}]")
                elif i == 2:
                    print(f"      ... processing {len(tp_info['chunks']) - 4} more steps ...")

            except Exception as e:
                print(f"      Step {step_idx:3d}: FAILED - {str(e)[:50]}")
                decode_stats['failed'] += 1
                # Fill with NaN for failed chunks
                timestep_data.append(np.full(GEFS_GRID_SHAPE, np.nan, dtype=np.float32))

        # Stack into 3D array (time, lat, lon)
        data_3d = np.stack(timestep_data, axis=0).astype(np.float32)

        # Subset to East Africa region
        lat_mask = (GEFS_LATS >= LAT_MIN) & (GEFS_LATS <= LAT_MAX)
        lon_mask = (GEFS_LONS >= LON_MIN) & (GEFS_LONS <= LON_MAX)

        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        data_subset = data_3d[:, lat_indices[0]:lat_indices[-1]+1,
                              lon_indices[0]:lon_indices[-1]+1]
        lats_subset = GEFS_LATS[lat_indices[0]:lat_indices[-1]+1]
        lons_subset = GEFS_LONS[lon_indices[0]:lon_indices[-1]+1]

        elapsed = time.time() - start_time
        print(f"    Completed: shape={data_subset.shape}, "
              f"gribberish={decode_stats['gribberish']}, "
              f"cfgrib={decode_stats['cfgrib']}, "
              f"failed={decode_stats['failed']}, "
              f"time={elapsed:.1f}s")

        return data_subset, lats_subset, lons_subset

    except Exception as e:
        print(f"    Error streaming {member_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ==============================================================================
# ALTERNATIVE: FSSPEC-BASED STREAMING (slower but works without gribberish)
# ==============================================================================

def stream_precipitation_fsspec(parquet_path: str) -> Optional[np.ndarray]:
    """
    Stream precipitation data using fsspec reference filesystem.
    This is the traditional approach, slower but doesn't require gribberish.
    """
    member_name = Path(parquet_path).stem.split('_')[0]
    print(f"\n  Streaming {member_name} (fsspec method)...")

    start_time = time.time()

    try:
        # Read parquet references
        zstore = read_parquet_refs(parquet_path)

        # Remove version key if present
        if 'version' in zstore:
            del zstore['version']

        # Create reference filesystem
        fs = fsspec.filesystem(
            "reference",
            fo={'refs': zstore, 'version': 1},
            remote_protocol='s3',
            remote_options={'anon': True}
        )
        mapper = fs.get_mapper("")

        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

        # Navigate to precipitation data
        data_var = dt['/tp/accum/surface'].ds['tp']

        # Extract region
        regional_data = data_var.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX)
        )

        # Compute numpy array
        regional_numpy = regional_data.compute()

        elapsed = time.time() - start_time
        print(f"    Completed: shape={regional_numpy.shape}, time={elapsed:.1f}s")

        lats = regional_data.latitude.values
        lons = regional_data.longitude.values

        return regional_numpy.values, lats, lons

    except Exception as e:
        print(f"    Error streaming {member_name}: {e}")
        return None, None, None


# ==============================================================================
# 24-HOUR ACCUMULATION AND PROBABILITY CALCULATIONS
# ==============================================================================

def calculate_24h_accumulations(precip_data: np.ndarray) -> np.ndarray:
    """
    Calculate 24-hour accumulated precipitation from 3-hourly data.

    Parameters:
        precip_data: numpy array with shape (time, lat, lon)

    Returns:
        24-hour accumulations with shape (days, lat, lon)
    """
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


def calculate_exceedance_probabilities(
    ensemble_24h: Dict[str, np.ndarray],
    thresholds: List[float]
) -> Tuple[Dict, int]:
    """
    Calculate probability of exceeding thresholds for ensemble data.

    Returns:
        probabilities: dict with structure {day: {threshold: probability_array}}
        n_members: number of ensemble members
    """
    # Get dimensions from first member
    first_member = list(ensemble_24h.values())[0]
    n_days = first_member.shape[0]
    n_members = len(ensemble_24h)

    probabilities = {}

    for day in range(n_days):
        probabilities[day] = {}

        # Stack all member data for this day
        day_data = []
        for member_data in ensemble_24h.values():
            if member_data is not None:
                day_data.append(member_data[day])

        if day_data:
            day_stack = np.stack(day_data, axis=0)

            # Calculate probabilities for each threshold
            for threshold in thresholds:
                exceedance_count = np.sum(day_stack >= threshold, axis=0)
                probability = (exceedance_count / len(day_data)) * 100
                probabilities[day][threshold] = probability

    return probabilities, n_members


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def create_probability_plot(
    probabilities: Dict,
    lons: np.ndarray,
    lats: np.ndarray,
    n_members: int,
    n_days: int,
    output_dir: Path = None
) -> Optional[str]:
    """Create multi-panel plot showing 24h rainfall exceedance probabilities."""

    if not PLOTTING_AVAILABLE:
        print("  Plotting not available - skipping visualization")
        return None

    # Parse date for titles
    model_date = datetime.strptime(TARGET_DATE, "%Y%m%d")
    model_run_hour = int(TARGET_RUN)
    base_datetime = model_date + timedelta(hours=model_run_hour)

    # Create figure
    fig, axes = plt.subplots(
        n_days, len(THRESHOLDS_24H),
        figsize=(4*len(THRESHOLDS_24H), 4*n_days),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Handle single row case
    if n_days == 1:
        axes = axes.reshape(1, -1)

    # Color levels
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933',
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']

    # Plot each panel
    for day in range(n_days):
        for t_idx, threshold in enumerate(THRESHOLDS_24H):
            ax = axes[day, t_idx]

            prob_data = probabilities[day][threshold]

            # Create contour plot
            cf = ax.contourf(lons, lats, prob_data, levels=levels, colors=colors,
                           transform=ccrs.PlateCarree(), extend='neither')

            # Add 50% contour line
            ax.contour(lons, lats, prob_data, levels=[50],
                      colors='black', linewidths=1, alpha=0.7,
                      transform=ccrs.PlateCarree())

            # Add coastlines and features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.1)

            # Set extent
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

            # Calculate dates
            start_datetime = base_datetime + timedelta(hours=day*24)
            end_datetime = base_datetime + timedelta(hours=(day+1)*24)
            start_eat = start_datetime + timedelta(hours=EAT_OFFSET)
            end_eat = end_datetime + timedelta(hours=EAT_OFFSET)

            max_prob = np.nanmax(prob_data)

            # Titles
            if t_idx == 0:
                if day == 0:
                    title = f'>{threshold}mm\n'
                else:
                    title = ''
                title += f'{start_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'to {end_eat.strftime("%Y-%m-%d %H:%M")} EAT\n'
                title += f'Max: {max_prob:.0f}%'
                ax.set_title(title, fontsize=9, pad=10)

                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                                 color='gray', alpha=0.3, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
            else:
                if day == 0:
                    ax.set_title(f'>{threshold}mm\nMax: {max_prob:.0f}%', fontsize=10)
                else:
                    ax.set_title(f'Max: {max_prob:.0f}%', fontsize=10)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)

    # Overall title
    model_run_str = f'{model_date.strftime("%Y-%m-%d")} {model_run_hour:02d}:00 UTC'
    model_run_eat = f'{model_date.strftime("%Y-%m-%d")} {(model_run_hour + EAT_OFFSET) % 24:02d}:00 EAT'

    fig.suptitle(
        f'GEFS 24-Hour Rainfall Exceedance Probabilities\n'
        f'Model Run: {model_run_str} ({model_run_eat})\n'
        f'Based on {n_members} ensemble members | Region: East Africa',
        fontsize=14, y=0.98
    )

    # Save figure
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'gefs_24h_probability_{TARGET_DATE}_{TARGET_RUN}z.png'
    else:
        output_file = f'gefs_24h_probability_{TARGET_DATE}_{TARGET_RUN}z.png'

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {output_file}")
    plt.close()

    return str(output_file)


# ==============================================================================
# MAIN ROUTINE
# ==============================================================================

def main():
    """Main processing routine."""
    print("\nStarting GEFS Data Streaming and Analysis\n")

    start_time = time.time()

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

    # Select streaming method based on gribberish availability
    if GRIBBERISH_AVAILABLE:
        print("\nUsing GRIBBERISH for fast data streaming (~80x faster)")
        stream_func = stream_precipitation_gribberish
    else:
        print("\nUsing FSSPEC for data streaming (gribberish not available)")
        stream_func = stream_precipitation_fsspec

    # Stream data for all ensemble members
    print("\n[Step 1] Streaming Precipitation Data")
    print("="*70)

    ensemble_data = {}
    lats = None
    lons = None

    for pf in parquet_files:
        member_name = pf.stem.split('_')[0]

        if member_name not in ENSEMBLE_MEMBERS:
            continue

        result = stream_func(str(pf))

        if result[0] is not None:
            data, member_lats, member_lons = result
            ensemble_data[member_name] = data

            if lats is None:
                lats = member_lats
                lons = member_lons

    print(f"\nSuccessfully loaded {len(ensemble_data)} ensemble members")

    if len(ensemble_data) == 0:
        print("Error: No data loaded successfully!")
        return False

    # Calculate 24-hour accumulations
    print("\n[Step 2] Calculating 24-Hour Accumulations")
    print("="*70)

    ensemble_24h = {}

    for member, data in ensemble_data.items():
        daily_accum = calculate_24h_accumulations(data)
        ensemble_24h[member] = daily_accum
        print(f"  {member}: {daily_accum.shape[0]} days processed")

    # Calculate exceedance probabilities
    print("\n[Step 3] Calculating Exceedance Probabilities")
    print("="*70)

    probabilities, n_members = calculate_exceedance_probabilities(
        ensemble_24h, THRESHOLDS_24H
    )

    n_days = len(probabilities)
    print(f"  Days: {n_days}")
    print(f"  Thresholds: {THRESHOLDS_24H} mm")
    print(f"  Members: {n_members}")

    # Print summary statistics
    print("\n  Summary Statistics:")
    for day in range(min(3, n_days)):  # Show first 3 days
        print(f"\n    Day {day+1}:")
        for threshold in THRESHOLDS_24H[:3]:  # Show first 3 thresholds
            max_prob = np.nanmax(probabilities[day][threshold])
            area_50 = np.sum(probabilities[day][threshold] >= 50)
            print(f"      >{threshold:3d}mm: Max={max_prob:5.1f}%, P>=50% at {area_50} points")

    # Create visualization
    print("\n[Step 4] Creating Visualization")
    print("="*70)

    plot_file = create_probability_plot(
        probabilities, lons, lats, n_members, n_days, OUTPUT_DIR
    )

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Target Date: {TARGET_DATE} {TARGET_RUN}Z")
    print(f"  Ensemble Members: {n_members}")
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


if __name__ == "__main__":
    success = main()
    if success:
        print("\nData streaming completed successfully!")
    else:
        print("\nData streaming failed. Check error messages above.")
        sys.exit(1)
