#!/usr/bin/env python3
"""
ECMWF Data Streaming with Gribberish
=====================================

This script demonstrates how to stream ECMWF data using the gribberish library
for fast GRIB decoding (~80x faster than cfgrib).

Uses parquet files created by run_ecmwf_tutorial.py with step_XXX format:
  step_XXX/varname/level/member/0.0.0

Creates 24-hour precipitation exceedance probability plots for East Africa.

Prerequisites:
    pip install gribberish matplotlib cartopy numpy pandas fsspec s3fs

Usage:
    python run_ecmwf_data_streaming.py

Author: ICPAC GIK Team
"""

import os
import sys
import json
import warnings
import time
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import fsspec

# Try to import gribberish
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available. Install with: pip install gribberish")

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/cartopy not available for plotting")

warnings.filterwarnings('ignore')

# Set up anonymous S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input/Output directories
PARQUET_DIR = Path("output_parquet")
OUTPUT_DIR = Path("output_plots")

# ECMWF IFS grid specification (0.25 degree global)
ECMWF_GRID_SHAPE = (721, 1440)  # lat x lon
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# East Africa region
EA_LAT_MIN, EA_LAT_MAX = -12, 23
EA_LON_MIN, EA_LON_MAX = 21, 53

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# Precipitation thresholds for 24h accumulation (mm/day)
PRECIP_THRESHOLDS_24H = [20, 40, 60, 80, 100]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

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
                value = value

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
    Find all chunks for a variable using the step_XXX format.

    ECMWF format: step_XXX/varname/level/member/0.0.0

    Returns list of (step_hours, key, reference) tuples.
    """
    chunks = []

    # Pattern: step_XXX/varname/level/member/0.0.0
    pattern = re.compile(rf'^step_(\d+)/{re.escape(var_name)}/{re.escape(level)}/[^/]+/0\.0\.0$')

    for key, value in zstore.items():
        match = pattern.match(key)
        if match and isinstance(value, list) and len(value) >= 3:
            step_hours = int(match.group(1))
            chunks.append((step_hours, key, value))

    chunks.sort(key=lambda x: x[0])
    return chunks


def fetch_grib_bytes(ref: List, fs) -> Tuple[bytes, int]:
    """Fetch GRIB bytes from S3 using the reference."""
    if not isinstance(ref, list) or len(ref) < 3:
        raise ValueError(f"Invalid reference format: {ref}")

    url, offset, length = ref[0], ref[1], ref[2]

    # Add .grib2 extension if missing (ECMWF S3 files require it)
    if not url.endswith('.grib2'):
        url = url + '.grib2'

    with fs.open(url, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_with_gribberish_subprocess(grib_bytes: bytes, grid_shape: Tuple[int, int] = ECMWF_GRID_SHAPE) -> Tuple[Optional[np.ndarray], bool]:
    """Decode GRIB using gribberish in a subprocess to safely catch Rust panics."""
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

    if os.path.exists(tmp_in_path):
        os.unlink(tmp_in_path)
    if os.path.exists(tmp_out_path):
        os.unlink(tmp_out_path)

    return None, False


def decode_grib(grib_bytes: bytes) -> Tuple[np.ndarray, str]:
    """Decode GRIB with gribberish, fallback to cfgrib on failure."""
    if GRIBBERISH_AVAILABLE:
        array, success = decode_with_gribberish_subprocess(grib_bytes)
        if success:
            return array, 'gribberish'

    # Fallback to cfgrib
    try:
        import xarray as xr

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

        return array_2d, 'cfgrib'
    except Exception as e:
        raise RuntimeError(f"Failed to decode GRIB: {e}")


def get_ea_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get East Africa subset indices and coordinates."""
    lat_mask = (ECMWF_LATS >= EA_LAT_MIN) & (ECMWF_LATS <= EA_LAT_MAX)
    lon_mask = (ECMWF_LONS >= EA_LON_MIN) & (ECMWF_LONS <= EA_LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lats = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]

    return lat_indices, lon_indices, lats, lons


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
        print("    Warning: No 24h periods found, using cumulative values converted to mm")
        return data_3d * 1000.0, timesteps

    return np.stack(periods_24h, axis=0).astype(np.float32), valid_hours


def load_member_precipitation(parquet_file: Path, timesteps: List[int], fs,
                              lat_indices: np.ndarray, lon_indices: np.ndarray) -> Tuple[Optional[np.ndarray], str, List[int]]:
    """
    Load precipitation data for a single ensemble member using gribberish.

    Returns 3D array (time, lat, lon) for East Africa region, member label, and actual timesteps loaded.
    """
    member = parquet_file.stem
    # Extract member label from filename
    match = re.search(r'(control|ens_?\d+)', member)
    member_label = match.group(1) if match else member

    try:
        zstore = read_parquet_to_zstore(str(parquet_file))

        # Find precipitation chunks using step_XXX format
        chunks = find_variable_chunks_step_format(zstore, 'tp', level='sfc')

        if not chunks:
            print(f"    {member_label}: No tp chunks found")
            return None, member_label, []

        # Create chunk lookup by step
        chunk_lookup = {step: ref for step, key, ref in chunks}

        # Determine which timesteps to load
        available_steps = sorted(chunk_lookup.keys())
        steps_to_load = [ts for ts in timesteps if ts in chunk_lookup]

        if not steps_to_load:
            print(f"    {member_label}: No matching timesteps found (available: {available_steps[:5]}...)")
            return None, member_label, []

        # Load data for each timestep
        n_lats = lat_indices[-1] - lat_indices[0] + 1
        n_lons = lon_indices[-1] - lon_indices[0] + 1
        data_3d = np.full((len(steps_to_load), n_lats, n_lons), np.nan, dtype=np.float32)

        decode_times = []
        for i, ts in enumerate(steps_to_load):
            ref = chunk_lookup[ts]

            t0 = time.time()
            grib_bytes, _ = fetch_grib_bytes(ref, fs)
            data_2d, decoder = decode_grib(grib_bytes)
            decode_times.append((time.time() - t0) * 1000)

            # Subset to East Africa
            data_3d[i] = data_2d[lat_indices[0]:lat_indices[-1]+1,
                                 lon_indices[0]:lon_indices[-1]+1]

        avg_decode = np.mean(decode_times)
        max_val = np.nanmax(data_3d) * 1000  # Convert to mm for display
        print(f"    {member_label}: Loaded {len(steps_to_load)} timesteps, avg {avg_decode:.1f}ms/chunk, max={max_val:.2f}mm")

        return data_3d, member_label, steps_to_load

    except Exception as e:
        print(f"    {member_label}: Error - {type(e).__name__}: {str(e)[:50]}")
        return None, member_label, []


def create_probability_plot(ensemble_data_24h: Dict[str, np.ndarray],
                           lats: np.ndarray, lons: np.ndarray,
                           model_date: datetime, run_hour: int,
                           timestep_24h: int, thresholds: List[float],
                           output_dir: Path) -> str:
    """Create 24h precipitation exceedance probability plot."""
    if not PLOTTING_AVAILABLE:
        print("  Plotting libraries not available")
        return None

    n_members = len(ensemble_data_24h)
    if n_members == 0:
        return None

    # Stack all member data
    members = sorted(ensemble_data_24h.keys())
    data_stack = np.stack([ensemble_data_24h[m] for m in members], axis=0)

    # Calculate time info
    forecast_hour = timestep_24h
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    # Create figure
    n_thresholds = len(thresholds)
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

        # Calculate probability of exceeding threshold
        exceedance_count = np.sum(data_stack >= threshold, axis=0)
        probability = (exceedance_count / n_members) * 100

        # Plot
        im = ax.contourf(lons, lats, probability, levels=prob_levels,
                        colors=prob_colors, transform=ccrs.PlateCarree())

        # Add 50% contour
        ax.contour(lons, lats, probability, levels=[50], colors='black',
                  linewidths=1.5, transform=ccrs.PlateCarree())

        # Map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

        # Title
        max_prob = np.nanmax(probability)
        ax.set_title(f'P(24h precip > {threshold:.0f}mm)\nMax: {max_prob:.0f}%', fontsize=11)

    # Hide unused axes
    for idx in range(len(thresholds), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)

    # Main title
    date_str = model_date.strftime('%Y%m%d')
    fig.suptitle(f'ECMWF IFS 24h Precipitation Exceedance Probabilities (Gribberish Decoder)\n'
                f'{date_str} {run_hour:02d}Z Run - T+{forecast_hour}h (24h ending)\n'
                f'Valid: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT) | {n_members} members',
                fontsize=14, y=0.98)

    # Save
    output_file = output_dir / f'ecmwf_24h_probability_{date_str}_{run_hour:02d}z.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved: {output_file}")
    return str(output_file)


def create_ensemble_mean_plot(ensemble_data_24h: Dict[str, np.ndarray],
                              lats: np.ndarray, lons: np.ndarray,
                              model_date: datetime, run_hour: int,
                              timestep_24h: int, output_dir: Path) -> str:
    """Create ensemble mean 24h precipitation plot."""
    if not PLOTTING_AVAILABLE:
        return None

    n_members = len(ensemble_data_24h)
    if n_members == 0:
        return None

    # Calculate ensemble mean
    members = sorted(ensemble_data_24h.keys())
    data_stack = np.stack([ensemble_data_24h[m] for m in members], axis=0)
    ensemble_mean = np.nanmean(data_stack, axis=0)

    # Calculate time info
    forecast_hour = timestep_24h
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Colormap and levels
    vmax = max(10, np.ceil(np.nanmax(ensemble_mean) * 1.1 / 10) * 10)
    levels = np.linspace(0, vmax, 11)

    cf = ax.contourf(lons, lats, ensemble_mean, levels=levels, cmap='Blues',
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
    cbar.set_label('24h Precipitation (mm)', rotation=270, labelpad=20)

    # Title
    date_str = model_date.strftime('%Y%m%d')
    plt.title(f'ECMWF IFS Ensemble Mean 24h Precipitation (Gribberish Decoder)\n'
              f'{date_str} {run_hour:02d}Z Run - T+{forecast_hour}h (24h ending)\n'
              f'Valid: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT)\n'
              f'{n_members} members | Max: {np.nanmax(ensemble_mean):.1f}mm',
              fontsize=12)

    # Save
    output_file = output_dir / f'ecmwf_24h_ensemble_mean_{date_str}_{run_hour:02d}z.png'
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: {output_file}")
    return str(output_file)


# ==============================================================================
# MAIN ROUTINE
# ==============================================================================

def main():
    """Main processing function."""
    print("="*70)
    print("ECMWF Data Streaming with Gribberish")
    print("="*70)

    if not GRIBBERISH_AVAILABLE:
        print("\nWARNING: gribberish is not installed, will use cfgrib fallback (slower)")
        print("Install with: pip install gribberish")

    start_time = time.time()

    # Check parquet directory
    if not PARQUET_DIR.exists():
        print(f"\nERROR: Parquet directory not found: {PARQUET_DIR}")
        print("Run run_ecmwf_tutorial.py first to create parquet files.")
        return False

    # Find parquet files (stage3_ens_XX_final.parquet or stage3_control_final.parquet)
    parquet_files = sorted(PARQUET_DIR.glob("stage3_ens_*_final.parquet"))
    control_file = PARQUET_DIR / "stage3_control_final.parquet"
    if control_file.exists():
        parquet_files.insert(0, control_file)

    if not parquet_files:
        print(f"\nERROR: No parquet files found in {PARQUET_DIR}")
        print("Expected files named: stage3_control_final.parquet, stage3_ens_XX_final.parquet")
        return False

    print(f"\nFound {len(parquet_files)} parquet files")
    for pf in parquet_files[:3]:
        print(f"  - {pf.name}")
    if len(parquet_files) > 3:
        print(f"  ... and {len(parquet_files) - 3} more")

    # Try to extract date from first parquet file
    model_date = datetime.now()
    run_hour = 0

    try:
        zstore = read_parquet_to_zstore(str(parquet_files[0]))
        for key, value in zstore.items():
            if isinstance(value, list) and len(value) >= 1:
                url = str(value[0])
                if 's3://' in url:
                    match = re.search(r'/(\d{8})/(\d{2})z/', url)
                    if match:
                        model_date = datetime.strptime(match.group(1), '%Y%m%d')
                        run_hour = int(match.group(2))
                        print(f"\nExtracted model date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")
                        break
    except Exception as e:
        print(f"  Could not extract date: {e}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create S3 filesystem
    fs = fsspec.filesystem('s3', anon=True)

    # Get East Africa indices
    lat_indices, lon_indices, lats, lons = get_ea_indices()
    print(f"\nEast Africa region: {len(lats)} x {len(lons)} grid points")

    # Define timesteps to load (need at least 0-24h for 24h accumulation)
    # Load more timesteps to get multiple 24h periods
    timesteps = list(range(0, 73, 3))  # 0, 3, 6, ..., 72h

    print(f"\nLoading precipitation data for timesteps: {timesteps[0]}-{timesteps[-1]}h")
    print(f"This will create 24h periods ending at: T+24, T+27, T+30, ..., T+72h")

    # Load precipitation data for each member
    print(f"\nLoading ensemble members...")
    raw_member_data = {}

    for pf in parquet_files:
        data, member_label, actual_timesteps = load_member_precipitation(pf, timesteps, fs, lat_indices, lon_indices)
        if data is not None:
            raw_member_data[member_label] = (data, actual_timesteps)

    if len(raw_member_data) == 0:
        print("\nERROR: No precipitation data loaded!")
        print("This might be because:")
        print("  1. Parquet files don't have step_XXX/tp/sfc/member/0.0.0 format")
        print("  2. S3 data access failed")
        print("\nChecking parquet structure...")

        # Debug: show what's in the parquet file
        try:
            zstore = read_parquet_to_zstore(str(parquet_files[0]))
            step_keys = [k for k in zstore.keys() if k.startswith('step_')]
            print(f"  Found {len(step_keys)} step_XXX keys")
            if step_keys:
                print(f"  Sample keys: {step_keys[:3]}")

            tp_keys = [k for k in zstore.keys() if '/tp/' in k]
            print(f"  Found {len(tp_keys)} keys with '/tp/'")
            if tp_keys:
                print(f"  Sample tp keys: {tp_keys[:3]}")
        except Exception as e:
            print(f"  Error reading parquet: {e}")

        return False

    print(f"\nSuccessfully loaded: {len(raw_member_data)} members")

    # Convert cumulative precipitation to 24-hour accumulation
    print(f"\nConverting to 24h precipitation accumulation...")
    ensemble_data_24h = {}
    valid_24h_hours = None

    for member_label, (data, ts_list) in raw_member_data.items():
        data_24h, hours_24h = convert_cumulative_to_24h_precip(data, ts_list)

        if valid_24h_hours is None:
            valid_24h_hours = hours_24h
            print(f"  24h periods ending at: T+{hours_24h}")

        # For plotting, use the first 24h period (T+24)
        if len(data_24h.shape) == 3 and data_24h.shape[0] > 0:
            ensemble_data_24h[member_label] = data_24h[0]  # First 24h period

    max_24h = max(np.nanmax(d) for d in ensemble_data_24h.values())
    print(f"  Max 24h precipitation across all members: {max_24h:.1f} mm")

    # Create plots
    print(f"\n" + "="*70)
    print("Creating Plots")
    print("="*70)

    # Get the first valid 24h period ending hour
    plot_hour = valid_24h_hours[0] if valid_24h_hours else 24

    # Ensemble mean plot
    print("\nCreating ensemble mean plot...")
    create_ensemble_mean_plot(ensemble_data_24h, lats, lons, model_date, run_hour, plot_hour, OUTPUT_DIR)

    # Probability plot
    print("\nCreating probability plot...")
    create_probability_plot(ensemble_data_24h, lats, lons, model_date, run_hour,
                           plot_hour, PRECIP_THRESHOLDS_24H, OUTPUT_DIR)

    # Summary
    elapsed = time.time() - start_time

    print(f"\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Members processed: {len(ensemble_data_24h)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\nOutput files:")
    for pf in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {pf.name}")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
