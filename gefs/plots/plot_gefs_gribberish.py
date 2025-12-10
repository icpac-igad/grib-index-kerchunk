#!/usr/bin/env python3
"""
GEFS Plotting with Gribberish - Direct GRIB Decoding

This script reads parquet files to get byte offset/length information,
then uses gribberish to decode GRIB data directly from S3,
and creates precipitation plots without using zarr/datatree.

Usage:
    python plot_gefs_gribberish.py [--parquet_dir DIR] [--timestep IDX] [--members N]

Example:
    python plot_gefs_gribberish.py --parquet_dir 20250918_00 --timestep 4 --members 5
"""

import os
import sys
import argparse
import json
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import fsspec

# Try to import gribberish
try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False
    print("Warning: gribberish not available")

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# GEFS grid specification (0.25 degree global)
GEFS_GRID_SHAPE = (721, 1440)  # lat x lon
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# East Africa region
EA_LAT_MIN, EA_LAT_MAX = -12, 23
EA_LON_MIN, EA_LON_MAX = 21, 53

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3


def read_parquet_to_zstore(parquet_path: str) -> dict:
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


def find_tp_chunks(zstore: dict) -> List[Tuple[int, str]]:
    """Find all total precipitation (tp) chunks in the zstore."""
    chunks = []

    # Look for tp/accum/surface pattern (accumulated precipitation)
    for key in zstore.keys():
        if key.startswith('tp/accum/surface/') and key.endswith('.0.0'):
            # Extract timestep index from key like "tp/accum/surface/4.0.0"
            parts = key.split('/')
            if len(parts) == 4:
                try:
                    timestep_str = parts[3].split('.')[0]
                    timestep = int(timestep_str)
                    chunks.append((timestep, key))
                except ValueError:
                    pass

    # Sort by timestep
    chunks.sort(key=lambda x: x[0])
    return chunks


def fetch_grib_bytes(zstore: dict, chunk_key: str, fs) -> Tuple[bytes, int]:
    """Fetch GRIB bytes from S3 using the reference in zstore."""
    ref = zstore[chunk_key]

    if isinstance(ref, list) and len(ref) >= 3:
        url, offset, length = ref[0], ref[1], ref[2]
    else:
        raise ValueError(f"Invalid reference format for {chunk_key}: {ref}")

    with fs.open(url, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_with_gribberish(grib_bytes: bytes, grid_shape: Tuple[int, int] = GEFS_GRID_SHAPE) -> np.ndarray:
    """Decode GRIB bytes using gribberish."""
    if not GRIBBERISH_AVAILABLE:
        raise RuntimeError("gribberish not available")

    flat_array = gribberish.parse_grib_array(grib_bytes, 0)
    array_2d = flat_array.reshape(grid_shape)
    return array_2d


def subset_to_east_africa(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset data to East Africa region."""
    # Find indices for subsetting
    # Note: GEFS latitudes go from 90 to -90 (decreasing)
    lat_mask = (GEFS_LATS >= EA_LAT_MIN) & (GEFS_LATS <= EA_LAT_MAX)
    lon_mask = (GEFS_LONS >= EA_LON_MIN) & (GEFS_LONS <= EA_LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    # Subset
    data_subset = data[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    lats_subset = GEFS_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons_subset = GEFS_LONS[lon_indices[0]:lon_indices[-1]+1]

    return data_subset, lats_subset, lons_subset


def load_member_precipitation(parquet_file: Path, timestep: int, fs) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load precipitation data for a single member at a specific timestep using gribberish."""
    member = parquet_file.stem

    try:
        # Read parquet to get zstore
        zstore = read_parquet_to_zstore(str(parquet_file))

        # Find tp chunks
        tp_chunks = find_tp_chunks(zstore)

        if not tp_chunks:
            print(f"  {member}: No TP chunks found")
            return None

        # Find the chunk for the requested timestep
        chunk_key = None
        for ts, key in tp_chunks:
            if ts == timestep:
                chunk_key = key
                break

        if chunk_key is None:
            print(f"  {member}: Timestep {timestep} not found (available: {[t for t, _ in tp_chunks[:5]]}...)")
            return None

        # Fetch and decode GRIB bytes
        t0 = time.time()
        grib_bytes, byte_length = fetch_grib_bytes(zstore, chunk_key, fs)

        # Decode with gribberish
        data_2d = decode_with_gribberish(grib_bytes)
        decode_time = (time.time() - t0) * 1000

        # Subset to East Africa
        data_ea, lats_ea, lons_ea = subset_to_east_africa(data_2d)

        print(f"  {member}: Decoded in {decode_time:.1f}ms, max={np.nanmax(data_ea):.2f}mm")

        return data_ea, lats_ea, lons_ea

    except Exception as e:
        print(f"  {member}: Error - {type(e).__name__}: {str(e)[:60]}")
        return None


def create_ensemble_plot(ensemble_data: Dict[str, np.ndarray],
                        lats: np.ndarray, lons: np.ndarray,
                        model_date: datetime, run_hour: int,
                        timestep: int, output_dir: Path) -> str:
    """Create ensemble comparison plot."""
    n_members = len(ensemble_data)
    if n_members == 0:
        print("No data to plot!")
        return None

    # Calculate grid size
    n_cols = min(5, n_members)
    n_rows = (n_members + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    if n_members == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    # Calculate time info
    forecast_hour = timestep * 3
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    # Calculate colorbar range from all data
    all_max = max(np.nanmax(d) for d in ensemble_data.values())
    vmax = max(5, np.ceil(all_max * 1.1 / 5) * 5)  # Round up to nearest 5
    levels = np.linspace(0, vmax, 11)

    # Plot each member
    members = sorted(ensemble_data.keys())
    im = None

    for idx, member in enumerate(members):
        ax = axes_flat[idx]
        data = ensemble_data[member]

        # Create contour plot
        im = ax.contourf(lons, lats, data, levels=levels, cmap='Blues',
                        transform=ccrs.PlateCarree(), extend='max')

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, alpha=0.1)

        # Set extent
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

        # Title
        data_max = np.nanmax(data)
        ax.set_title(f'{member}\nMax: {data_max:.1f}mm', fontsize=10)

    # Hide unused axes
    for idx in range(len(members), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Precipitation (mm)', rotation=270, labelpad=20)

    # Main title
    date_str = model_date.strftime('%Y%m%d')
    fig.suptitle(f'GEFS Ensemble Precipitation (Gribberish Decoder)\n'
                f'{date_str} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
                f'Valid: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT) | {n_members} members',
                fontsize=14, y=0.98)

    # Save
    output_file = output_dir / f'gribberish_ensemble_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved: {output_file}")
    return str(output_file)


def create_probability_plot(ensemble_data: Dict[str, np.ndarray],
                           lats: np.ndarray, lons: np.ndarray,
                           model_date: datetime, run_hour: int,
                           timestep: int, thresholds: List[float],
                           output_dir: Path) -> str:
    """Create exceedance probability plot."""
    n_members = len(ensemble_data)
    if n_members == 0:
        return None

    # Stack all member data
    members = sorted(ensemble_data.keys())
    data_stack = np.stack([ensemble_data[m] for m in members], axis=0)

    # Calculate probabilities for each threshold
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

    # Probability color levels
    prob_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    prob_colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933',
                   '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']

    # Calculate time info
    forecast_hour = timestep * 3
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

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
                  linewidths=1, transform=ccrs.PlateCarree())

        # Map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

        # Title
        max_prob = np.nanmax(probability)
        ax.set_title(f'P(precip > {threshold}mm)\nMax: {max_prob:.0f}%', fontsize=11)

    # Hide unused axes
    for idx in range(len(thresholds), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)

    # Main title
    date_str = model_date.strftime('%Y%m%d')
    fig.suptitle(f'GEFS Exceedance Probabilities (Gribberish Decoder)\n'
                f'{date_str} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
                f'Valid: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT) | {n_members} members',
                fontsize=14, y=0.98)

    # Save
    output_file = output_dir / f'gribberish_probability_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Probability plot saved: {output_file}")
    return str(output_file)


def main(parquet_dir: str, timestep: int = 4, max_members: int = None):
    """Main processing function."""
    print("=" * 70)
    print("GEFS Plotting with Gribberish (Direct GRIB Decoding)")
    print("=" * 70)

    if not GRIBBERISH_AVAILABLE:
        print("ERROR: gribberish is required but not installed")
        print("Install with: pip install gribberish")
        return False

    start_time = time.time()

    # Setup paths
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"ERROR: Directory {parquet_path} not found")
        return False

    # Extract date and run from directory name
    dir_name = parquet_path.name
    if "_" in dir_name:
        date_str, run_str = dir_name.split("_")
        model_date = datetime.strptime(date_str, "%Y%m%d")
        run_hour = int(run_str)
    else:
        print(f"WARNING: Cannot parse date from {dir_name}, using defaults")
        model_date = datetime.now()
        run_hour = 0

    # Calculate forecast info
    forecast_hour = timestep * 3
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    print(f"\nConfiguration:")
    print(f"  Parquet directory: {parquet_path}")
    print(f"  Model date: {model_date.strftime('%Y-%m-%d')}")
    print(f"  Model run: {run_hour:02d}Z")
    print(f"  Timestep: {timestep} (T+{forecast_hour}h)")
    print(f"  Valid time: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT)")

    # Find parquet files
    parquet_files = sorted(parquet_path.glob("gep*.par"))
    if max_members:
        parquet_files = parquet_files[:max_members]

    print(f"  Ensemble members: {len(parquet_files)}")

    # Create S3 filesystem
    fs = fsspec.filesystem('s3', anon=True)

    # Load data for each member
    print(f"\nLoading precipitation data using gribberish...")
    ensemble_data = {}
    lats = None
    lons = None

    for pf in parquet_files:
        result = load_member_precipitation(pf, timestep, fs)
        if result is not None:
            data, lats, lons = result
            ensemble_data[pf.stem] = data

    if len(ensemble_data) == 0:
        print("\nERROR: No data loaded successfully!")
        return False

    print(f"\nSuccessfully loaded: {len(ensemble_data)} members")

    # Create output directory (same as input)
    output_dir = parquet_path

    # Create ensemble comparison plot
    print(f"\nCreating ensemble comparison plot...")
    create_ensemble_plot(ensemble_data, lats, lons, model_date, run_hour, timestep, output_dir)

    # Create probability plot
    print(f"\nCreating probability plot...")
    thresholds = [1, 5, 10, 15, 20, 25]
    create_probability_plot(ensemble_data, lats, lons, model_date, run_hour,
                           timestep, thresholds, output_dir)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 70)
    print(f"COMPLETE!")
    print(f"=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Members processed: {len(ensemble_data)}")
    print(f"  Output directory: {output_dir}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot GEFS data using gribberish for direct GRIB decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot 5 members at timestep 4 (12h forecast)
  python plot_gefs_gribberish.py --parquet_dir 20250918_00 --timestep 4 --members 5

  # Plot all members at timestep 8 (24h forecast)
  python plot_gefs_gribberish.py --parquet_dir 20250918_00 --timestep 8

  # Plot for 6h forecast
  python plot_gefs_gribberish.py --parquet_dir 20250918_00 --timestep 2
        """
    )

    parser.add_argument('--parquet_dir', type=str, default='20250918_00',
                       help='Directory containing parquet files (default: 20250918_00)')
    parser.add_argument('--timestep', type=int, default=4,
                       help='Timestep index (default: 4 = 12h forecast)')
    parser.add_argument('--members', type=int, default=None,
                       help='Maximum number of members to process (default: all)')

    args = parser.parse_args()

    success = main(args.parquet_dir, args.timestep, args.members)
    if not success:
        sys.exit(1)
