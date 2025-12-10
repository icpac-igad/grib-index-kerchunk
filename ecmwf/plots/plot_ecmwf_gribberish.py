#!/usr/bin/env python3
"""
ECMWF IFS Plotting with Gribberish - Direct GRIB Decoding

This script reads parquet files to get byte offset/length information,
then uses gribberish to decode GRIB data directly from S3,
and creates precipitation plots without using zarr/datatree.

Adapted from GEFS version for ECMWF IFS parquet structure.

Key differences from GEFS:
- ECMWF uses lon range -180 to 180 (GEFS uses 0-360)
- ECMWF parquet has step_XXX format: step_XXX/varname/level/member/0.0.0
- Single ensemble member per parquet file

Usage:
    python plot_ecmwf_gribberish.py [--parquet_dir DIR] [--timestep HOURS] [--members N] [--variable VAR]

Example:
    python plot_ecmwf_gribberish.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --timestep 12 --members 5
"""

import os
import sys
import argparse
import json
import warnings
import time
import re
import tempfile
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

# ECMWF variable name mappings (output_var -> parquet_var)
ECMWF_VAR_NAMES = {
    't2m': '2t',
    'u10': '10u',
    'v10': '10v',
    'd2m': '2d',
}


def get_parquet_var_name(var_name: str) -> str:
    """Get the parquet variable name from the output variable name."""
    return ECMWF_VAR_NAMES.get(var_name, var_name)


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


def find_variable_chunks(zstore: dict, var_name: str, level: str = 'sfc') -> List[Tuple[int, str, List]]:
    """
    Find all chunks for a variable using the step_XXX format.

    ECMWF format: step_XXX/varname/level/member/0.0.0

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


def decode_with_cfgrib(grib_bytes: bytes) -> np.ndarray:
    """Decode GRIB bytes using cfgrib (fallback path)."""
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

    return array_2d


def decode_grib(grib_bytes: bytes, grid_shape: Tuple[int, int] = ECMWF_GRID_SHAPE) -> Tuple[np.ndarray, str]:
    """Decode GRIB with gribberish, fallback to cfgrib on failure."""
    if GRIBBERISH_AVAILABLE:
        array, success = decode_with_gribberish_subprocess(grib_bytes, grid_shape)
        if success:
            return array, 'gribberish'

    # Fallback to cfgrib
    array = decode_with_cfgrib(grib_bytes)
    return array, 'cfgrib'


def subset_to_east_africa(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset data to East Africa region."""
    lat_mask = (ECMWF_LATS >= EA_LAT_MIN) & (ECMWF_LATS <= EA_LAT_MAX)
    lon_mask = (ECMWF_LONS >= EA_LON_MIN) & (ECMWF_LONS <= EA_LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    # Subset
    data_subset = data[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    lats_subset = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons_subset = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]

    return data_subset, lats_subset, lons_subset


def load_member_data(parquet_file: Path, var_name: str, timestep: int, fs) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load variable data for a single member at a specific timestep using gribberish."""
    member = parquet_file.stem
    # Extract member identifier
    match = re.search(r'(control|ens_?\d+)', member)
    member_label = match.group(1) if match else member

    try:
        # Read parquet to get zstore
        zstore = read_parquet_to_zstore(str(parquet_file))

        # Find variable chunks using step_XXX format
        chunks = find_variable_chunks(zstore, var_name, level='sfc')

        if not chunks:
            print(f"  {member_label}: No {var_name} chunks found")
            return None

        # Find the chunk for the requested timestep
        ref = None
        for step, key, chunk_ref in chunks:
            if step == timestep:
                ref = chunk_ref
                break

        if ref is None:
            available = [s for s, _, _ in chunks[:10]]
            print(f"  {member_label}: Timestep {timestep} not found (available: {available}...)")
            return None

        # Fetch and decode GRIB bytes
        t0 = time.time()
        grib_bytes, byte_length = fetch_grib_bytes(ref, fs)

        # Decode
        data_2d, decoder = decode_grib(grib_bytes)
        decode_time = (time.time() - t0) * 1000

        # Subset to East Africa
        data_ea, lats_ea, lons_ea = subset_to_east_africa(data_2d)

        print(f"  {member_label}: Decoded in {decode_time:.1f}ms [{decoder}], max={np.nanmax(data_ea):.4f}")

        return data_ea, lats_ea, lons_ea, member_label

    except Exception as e:
        print(f"  {member_label}: Error - {type(e).__name__}: {str(e)[:60]}")
        return None


def create_ensemble_plot(ensemble_data: Dict[str, np.ndarray],
                        lats: np.ndarray, lons: np.ndarray,
                        var_name: str,
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
    forecast_hour = timestep
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    # Prepare data for plotting based on variable
    plot_data = {}
    for member, data in ensemble_data.items():
        if var_name == 'tp':
            plot_data[member] = data * 1000  # Convert to mm
        elif var_name == 't2m':
            plot_data[member] = data - 273.15  # Convert to C
        else:
            plot_data[member] = data

    # Calculate colorbar range from all data
    all_max = max(np.nanmax(d) for d in plot_data.values())
    all_min = min(np.nanmin(d) for d in plot_data.values())

    if var_name == 'tp':
        vmax = max(5, np.ceil(all_max * 1.1 / 5) * 5)
        levels = np.linspace(0, vmax, 11)
        cmap = 'Blues'
        cbar_label = 'Precipitation (mm)'
    elif var_name == 't2m':
        levels = np.linspace(all_min, all_max, 11)
        cmap = 'RdYlBu_r'
        cbar_label = 'Temperature (C)'
    else:
        vmax = max(1, np.ceil(all_max * 1.1))
        levels = np.linspace(0, vmax, 11)
        cmap = 'viridis'
        cbar_label = var_name

    # Plot each member
    members = sorted(plot_data.keys())
    im = None

    for idx, member in enumerate(members):
        ax = axes_flat[idx]
        data = plot_data[member]

        # Create contour plot
        im = ax.contourf(lons, lats, data, levels=levels, cmap=cmap,
                        transform=ccrs.PlateCarree(), extend='max')

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, alpha=0.1)

        # Set extent
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

        # Title
        data_max = np.nanmax(data)
        ax.set_title(f'{member}\nMax: {data_max:.2f}', fontsize=10)

    # Hide unused axes
    for idx in range(len(members), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)

    # Main title
    date_str = model_date.strftime('%Y%m%d')
    fig.suptitle(f'ECMWF IFS Ensemble {var_name.upper()} (Gribberish Decoder)\n'
                f'{date_str} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
                f'Valid: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT) | {n_members} members',
                fontsize=14, y=0.98)

    # Save
    output_file = output_dir / f'ecmwf_gribberish_ensemble_{var_name}_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved: {output_file}")
    return str(output_file)


def create_probability_plot(ensemble_data: Dict[str, np.ndarray],
                           lats: np.ndarray, lons: np.ndarray,
                           var_name: str,
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
    forecast_hour = timestep
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

        # Title - format threshold nicely
        if var_name == 'tp':
            threshold_display = f'{threshold*1000:.1f}mm'
        else:
            threshold_display = f'{threshold}'

        max_prob = np.nanmax(probability)
        ax.set_title(f'P({var_name} > {threshold_display})\nMax: {max_prob:.0f}%', fontsize=11)

    # Hide unused axes
    for idx in range(len(thresholds), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)

    # Main title
    date_str = model_date.strftime('%Y%m%d')
    fig.suptitle(f'ECMWF IFS Exceedance Probabilities (Gribberish Decoder)\n'
                f'{date_str} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
                f'Valid: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT) | {n_members} members',
                fontsize=14, y=0.98)

    # Save
    output_file = output_dir / f'ecmwf_gribberish_probability_{var_name}_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Probability plot saved: {output_file}")
    return str(output_file)


def main(parquet_dir: str, variable: str = "tp", timestep: int = 12, max_members: int = None):
    """Main processing function."""
    print("=" * 70)
    print("ECMWF IFS Plotting with Gribberish (Direct GRIB Decoding)")
    print("=" * 70)

    if not GRIBBERISH_AVAILABLE:
        print("WARNING: gribberish is not installed, will use cfgrib fallback")
        print("Install with: pip install gribberish")

    start_time = time.time()

    # Setup paths
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"ERROR: Directory {parquet_path} not found")
        return False

    # Try to extract date from parquet files
    model_date = datetime.now()
    run_hour = 0

    # Find parquet files
    parquet_files = sorted(parquet_path.glob("stage3_ens_*_final.parquet"))
    control_file = parquet_path / "stage3_control_final.parquet"
    if control_file.exists():
        parquet_files.insert(0, control_file)

    if max_members:
        parquet_files = parquet_files[:max_members]

    if not parquet_files:
        print(f"ERROR: No parquet files found in {parquet_path}")
        return False

    # Try to get date from first parquet file
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
                        break
    except:
        pass

    # Calculate forecast info
    forecast_hour = timestep
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    print(f"\nConfiguration:")
    print(f"  Parquet directory: {parquet_path}")
    print(f"  Variable: {variable}")
    print(f"  Model date: {model_date.strftime('%Y-%m-%d')}")
    print(f"  Model run: {run_hour:02d}Z")
    print(f"  Timestep: T+{forecast_hour}h")
    print(f"  Valid time: {utc_hour:02d}:00 UTC ({eat_hour:02d}:00 EAT)")
    print(f"  Ensemble members: {len(parquet_files)}")

    # Create S3 filesystem
    fs = fsspec.filesystem('s3', anon=True)

    # Load data for each member
    print(f"\nLoading {variable} data using gribberish...")
    ensemble_data = {}
    lats = None
    lons = None

    for pf in parquet_files:
        result = load_member_data(pf, variable, timestep, fs)
        if result is not None:
            data, lats, lons, member_label = result
            ensemble_data[member_label] = data

    if len(ensemble_data) == 0:
        print("\nERROR: No data loaded successfully!")
        return False

    print(f"\nSuccessfully loaded: {len(ensemble_data)} members")

    # Create output directory (same as input)
    output_dir = parquet_path

    # Create ensemble comparison plot
    print(f"\nCreating ensemble comparison plot...")
    create_ensemble_plot(ensemble_data, lats, lons, variable, model_date, run_hour, timestep, output_dir)

    # Create probability plot
    print(f"\nCreating probability plot...")
    if variable == 'tp':
        # Thresholds in meters for precipitation
        thresholds = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
    elif variable == 't2m':
        # Temperature thresholds in Kelvin
        thresholds = [288, 293, 298, 303, 308, 313]  # 15C, 20C, 25C, 30C, 35C, 40C
    else:
        # Generic thresholds
        all_vals = np.concatenate([d.flatten() for d in ensemble_data.values()])
        p25, p50, p75, p90, p95, p99 = np.nanpercentile(all_vals, [25, 50, 75, 90, 95, 99])
        thresholds = [p50, p75, p90, p95, p99]

    create_probability_plot(ensemble_data, lats, lons, variable, model_date, run_hour,
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
        description="Plot ECMWF IFS data using gribberish for direct GRIB decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot 5 members at T+12h for precipitation
  python plot_ecmwf_gribberish.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --timestep 12 --members 5

  # Plot all members at T+24h
  python plot_ecmwf_gribberish.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --timestep 24

  # Plot temperature
  python plot_ecmwf_gribberish.py --parquet_dir test_ecmwf_three_stage_prebuilt_output --variable t2m --timestep 12
        """
    )

    parser.add_argument('--parquet_dir', type=str, default='test_ecmwf_three_stage_prebuilt_output',
                       help='Directory containing ECMWF parquet files (default: test_ecmwf_three_stage_prebuilt_output)')
    parser.add_argument('--variable', type=str, default='tp',
                       help='Variable to plot (default: tp)')
    parser.add_argument('--timestep', type=int, default=12,
                       help='Forecast hour (default: 12)')
    parser.add_argument('--members', type=int, default=None,
                       help='Maximum number of members to process (default: all)')

    args = parser.parse_args()

    success = main(args.parquet_dir, args.variable, args.timestep, args.members)
    if not success:
        sys.exit(1)
