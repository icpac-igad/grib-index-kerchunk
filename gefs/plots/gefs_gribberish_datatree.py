#!/usr/bin/env python3
"""
GEFS Ensemble Processing with Gribberish and Zarr DataTree

This script:
1. Reads parquet files to get byte offset/length references
2. Uses gribberish for fast GRIB decoding (~80x faster than cfgrib)
3. Builds an xarray DataTree with all ensemble members
4. Stores data in zarr format for efficient access
5. Calculates empirical probabilities and creates plots

Usage:
    python gefs_gribberish_datatree.py [options]

Example:
    python gefs_gribberish_datatree.py --parquet_dir 20250918_00 --timesteps 0-24 --members 30
"""

import os
import sys
import argparse
import json
import warnings
import time
import gc
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

# Variable metadata
VARIABLE_METADATA = {
    'tp': {
        'long_name': 'Total precipitation',
        'units': 'mm',
        'standard_name': 'precipitation_amount'
    },
    'probability': {
        'long_name': 'Exceedance probability',
        'units': '%',
    }
}


class GribberishDecoder:
    """Fast GRIB decoder using gribberish with S3 byte fetching."""

    def __init__(self, grid_shape: Tuple[int, int] = GEFS_GRID_SHAPE):
        self.grid_shape = grid_shape
        self.fs = fsspec.filesystem('s3', anon=True)
        self.decode_stats = {'gribberish': 0, 'failed': 0}
        self.total_decode_time = 0

    def fetch_grib_bytes(self, url: str, offset: int, length: int) -> bytes:
        """Fetch GRIB bytes from S3."""
        with self.fs.open(url, 'rb') as f:
            f.seek(offset)
            return f.read(length)

    def decode(self, grib_bytes: bytes) -> np.ndarray:
        """Decode GRIB bytes using gribberish."""
        if not GRIBBERISH_AVAILABLE:
            raise RuntimeError("gribberish not available")

        t0 = time.time()
        flat_array = gribberish.parse_grib_array(grib_bytes, 0)
        array_2d = flat_array.reshape(self.grid_shape)
        self.total_decode_time += (time.time() - t0) * 1000
        self.decode_stats['gribberish'] += 1

        return array_2d

    def decode_from_reference(self, ref: List) -> Optional[np.ndarray]:
        """Decode GRIB data from a zstore reference [url, offset, length]."""
        if not isinstance(ref, list) or len(ref) < 3:
            return None

        url, offset, length = ref[0], ref[1], ref[2]

        try:
            grib_bytes = self.fetch_grib_bytes(url, offset, length)
            return self.decode(grib_bytes)
        except Exception as e:
            self.decode_stats['failed'] += 1
            return None

    def get_stats(self) -> Dict:
        """Return decoding statistics."""
        return {
            'decoded': self.decode_stats['gribberish'],
            'failed': self.decode_stats['failed'],
            'total_time_ms': self.total_decode_time,
            'avg_time_ms': self.total_decode_time / max(1, self.decode_stats['gribberish'])
        }


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


def find_variable_chunks(zstore: dict, var_path: str = 'tp/accum/surface') -> List[Tuple[int, str, List]]:
    """
    Find all chunks for a variable in the zstore.

    Returns list of (timestep, key, reference) tuples.
    """
    chunks = []

    for key, value in zstore.items():
        if key.startswith(f'{var_path}/') and isinstance(value, list) and len(value) >= 3:
            # Extract timestep from key like "tp/accum/surface/4.0.0"
            parts = key.split('/')
            if len(parts) >= 4:
                try:
                    timestep_str = parts[-1].split('.')[0]
                    timestep = int(timestep_str)
                    chunks.append((timestep, key, value))
                except ValueError:
                    pass

    chunks.sort(key=lambda x: x[0])
    return chunks


def get_ea_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get East Africa subset indices and coordinates."""
    lat_mask = (GEFS_LATS >= EA_LAT_MIN) & (GEFS_LATS <= EA_LAT_MAX)
    lon_mask = (GEFS_LONS >= EA_LON_MIN) & (GEFS_LONS <= EA_LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lats = GEFS_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons = GEFS_LONS[lon_indices[0]:lon_indices[-1]+1]

    return lat_indices, lon_indices, lats, lons


def load_member_data(parquet_file: Path, decoder: GribberishDecoder,
                     timesteps: List[int], lat_indices: np.ndarray,
                     lon_indices: np.ndarray) -> Optional[np.ndarray]:
    """
    Load precipitation data for a single ensemble member.

    Returns 3D array (time, lat, lon) for East Africa region.
    """
    member = parquet_file.stem

    try:
        # Read zstore from parquet
        zstore = read_parquet_to_zstore(str(parquet_file))

        # Find tp chunks
        tp_chunks = find_variable_chunks(zstore, 'tp/accum/surface')

        if not tp_chunks:
            print(f"    {member}: No TP chunks found")
            return None

        # Create chunk lookup
        chunk_lookup = {ts: ref for ts, key, ref in tp_chunks}

        # Load data for requested timesteps
        n_lats = lat_indices[-1] - lat_indices[0] + 1
        n_lons = lon_indices[-1] - lon_indices[0] + 1
        data_3d = np.full((len(timesteps), n_lats, n_lons), np.nan, dtype=np.float32)

        for i, ts in enumerate(timesteps):
            if ts in chunk_lookup:
                ref = chunk_lookup[ts]
                data_2d = decoder.decode_from_reference(ref)

                if data_2d is not None:
                    # Subset to East Africa
                    data_3d[i] = data_2d[lat_indices[0]:lat_indices[-1]+1,
                                         lon_indices[0]:lon_indices[-1]+1]

        return data_3d

    except Exception as e:
        print(f"    {member}: Error - {type(e).__name__}: {str(e)[:50]}")
        return None


def build_ensemble_datatree(parquet_dir: Path, timesteps: List[int],
                           max_members: int = None,
                           model_date: datetime = None,
                           run_hour: int = 0) -> Tuple[DataTree, Dict]:
    """
    Build xarray DataTree with all ensemble members using gribberish decoding.

    Structure:
        /
        ├── gep01/
        │   └── tp (time, latitude, longitude)
        ├── gep02/
        │   └── tp (time, latitude, longitude)
        ...
        └── ensemble_stats/
            ├── mean
            ├── std
            └── probability_X (for various thresholds)
    """
    print("\n" + "="*70)
    print("Building Ensemble DataTree with Gribberish")
    print("="*70)

    # Get parquet files
    parquet_files = sorted(parquet_dir.glob("gep*.par"))
    if max_members:
        parquet_files = parquet_files[:max_members]

    print(f"  Found {len(parquet_files)} parquet files")
    print(f"  Timesteps to load: {timesteps[0]} to {timesteps[-1]} ({len(timesteps)} total)")

    # Get East Africa indices
    lat_indices, lon_indices, lats, lons = get_ea_indices()
    print(f"  East Africa region: {len(lats)} x {len(lons)} grid points")

    # Create decoder
    decoder = GribberishDecoder()

    # Calculate valid times
    if model_date is None:
        model_date = datetime.now()

    base_time = model_date + timedelta(hours=run_hour)
    valid_times = [base_time + timedelta(hours=ts*3) for ts in timesteps]
    forecast_hours = [ts * 3 for ts in timesteps]

    # Load data for all members
    print(f"\n  Loading ensemble member data...")
    member_datasets = {}

    for pf in parquet_files:
        member = pf.stem
        print(f"    Loading {member}...", end=" ", flush=True)

        t0 = time.time()
        data = load_member_data(pf, decoder, timesteps, lat_indices, lon_indices)
        elapsed = time.time() - t0

        if data is not None:
            # Create dataset for this member
            ds = xr.Dataset(
                data_vars={
                    'tp': (['time', 'latitude', 'longitude'], data, VARIABLE_METADATA['tp'])
                },
                coords={
                    'time': valid_times,
                    'latitude': lats,
                    'longitude': lons,
                    'forecast_hour': ('time', forecast_hours)
                },
                attrs={
                    'member': member,
                    'model_date': model_date.strftime('%Y-%m-%d'),
                    'run_hour': run_hour
                }
            )
            member_datasets[member] = ds
            print(f"OK ({elapsed:.1f}s, max={np.nanmax(data):.2f}mm)")
        else:
            print("FAILED")

    # Print decoder stats
    stats = decoder.get_stats()
    print(f"\n  Decoder stats: {stats['decoded']} chunks decoded, "
          f"{stats['failed']} failed, avg {stats['avg_time_ms']:.1f}ms/chunk")

    # Build DataTree
    print(f"\n  Building DataTree structure...")
    tree_dict = {f"/{member}": ds for member, ds in member_datasets.items()}

    # Calculate ensemble statistics
    if len(member_datasets) > 0:
        print(f"  Calculating ensemble statistics...")

        # Stack all members
        all_data = np.stack([ds['tp'].values for ds in member_datasets.values()], axis=0)
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
                        {'long_name': 'Ensemble mean precipitation', 'units': 'mm'}),
                'std': (['time', 'latitude', 'longitude'], ensemble_std,
                       {'long_name': 'Ensemble standard deviation', 'units': 'mm'}),
                'max': (['time', 'latitude', 'longitude'], ensemble_max,
                       {'long_name': 'Ensemble maximum', 'units': 'mm'}),
                'min': (['time', 'latitude', 'longitude'], ensemble_min,
                       {'long_name': 'Ensemble minimum', 'units': 'mm'}),
            },
            coords={
                'time': valid_times,
                'latitude': lats,
                'longitude': lons,
                'forecast_hour': ('time', forecast_hours)
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
        'title': 'GEFS Ensemble Forecast Data',
        'institution': 'NOAA/NCEP',
        'source': 'GEFS (Global Ensemble Forecast System)',
        'decoder': 'gribberish',
        'model_date': model_date.strftime('%Y-%m-%d'),
        'run_hour': run_hour,
        'n_members': len(member_datasets),
        'region': 'East Africa',
        'created': datetime.now().isoformat()
    }

    print(f"\n  DataTree created with {len(member_datasets)} members")

    return dt, {'lats': lats, 'lons': lons, 'n_members': len(member_datasets)}


def calculate_exceedance_probabilities(dt: DataTree, thresholds: List[float]) -> xr.Dataset:
    """
    Calculate exceedance probabilities from ensemble DataTree.

    Returns Dataset with probability fields for each threshold.
    """
    print(f"\n  Calculating exceedance probabilities for thresholds: {thresholds}")

    # Get member names (only gep* members)
    members = sorted([name for name in dt.children.keys() if name.startswith('gep')])
    n_members = len(members)

    if n_members == 0:
        return None

    # Stack all member data
    all_data = np.stack([dt[member].ds['tp'].values for member in members], axis=0)

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

        var_name = f'prob_gt_{int(threshold)}mm'
        prob_vars[var_name] = (
            ['time', 'latitude', 'longitude'],
            probability.astype(np.float32),
            {
                'long_name': f'Probability of precipitation > {threshold}mm',
                'units': '%',
                'threshold': threshold,
                'n_members': n_members
            }
        )

        max_prob = np.nanmax(probability)
        print(f"    {threshold}mm: max probability = {max_prob:.1f}%")

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
            'thresholds': thresholds
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


def plot_ensemble_mean(dt: DataTree, timestep_idx: int, output_dir: Path,
                       model_date: datetime, run_hour: int) -> str:
    """Plot ensemble mean precipitation."""
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

    # Plot
    vmax = max(5, np.ceil(np.nanmax(mean_data) * 1.1 / 5) * 5)
    levels = np.linspace(0, vmax, 11)

    cf = ax.contourf(lons, lats, mean_data, levels=levels, cmap='Blues',
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
    cbar.set_label('Precipitation (mm)', rotation=270, labelpad=20)

    # Title
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24
    n_members = stats.attrs.get('n_members', '?')

    plt.title(f'GEFS Ensemble Mean Precipitation\n'
              f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
              f'Valid: {valid_time.strftime("%Y-%m-%d %H:%M")} UTC ({eat_hour:02d}:00 EAT)\n'
              f'{n_members} members | Max: {np.nanmax(mean_data):.1f}mm',
              fontsize=12)

    # Save
    output_file = output_dir / f'ensemble_mean_T{forecast_hour:03d}.png'
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return str(output_file)


def plot_probability_panels(prob_ds: xr.Dataset, timestep_idx: int, output_dir: Path,
                           model_date: datetime, run_hour: int) -> str:
    """Plot probability panels for all thresholds."""
    thresholds = prob_ds.attrs.get('thresholds', [5, 10, 15, 20, 25])
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

        var_name = f'prob_gt_{int(threshold)}mm'
        prob_data = prob_ds[var_name].isel(time=timestep_idx).values

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

        # Title
        max_prob = np.nanmax(prob_data)
        ax.set_title(f'P(precip > {threshold}mm)\nMax: {max_prob:.0f}%', fontsize=11)

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

    fig.suptitle(f'GEFS Exceedance Probabilities (Gribberish + Zarr DataTree)\n'
                 f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}Z Run - T+{forecast_hour}h\n'
                 f'Valid: {valid_time.strftime("%Y-%m-%d %H:%M")} UTC ({eat_hour:02d}:00 EAT) | {n_members} members',
                 fontsize=13, y=0.98)

    # Save
    output_file = output_dir / f'probability_panels_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return str(output_file)


def plot_member_comparison(dt: DataTree, timestep_idx: int, output_dir: Path,
                          model_date: datetime, run_hour: int) -> str:
    """Plot all ensemble members side by side."""
    # Filter to only gep* members (exclude ensemble_stats and probabilities)
    members = sorted([name for name in dt.children.keys()
                     if name.startswith('gep')])
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
    all_max = max(np.nanmax(dt[m].ds['tp'].isel(time=timestep_idx).values) for m in members)
    vmax = max(5, np.ceil(all_max * 1.1 / 5) * 5)
    levels = np.linspace(0, vmax, 11)

    im = None
    for idx, member in enumerate(members):
        ax = axes_flat[idx]
        data = dt[member].ds['tp'].isel(time=timestep_idx).values

        im = ax.contourf(lons, lats, data, levels=levels, cmap='Blues',
                        transform=ccrs.PlateCarree(), extend='max')

        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2)
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

        data_max = np.nanmax(data)
        ax.set_title(f'{member}\nMax: {data_max:.1f}mm', fontsize=9)

    # Hide unused
    for idx in range(n_members, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Precipitation (mm)', rotation=270, labelpad=15)

    # Title
    utc_hour = (run_hour + forecast_hour) % 24
    eat_hour = (utc_hour + EAT_OFFSET) % 24

    fig.suptitle(f'GEFS Ensemble Members (Gribberish Decoder)\n'
                 f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}Z - T+{forecast_hour}h '
                 f'({utc_hour:02d}:00 UTC / {eat_hour:02d}:00 EAT)',
                 fontsize=13, y=0.99)

    # Save
    output_file = output_dir / f'member_comparison_T{forecast_hour:03d}.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return str(output_file)


def main(parquet_dir: str, timesteps_str: str = "0-24", max_members: int = None,
         plot_timestep: int = 4, thresholds_str: str = "5,10,15,20,25",
         save_zarr: bool = True):
    """Main processing function."""
    print("=" * 70)
    print("GEFS Ensemble Processing with Gribberish and Zarr DataTree")
    print("=" * 70)

    if not GRIBBERISH_AVAILABLE:
        print("\nERROR: gribberish is required but not installed")
        print("Install with: pip install gribberish")
        return False

    start_time = time.time()

    # Parse inputs
    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"\nERROR: Directory {parquet_path} not found")
        return False

    # Parse timesteps (e.g., "0-24" or "0,4,8,12")
    if '-' in timesteps_str:
        start, end = map(int, timesteps_str.split('-'))
        timesteps = list(range(start, end + 1))
    else:
        timesteps = [int(t) for t in timesteps_str.split(',')]

    # Parse thresholds
    thresholds = [float(t) for t in thresholds_str.split(',')]

    # Extract date/run from directory name
    dir_name = parquet_path.name
    if "_" in dir_name:
        date_str, run_str = dir_name.split("_")
        model_date = datetime.strptime(date_str, "%Y%m%d")
        run_hour = int(run_str)
    else:
        model_date = datetime.now()
        run_hour = 0

    print(f"\nConfiguration:")
    print(f"  Parquet directory: {parquet_path}")
    print(f"  Model date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")
    print(f"  Timesteps: {timesteps[0]}-{timesteps[-1]} ({len(timesteps)} steps)")
    print(f"  Max members: {max_members or 'all'}")
    print(f"  Plot timestep: {plot_timestep} (T+{plot_timestep*3}h)")
    print(f"  Thresholds: {thresholds} mm")

    # Build DataTree
    dt, meta = build_ensemble_datatree(
        parquet_path, timesteps, max_members, model_date, run_hour
    )

    if meta['n_members'] == 0:
        print("\nERROR: No members loaded successfully!")
        return False

    # Save to zarr
    output_dir = parquet_path

    if save_zarr:
        zarr_path = output_dir / f'ensemble_datatree_{model_date.strftime("%Y%m%d")}_{run_hour:02d}z.zarr'
        save_datatree_to_zarr(dt, zarr_path)

    # Calculate probabilities
    print("\n" + "="*70)
    print("Calculating Exceedance Probabilities")
    print("="*70)

    prob_ds = calculate_exceedance_probabilities(dt, thresholds)

    # Add probabilities to DataTree
    if prob_ds is not None:
        dt['/probabilities'] = DataTree(prob_ds)

    # Create plots
    print("\n" + "="*70)
    print("Creating Plots")
    print("="*70)

    # Ensure plot_timestep is in range
    if plot_timestep >= len(timesteps):
        plot_timestep = len(timesteps) - 1
        print(f"  Adjusted plot_timestep to {plot_timestep}")

    plot_files = []

    # Ensemble mean plot
    print("\n  Creating ensemble mean plot...")
    plot_files.append(plot_ensemble_mean(dt, plot_timestep, output_dir, model_date, run_hour))

    # Probability panels
    if prob_ds is not None:
        print("\n  Creating probability panels...")
        plot_files.append(plot_probability_panels(prob_ds, plot_timestep, output_dir, model_date, run_hour))

    # Member comparison
    print("\n  Creating member comparison plot...")
    plot_files.append(plot_member_comparison(dt, plot_timestep, output_dir, model_date, run_hour))

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
        description="GEFS ensemble processing with gribberish and zarr DataTree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 5 members, timesteps 0-8, plot at T+12h
  python gefs_gribberish_datatree.py --parquet_dir 20250918_00 --members 5 --timesteps 0-8 --plot_timestep 4

  # Process all members, full 10-day forecast
  python gefs_gribberish_datatree.py --parquet_dir 20250918_00 --timesteps 0-80

  # Custom thresholds
  python gefs_gribberish_datatree.py --parquet_dir 20250918_00 --thresholds "1,5,10,25,50"
        """
    )

    parser.add_argument('--parquet_dir', type=str, default='20250918_00',
                       help='Directory containing parquet files')
    parser.add_argument('--timesteps', type=str, default='0-24',
                       help='Timesteps to load (e.g., "0-24" or "0,4,8,12")')
    parser.add_argument('--members', type=int, default=None,
                       help='Max number of members to process')
    parser.add_argument('--plot_timestep', type=int, default=4,
                       help='Timestep index for plotting (default: 4 = T+12h)')
    parser.add_argument('--thresholds', type=str, default='5,10,15,20,25',
                       help='Precipitation thresholds in mm')
    parser.add_argument('--no_zarr', action='store_true',
                       help='Skip saving to zarr')

    args = parser.parse_args()

    success = main(
        args.parquet_dir,
        args.timesteps,
        args.members,
        args.plot_timestep,
        args.thresholds,
        save_zarr=not args.no_zarr
    )

    if not success:
        sys.exit(1)
