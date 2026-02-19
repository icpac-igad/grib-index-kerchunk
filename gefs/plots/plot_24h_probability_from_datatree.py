#!/usr/bin/env python3
"""
Plot 24-Hour Rainfall Exceedance Probabilities from DataTree

This script reads the zarr DataTree created by gefs_gribberish_datatree.py
and generates a multi-panel plot showing 24-hour rainfall exceedance
probabilities for multiple days and thresholds.

The plot format matches run_gefs_24h_accumulation.py output:
- Rows: Each 24-hour period
- Columns: Each precipitation threshold
- Shows probability of exceedance (0-100%)

Usage:
    python plot_24h_probability_from_datatree.py [options]

Example:
    python plot_24h_probability_from_datatree.py --zarr_path 20250918_00/ensemble_datatree_20250918_00z.zarr
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from xarray import open_datatree
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gp

warnings.filterwarnings('ignore')

# East Africa region
EA_LAT_MIN, EA_LAT_MAX = -12, 23
EA_LON_MIN, EA_LON_MAX = 21, 53

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# GEFS timestep is 3 hours, so 24 hours = 8 timesteps
TIMESTEPS_PER_DAY = 8

# 24-hour rainfall thresholds (mm)
THRESHOLDS_24H = [5, 25, 50, 75, 100, 125]

# Boundary file
BOUNDARY_JSON = "ea_ghcf_simple.geojson"


def load_geojson_boundaries(json_file: str) -> Optional[gp.GeoDataFrame]:
    """Load GeoJSON boundaries from file."""
    try:
        json_path = Path(json_file)
        if not json_path.exists():
            # Try looking in the script directory
            script_dir = Path(__file__).parent
            json_path = script_dir / json_file

        if json_path.exists():
            with open(json_path, 'r') as f:
                geom = json.load(f)
            gdf = gp.GeoDataFrame.from_features(geom)
            return gdf
        else:
            print(f"  Warning: Boundary file not found: {json_file}")
            return None
    except Exception as e:
        print(f"  Warning: Could not load boundary file: {e}")
        return None


def load_datatree(zarr_path: str) -> xr.DataTree:
    """Load DataTree from zarr store."""
    print(f"  Loading DataTree from: {zarr_path}")
    dt = open_datatree(zarr_path, engine="zarr")
    return dt


def get_ensemble_members(dt: xr.DataTree) -> List[str]:
    """Get list of ensemble member names from DataTree."""
    return sorted([name for name in dt.children.keys() if name.startswith('gep')])


def accumulate_24h_rainfall(dt: xr.DataTree, members: List[str]) -> Dict[str, np.ndarray]:
    """
    Accumulate precipitation data into 24-hour periods for each member.

    Parameters:
    -----------
    dt : DataTree
        The loaded DataTree with ensemble data
    members : list
        List of member names

    Returns:
    --------
    dict : Dictionary with member names as keys and 24h accumulated arrays as values
           Shape of each array: (n_days, n_lats, n_lons)
    """
    print(f"  Accumulating 24-hour rainfall for {len(members)} members...")

    ensemble_24h = {}

    for member in members:
        # Get precipitation data for this member
        tp_data = dt[member]['tp'].values  # Shape: (time, lat, lon)
        n_timesteps = tp_data.shape[0]

        # Skip timestep 0 (initial condition with NaN) and work with forecast timesteps
        # GEFS provides 3-hourly precipitation amounts
        forecast_data = tp_data[1:]  # Skip first timestep
        forecast_timesteps = forecast_data.shape[0]

        # Calculate number of complete 24-hour periods
        n_days = forecast_timesteps // TIMESTEPS_PER_DAY

        # Sum 8 consecutive 3-hour timesteps to get 24-hour totals
        daily_accum = np.zeros((n_days,) + tp_data.shape[1:], dtype=np.float32)

        for day in range(n_days):
            start_idx = day * TIMESTEPS_PER_DAY
            end_idx = (day + 1) * TIMESTEPS_PER_DAY

            if end_idx <= forecast_timesteps:
                daily_accum[day] = np.nansum(forecast_data[start_idx:end_idx], axis=0)

        ensemble_24h[member] = daily_accum

    # Get number of days from first member
    n_days = list(ensemble_24h.values())[0].shape[0]
    print(f"  Accumulated {n_days} days of 24-hour totals")

    return ensemble_24h


def calculate_exceedance_probabilities(ensemble_24h: Dict[str, np.ndarray],
                                       thresholds: List[float]) -> Tuple[Dict, int]:
    """
    Calculate probability of exceeding thresholds for ensemble data.

    Parameters:
    -----------
    ensemble_24h : dict
        Dictionary with member names as keys and 24h accumulated data as values
    thresholds : list
        List of threshold values in mm

    Returns:
    --------
    probabilities : dict
        Dictionary with structure {day: {threshold: probability_array}}
    n_members : int
        Number of ensemble members
    """
    # Get dimensions from first member
    first_member = list(ensemble_24h.values())[0]
    n_days = first_member.shape[0]
    n_members = len(ensemble_24h)

    print(f"  Calculating exceedance probabilities for {n_days} days, {len(thresholds)} thresholds...")

    # Initialize probabilities dictionary
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


def create_24h_probability_plot(probabilities: Dict, lons: np.ndarray, lats: np.ndarray,
                                n_members: int, n_days: int,
                                model_date: datetime, run_hour: int,
                                output_dir: Path, thresholds: List[float] = THRESHOLDS_24H) -> str:
    """
    Create multi-panel plot showing 24h rainfall exceedance probabilities.

    The plot has rows for each 24-hour period and columns for each threshold.
    """
    print(f"\n  Creating 24-hour probability plot...")
    print(f"    Days: {n_days}, Thresholds: {thresholds}")

    # Create figure
    fig, axes = plt.subplots(n_days, len(thresholds),
                            figsize=(4*len(thresholds), 4*n_days),
                            subplot_kw={'projection': ccrs.PlateCarree()})

    # Handle single row or column
    if n_days == 1:
        axes = axes.reshape(1, -1)
    elif len(thresholds) == 1:
        axes = axes.reshape(-1, 1)

    # Common color levels
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933',
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']

    # Load boundaries
    gdf = load_geojson_boundaries(BOUNDARY_JSON)

    # Calculate base datetime (model run time)
    base_datetime = model_date + timedelta(hours=run_hour)

    # Plot each panel
    cf = None
    for day in range(n_days):
        for t_idx, threshold in enumerate(thresholds):
            ax = axes[day, t_idx]

            # Get probability data
            prob_data = probabilities[day][threshold]

            # Create contour plot
            cf = ax.contourf(lons, lats, prob_data, levels=levels, colors=colors,
                           transform=ccrs.PlateCarree(), extend='neither')

            # Add 50% contour line
            try:
                cs = ax.contour(lons, lats, prob_data, levels=[50],
                              colors='black', linewidths=1, alpha=0.7,
                              transform=ccrs.PlateCarree())
            except:
                pass  # Skip if no 50% contour exists

            # Add boundaries from GeoJSON
            if gdf is not None:
                ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                                facecolor="none", edgecolor="black", linewidth=0.8)

            # Add coastlines and features (as backup)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', alpha=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.1)

            # Set extent
            ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])

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
    model_run_str = f'{model_date.strftime("%Y-%m-%d")} {run_hour:02d}:00 UTC'
    model_run_eat = f'{model_date.strftime("%Y-%m-%d")} {(run_hour + EAT_OFFSET) % 24:02d}:00 EAT'

    fig.suptitle(f'GEFS 24-Hour Rainfall Exceedance Probabilities\n'
                 f'Model Run: {model_run_str} ({model_run_eat})\n'
                 f'Based on {n_members} ensemble members | Coverage: {EA_LAT_MIN}°-{EA_LAT_MAX}°N, {EA_LON_MIN}°-{EA_LON_MAX}°E\n'
                 f'(Generated from Zarr DataTree using Gribberish)',
                 fontsize=14, y=0.99)

    # Save figure with date and run info
    date_str = model_date.strftime('%Y%m%d')
    run_str = f'{run_hour:02d}'

    output_file = output_dir / f'probability_24h_from_datatree_{date_str}_{run_str}z.png'

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    return str(output_file)


def print_summary_statistics(probabilities: Dict, thresholds: List[float], n_days: int):
    """Print summary statistics for the probabilities."""
    print("\n  Summary Statistics:")

    for day in range(n_days):
        print(f"\n    Day {day+1} (Hours {day*24}-{(day+1)*24}):")
        for threshold in thresholds:
            max_prob = np.nanmax(probabilities[day][threshold])
            area_50 = np.sum(probabilities[day][threshold] >= 50)
            print(f"      >{threshold:3d}mm: Max probability = {max_prob:5.1f}%, "
                  f"Grid points with P>=50% = {area_50:4d}")


def main(zarr_path: str, output_dir: str = None, max_days: int = None,
         thresholds_str: str = None):
    """Main processing function."""
    print("=" * 70)
    print("24-Hour Rainfall Probability Plot from DataTree")
    print("=" * 70)

    import time
    start_time = time.time()

    # Parse zarr path
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        print(f"\nERROR: Zarr store not found: {zarr_path}")
        return False

    # Parse output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = zarr_path.parent
    output_path.mkdir(exist_ok=True)

    # Parse thresholds
    if thresholds_str:
        thresholds = [int(t) for t in thresholds_str.split(',')]
    else:
        thresholds = THRESHOLDS_24H

    print(f"\n  Zarr path: {zarr_path}")
    print(f"  Output directory: {output_path}")
    print(f"  Thresholds: {thresholds} mm")

    # Load DataTree
    print("\n1. Loading DataTree...")
    dt = load_datatree(str(zarr_path))

    # Get ensemble members
    members = get_ensemble_members(dt)
    print(f"  Found {len(members)} ensemble members")

    if len(members) == 0:
        print("\nERROR: No ensemble members found in DataTree!")
        return False

    # Extract model info from DataTree attributes
    model_date_str = dt.attrs.get('model_date', datetime.now().strftime('%Y-%m-%d'))
    model_date = datetime.strptime(model_date_str, '%Y-%m-%d')
    run_hour = dt.attrs.get('run_hour', 0)

    print(f"  Model date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")

    # Get coordinates
    first_member = members[0]
    lats = dt[first_member]['tp'].latitude.values
    lons = dt[first_member]['tp'].longitude.values

    print(f"  Grid: {len(lats)} x {len(lons)}")

    # Accumulate 24-hour rainfall
    print("\n2. Accumulating 24-hour rainfall...")
    ensemble_24h = accumulate_24h_rainfall(dt, members)

    # Limit days if requested
    if max_days:
        n_days = min(max_days, list(ensemble_24h.values())[0].shape[0])
        for member in ensemble_24h:
            ensemble_24h[member] = ensemble_24h[member][:n_days]
        print(f"  Limited to {n_days} days")
    else:
        n_days = list(ensemble_24h.values())[0].shape[0]

    # Calculate exceedance probabilities
    print("\n3. Calculating exceedance probabilities...")
    probabilities, n_members = calculate_exceedance_probabilities(ensemble_24h, thresholds)

    # Print summary statistics
    print_summary_statistics(probabilities, thresholds, n_days)

    # Create plot
    print("\n4. Creating plot...")
    plot_file = create_24h_probability_plot(
        probabilities, lons, lats, n_members, n_days,
        model_date, run_hour, output_path, thresholds
    )

    # Summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Ensemble members: {n_members}")
    print(f"  Days plotted: {n_days}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Output: {plot_file}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 24-hour rainfall probabilities from DataTree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python plot_24h_probability_from_datatree.py --zarr_path 20250918_00/ensemble_datatree_20250918_00z.zarr

  # Limit to first 5 days
  python plot_24h_probability_from_datatree.py --zarr_path 20250918_00/ensemble_datatree_20250918_00z.zarr --max_days 5

  # Custom thresholds
  python plot_24h_probability_from_datatree.py --zarr_path 20250918_00/ensemble_datatree_20250918_00z.zarr --thresholds "10,25,50,100"

  # Custom output directory
  python plot_24h_probability_from_datatree.py --zarr_path 20250918_00/ensemble_datatree_20250918_00z.zarr --output_dir ./plots
        """
    )

    parser.add_argument('--zarr_path', type=str, required=True,
                       help='Path to zarr DataTree store')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as zarr parent)')
    parser.add_argument('--max_days', type=int, default=None,
                       help='Maximum number of days to plot (default: all)')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated thresholds in mm (default: 5,25,50,75,100,125)')

    args = parser.parse_args()

    success = main(args.zarr_path, args.output_dir, args.max_days, args.thresholds)

    if not success:
        sys.exit(1)
