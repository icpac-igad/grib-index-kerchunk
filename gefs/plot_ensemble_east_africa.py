#!/usr/bin/env python3
"""
East Africa Ensemble Data Plotting Script
This script reads ensemble T2M NetCDF files (mean, std) and creates visualization
plots for the East Africa region with boundary overlays.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gp
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
NETCDF_FILE = "/home/roller/Downloads/ensemble_t2m.nc"
MEAN_FILE = "/home/roller/Downloads/ensemble_mean_t2m.nc"
STD_FILE = "/home/roller/Downloads/ensemble_std_t2m.nc"
BOUNDARY_JSON = "ea_ghcf_simple.geojson"

# East Africa coverage area
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53


def load_geojson_boundaries(json_file):
    """Load GeoJSON boundaries from file."""
    try:
        with open(json_file, 'r') as f:
            geom = json.load(f)

        # Create GeoDataFrame
        gdf = gp.GeoDataFrame.from_features(geom)
        return gdf
    except Exception as e:
        print(f"Warning: Could not load boundary file: {e}")
        return None


def plot_individual_member(ds, member_idx, time_idx, gdf, output_file):
    """
    Plot individual ensemble member temperature.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ensemble data
    member_idx : int
        Ensemble member index (0-based)
    time_idx : int
        Time index to plot
    gdf : GeoDataFrame
        Boundary geometries
    output_file : str
        Output filename for the plot
    """
    # Extract data for specific member and time
    temp_data = ds['t2m'].isel(member=member_idx, valid_times=time_idx)
    lons = ds['longitude'].values
    lats = ds['latitude'].values

    # Convert from Kelvin to Celsius
    temp_celsius = temp_data - 273.15

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Create contour plot
    levels = np.arange(10, 45, 2)  # Temperature levels from 10°C to 45°C
    cf = ax.contourf(lons, lats, temp_celsius, levels=levels,
                     cmap='RdYlBu_r', transform=ccrs.PlateCarree(), extend='both')

    # Add contour lines
    cs = ax.contour(lons, lats, temp_celsius, levels=levels[::2],
                    colors='black', linewidths=0.5, alpha=0.6,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d°C')

    # Add boundaries from GeoJSON
    if gdf is not None:
        ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                         facecolor="none", edgecolor="black", linewidth=1.0)

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='navy')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')
    ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')

    # Set extent
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # Calculate statistics
    mean_temp = float(np.nanmean(temp_celsius))
    min_temp = float(np.nanmin(temp_celsius))
    max_temp = float(np.nanmax(temp_celsius))

    # Title
    plt.title(f'GEFS Ensemble Member {member_idx+1} - 2m Temperature\n'
              f'Time Index: {time_idx} | Mean: {mean_temp:.1f}°C | Range: {min_temp:.1f}°C to {max_temp:.1f}°C',
              fontsize=14, pad=20)

    # Add text box with statistics
    textstr = f'Member {member_idx+1}\n'
    textstr += f'Time: {time_idx}\n'
    textstr += f'Mean: {mean_temp:.1f}°C\n'
    textstr += f'Min: {min_temp:.1f}°C\n'
    textstr += f'Max: {max_temp:.1f}°C'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Member {member_idx+1} plot saved: {output_file}")
    plt.close()

    return output_file


def compute_ensemble_statistics(ds, time_idx):
    """
    Compute ensemble mean and standard deviation from individual members.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ensemble data
    time_idx : int
        Time index to process

    Returns:
    --------
    tuple : (mean_data, std_data)
        Ensemble mean and standard deviation
    """
    # Extract data for specific time
    temp_data = ds['t2m'].isel(valid_times=time_idx)

    # Compute statistics across ensemble members
    mean_data = temp_data.mean(dim='member')
    std_data = temp_data.std(dim='member')

    return mean_data, std_data


def plot_ensemble_mean(ds, gdf, output_file="ensemble_mean_t2m.png", time_idx=0):
    """
    Plot ensemble mean temperature with East Africa boundaries.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ensemble data (either individual members or pre-computed mean)
    gdf : GeoDataFrame
        Boundary geometries
    output_file : str
        Output filename for the plot
    time_idx : int
        Time index to plot (for ensemble data)
    """
    # Check if we have pre-computed mean or need to compute it
    if 't2m' in ds.data_vars and 'member' not in ds.dims:
        # Pre-computed mean (t2m without member dimension)
        mean_data = ds['t2m']
        if len(mean_data.dims) > 2:  # has time dimension
            mean_data = mean_data.isel(valid_times=time_idx)
    elif 't2m' in ds.data_vars and 'member' in ds.dims:
        # Individual ensemble members - compute mean
        mean_data, _ = compute_ensemble_statistics(ds, time_idx)
    else:
        print("❌ Could not find temperature data in dataset")
        return None

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    # Convert from Kelvin to Celsius
    temp_celsius = mean_data - 273.15

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Create contour plot
    levels = np.arange(10, 45, 2)  # Temperature levels from 10°C to 45°C
    cf = ax.contourf(lons, lats, temp_celsius, levels=levels,
                     cmap='RdYlBu_r', transform=ccrs.PlateCarree(), extend='both')

    # Add contour lines
    cs = ax.contour(lons, lats, temp_celsius, levels=levels[::2],
                    colors='black', linewidths=0.5, alpha=0.6,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d°C')

    # Add boundaries from GeoJSON
    if gdf is not None:
        ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                         facecolor="none", edgecolor="black", linewidth=1.0)

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='navy')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')
    ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')

    # Set extent
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # Calculate statistics
    mean_temp = np.nanmean(temp_celsius)
    min_temp = np.nanmin(temp_celsius)
    max_temp = np.nanmax(temp_celsius)

    # Title
    plt.title(f'GEFS Ensemble Mean 2m Temperature - East Africa\n'
              f'Mean: {mean_temp:.1f}°C | Range: {min_temp:.1f}°C to {max_temp:.1f}°C',
              fontsize=14, pad=20)

    # Add text box with statistics
    textstr = f'Statistics:\n'
    textstr += f'Mean: {mean_temp:.1f}°C\n'
    textstr += f'Min: {min_temp:.1f}°C\n'
    textstr += f'Max: {max_temp:.1f}°C'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Ensemble mean plot saved: {output_file}")
    plt.close()

    return output_file


def plot_ensemble_std(ds, gdf, output_file="ensemble_std_t2m.png", time_idx=0):
    """
    Plot ensemble standard deviation with East Africa boundaries.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ensemble data (either individual members or pre-computed std)
    gdf : GeoDataFrame
        Boundary geometries
    output_file : str
        Output filename for the plot
    time_idx : int
        Time index to plot (for ensemble data)
    """
    # Check if we have pre-computed std or need to compute it
    if 't2m' in ds.data_vars and 'member' not in ds.dims:
        # Pre-computed std (t2m without member dimension)
        std_data = ds['t2m']
        if len(std_data.dims) > 2:  # has time dimension
            std_data = std_data.isel(valid_times=time_idx)
    elif 't2m' in ds.data_vars and 'member' in ds.dims:
        # Individual ensemble members - compute std
        _, std_data = compute_ensemble_statistics(ds, time_idx)
    else:
        print("❌ Could not find temperature data in dataset")
        return None

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Create contour plot
    levels = np.arange(0, 6, 0.5)  # Standard deviation levels
    cf = ax.contourf(lons, lats, std_data, levels=levels,
                     cmap='viridis', transform=ccrs.PlateCarree(), extend='max')

    # Add contour lines
    cs = ax.contour(lons, lats, std_data, levels=levels[::2],
                    colors='black', linewidths=0.5, alpha=0.6,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Add boundaries from GeoJSON
    if gdf is not None:
        ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                         facecolor="none", edgecolor="black", linewidth=1.0)

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='navy')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')
    ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')

    # Set extent
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Temperature Standard Deviation (K)', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # Calculate statistics
    mean_std = np.nanmean(std_data)
    min_std = np.nanmin(std_data)
    max_std = np.nanmax(std_data)

    # Title
    plt.title(f'GEFS Ensemble Temperature Standard Deviation - East Africa\n'
              f'Mean Std: {mean_std:.2f}K | Range: {min_std:.2f}K to {max_std:.2f}K',
              fontsize=14, pad=20)

    # Add text box with statistics
    textstr = f'Statistics:\n'
    textstr += f'Mean Std: {mean_std:.2f}K\n'
    textstr += f'Min Std: {min_std:.2f}K\n'
    textstr += f'Max Std: {max_std:.2f}K'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Ensemble std plot saved: {output_file}")
    plt.close()

    return output_file


def plot_temperature_difference(ds, gdf, output_file="temperature_difference_t2m.png"):
    """
    Plot temperature difference between max and min ensemble members.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ensemble data
    gdf : GeoDataFrame
        Boundary geometries
    output_file : str
        Output filename for the plot
    """
    # Calculate temperature range (assuming we have individual members or can derive this)
    # For now, use 2 * std as an approximation of the range
    temp_range = 2 * ds['t2m_std']
    lons = ds['longitude'].values
    lats = ds['latitude'].values

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Create contour plot
    levels = np.arange(0, 12, 1)  # Temperature range levels
    cf = ax.contourf(lons, lats, temp_range, levels=levels,
                     cmap='Reds', transform=ccrs.PlateCarree(), extend='max')

    # Add contour lines
    cs = ax.contour(lons, lats, temp_range, levels=levels[::2],
                    colors='black', linewidths=0.5, alpha=0.6,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Add boundaries from GeoJSON
    if gdf is not None:
        ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                         facecolor="none", edgecolor="black", linewidth=1.0)

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='navy')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')
    ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')

    # Set extent
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Temperature Range (≈2σ, K)', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # Calculate statistics
    mean_range = np.nanmean(temp_range)
    min_range = np.nanmin(temp_range)
    max_range = np.nanmax(temp_range)

    # Title
    plt.title(f'GEFS Ensemble Temperature Range - East Africa\n'
              f'Mean Range: {mean_range:.2f}K | Max Range: {max_range:.2f}K',
              fontsize=14, pad=20)

    # Add text box with statistics
    textstr = f'Statistics:\n'
    textstr += f'Mean Range: {mean_range:.2f}K\n'
    textstr += f'Min Range: {min_range:.2f}K\n'
    textstr += f'Max Range: {max_range:.2f}K\n'
    textstr += f'(Range ≈ 2×std dev)'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Temperature range plot saved: {output_file}")
    plt.close()

    return output_file


def create_combined_plot(ds, gdf, output_file="combined_ensemble_t2m.png"):
    """
    Create a combined plot showing mean, std, and range in subplots.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing ensemble data
    gdf : GeoDataFrame
        Boundary geometries
    output_file : str
        Output filename for the plot
    """
    # Extract data
    if 't2m' in ds.data_vars and 'member' not in ds.dims:
        # Pre-computed data - check if it's mean or std based on data range
        temp_data = ds['t2m']
        if temp_data.max() > 50:  # Likely temperature in Kelvin (mean data)
            mean_data = temp_data - 273.15  # Convert to Celsius
            std_data = temp_data * 0 + 1  # Placeholder std data
        else:  # Likely std data in Kelvin
            mean_data = temp_data * 0 + 20  # Placeholder mean data
            std_data = temp_data
    else:
        print("❌ Cannot create combined plot - need both mean and std data")
        return None
    temp_range = 2 * std_data
    lons = ds['longitude'].values
    lats = ds['latitude'].values

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    # Plot 1: Ensemble Mean
    ax = axes[0]
    levels_mean = np.arange(10, 45, 2)
    cf1 = ax.contourf(lons, lats, mean_data, levels=levels_mean,
                      cmap='RdYlBu_r', transform=ccrs.PlateCarree(), extend='both')

    # Plot 2: Ensemble Std
    ax = axes[1]
    levels_std = np.arange(0, 6, 0.5)
    cf2 = ax.contourf(lons, lats, std_data, levels=levels_std,
                      cmap='viridis', transform=ccrs.PlateCarree(), extend='max')

    # Plot 3: Temperature Range
    ax = axes[2]
    levels_range = np.arange(0, 12, 1)
    cf3 = ax.contourf(lons, lats, temp_range, levels=levels_range,
                      cmap='Reds', transform=ccrs.PlateCarree(), extend='max')

    # Hide the 4th subplot
    axes[3].set_visible(False)

    # Common formatting for all subplots
    titles = ['Ensemble Mean Temperature (°C)',
              'Ensemble Standard Deviation (K)',
              'Temperature Range ≈2σ (K)']
    colorbars = [cf1, cf2, cf3]
    cbar_labels = ['Temperature (°C)', 'Standard Deviation (K)', 'Temperature Range (K)']

    for i, (ax, title, cf, cbar_label) in enumerate(zip(axes[:3], titles, colorbars, cbar_labels)):
        # Add boundaries
        if gdf is not None:
            ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                             facecolor="none", edgecolor="black", linewidth=0.8)

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='navy')
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, color='gray')
        ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')

        # Set extent
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

        # Add title
        ax.set_title(title, fontsize=12, pad=10)

        # Add gridlines for first subplot only
        if i == 0:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.3, color='gray', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

        # Add colorbar
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # Overall title
    fig.suptitle('GEFS Ensemble Temperature Analysis - East Africa', fontsize=16, y=0.95)

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Combined ensemble plot saved: {output_file}")
    plt.close()

    return output_file


def create_combined_plot_from_ensemble(ds, gdf, time_idx, output_file="combined_ensemble_analysis.png"):
    """
    Create a combined plot from ensemble data (computing statistics on the fly).

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing individual ensemble members
    gdf : GeoDataFrame
        Boundary geometries
    time_idx : int
        Time index to plot
    output_file : str
        Output filename for the plot
    """
    # Compute statistics
    mean_data, std_data = compute_ensemble_statistics(ds, time_idx)
    temp_range = 2 * std_data
    lons = ds['longitude'].values
    lats = ds['latitude'].values

    # Convert mean to Celsius
    mean_celsius = mean_data - 273.15

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    # Plot 1: Ensemble Mean
    ax = axes[0]
    levels_mean = np.arange(10, 45, 2)
    cf1 = ax.contourf(lons, lats, mean_celsius, levels=levels_mean,
                      cmap='RdYlBu_r', transform=ccrs.PlateCarree(), extend='both')

    # Plot 2: Ensemble Std
    ax = axes[1]
    levels_std = np.arange(0, 6, 0.5)
    cf2 = ax.contourf(lons, lats, std_data, levels=levels_std,
                      cmap='viridis', transform=ccrs.PlateCarree(), extend='max')

    # Plot 3: Temperature Range
    ax = axes[2]
    levels_range = np.arange(0, 12, 1)
    cf3 = ax.contourf(lons, lats, temp_range, levels=levels_range,
                      cmap='Reds', transform=ccrs.PlateCarree(), extend='max')

    # Hide the 4th subplot
    axes[3].set_visible(False)

    # Common formatting for all subplots
    titles = [f'Ensemble Mean Temperature (°C) - Time {time_idx}',
              f'Ensemble Standard Deviation (K) - Time {time_idx}',
              f'Temperature Range ≈2σ (K) - Time {time_idx}']
    colorbars = [cf1, cf2, cf3]
    cbar_labels = ['Temperature (°C)', 'Standard Deviation (K)', 'Temperature Range (K)']

    for i, (ax, title, cf, cbar_label) in enumerate(zip(axes[:3], titles, colorbars, cbar_labels)):
        # Add boundaries
        if gdf is not None:
            ax.add_geometries(gdf["geometry"], crs=ccrs.PlateCarree(),
                             facecolor="none", edgecolor="black", linewidth=0.8)

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='navy')
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, color='gray')
        ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')

        # Set extent
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])

        # Add title
        ax.set_title(title, fontsize=12, pad=10)

        # Add gridlines for first subplot only
        if i == 0:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.3, color='gray', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

        # Add colorbar
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # Overall title
    fig.suptitle(f'GEFS Ensemble Temperature Analysis - East Africa (Time Index: {time_idx})',
                 fontsize=16, y=0.95)

    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Combined ensemble analysis plot saved: {output_file}")
    plt.close()

    return output_file


def main():
    """Main processing function."""
    print("="*80)
    print("East Africa Ensemble Temperature Plotting")
    print("="*80)

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Load boundary data
    print(f"\n🗺️  Loading boundary data from: {BOUNDARY_JSON}")
    gdf = load_geojson_boundaries(BOUNDARY_JSON)
    if gdf is not None:
        print(f"✅ Boundary data loaded: {len(gdf)} features")

    plot_files = []

    # Try to load different types of files
    datasets_to_try = [
        (NETCDF_FILE, "ensemble data with individual members"),
        (MEAN_FILE, "pre-computed ensemble mean"),
        (STD_FILE, "pre-computed ensemble standard deviation")
    ]

    loaded_datasets = {}

    for file_path, description in datasets_to_try:
        if Path(file_path).exists():
            try:
                print(f"\n📊 Loading {description} from: {file_path}")
                ds = xr.open_dataset(file_path)
                print(f"✅ Dataset loaded successfully")
                print(f"   Variables: {list(ds.data_vars.keys())}")
                print(f"   Dimensions: {dict(ds.dims)}")
                loaded_datasets[file_path] = ds

                # Show data info
                if 't2m' in ds.data_vars:
                    temp_data = ds['t2m']
                    if 'member' in ds.dims:
                        print(f"   Ensemble members: {ds.dims['member']}")
                        print(f"   Time steps: {ds.dims.get('valid_times', 'N/A')}")
                        # Show range for first member, first time
                        sample_data = temp_data.isel(member=0, valid_times=0)
                        print(f"   Sample temperature range: {float(sample_data.min()):.1f}K to {float(sample_data.max()):.1f}K")
                        print(f"   ({float(sample_data.min()-273.15):.1f}°C to {float(sample_data.max()-273.15):.1f}°C)")
                    else:
                        print(f"   Temperature range: {float(temp_data.min()):.1f}K to {float(temp_data.max()):.1f}K")
                        print(f"   ({float(temp_data.min()-273.15):.1f}°C to {float(temp_data.max()-273.15):.1f}°C)")

                elif 't2m' in ds.data_vars and 'member' not in ds.dims:
                    # Pre-computed mean or std data
                    temp_data = ds['t2m']
                    if temp_data.max() > 50:  # Likely temperature in Kelvin (mean data)
                        print(f"   Temperature range (mean): {float(temp_data.min()):.1f}K to {float(temp_data.max()):.1f}K")
                        print(f"   ({float(temp_data.min()-273.15):.1f}°C to {float(temp_data.max()-273.15):.1f}°C)")
                    else:  # Likely std data
                        print(f"   Std dev range: {float(temp_data.min()):.2f}K to {float(temp_data.max()):.2f}K")

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

    if not loaded_datasets:
        print("❌ No datasets could be loaded!")
        return False

    # Process the main ensemble file (individual members)
    print(f"\n🎨 Creating plots...")
    time_idx = 0  # Use first time step

    if NETCDF_FILE in loaded_datasets:
        ds = loaded_datasets[NETCDF_FILE]

        # Plot individual ensemble members (first few)
        if 't2m' in ds.data_vars and 'member' in ds.dims:
            print(f"\n📊 Plotting individual ensemble members...")
            n_members = min(5, ds.dims['member'])  # Plot first 5 members

            for i in range(n_members):
                output_file = output_dir / f"member_{i+1:02d}_t2m_time_{time_idx:03d}.png"
                plot_file = plot_individual_member(ds, i, time_idx, gdf, output_file)
                plot_files.append(plot_file)

            # Plot ensemble mean and std computed from individual members
            print(f"\n📊 Computing and plotting ensemble statistics...")

            plot_file = plot_ensemble_mean(ds, gdf, output_dir / "computed_ensemble_mean_t2m.png", time_idx)
            if plot_file:
                plot_files.append(plot_file)

            plot_file = plot_ensemble_std(ds, gdf, output_dir / "computed_ensemble_std_t2m.png", time_idx)
            if plot_file:
                plot_files.append(plot_file)

    # Process pre-computed mean file
    if MEAN_FILE in loaded_datasets:
        ds_mean = loaded_datasets[MEAN_FILE]
        print(f"\n📊 Plotting pre-computed ensemble mean...")
        plot_file = plot_ensemble_mean(ds_mean, gdf, output_dir / "precomputed_ensemble_mean_t2m.png")
        if plot_file:
            plot_files.append(plot_file)

    # Process pre-computed std file
    if STD_FILE in loaded_datasets:
        ds_std = loaded_datasets[STD_FILE]
        print(f"\n📊 Plotting pre-computed ensemble std...")
        plot_file = plot_ensemble_std(ds_std, gdf, output_dir / "precomputed_ensemble_std_t2m.png")
        if plot_file:
            plot_files.append(plot_file)

    # Create combined plot if we have both mean and std from the same source
    main_ds = loaded_datasets.get(NETCDF_FILE)
    if main_ds and 't2m' in main_ds.data_vars and 'member' in main_ds.dims:
        print(f"\n📊 Creating combined plot...")
        plot_file = create_combined_plot_from_ensemble(main_ds, gdf, time_idx,
                                                     output_dir / "combined_ensemble_analysis.png")
        if plot_file:
            plot_files.append(plot_file)

    print(f"\n✅ Plotting completed successfully!")
    print(f"   Created {len(plot_files)} plots:")
    for pf in plot_files:
        print(f"   - {pf}")

    # Print summary for main dataset
    if main_ds:
        print(f"\n📊 Dataset Summary:")
        print(f"   Latitude range: {float(main_ds.latitude.min()):.1f}° to {float(main_ds.latitude.max()):.1f}°")
        print(f"   Longitude range: {float(main_ds.longitude.min()):.1f}° to {float(main_ds.longitude.max()):.1f}°")
        print(f"   Grid points: {main_ds.dims.get('latitude', 'N/A')} × {main_ds.dims.get('longitude', 'N/A')}")
        if 'member' in main_ds.dims:
            print(f"   Ensemble members: {main_ds.dims['member']}")
        if 'valid_times' in main_ds.dims:
            print(f"   Time steps: {main_ds.dims['valid_times']}")

    # Close datasets
    for ds in loaded_datasets.values():
        ds.close()

    print("="*80)
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)