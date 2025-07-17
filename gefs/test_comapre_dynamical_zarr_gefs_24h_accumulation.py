#!/usr/bin/env python3
"""
GEFS 24-hour Rainfall Accumulation Processing from Zarr Store
This script processes GEFS ensemble data from a zarr store to create 24-hour rainfall accumulation
plots with threshold exceedance probabilities.
"""

import numpy as np
import xarray as xr
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import geopandas as gp
import time

warnings.filterwarnings('ignore')

# Configuration
ZARR_URL = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com"
BOUNDARY_JSON = "ea_ghcf_simple.geojson"  # GeoJSON file for boundaries

# Date Configuration - Set TARGET_DATE to specify which date to process
# Format: 'YYYY-MM-DD' or None for latest available date
TARGET_DATE = None  # Example: '2025-07-15' or None

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# Coverage area for the plot
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53

# 24-hour rainfall thresholds (mm)
THRESHOLDS_24H = [5, 25, 50, 75, 100, 125]

# GEFS timestep is 3 hours, so 24 hours = 8 timesteps
TIMESTEPS_PER_DAY = 8


def load_zarr_dataset():
    """Load the GEFS zarr dataset and return relevant information."""
    print("📊 Loading GEFS zarr dataset...")
    start_time = time.time()
    
    try:
        ds = xr.open_zarr(ZARR_URL)
        load_time = time.time() - start_time
        print(f"✅ Successfully loaded zarr dataset in {load_time:.1f}s")
        
        # Get dataset info
        print(f"   Dataset shape: {ds.precipitation_surface.shape}")
        print(f"   Dimensions: {ds.precipitation_surface.dims}")
        print(f"   Latest init_time: {ds.init_time[-1].values}")
        print(f"   Ensemble members: {len(ds.ensemble_member)}")
        print(f"   Lead times: {len(ds.lead_time)}")
        
        return ds
        
    except Exception as e:
        print(f"❌ Error loading zarr dataset: {e}")
        return None


def extract_regional_precipitation(ds, region_name="East Africa"):
    """
    Extract precipitation data for the specified region and convert from rate to accumulation.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The GEFS dataset
    region_name : str
        Name of the region (for display purposes)
    
    Returns:
    --------
    regional_precip : xarray.DataArray
        Regional precipitation data in mm per 3-hour period
    """
    print(f"🌍 Extracting {region_name} precipitation data...")
    
    # Get the precipitation variable
    precip_var = ds['precipitation_surface']
    
    # Select init_time based on TARGET_DATE
    if TARGET_DATE is None:
        # Use the latest available init_time
        selected_init = ds.init_time[-1]
        print(f"   Using latest init_time: {selected_init.values}")
    else:
        # Parse target date and find matching init_time
        try:
            target_datetime = datetime.strptime(TARGET_DATE, '%Y-%m-%d')
            target_np_datetime = np.datetime64(target_datetime)
            
            # Find the closest available init_time
            available_dates = ds.init_time.values
            
            # Check if exact date exists
            exact_matches = available_dates[available_dates.astype('datetime64[D]') == target_np_datetime.astype('datetime64[D]')]
            
            if len(exact_matches) > 0:
                # Use the first (usually 00:00 UTC) run from the target date
                selected_init = ds.init_time.sel(init_time=exact_matches[0])
                print(f"   Using exact date match: {selected_init.values}")
            else:
                # Find the closest date
                time_diffs = np.abs(available_dates.astype('datetime64[D]') - target_np_datetime.astype('datetime64[D]'))
                closest_idx = np.argmin(time_diffs)
                selected_init = ds.init_time[closest_idx]
                closest_date = available_dates[closest_idx]
                print(f"   Target date {TARGET_DATE} not found.")
                print(f"   Using closest available date: {closest_date}")
                
        except ValueError as e:
            print(f"   ❌ Error parsing date '{TARGET_DATE}': {e}")
            print(f"   Using latest available date instead.")
            selected_init = ds.init_time[-1]
            print(f"   Using init_time: {selected_init.values}")
    
    # Extract regional data for selected init_time
    regional_precip = precip_var.sel(
        init_time=selected_init,
        latitude=slice(LAT_MAX, LAT_MIN),  # Note: slice order for decreasing values
        longitude=slice(LON_MIN, LON_MAX)
    )
    
    # The data is in mm/s (average rate since previous forecast step)
    # Convert to mm per 3-hour period by multiplying by 3*3600 seconds
    # This gives us the actual precipitation amount for each 3-hour timestep
    regional_precip = regional_precip * 3 * 3600  # Convert from mm/s to mm per 3-hour
    
    print(f"   Regional data shape: {regional_precip.shape}")
    print(f"   Available lead times: {len(regional_precip.lead_time)}")
    print(f"   Coordinate ranges - Lat: {regional_precip.latitude.min().values:.1f} to {regional_precip.latitude.max().values:.1f}")
    print(f"                       Lon: {regional_precip.longitude.min().values:.1f} to {regional_precip.longitude.max().values:.1f}")
    
    return regional_precip


def accumulate_24h_rainfall_zarr(precip_data):
    """
    Accumulate precipitation data into 24-hour periods from zarr data.
    
    Parameters:
    -----------
    precip_data : xarray.DataArray
        Precipitation data with dimensions (ensemble_member, lead_time, latitude, longitude)
        Values are in mm per 3-hour period
    
    Returns:
    --------
    accumulated : numpy.ndarray
        24-hour accumulated precipitation with shape (ensemble_member, n_days, lat, lon)
    """
    print("📊 Processing 24-hour accumulations...")
    
    # Get dimensions
    n_members = precip_data.shape[0]
    n_timesteps = precip_data.shape[1]
    
    # Calculate number of complete 24-hour periods
    n_days = n_timesteps // TIMESTEPS_PER_DAY
    print(f"   Processing {n_days} complete 24-hour periods from {n_timesteps} timesteps")
    
    # Initialize output array
    output_shape = (n_members, n_days) + precip_data.shape[2:]
    daily_accumulations = np.zeros(output_shape)
    
    # Process each ensemble member
    for member_idx in range(n_members):
        member_data = precip_data[member_idx].values
        
        # Accumulate for each 24-hour period
        for day in range(n_days):
            start_idx = day * TIMESTEPS_PER_DAY
            end_idx = (day + 1) * TIMESTEPS_PER_DAY
            
            # Sum 8 consecutive 3-hour periods to get 24-hour total
            daily_accumulations[member_idx, day] = np.sum(member_data[start_idx:end_idx], axis=0)
    
    print(f"   Completed accumulation processing")
    print(f"   Output shape: {daily_accumulations.shape}")
    
    return daily_accumulations


def calculate_exceedance_probabilities(ensemble_24h_data, thresholds):
    """
    Calculate probability of exceeding thresholds for ensemble data.
    
    Parameters:
    -----------
    ensemble_24h_data : numpy.ndarray
        Array with shape (n_members, n_days, lat, lon)
    thresholds : list
        List of threshold values
    
    Returns:
    --------
    probabilities : dict
        Dictionary with structure {day: {threshold: probability_array}}
    """
    n_members, n_days = ensemble_24h_data.shape[:2]
    
    print(f"📈 Calculating exceedance probabilities for {n_members} members, {n_days} days...")
    
    # Initialize probabilities dictionary
    probabilities = {}
    
    for day in range(n_days):
        probabilities[day] = {}
        
        # Get data for this day across all members
        day_data = ensemble_24h_data[:, day, :, :]  # Shape: (n_members, lat, lon)
        
        # Calculate probabilities for each threshold
        for threshold in thresholds:
            exceedance_count = np.sum(day_data >= threshold, axis=0)
            probability = (exceedance_count / n_members) * 100
            probabilities[day][threshold] = probability
    
    print(f"   Completed probability calculations")
    return probabilities, n_members


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


def create_24h_probability_plot_zarr(probabilities, lons, lats, n_members, n_days, init_time, output_dir=None):
    """
    Create multi-panel plot showing 24h rainfall exceedance probabilities.
    
    The plot has rows for each 24-hour period and columns for each threshold.
    """
    print("🎨 Creating 24-hour accumulation plots...")
    
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
    
    # Calculate base datetime from init_time
    base_datetime = pd.to_datetime(init_time)
    
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
            cs = ax.contour(lons, lats, prob_data, levels=[50], 
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
    init_datetime_str = base_datetime.strftime("%Y-%m-%d %H:%M UTC")
    init_eat_str = (base_datetime + timedelta(hours=EAT_OFFSET)).strftime("%Y-%m-%d %H:%M EAT")
    
    fig.suptitle(f'GEFS 24-Hour Rainfall Exceedance Probabilities (Zarr Source)\n'
                 f'Model Init: {init_datetime_str} ({init_eat_str})\n'
                 f'Based on {n_members} ensemble members | Coverage: {LAT_MIN}°-{LAT_MAX}°N, {LON_MIN}°-{LON_MAX}°E',
                 fontsize=14, y=0.98)
    
    # Save figure
    if output_dir:
        output_file = Path(output_dir) / 'zarr_probability_24h_accumulation_all_thresholds.png'
    else:
        output_file = 'zarr_probability_24h_accumulation_all_thresholds.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ 24-hour accumulation plot saved: {output_file}")
    plt.close()
    
    return str(output_file)


def main():
    """Main processing function."""
    print("="*80)
    print("GEFS 24-Hour Rainfall Accumulation Processing from Zarr Store")
    print("="*80)
    
    if TARGET_DATE:
        print(f"🎯 Target date: {TARGET_DATE}")
    else:
        print("📅 Using latest available date")
    
    start_time = time.time()
    
    # Load zarr dataset
    ds = load_zarr_dataset()
    if ds is None:
        return False
    
    # Extract regional precipitation data
    regional_precip = extract_regional_precipitation(ds)
    if regional_precip is None:
        return False
    
    # Get model run information
    init_time = regional_precip.init_time.values
    print(f"\n📅 Model Run Information:")
    print(f"   Init Time: {pd.to_datetime(init_time)}")
    
    # Process accumulations
    print("\n📊 Processing 24-hour accumulations...")
    accum_start_time = time.time()
    ensemble_24h = accumulate_24h_rainfall_zarr(regional_precip)
    accum_end_time = time.time()
    print(f"⏱️  Accumulation processing time: {(accum_end_time - accum_start_time):.1f} seconds")
    
    # Get coordinates
    lons = regional_precip.longitude.values
    lats = regional_precip.latitude.values
    
    # Calculate exceedance probabilities
    print("\n📈 Calculating exceedance probabilities...")
    prob_start_time = time.time()
    probabilities, n_members = calculate_exceedance_probabilities(ensemble_24h, THRESHOLDS_24H)
    
    # Get number of days
    n_days = len(probabilities)
    prob_end_time = time.time()
    print(f"  Days: {n_days}")
    print(f"  Thresholds: {THRESHOLDS_24H} mm")
    print(f"⏱️  Probability calculation time: {(prob_end_time - prob_start_time):.1f} seconds")
    
    # Create visualization
    print("\n🎨 Creating 24-hour accumulation plots...")
    plot_start_time = time.time()
    plot_file = create_24h_probability_plot_zarr(probabilities, lons, lats, n_members, n_days, init_time)
    plot_end_time = time.time()
    print(f"⏱️  Plotting time: {(plot_end_time - plot_start_time):.1f} seconds")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    for day in range(n_days):
        print(f"\n  Day {day+1} (Hours {day*24}-{(day+1)*24}):")
        for threshold in THRESHOLDS_24H:
            max_prob = np.nanmax(probabilities[day][threshold])
            area_50 = np.sum(probabilities[day][threshold] >= 50)
            print(f"    >{threshold:3d}mm: Max probability = {max_prob:5.1f}%, "
                  f"Grid points with P≥50% = {area_50:4d}")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ Processing completed successfully!")
    print("="*80)
    
    # Detailed timing breakdown
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"   🔄 24h Accumulation:       {(accum_end_time - accum_start_time):.1f} seconds ({(accum_end_time - accum_start_time)/total_time*100:.1f}%)")
    print(f"   📈 Probability Calculation: {(prob_end_time - prob_start_time):.1f} seconds ({(prob_end_time - prob_start_time)/total_time*100:.1f}%)")
    print(f"   🎨 Plot Generation:        {(plot_end_time - plot_start_time):.1f} seconds ({(plot_end_time - plot_start_time)/total_time*100:.1f}%)")
    print(f"   ⏱️  TOTAL TIME:             {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\n📁 Output file: {plot_file}")
    print("="*80)
    
    return True


if __name__ == "__main__":
    # Import pandas here since it's used in some functions
    import pandas as pd
    
    success = main()
    if not success:
        exit(1)