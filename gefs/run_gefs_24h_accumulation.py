#!/usr/bin/env python3
"""
GEFS 24-hour Rainfall Accumulation Processing V2
This script processes GEFS ensemble data to create 24-hour rainfall accumulation
plots with threshold exceedance probabilities.
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import fsspec
import geopandas as gp
import time

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Configuration
PARQUET_DIR = Path("20240323_18")  # Directory containing parquet files
BOUNDARY_JSON = "ea_ghcf_simple.json"  # GeoJSON file for boundaries

# Extract date and run hour from directory name
dir_name = PARQUET_DIR.name
if "_" in dir_name:
    date_str, run_hour_str = dir_name.split("_")
    MODEL_DATE = datetime.strptime(date_str, "%Y%m%d")
    MODEL_RUN_HOUR = int(run_hour_str)
else:
    # Fallback if format is different
    MODEL_DATE = datetime.now()
    MODEL_RUN_HOUR = 0

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

# Coverage area for the plot
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53

# 24-hour rainfall thresholds (mm)
THRESHOLDS_24H = [5, 25, 50, 75, 100, 125]

# GEFS timestep is 3 hours, so 24 hours = 8 timesteps
TIMESTEPS_PER_DAY = 8


def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling - from original script."""
    import ast
    
    df = pd.read_parquet(parquet_path)
    
    if 'refs' in df['key'].values and len(df) <= 2:
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
        print(f"✅ Extracted {len(zstore)} entries from old format")
    else:
        zstore = {}
        for _, row in df.iterrows():
            key = row['key']
            value = row['value']
            
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value)
                    except:
                        pass
            
            zstore[key] = value
        
        print(f"✅ Loaded {len(zstore)} entries from new format")
    
    if 'version' in zstore:
        del zstore['version']
    
    return zstore


def stream_single_member_precipitation(parquet_file, variable='tp'):
    """Stream precipitation data for a single ensemble member."""
    member = parquet_file.stem
    member_start_time = time.time()
    print(f"\n📊 Processing {member}...")
    
    try:
        # Read zarr store from parquet
        zstore = read_parquet_fixed(parquet_file)
        
        # Create reference filesystem
        fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                              remote_options={'anon': True})
        mapper = fs.get_mapper("")
        
        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
        
        # Navigate to variable data
        if variable == 'tp':
            data_var = dt['/tp/accum/surface'].ds['tp']
        else:
            return None
        
        # Extract region
        regional_data = data_var.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX)
        )
        
        # Compute numpy array
        regional_numpy = regional_data.compute()
        
        member_end_time = time.time()
        print(f"✅ {member} data shape: {regional_numpy.shape} | Time: {(member_end_time - member_start_time):.1f}s")
        
        return regional_numpy, regional_data
        
    except Exception as e:
        print(f"❌ Error streaming {member}: {e}")
        return None, None


def accumulate_24h_rainfall(precip_data):
    """
    Accumulate precipitation data into 24-hour periods.
    
    Parameters:
    -----------
    precip_data : xarray.DataArray or numpy.ndarray
        Precipitation data with time as first dimension
    
    Returns:
    --------
    accumulated : numpy.ndarray
        24-hour accumulated precipitation
    """
    if isinstance(precip_data, xr.DataArray):
        data = precip_data.values
    else:
        data = precip_data
    
    # Get time dimension
    n_timesteps = data.shape[0]
    
    # Calculate number of complete 24-hour periods
    n_days = n_timesteps // TIMESTEPS_PER_DAY
    
    # Create array to store 24-hour accumulations
    daily_shape = (n_days,) + data.shape[1:]
    daily_accumulations = np.zeros(daily_shape)
    
    for day in range(n_days):
        start_idx = day * TIMESTEPS_PER_DAY
        end_idx = (day + 1) * TIMESTEPS_PER_DAY
        
        # Sum precipitation over 24-hour period
        daily_accumulations[day] = np.sum(data[start_idx:end_idx], axis=0)
    
    return daily_accumulations


def calculate_exceedance_probabilities(ensemble_24h_data, thresholds):
    """
    Calculate probability of exceeding thresholds for ensemble data.
    
    Parameters:
    -----------
    ensemble_24h_data : dict
        Dictionary with member names as keys and 24h accumulated data as values
    thresholds : list
        List of threshold values
    
    Returns:
    --------
    probabilities : dict
        Dictionary with structure {day: {threshold: probability_array}}
    """
    # Get dimensions from first member
    first_member = list(ensemble_24h_data.values())[0]
    n_days = first_member.shape[0]
    n_members = len(ensemble_24h_data)
    
    # Initialize probabilities dictionary
    probabilities = {}
    
    for day in range(n_days):
        probabilities[day] = {}
        
        # Stack all member data for this day
        day_data = []
        for member_data in ensemble_24h_data.values():
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


def create_24h_probability_plot(probabilities, lons, lats, n_members, n_days, output_dir=None):
    """
    Create multi-panel plot showing 24h rainfall exceedance probabilities.
    
    The plot has rows for each 24-hour period and columns for each threshold.
    """
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
    from datetime import timedelta
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
    model_run_str = f'{MODEL_DATE.strftime("%Y-%m-%d")} {MODEL_RUN_HOUR:02d}:00 UTC'
    model_run_eat = f'{MODEL_DATE.strftime("%Y-%m-%d")} {(MODEL_RUN_HOUR + EAT_OFFSET) % 24:02d}:00 EAT'
    
    fig.suptitle(f'GEFS 24-Hour Rainfall Exceedance Probabilities\n'
                 f'Model Run: {model_run_str} ({model_run_eat})\n'
                 f'Based on {n_members} ensemble members | Coverage: {LAT_MIN}°-{LAT_MAX}°N, {LON_MIN}°-{LON_MAX}°E',
                 fontsize=14, y=0.98)
    
    # Save figure
    if output_dir:
        output_file = output_dir / 'probability_24h_accumulation_all_thresholds.png'
    else:
        output_file = 'probability_24h_accumulation_all_thresholds.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ 24-hour accumulation plot saved: {output_file}")
    plt.close()
    
    return str(output_file)


def main():
    """Main processing function."""
    print("="*80)
    print("GEFS 24-Hour Rainfall Accumulation Processing V2")
    print("="*80)
    
    # Display model run information
    print(f"\n📅 Model Run Information:")
    print(f"   Date: {MODEL_DATE.strftime('%Y-%m-%d')}")
    print(f"   Run Hour: {MODEL_RUN_HOUR:02d}:00 UTC ({(MODEL_RUN_HOUR + EAT_OFFSET) % 24:02d}:00 EAT)")
    print(f"   Forecast starts: {MODEL_DATE.strftime('%Y-%m-%d')} {MODEL_RUN_HOUR:02d}:00 UTC")
    
    start_time = time.time()
    
    # Check if parquet directory exists
    if not PARQUET_DIR.exists():
        print(f"❌ Error: Directory {PARQUET_DIR} not found!")
        return False
    
    # Get list of parquet files
    parquet_files = sorted(PARQUET_DIR.glob("gep*.par"))
    print(f"📁 Found {len(parquet_files)} ensemble member files")
    
    # Use the same directory as input PAR files for output
    output_dir = PARQUET_DIR
    print(f"📝 Output will be saved in: {output_dir}")
    
    # Load ensemble data
    print("\n🌧️ Loading ensemble precipitation data...")
    load_start_time = time.time()
    ensemble_numpy = {}
    ensemble_xarray = {}
    
    for pf in parquet_files:  # Process all members
        numpy_data, xarray_data = stream_single_member_precipitation(pf)
        if numpy_data is not None:
            member_name = pf.stem
            ensemble_numpy[member_name] = numpy_data
            ensemble_xarray[member_name] = xarray_data
    
    load_end_time = time.time()
    print(f"\n✅ Successfully loaded {len(ensemble_numpy)} members")
    print(f"⏱️  Loading time: {(load_end_time - load_start_time):.1f} seconds")
    
    if len(ensemble_numpy) == 0:
        print("❌ No data loaded successfully!")
        return False
    
    # Get coordinates from first member
    first_member_xr = list(ensemble_xarray.values())[0]
    lons = first_member_xr.longitude.values
    lats = first_member_xr.latitude.values
    
    # Process accumulations
    print("\n📊 Processing 24-hour accumulations...")
    accum_start_time = time.time()
    ensemble_24h = {}
    
    for member, data in ensemble_numpy.items():
        # GEFS provides 3-hourly precipitation amounts (already incremental, not cumulative)
        # Timestep 0 is initial condition (NaN), actual forecast starts from timestep 1
        # We need to sum 8 consecutive 3-hour timesteps to get 24-hour totals
        n_timesteps = data.shape[0]
        
        # Skip timestep 0 (initial condition) and work with forecast timesteps 1-80
        forecast_timesteps = n_timesteps - 1  # 80 timesteps available for forecast
        n_days = forecast_timesteps // TIMESTEPS_PER_DAY  # 10 complete days
        
        # Extract forecast data (skip timestep 0 which is NaN)
        forecast_data = data[1:]  # timesteps 1-80 contain 3-hourly precipitation
        
        # Sum 8 consecutive forecast timesteps to get 24-hour accumulations
        daily_accum = np.zeros((n_days,) + data.shape[1:])
        
        for day in range(n_days):
            start_idx = day * TIMESTEPS_PER_DAY
            end_idx = (day + 1) * TIMESTEPS_PER_DAY
            
            if end_idx <= forecast_timesteps:
                # Sum the 8 timesteps (each 3-hour period) to get 24-hour total
                daily_accum[day] = np.sum(forecast_data[start_idx:end_idx], axis=0)
        
        ensemble_24h[member] = daily_accum
        print(f"  {member}: {n_days} days processed")
    
    accum_end_time = time.time()
    print(f"⏱️  Accumulation processing time: {(accum_end_time - accum_start_time):.1f} seconds")
    
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
    plot_file = create_24h_probability_plot(probabilities, lons, lats, n_members, n_days, output_dir)
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
    print(f"   📊 Data Loading:           {(load_end_time - load_start_time):.1f} seconds ({(load_end_time - load_start_time)/total_time*100:.1f}%)")
    print(f"   🔄 24h Accumulation:       {(accum_end_time - accum_start_time):.1f} seconds ({(accum_end_time - accum_start_time)/total_time*100:.1f}%)")
    print(f"   📈 Probability Calculation: {(prob_end_time - prob_start_time):.1f} seconds ({(prob_end_time - prob_start_time)/total_time*100:.1f}%)")
    print(f"   🎨 Plot Generation:        {(plot_end_time - plot_start_time):.1f} seconds ({(plot_end_time - plot_start_time)/total_time*100:.1f}%)")
    print(f"   ⏱️  TOTAL TIME:             {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\n📁 Output directory: {output_dir}")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)