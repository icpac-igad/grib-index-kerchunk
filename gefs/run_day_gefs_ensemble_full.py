#!/usr/bin/env python3
"""
GEFS Full Ensemble Processing for 2025-06-18 00Z Run
This script processes all 30 ensemble members now that reference mappings exist.
Modified version that saves parquet files in organized directories.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


# Import from gefs_util
from gefs_util import generate_axes
from gefs_util import filter_build_grib_tree 
from gefs_util import calculate_time_dimensions
from gefs_util import cs_create_mapped_index
from gefs_util import prepare_zarr_store
from gefs_util import process_unique_groups

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Option 1: Specify forecast hour directly (hours from model run time)
FORECAST_HOUR = 12  # 12 hours from 00Z = 12:00 UTC = 15:00 Nairobi time

# Option 2: Specify local time in target timezone
TARGET_LOCAL_TIME = 14  # 14:00 local time
TARGET_TIMEZONE_OFFSET = 3  # UTC+3 for Nairobi
TARGET_UTC_TIME = TARGET_LOCAL_TIME - TARGET_TIMEZONE_OFFSET  # 11:00 UTC
FORECAST_HOUR_FROM_LOCAL = TARGET_UTC_TIME if TARGET_UTC_TIME >= 0 else TARGET_UTC_TIME + 24

# Calculate timestep index (GEFS uses 3-hour intervals)
TIMESTEP_INDEX = FORECAST_HOUR // 3  # For 12 hours: 12/3 = 4

# Configuration for the specific run
TARGET_DATE_STR = '20250624'
TARGET_RUN = '00'  # 00Z run
REFERENCE_DATE_STR = '20241112'  # Date with existing parquet mappings
ENSEMBLE_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # All 30 members

# Option to keep parquet files
KEEP_PARQUET_FILES = True  # Set to True to keep files, False to delete them

# East Africa bounding box
EA_LAT_MIN, EA_LAT_MAX = -12, 15
EA_LON_MIN, EA_LON_MAX = 25, 52

print(f"🚀 Processing GEFS full ensemble data for {TARGET_DATE_STR} {TARGET_RUN}Z run")
print(f"👥 Ensemble members: {len(ENSEMBLE_MEMBERS)} members (gep01-gep30)")
print(f"📊 Using reference mappings from: {REFERENCE_DATE_STR}")
print(f"🌍 East Africa region: {EA_LAT_MIN}°S-{EA_LAT_MAX}°N, {EA_LON_MIN}°E-{EA_LON_MAX}°E")
print(f"💾 Parquet files will be {'KEPT' if KEEP_PARQUET_FILES else 'DELETED'} after processing")


# Option 1: Specify forecast hour directly (hours from model run time)
FORECAST_HOUR = 12  # 12 hours from 00Z = 12:00 UTC = 15:00 Nairobi time

# Option 2: Specify local time in target timezone
TARGET_LOCAL_TIME = 14  # 14:00 local time
TARGET_TIMEZONE_OFFSET = 3  # UTC+3 for Nairobi
TARGET_UTC_TIME = TARGET_LOCAL_TIME - TARGET_TIMEZONE_OFFSET  # 11:00 UTC
FORECAST_HOUR_FROM_LOCAL = TARGET_UTC_TIME if TARGET_UTC_TIME >= 0 else TARGET_UTC_TIME + 24

# Calculate timestep index (GEFS uses 3-hour intervals)
TIMESTEP_INDEX = FORECAST_HOUR // 3  # For 12 hours: 12/3 = 4

# For 6-hour forecast: TIMESTEP_INDEX = 2 (6/3 = 2)
# For 12-hour forecast: TIMESTEP_INDEX = 4 (12/3 = 4)
# For 24-hour forecast: TIMESTEP_INDEX = 8 (24/3 = 8)

print(f"📍 Time Configuration:")
print(f"   - Model run time: {TARGET_DATE_STR} {TARGET_RUN}Z")
print(f"   - Forecast hour: +{FORECAST_HOUR}h")
print(f"   - UTC time: {(int(TARGET_RUN) + FORECAST_HOUR) % 24:02d}:00 UTC")
print(f"   - Nairobi time: {((int(TARGET_RUN) + FORECAST_HOUR + 3) % 24):02d}:00 EAT")
print(f"   - Timestep index: {TIMESTEP_INDEX}")




def create_output_directory(date_str: str, run: str) -> Path:
    """Create output directory structure for parquet files."""
    output_dir = Path(f"{date_str}_{run}")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir


def create_parquet_file_fixed(zstore: dict, output_parquet_file: str):
    """Fixed version that stores each zarr reference as an individual row."""
    data = []
    
    for key, value in zstore.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        elif isinstance(value, (int, float, np.integer, np.floating)):
            encoded_value = str(value).encode('utf-8')
        else:
            encoded_value = str(value).encode('utf-8')
        
        data.append((key, encoded_value))
    
    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_parquet_file)
    print(f"✅ Fixed parquet file saved: {output_parquet_file} ({len(df)} rows)")
    
    return df


def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling."""
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


def process_single_ensemble_member(member, target_date_str, target_run, reference_date_str,
                                 axes, forecast_dict, time_dims, time_coords, times, valid_times, steps):
    """Process a single ensemble member and return its zarr store."""
    print(f"\n🎯 Processing ensemble member: {member}")
    
    # Define GEFS files for this member
    gefs_files = []
    for hour in [0, 3]:  # Only initial timesteps needed
        gefs_files.append(
            f"s3://noaa-gefs-pds/gefs.{target_date_str}/{target_run}/atmos/pgrb2sp25/"
            f"{member}.t{target_run}z.pgrb2s.0p25.f{hour:03d}"
        )
    
    try:
        # Build GRIB tree from files
        print(f"🔨 Building GRIB tree for {member}...")
        _, deflated_gefs_grib_tree_store = filter_build_grib_tree(gefs_files, forecast_dict)
        print(f"✅ GRIB tree built successfully for {member}")
        
        # Create zarr store using reference mappings
        print(f"🗃️ Creating zarr store for {member}...")
        gcs_bucket_name = 'gik-gefs-aws-tf'
        gcp_service_account_json = 'coiled-data.json'
        
        try:
            # Use the original function with reference date
            gefs_kind = cs_create_mapped_index(
                axes, gcs_bucket_name, target_date_str, member,
                gcp_service_account_json=gcp_service_account_json,
                reference_date_str=reference_date_str
            )
            
            zstore, chunk_index = prepare_zarr_store(deflated_gefs_grib_tree_store, gefs_kind)
            updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, 
                                                 times, valid_times, steps)
            
            print(f"✅ Zarr store created for {member} using reference mappings")
            
            return member, updated_zstore, True
            
        except Exception as e:
            print(f"⚠️ Reference mapping failed for {member}: {e}")
            print(f"🔄 Using direct zarr store for {member}...")
            return member, deflated_gefs_grib_tree_store, False
        
    except Exception as e:
        print(f"❌ Error processing {member}: {e}")
        return member, None, False


def process_ensemble_members_batch(members_batch, target_date_str, target_run, reference_date_str,
                                  axes, forecast_dict, time_dims, time_coords, times, valid_times, steps):
    """Process a batch of ensemble members."""
    results = {}
    
    for member in members_batch:
        member_name, member_store, success = process_single_ensemble_member(
            member, target_date_str, target_run, reference_date_str,
            axes, forecast_dict, time_dims, time_coords, times, valid_times, steps
        )
        
        if member_store is not None:
            results[member_name] = {
                'store': member_store,
                'success': success
            }
    
    return results


def stream_ensemble_precipitation(members_data, variable='tp', output_dir=None):
    """Stream precipitation data for all successful ensemble members."""
    print(f"\n🌧️ Streaming {variable} data for {len(members_data)} ensemble members...")
    
    ensemble_numpy = {}
    ensemble_xarray = {}
    
    for member, data in members_data.items():
        print(f"\n📊 Processing {member}...")
        
        try:
            # Create parquet file for this member
            if output_dir:
                # Save in organized directory structure
                parquet_file = output_dir / f"{member}.par"
                parquet_file_str = str(parquet_file)
            else:
                # Original behavior - save in current directory
                parquet_file_str = f'gefs_{member}_{TARGET_DATE_STR}_{TARGET_RUN}z_fixed.par'
            
            create_parquet_file_fixed(data['store'], parquet_file_str)
            
            # Read and stream data
            zstore = read_parquet_fixed(parquet_file_str)
            
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
                continue
            
            # Extract East Africa region
            ea_data = data_var.sel(
                latitude=slice(EA_LAT_MAX, EA_LAT_MIN),
                longitude=slice(EA_LON_MIN, EA_LON_MAX)
            )
            
            # Compute numpy array
            ea_numpy = ea_data.compute().values
            
            # Store results
            ensemble_numpy[member] = ea_numpy
            ensemble_xarray[member] = ea_data
            
            print(f"✅ {member} data shape: {ea_numpy.shape}")
            
            # Clean up parquet file if not keeping
            if not KEEP_PARQUET_FILES and not output_dir:
                os.remove(parquet_file_str)
                print(f"🗑️ Deleted temporary parquet file")
            
        except Exception as e:
            print(f"❌ Error streaming {member}: {e}")
            continue
    
    print(f"\n✅ Successfully streamed data for {len(ensemble_numpy)} members")
    
    if KEEP_PARQUET_FILES and output_dir:
        # List saved files
        parquet_files = sorted(output_dir.glob("*.par"))
        print(f"\n💾 Saved {len(parquet_files)} parquet files in {output_dir}:")
        for pf in parquet_files[:5]:  # Show first 5
            print(f"   - {pf.name}")
        if len(parquet_files) > 5:
            print(f"   ... and {len(parquet_files) - 5} more")
    
    return ensemble_numpy, ensemble_xarray


def old_create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, variable='tp', output_dir=None):
    """Create comprehensive ensemble comparison plot using correct TP plotting approach."""
    print(f"\n🎨 Creating ensemble comparison plot for {variable}...")
    
    num_members = len(ensemble_numpy)
    if num_members == 0:
        print("❌ No ensemble data to plot")
        return None
    
    # Sort members
    members = sorted(ensemble_numpy.keys())
    
    # Create figure - 6 rows x 5 columns for 30 members
    fig, axes = plt.subplots(6, 5, figsize=(25, 30), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Use timestep 8 (T+24h)
    timestep_idx = 8
    
    # Get coordinates from first member
    first_member = members[0]
    first_xr = ensemble_xarray[first_member]
    lons = first_xr.longitude.values
    lats = first_xr.latitude.values
    
    # Plot each member
    all_data = []
    
    for idx, member in enumerate(members):
        ax = axes[idx]
        member_numpy = ensemble_numpy[member]
        
        if timestep_idx < member_numpy.shape[0]:
            # Get data for timestep (use correct approach from diagnose_gefs_tp_data.py)
            data = member_numpy[timestep_idx]
            
            # Data is already in mm (kg/m²), no conversion needed
            plot_data = data
            
            # Store for ensemble statistics
            all_data.append(plot_data)
            
            # Calculate adaptive colorbar range
            finite_data = plot_data[np.isfinite(plot_data)]
            if len(finite_data) > 0:
                data_max = np.max(finite_data)
                vmax = np.ceil(data_max * 1.2 / 10) * 10  # Round up to nearest 10
                vmax = max(10, vmax)  # Minimum 10mm
            else:
                vmax = 30
            
            # Create plot
            levels = np.linspace(0, vmax, 11)
            im = ax.contourf(lons, lats, plot_data,
                           levels=levels, cmap='Blues',
                           transform=ccrs.PlateCarree(),
                           extend='max')
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            
            # Set extent
            ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])
            
            # Title
            ax.set_title(f'{member}\nMax: {data_max:.1f}mm', fontsize=10)
            
        else:
            ax.text(0.5, 0.5, f'{member}\nNo data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{member}')
    
    # Hide unused subplots
    for idx in range(len(members), len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Total Precipitation (mm)', rotation=270, labelpad=20)
    
    # Main title
    plt.suptitle(f'GEFS Ensemble Total Precipitation Forecast - East Africa\n'
                f'{TARGET_DATE_STR} {TARGET_RUN}Z Run - T+24h ({len(members)} Members)', 
                fontsize=16, y=0.98)
    
    # Save
    if output_dir:
        output_file = output_dir / f'ensemble_all_members_comparison.png'
    else:
        output_file = f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_all_members_comparison.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ Ensemble comparison plot saved: {output_file}")
    plt.close()
    
    # Calculate and save ensemble statistics
    if len(all_data) > 0:
        ensemble_stack = np.stack(all_data, axis=0)
        ensemble_mean = np.mean(ensemble_stack, axis=0)
        ensemble_std = np.std(ensemble_stack, axis=0)
        
        # Create ensemble statistics plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot ensemble mean
        ax = axes[0]
        mean_max = np.nanmax(ensemble_mean)
        vmax_mean = np.ceil(mean_max * 1.2 / 10) * 10
        levels = np.linspace(0, vmax_mean, 11)
        
        im1 = ax.contourf(lons, lats, ensemble_mean,
                         levels=levels, cmap='Blues',
                         transform=ccrs.PlateCarree(),
                         extend='max')
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])
        ax.set_title(f'Ensemble Mean\nMax: {mean_max:.1f}mm')
        
        plt.colorbar(im1, ax=ax, label='Mean Precipitation (mm)')
        
        # Plot ensemble standard deviation
        ax = axes[1]
        std_max = np.nanmax(ensemble_std)
        vmax_std = np.ceil(std_max * 1.2 / 5) * 5
        levels = np.linspace(0, vmax_std, 11)
        
        im2 = ax.contourf(lons, lats, ensemble_std,
                         levels=levels, cmap='Reds',
                         transform=ccrs.PlateCarree(),
                         extend='max')
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])
        ax.set_title(f'Ensemble Std Dev\nMax: {std_max:.1f}mm')
        
        plt.colorbar(im2, ax=ax, label='Std Dev Precipitation (mm)')
        
        plt.suptitle(f'GEFS Ensemble Statistics - East Africa\n'
                    f'{TARGET_DATE_STR} {TARGET_RUN}Z Run - T+24h', fontsize=14)
        
        if output_dir:
            stats_file = output_dir / 'ensemble_statistics.png'
        else:
            stats_file = f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_statistics.png'
        
        plt.tight_layout()
        plt.savefig(str(stats_file), dpi=150, bbox_inches='tight')
        print(f"✅ Ensemble statistics plot saved: {stats_file}")
        plt.close()
        
        # Save statistics as numpy arrays
        if output_dir:
            np.save(output_dir / 'ensemble_mean.npy', ensemble_mean)
            np.save(output_dir / 'ensemble_std.npy', ensemble_std)
        else:
            np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_mean.npy', ensemble_mean)
            np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_std.npy', ensemble_std)
        
        print(f"📊 Ensemble statistics:")
        print(f"   - Members: {len(members)}")
        print(f"   - Mean precipitation range: {np.nanmin(ensemble_mean):.2f} to {np.nanmax(ensemble_mean):.2f} mm")
        print(f"   - Max standard deviation: {np.nanmax(ensemble_std):.2f} mm")
    
    return output_file


def create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, variable='tp', output_dir=None):
    """Create comprehensive ensemble comparison plot using correct TP plotting approach."""
    print(f"\n🎨 Creating ensemble comparison plot for {variable}...")
    
    num_members = len(ensemble_numpy)
    if num_members == 0:
        print("❌ No ensemble data to plot")
        return None
    
    # Sort members
    members = sorted(ensemble_numpy.keys())
    
    # Create figure - 6 rows x 5 columns for 30 members
    fig, axes = plt.subplots(6, 5, figsize=(25, 30), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # CHANGE THIS LINE (previously: timestep_idx = 8)
    timestep_idx = TIMESTEP_INDEX  # Use configured timestep
    
    # Calculate actual times for title
    forecast_hour = timestep_idx * 3
    utc_hour = (int(TARGET_RUN) + forecast_hour) % 24
    nairobi_hour = (utc_hour + 3) % 24
    
    print(f"📊 Plotting data for:")
    print(f"   - Timestep {timestep_idx} (T+{forecast_hour}h)")
    print(f"   - {utc_hour:02d}:00 UTC")
    print(f"   - {nairobi_hour:02d}:00 Nairobi time")
    
    # Get coordinates from first member
    first_member = members[0]
    first_xr = ensemble_xarray[first_member]
    lons = first_xr.longitude.values
    lats = first_xr.latitude.values
    
    # Check if we're plotting accumulated or instantaneous precipitation
    # For shorter periods (6h, 12h), you might want to show accumulation
    if forecast_hour <= 12:
        plot_title_suffix = f"T+{forecast_hour}h ({forecast_hour}-hour accumulation)"
    else:
        plot_title_suffix = f"T+{forecast_hour}h"
    
    # Plot each member
    all_data = []
    
    for idx, member in enumerate(members):
        ax = axes[idx]
        member_numpy = ensemble_numpy[member]
        
        if timestep_idx < member_numpy.shape[0]:
            # Get data for timestep
            data = member_numpy[timestep_idx]
            
            # Data is already in mm (kg/m²), no conversion needed
            plot_data = data
            
            # Store for ensemble statistics
            all_data.append(plot_data)
            
            # Calculate adaptive colorbar range
            finite_data = plot_data[np.isfinite(plot_data)]
            if len(finite_data) > 0:
                data_max = np.max(finite_data)
                # Adjust colorbar scaling for shorter forecasts
                if forecast_hour <= 6:
                    vmax = np.ceil(data_max * 1.2 / 5) * 5  # Round to nearest 5mm
                    vmax = max(5, vmax)  # Minimum 5mm for 6h
                elif forecast_hour <= 12:
                    vmax = np.ceil(data_max * 1.2 / 10) * 10  # Round to nearest 10mm
                    vmax = max(10, vmax)  # Minimum 10mm for 12h
                else:
                    vmax = np.ceil(data_max * 1.2 / 10) * 10
                    vmax = max(10, vmax)
            else:
                vmax = 10 if forecast_hour <= 6 else 20
            
            # Create plot
            levels = np.linspace(0, vmax, 11)
            im = ax.contourf(lons, lats, plot_data,
                           levels=levels, cmap='Blues',
                           transform=ccrs.PlateCarree(),
                           extend='max')
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            
            # Set extent
            ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])
            
            # Title
            ax.set_title(f'{member}\nMax: {data_max:.1f}mm', fontsize=10)
            
        else:
            ax.text(0.5, 0.5, f'{member}\nNo data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{member}')
    
    # Hide unused subplots
    for idx in range(len(members), len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Total Precipitation (mm)', rotation=270, labelpad=20)
    
    # UPDATED TITLE with time information
    plt.suptitle(f'GEFS Ensemble Total Precipitation Forecast - East Africa\n'
                f'{TARGET_DATE_STR} {TARGET_RUN}Z Run - {plot_title_suffix}\n'
                f'Valid: {utc_hour:02d}:00 UTC ({nairobi_hour:02d}:00 Nairobi)', 
                fontsize=16, y=0.98)
    
    # Save with timestep in filename
    if output_dir:
        output_file = output_dir / f'ensemble_all_members_comparison_T{forecast_hour:03d}.png'
    else:
        output_file = f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_all_members_comparison_T{forecast_hour:03d}.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ Ensemble comparison plot saved: {output_file}")
    plt.close()
    
    # Rest of the function remains the same...
    # (ensemble statistics calculation)


# ============================================================================
# OPTIONAL: Function to plot multiple timesteps
# ============================================================================
def plot_multiple_timesteps(ensemble_numpy, ensemble_xarray, timesteps_to_plot, output_dir=None):
    """
    Plot ensemble data for multiple timesteps
    
    Parameters:
    -----------
    timesteps_to_plot : list of tuples
        Each tuple contains (timestep_index, description)
        Example: [(2, "6h"), (4, "12h"), (8, "24h")]
    """
    for timestep_idx, description in timesteps_to_plot:
        print(f"\n{'='*60}")
        print(f"Creating plot for {description} forecast (timestep {timestep_idx})")
        print(f"{'='*60}")
        
        # Temporarily set the global timestep index
        global TIMESTEP_INDEX
        original_timestep = TIMESTEP_INDEX
        TIMESTEP_INDEX = timestep_idx
        
        # Create plot
        create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, 'tp', output_dir)
        
        # Restore original
        TIMESTEP_INDEX = original_timestep


def calculate_ensemble_probabilities(ensemble_numpy, thresholds=[5, 10, 15, 20, 25], timestep_idx=4):
    """
    Calculate empirical probabilities for precipitation exceeding thresholds.
    
    Parameters:
    -----------
    ensemble_numpy : dict
        Dictionary with member names as keys and numpy arrays as values
    thresholds : list
        List of precipitation thresholds in mm
    timestep_idx : int
        Time step index to analyze (default: 4 = 12h forecast)
    
    Returns:
    --------
    probabilities : dict
        Dictionary with thresholds as keys and probability arrays as values
    """
    print(f"\n📊 Calculating ensemble probabilities for {len(thresholds)} thresholds...")
    
    # Get ensemble data for the specified timestep
    members = sorted(ensemble_numpy.keys())
    n_members = len(members)
    
    if n_members == 0:
        print("❌ No ensemble data available")
        return None
    
    # Get data shape from first member
    first_data = ensemble_numpy[members[0]][timestep_idx]
    data_shape = first_data.shape
    
    # Stack all member data for the timestep
    ensemble_stack = np.zeros((n_members, *data_shape))
    
    for i, member in enumerate(members):
        member_data = ensemble_numpy[member]
        if timestep_idx < member_data.shape[0]:
            ensemble_stack[i] = member_data[timestep_idx]
        else:
            ensemble_stack[i] = np.nan
    
    # Calculate probabilities for each threshold
    probabilities = {}
    
    for threshold in thresholds:
        # Count how many members exceed threshold at each grid point
        exceedance_count = np.sum(ensemble_stack >= threshold, axis=0)
        
        # Convert to probability (percentage)
        probability = (exceedance_count / n_members) * 100
        
        probabilities[threshold] = probability
        
        # Print statistics
        max_prob = np.nanmax(probability)
        area_above_50 = np.sum(probability >= 50)
        print(f"   Threshold {threshold}mm: Max probability = {max_prob:.1f}%, "
              f"Grid points with P>=50% = {area_above_50}")
    
    return probabilities, n_members


def plot_probability_map(probability, threshold, lons, lats, timestep_idx, 
                        target_date_str, target_run, n_members, output_dir=None):
    """
    Plot probability map for a single threshold with Nairobi location.
    
    Parameters:
    -----------
    probability : numpy array
        Probability values (0-100%)
    threshold : float
        Precipitation threshold in mm
    lons, lats : numpy arrays
        Longitude and latitude coordinates
    """
    # Nairobi coordinates
    NAIROBI_LAT = -1.2921
    NAIROBI_LON = 36.8219
    
    # Calculate forecast hour and times
    forecast_hour = timestep_idx * 3
    utc_hour = (int(target_run) + forecast_hour) % 24
    nairobi_hour = (utc_hour + 3) % 24
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Define probability levels and colors
    # Using a color scheme from white (0%) through yellows to dark red (100%)
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933', 
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']
    
    # Create contour plot
    cf = ax.contourf(lons, lats, probability, levels=levels, colors=colors,
                     transform=ccrs.PlateCarree(), extend='neither')
    
    # Add contour lines at key probability levels
    cs = ax.contour(lons, lats, probability, levels=[25, 50, 75], 
                    colors='black', linewidths=0.5, alpha=0.5,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d%%')
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='navy')
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.LAND, alpha=0.1, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.1, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')
    
    # Add rivers for context
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5, color='blue')
    
    # Set extent to East Africa
    ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])
    
    # Add Nairobi marker
    # Plot as a star with label
    ax.plot(NAIROBI_LON, NAIROBI_LAT, marker='*', markersize=20, 
            color='red', markeredgecolor='darkred', markeredgewidth=1,
            transform=ccrs.PlateCarree(), zorder=10)
    
    # Add Nairobi label with background
    ax.text(NAIROBI_LON + 0.3, NAIROBI_LAT + 0.3, 'Nairobi', 
            transform=ccrs.PlateCarree(), fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='darkred', alpha=0.8),
            zorder=11)
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label(f'Probability of exceeding {threshold}mm (%)', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    
    # Title with comprehensive information
    plt.title(f'GEFS Ensemble Probability: Precipitation > {threshold}mm\n'
              f'{target_date_str} {target_run}Z Run - T+{forecast_hour}h '
              f'({utc_hour:02d}:00 UTC / {nairobi_hour:02d}:00 EAT)\n'
              f'Based on {n_members} ensemble members',
              fontsize=14, pad=20)
    
    # Add text box with key information
    textstr = f'Max Probability: {np.nanmax(probability):.1f}%\n'
    textstr += f'Area with P≥50%: {np.sum(probability >= 50)} grid points\n'
    textstr += f'Area with P≥75%: {np.sum(probability >= 75)} grid points'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save figure
    if output_dir:
        output_file = output_dir / f'probability_exceeding_{threshold:02d}mm_T{forecast_hour:03d}.png'
    else:
        output_file = f'gefs_probability_{target_date_str}_{target_run}z_{threshold:02d}mm_T{forecast_hour:03d}.png'
    
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ Probability map saved: {output_file}")
    plt.close()
    
    return str(output_file)


def create_probability_summary_plot(probabilities, thresholds, lons, lats, timestep_idx,
                                   target_date_str, target_run, n_members, output_dir=None):
    """
    Create a multi-panel plot showing probabilities for all thresholds.
    """
    # Calculate times
    forecast_hour = timestep_idx * 3
    utc_hour = (int(target_run) + forecast_hour) % 24
    nairobi_hour = (utc_hour + 3) % 24
    
    # Nairobi coordinates
    NAIROBI_LAT = -1.2921
    NAIROBI_LON = 36.8219
    
    # Create figure with subplots
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Common color levels for all subplots
    levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['white', '#FFFFCC', '#FFFF99', '#FFCC66', '#FF9933', 
              '#FF6600', '#FF3300', '#CC0000', '#990000', '#660000']
    
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        probability = probabilities[threshold]
        
        # Create contour plot
        cf = ax.contourf(lons, lats, probability, levels=levels, colors=colors,
                         transform=ccrs.PlateCarree(), extend='neither')
        
        # Add 50% contour line
        cs = ax.contour(lons, lats, probability, levels=[50], 
                       colors='black', linewidths=1, alpha=0.7,
                       transform=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='navy')
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, color='gray')
        ax.add_feature(cfeature.LAND, alpha=0.1)
        
        # Set extent
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX])
        
        # Add Nairobi marker
        ax.plot(NAIROBI_LON, NAIROBI_LAT, marker='*', markersize=0.3, 
                color='red', markeredgecolor='darkred', markeredgewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=10)
        
        # Add title for each subplot
        max_prob = np.nanmax(probability)
        ax.set_title(f'P(TP > {threshold}mm) - Max: {max_prob:.0f}%', fontsize=12)
        
        # Add gridlines for first subplot only
        if i == 0:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.3, color='gray', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
    
    # Hide the 6th subplot if we have 5 thresholds
    if n_thresholds < 6:
        axes[5].set_visible(False)
    
    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Probability (%)', rotation=270, labelpad=20)
    
    # Overall title
    fig.suptitle(f'GEFS Ensemble Exceedance Probabilities - East Africa\n'
                 f'{target_date_str} {target_run}Z Run - T+{forecast_hour}h '
                 f'({utc_hour:02d}:00 UTC / {nairobi_hour:02d}:00 EAT)\n'
                 f'{n_members} ensemble members',
                 fontsize=16, y=0.98)
    
    # Add legend for Nairobi
    nairobi_marker = mpatches.Patch(color='red', label='★ Nairobi')
    plt.figlegend(handles=[nairobi_marker], loc='lower right', 
                 bbox_to_anchor=(0.9, 0.05), fontsize=12)
    
    # Save figure
    if output_dir:
        output_file = output_dir / f'probability_summary_all_thresholds_T{forecast_hour:03d}.png'
    else:
        output_file = f'gefs_probability_summary_{target_date_str}_{target_run}z_T{forecast_hour:03d}.png'
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    print(f"✅ Probability summary plot saved: {output_file}")
    plt.close()
    
    return str(output_file)


def process_ensemble_probabilities(ensemble_numpy, ensemble_xarray, timestep_idx=4, 
                                  thresholds=[5, 10, 15, 20, 25], output_dir=None):
    """
    Main function to process and plot ensemble probabilities.
    
    Parameters:
    -----------
    ensemble_numpy : dict
        Dictionary of ensemble member data
    ensemble_xarray : dict
        Dictionary of xarray datasets
    timestep_idx : int
        Timestep index to analyze (default: 4 = 12h)
    thresholds : list
        List of precipitation thresholds in mm
    output_dir : Path
        Output directory for plots
    """
    print(f"\n{'='*80}")
    print(f"Processing Ensemble Exceedance Probabilities")
    print(f"{'='*80}")
    
    if len(ensemble_numpy) == 0:
        print("❌ No ensemble data available")
        return None
    
    # Get coordinates from first member
    first_member = sorted(ensemble_xarray.keys())[0]
    first_xr = ensemble_xarray[first_member]
    lons = first_xr.longitude.values
    lats = first_xr.latitude.values
    
    # Calculate probabilities
    probabilities, n_members = calculate_ensemble_probabilities(
        ensemble_numpy, thresholds, timestep_idx
    )
    
    if probabilities is None:
        return None
    
    # Create individual plots for each threshold
    print(f"\n📊 Creating individual probability maps...")
    plot_files = []
    
    for threshold in thresholds:
        plot_file = plot_probability_map(
            probabilities[threshold], threshold, lons, lats, timestep_idx,
            TARGET_DATE_STR, TARGET_RUN, n_members, output_dir
        )
        plot_files.append(plot_file)
    
    # Create summary plot
    print(f"\n📊 Creating summary probability plot...")
    summary_file = create_probability_summary_plot(
        probabilities, thresholds, lons, lats, timestep_idx,
        TARGET_DATE_STR, TARGET_RUN, n_members, output_dir
    )
    
    print(f"\n✅ Probability processing complete!")
    print(f"   - Individual plots: {len(plot_files)}")
    print(f"   - Summary plot: {summary_file}")
    
    return probabilities, plot_files, summary_file




def main():
    """Main processing function for full ensemble GEFS data."""
    print("="*80)
    print("GEFS Full Ensemble Processing (30 Members)")
    print("="*80)
    
    start_time = time.time()
    
    # Create output directory structure
    output_dir = None
    if KEEP_PARQUET_FILES:
        output_dir = create_output_directory(TARGET_DATE_STR, TARGET_RUN)
    
    # 1. Generate axes for the target date
    print(f"\n📅 Generating axes for {TARGET_DATE_STR}...")
    axes = generate_axes(TARGET_DATE_STR)
    
    # 2. Define forecast variables
    forecast_dict = {
        "Surface pressure": "PRES:surface",
        "Downward short-wave radiation flux": "DSWRF:surface", 
        "Convective available potential energy": "CAPE:surface",
        "Upward short-wave radiation flux": "USWRF:surface",
        "Total Precipitation": "APCP:surface",
        "Wind speed (gust)": "GUST:surface",
        "2 metre temperature": "TMP:2 m above ground",
        "2 metre relative humidity": "RH:2 m above ground",
        "10 metre U wind component": "UGRD:10 m above ground",
        "10 metre V wind component": "VGRD:10 m above ground",
    }
    
    # 3. Calculate time dimensions (same for all members)
    print(f"\n⏰ Calculating time dimensions...")
    time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
    print(f"✅ Time dimensions: {len(times)} timesteps")
    
    # 4. Process ensemble members in batches
    print(f"\n🚀 Processing {len(ENSEMBLE_MEMBERS)} ensemble members...")
    
    # Process in batches of 5 to manage resources
    batch_size = 5
    all_results = {}
    
    for i in range(0, len(ENSEMBLE_MEMBERS), batch_size):
        batch = ENSEMBLE_MEMBERS[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(ENSEMBLE_MEMBERS) + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches}: {', '.join(batch)}")
        
        batch_results = process_ensemble_members_batch(
            batch, TARGET_DATE_STR, TARGET_RUN, REFERENCE_DATE_STR,
            axes, forecast_dict, time_dims, time_coords, times, valid_times, steps
        )
        
        all_results.update(batch_results)
        
        print(f"✅ Batch {batch_num} completed: {len(batch_results)} successful")
    
    # Summary
    successful = [m for m, data in all_results.items() if data['success']]
    partial = [m for m, data in all_results.items() if not data['success'] and data['store'] is not None]
    failed = [m for m in ENSEMBLE_MEMBERS if m not in all_results]
    
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successful (with reference mappings): {len(successful)} members")
    if successful:
        print(f"      {', '.join(successful[:10])}{'...' if len(successful) > 10 else ''}")
    
    if partial:
        print(f"   ⚠️ Partial (fallback zarr store): {len(partial)} members")
        print(f"      {', '.join(partial[:10])}{'...' if len(partial) > 10 else ''}")
    
    if failed:
        print(f"   ❌ Failed: {len(failed)} members")
        print(f"      {', '.join(failed)}")
    
    if len(all_results) == 0:
        print("\n❌ No ensemble members processed successfully!")
        return False
    
    # 5. Stream precipitation data for all members
    ensemble_numpy, ensemble_xarray = stream_ensemble_precipitation(all_results, 'tp', output_dir)
    
    # 6. Create ensemble comparison plots
    if len(ensemble_numpy) > 0:
        plot_file = create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, 'tp', output_dir)
    # Option 2: Plot multiple timesteps
    timesteps_to_plot = [
        (2, "6h"),   # 06:00 UTC = 09:00 Nairobi
        (4, "12h"),  # 12:00 UTC = 15:00 Nairobi  
        (6, "18h"),  # 18:00 UTC = 21:00 Nairobi
    ]
    plot_multiple_timesteps(ensemble_numpy, ensemble_xarray, timesteps_to_plot, output_dir)

    # Process ensemble probabilities
    if len(ensemble_numpy) > 0:
        # Define precipitation thresholds (mm/3hr)
        thresholds = [5, 10, 15, 20, 25]
        
        # Choose timestep (4 = 12h = 15:00 Nairobi time for 00Z run)
        timestep_idx = 4
        
        # Process probabilities
        probabilities, prob_files, summary_file = process_ensemble_probabilities(
            ensemble_numpy, ensemble_xarray, 
            timestep_idx=timestep_idx,
            thresholds=thresholds,
            output_dir=output_dir
        )

    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ GEFS full ensemble processing completed!")
    print("="*80)
    
    print(f"\n📁 Results:")
    print(f"   - Total processing time: {total_time/60:.1f} minutes")
    print(f"   - Ensemble members with data: {len(ensemble_numpy)}")
    
    if KEEP_PARQUET_FILES and output_dir:
        print(f"   - Output directory: {output_dir}")
        print(f"   - Parquet files: {len(list(output_dir.glob('*.par')))}")
        print(f"   - PNG plots: {len(list(output_dir.glob('*.png')))}")
        print(f"   - NumPy arrays: {len(list(output_dir.glob('*.npy')))}")
    else:
        print(f"   - Parquet files: Temporary (deleted after use)")
        
    if 'plot_file' in locals():
        print(f"   - Ensemble comparison plot: {plot_file}")
    print(f"   - Ensemble statistics: numpy arrays and plots")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Full ensemble processing completed successfully!")
    else:
        print("\n❌ Full ensemble processing failed. Check error messages above.")
        exit(1)