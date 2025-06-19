#!/usr/bin/env python3
"""
GEFS Full Ensemble Processing for 2025-06-18 00Z Run
This script processes all 30 ensemble members now that reference mappings exist.
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

# Configuration for the specific run
TARGET_DATE_STR = '20250618'
TARGET_RUN = '00'  # 00Z run
REFERENCE_DATE_STR = '20241112'  # Date with existing parquet mappings
ENSEMBLE_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # All 30 members

# East Africa bounding box
EA_LAT_MIN, EA_LAT_MAX = -12, 15
EA_LON_MIN, EA_LON_MAX = 25, 52

print(f"🚀 Processing GEFS full ensemble data for {TARGET_DATE_STR} {TARGET_RUN}Z run")
print(f"👥 Ensemble members: {len(ENSEMBLE_MEMBERS)} members (gep01-gep30)")
print(f"📊 Using reference mappings from: {REFERENCE_DATE_STR}")
print(f"🌍 East Africa region: {EA_LAT_MIN}°S-{EA_LAT_MAX}°N, {EA_LON_MIN}°E-{EA_LON_MAX}°E")


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


def stream_ensemble_precipitation(members_data, variable='tp'):
    """Stream precipitation data for all successful ensemble members."""
    print(f"\n🌧️ Streaming {variable} data for {len(members_data)} ensemble members...")
    
    ensemble_numpy = {}
    ensemble_xarray = {}
    
    for member, data in members_data.items():
        print(f"\n📊 Processing {member}...")
        
        try:
            # Create parquet file for this member
            parquet_file = f'gefs_{member}_{TARGET_DATE_STR}_{TARGET_RUN}z_fixed.par'
            create_parquet_file_fixed(data['store'], parquet_file)
            
            # Read and stream data
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
            
            # Clean up parquet file
            os.remove(parquet_file)
            
        except Exception as e:
            print(f"❌ Error streaming {member}: {e}")
            continue
    
    print(f"\n✅ Successfully streamed data for {len(ensemble_numpy)} members")
    return ensemble_numpy, ensemble_xarray


def create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, variable='tp'):
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
    output_file = f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_all_members_comparison.png'
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
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
        
        stats_file = f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_statistics.png'
        plt.tight_layout()
        plt.savefig(stats_file, dpi=150, bbox_inches='tight')
        print(f"✅ Ensemble statistics plot saved: {stats_file}")
        plt.close()
        
        # Save statistics as numpy arrays
        np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_mean.npy', ensemble_mean)
        np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_std.npy', ensemble_std)
        
        print(f"📊 Ensemble statistics:")
        print(f"   - Members: {len(members)}")
        print(f"   - Mean precipitation range: {np.nanmin(ensemble_mean):.2f} to {np.nanmax(ensemble_mean):.2f} mm")
        print(f"   - Max standard deviation: {np.nanmax(ensemble_std):.2f} mm")
    
    return output_file


def main():
    """Main processing function for full ensemble GEFS data."""
    print("="*80)
    print("GEFS Full Ensemble Processing (30 Members)")
    print("="*80)
    
    start_time = time.time()
    
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
    ensemble_numpy, ensemble_xarray = stream_ensemble_precipitation(all_results, 'tp')
    
    # 6. Create ensemble comparison plots
    if len(ensemble_numpy) > 0:
        plot_file = create_ensemble_comparison_plot(ensemble_numpy, ensemble_xarray, 'tp')
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ GEFS full ensemble processing completed!")
    print("="*80)
    
    print(f"\n📁 Results:")
    print(f"   - Total processing time: {total_time/60:.1f} minutes")
    print(f"   - Ensemble members with data: {len(ensemble_numpy)}")
    print(f"   - Individual parquet files: {len(all_results)}")
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