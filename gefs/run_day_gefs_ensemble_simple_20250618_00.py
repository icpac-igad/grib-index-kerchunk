#!/usr/bin/env python3
"""
Simplified GEFS Multi-Ensemble Member Processing for 2025-06-18 00Z Run
This script processes multiple ensemble members independently and then
combines the results for analysis.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Configuration for the specific run
TARGET_DATE_STR = '20250618'
TARGET_RUN = '00'  # 00Z run
REFERENCE_DATE_STR = '20241112'  # Date with existing parquet mappings
ENSEMBLE_MEMBERS = ['gep01', 'gep02', 'gep03', 'gep04', 'gep05']  # Test with 5 members

# East Africa bounding box
EA_LAT_MIN, EA_LAT_MAX = -12, 15
EA_LON_MIN, EA_LON_MAX = 25, 52

print(f"🚀 Processing GEFS ensemble data for {TARGET_DATE_STR} {TARGET_RUN}Z run")
print(f"👥 Ensemble members: {', '.join(ENSEMBLE_MEMBERS)}")
print(f"📊 Using reference mappings from: {REFERENCE_DATE_STR}")
print(f"🌍 East Africa region: {EA_LAT_MIN}°S-{EA_LAT_MAX}°N, {EA_LON_MIN}°E-{EA_LON_MAX}°E")


def process_single_member(member):
    """
    Process a single ensemble member by running the existing script.
    Returns the member name and output files.
    """
    print(f"\n{'='*60}")
    print(f"🎯 Processing ensemble member: {member}")
    print(f"{'='*60}")
    
    # Create a modified version of the script for this member
    script_name = f'run_gefs_{member}_{TARGET_DATE_STR}_{TARGET_RUN}z.py'
    
    # Read the original script
    with open('run_day_gefs_gik_20250618_00.py', 'r') as f:
        script_content = f.read()
    
    # Modify the ensemble member
    script_content = script_content.replace("ENSEMBLE_MEMBER = 'gep01'", f"ENSEMBLE_MEMBER = '{member}'")
    
    # Write modified script
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    # Run the script
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully processed {member}")
            
            # Expected output files
            parquet_file = f'gefs_{member}_{TARGET_DATE_STR}_{TARGET_RUN}z_fixed.par'
            numpy_file = f'gefs_{member}_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_east_africa.npy'
            
            # Check if files exist
            if os.path.exists(parquet_file) and os.path.exists(numpy_file):
                return member, parquet_file, numpy_file
            else:
                print(f"⚠️ Output files not found for {member}")
                return member, None, None
        else:
            print(f"❌ Failed to process {member}")
            print(f"Error: {result.stderr}")
            return member, None, None
            
    except Exception as e:
        print(f"❌ Exception processing {member}: {e}")
        return member, None, None
    finally:
        # Clean up temporary script
        if os.path.exists(script_name):
            os.remove(script_name)


def read_parquet_fixed(parquet_path):
    """
    Read parquet files created by either the old or new format.
    """
    import ast
    
    df = pd.read_parquet(parquet_path)
    
    # Check if it's old format (has 'refs' key with only 2 rows)
    if 'refs' in df['key'].values and len(df) <= 2:
        print("🔧 Detected old format parquet, extracting zarr store...")
        
        # Extract from refs key
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        
        # Parse using ast.literal_eval (Python dict format)
        zstore = ast.literal_eval(refs_value)
        print(f"✅ Extracted {len(zstore)} entries from old format")
        
    else:
        print("📖 Reading new format parquet file...")
        
        # New format - build dict from rows
        zstore = {}
        
        for _, row in df.iterrows():
            key = row['key']
            value = row['value']
            
            # Decode bytes
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            # Parse JSON values back to Python objects
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    # Zarr reference list
                    try:
                        value = json.loads(value)
                    except:
                        pass  # Keep as string if parsing fails
                # Keep JSON strings as strings for zarr metadata
            
            zstore[key] = value
        
        print(f"✅ Loaded {len(zstore)} entries from new format")
    
    # Remove version key if present (causes fsspec issues)
    if 'version' in zstore:
        del zstore['version']
    
    return zstore


def stream_data_for_plotting(parquet_path, variable='tp'):
    """
    Stream GEFS data from parquet and extract East Africa region.
    """
    print(f"🌊 Streaming {variable} data from {parquet_path}...")
    
    # Read parquet
    zstore = read_parquet_fixed(parquet_path)
    
    # Create reference filesystem
    fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                          remote_options={'anon': True})
    mapper = fs.get_mapper("")
    
    # Open as datatree
    dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
    
    # Navigate to variable data
    if variable == 'tp':
        data = dt['/tp/accum/surface'].ds['tp']
    elif variable == 'cape':
        data = dt['/cape/instant/surface'].ds['cape']
    else:
        raise ValueError(f"Variable {variable} not implemented")
    
    # Extract East Africa region
    ea_data = data.sel(
        latitude=slice(EA_LAT_MAX, EA_LAT_MIN),
        longitude=slice(EA_LON_MIN, EA_LON_MAX)
    )
    
    # Compute and return
    ea_numpy = ea_data.compute().values
    
    return ea_numpy, ea_data


def create_ensemble_comparison_plot(ensemble_data, variable='tp'):
    """
    Create a plot comparing all ensemble members using the correct plotting approach
    from diagnose_gefs_tp_data.py.
    """
    print(f"\n🎨 Creating ensemble comparison plot for {variable}...")
    
    num_members = len(ensemble_data)
    if num_members == 0:
        print("❌ No ensemble data to plot")
        return
    
    # Sort members
    members = sorted(ensemble_data.keys())
    
    # Create figure - 2 rows, 3 columns (5 members + 1 mean)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Use timestep index 8 (T+24h) - after initial spin-up
    timestep_idx = 8
    
    # Get coordinates from first member
    first_member = members[0]
    _, first_xr = ensemble_data[first_member]
    lons = first_xr.longitude.values
    lats = first_xr.latitude.values
    
    # Collect data for ensemble mean
    all_member_data = []
    
    # Plot each member
    for idx, member in enumerate(members):
        ax = axes[idx]
        member_numpy, member_xr = ensemble_data[member]
        
        # Get data for timestep (skip first timestep for TP)
        if timestep_idx < member_numpy.shape[0]:
            data = member_numpy[timestep_idx]
            
            # Check if data is already in mm (kg/m²)
            if hasattr(member_xr, 'attrs') and 'GRIB_units' in member_xr.attrs:
                grib_units = member_xr.attrs['GRIB_units']
                if grib_units == 'kg m**-2':
                    # Data is already in mm
                    plot_data = data
                else:
                    # Convert if needed
                    plot_data = data * 1000
            else:
                # Assume mm if units not specified
                plot_data = data
            
            # Store for ensemble mean
            all_member_data.append(plot_data)
            
            # Handle NaN values
            finite_data = plot_data[np.isfinite(plot_data)]
            if len(finite_data) > 0:
                data_max = np.max(finite_data)
                # Set vmax with some headroom
                vmax = np.ceil(data_max * 1.2 / 10) * 10  # Round up to nearest 10
                vmax = max(10, vmax)  # Minimum 10mm
            else:
                vmax = 50
            
            # Create plot with adaptive colorbar
            levels = np.linspace(0, vmax, 11)
            im = ax.contourf(lons, lats, plot_data,
                           levels=levels, cmap='Blues', 
                           transform=ccrs.PlateCarree(),
                           extend='max')
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.add_feature(cfeature.OCEAN, alpha=0.1)
            ax.add_feature(cfeature.LAKES, alpha=0.3)
            
            # Set extent
            ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX], 
                         crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            
            # Title
            ax.set_title(f'{member} - T+24h\nMax: {data_max:.1f} mm')
    
    # Plot ensemble mean in the last panel
    if len(all_member_data) > 0:
        ax = axes[5]
        ensemble_mean = np.mean(all_member_data, axis=0)
        
        # Plot mean
        finite_mean = ensemble_mean[np.isfinite(ensemble_mean)]
        if len(finite_mean) > 0:
            mean_max = np.max(finite_mean)
            vmax = np.ceil(mean_max * 1.2 / 10) * 10
            vmax = max(10, vmax)
        else:
            vmax = 50
            
        levels = np.linspace(0, vmax, 11)
        im = ax.contourf(lons, lats, ensemble_mean,
                       levels=levels, cmap='Blues', 
                       transform=ccrs.PlateCarree(),
                       extend='max')
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.add_feature(cfeature.OCEAN, alpha=0.1)
        ax.add_feature(cfeature.LAKES, alpha=0.3)
        
        # Set extent
        ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX], 
                     crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Title
        ax.set_title(f'Ensemble Mean - T+24h\nMax: {mean_max:.1f} mm')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Total Precipitation (mm)', rotation=270, labelpad=20)
    
    # Main title
    plt.suptitle(f'GEFS Ensemble Total Precipitation Forecast - East Africa\n'
                f'{TARGET_DATE_STR} {TARGET_RUN}Z Run - T+24h ({num_members} Members)', 
                fontsize=16, y=0.98)
    
    # Save
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    output_file = f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_comparison_fixed.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Ensemble comparison plot saved: {output_file}")
    plt.close()
    
    return output_file


def calculate_ensemble_statistics(ensemble_numpy_files):
    """
    Calculate ensemble statistics from numpy files.
    """
    print(f"\n📊 Calculating ensemble statistics...")
    
    # Load all numpy arrays
    arrays = []
    members = []
    
    for member, numpy_file in ensemble_numpy_files.items():
        if numpy_file and os.path.exists(numpy_file):
            data = np.load(numpy_file)
            arrays.append(data)
            members.append(member)
            print(f"   ✅ Loaded {member}: shape {data.shape}")
    
    if len(arrays) == 0:
        print("❌ No data arrays to calculate statistics")
        return None
    
    # Stack arrays
    ensemble_stack = np.stack(arrays, axis=0)  # Shape: (members, time, lat, lon)
    print(f"📊 Ensemble stack shape: {ensemble_stack.shape}")
    
    # Calculate statistics
    ensemble_mean = np.mean(ensemble_stack, axis=0)
    ensemble_std = np.std(ensemble_stack, axis=0)
    ensemble_min = np.min(ensemble_stack, axis=0)
    ensemble_max = np.max(ensemble_stack, axis=0)
    
    # Save statistics
    np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_mean.npy', ensemble_mean)
    np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_std.npy', ensemble_std)
    np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_min.npy', ensemble_min)
    np.save(f'gefs_ensemble_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_max.npy', ensemble_max)
    
    print(f"✅ Ensemble statistics saved:")
    print(f"   - Mean range: {np.nanmin(ensemble_mean):.2f} to {np.nanmax(ensemble_mean):.2f} mm")
    print(f"   - Max std: {np.nanmax(ensemble_std):.2f} mm")
    print(f"   - Overall max: {np.nanmax(ensemble_max):.2f} mm")
    
    return {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'min': ensemble_min,
        'max': ensemble_max,
        'members': members
    }


def main():
    """
    Main processing function for ensemble GEFS data.
    """
    print("="*80)
    print("GEFS Simplified Multi-Ensemble Processing")
    print("="*80)
    
    # Process each ensemble member independently
    print(f"\n🚀 Processing {len(ENSEMBLE_MEMBERS)} ensemble members...")
    
    results = {}
    parquet_files = {}
    numpy_files = {}
    
    # Process sequentially to avoid resource conflicts
    for member in ENSEMBLE_MEMBERS:
        member_name, parquet_file, numpy_file = process_single_member(member)
        
        if parquet_file and numpy_file:
            results[member_name] = True
            parquet_files[member_name] = parquet_file
            numpy_files[member_name] = numpy_file
        else:
            results[member_name] = False
    
    # Summary
    successful = [m for m, success in results.items() if success]
    failed = [m for m, success in results.items() if not success]
    
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successful: {len(successful)} members - {', '.join(successful)}")
    if failed:
        print(f"   ❌ Failed: {len(failed)} members - {', '.join(failed)}")
    
    if len(successful) == 0:
        print("\n❌ No ensemble members processed successfully!")
        return False
    
    # Read data for plotting
    print(f"\n📖 Reading data for ensemble comparison...")
    ensemble_data = {}
    
    for member, parquet_file in parquet_files.items():
        try:
            numpy_data, xr_data = stream_data_for_plotting(parquet_file, 'tp')
            ensemble_data[member] = (numpy_data, xr_data)
            print(f"   ✅ Loaded {member} data: shape {numpy_data.shape}")
        except Exception as e:
            print(f"   ❌ Failed to load {member}: {e}")
    
    # Create ensemble comparison plot
    if len(ensemble_data) > 0:
        plot_file = create_ensemble_comparison_plot(ensemble_data, 'tp')
    
    # Calculate ensemble statistics
    stats = calculate_ensemble_statistics(numpy_files)
    
    print("\n" + "="*80)
    print("✅ GEFS ensemble processing completed!")
    print("="*80)
    
    print(f"\n📁 Output files:")
    print(f"   Individual parquet files: {len(parquet_files)}")
    print(f"   Individual numpy arrays: {len(numpy_files)}")
    if 'plot_file' in locals():
        print(f"   Ensemble comparison plot: {plot_file}")
    print(f"   Ensemble statistics: mean, std, min, max numpy arrays")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All ensemble processing completed successfully!")
    else:
        print("\n❌ Ensemble processing failed. Check error messages above.")
        exit(1)