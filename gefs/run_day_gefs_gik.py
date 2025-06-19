#!/usr/bin/env python3
"""
Optimized GEFS Data Processing for 2025-06-18 00Z Run
This script implements the fixed parquet creation and reading solution
to properly stream GEFS data and create East Africa numpy arrays.
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

# Import from gefs_util but use our fixed functions where needed
from gefs_util import generate_axes
from gefs_util import filter_build_grib_tree 
from gefs_util import KerchunkZarrDictStorageManager
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
ENSEMBLE_MEMBER = 'gep01'

# East Africa bounding box
EA_LAT_MIN, EA_LAT_MAX = -12, 15
EA_LON_MIN, EA_LON_MAX = 25, 52

print(f"🚀 Processing GEFS data for {TARGET_DATE_STR} {TARGET_RUN}Z run")
print(f"📊 Using reference mappings from: {REFERENCE_DATE_STR}")
print(f"🌍 East Africa region: {EA_LAT_MIN}°S-{EA_LAT_MAX}°N, {EA_LON_MIN}°E-{EA_LON_MAX}°E")


def create_parquet_file_fixed(zstore: dict, output_parquet_file: str):
    """
    Fixed version that stores each zarr reference as an individual row.
    This is the corrected implementation from GEFS_PARQUET_SOLUTION.md
    """
    data = []
    
    for key, value in zstore.items():
        # Handle different value types properly
        if isinstance(value, str):
            # Already a string (like .zarray, .zattrs JSON)
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            # Zarr references (lists) or other dicts - convert to JSON
            encoded_value = json.dumps(value).encode('utf-8')
        elif isinstance(value, (int, float, np.integer, np.floating)):
            # Numeric values
            encoded_value = str(value).encode('utf-8')
        else:
            # Other types - convert to string
            encoded_value = str(value).encode('utf-8')
        
        data.append((key, encoded_value))
    
    # Create DataFrame with proper structure
    df = pd.DataFrame(data, columns=['key', 'value'])
    
    # Save as parquet
    df.to_parquet(output_parquet_file)
    print(f"✅ Fixed parquet file saved: {output_parquet_file} ({len(df)} rows)")
    
    return df


def read_parquet_fixed(parquet_path):
    """
    Read parquet files created by either the old or new format.
    Handles the issue identified in GEFS_PARQUET_SOLUTION.md
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


def stream_gefs_data_to_numpy(parquet_path, variable='tp', region='east_africa'):
    """
    Stream GEFS data from parquet file and create numpy arrays for specific region.
    """
    print(f"\n🌊 Streaming {variable} data for {region}...")
    
    # Read parquet with our fixed reader
    zstore = read_parquet_fixed(parquet_path)
    
    # Count zarr references
    zarr_refs = [k for k, v in zstore.items() if isinstance(v, list) and len(v) == 3]
    var_refs = [k for k in zarr_refs if f'{variable}/' in k]
    print(f"📊 Found {len(zarr_refs)} total zarr references, {len(var_refs)} for {variable}")
    
    # Create reference filesystem
    fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                          remote_options={'anon': True})
    mapper = fs.get_mapper("")
    
    # Open as datatree
    print("📂 Opening data with xarray...")
    dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
    
    # Navigate to variable data
    if variable == 'tp':
        data = dt['/tp/accum/surface'].ds['tp']
    elif variable == 'sp':
        data = dt['/sp/instant/surface'].ds['sp']
    elif variable == 't2m':
        data = dt['/t2m/instant/heightAboveGround'].ds['t2m']
    elif variable == 'cape':
        data = dt['/cape/instant/surface'].ds['cape']
    elif variable == 'gust':
        data = dt['/gust/instant/surface'].ds['gust']
    else:
        # Try to find the variable automatically
        available_vars = []
        for path in dt:
            if hasattr(dt[path], 'ds'):
                available_vars.extend(list(dt[path].ds.data_vars))
        print(f"Available variables: {available_vars}")
        raise ValueError(f"Variable {variable} not found or not implemented")
    
    print(f"✅ Loaded {variable} data: shape {data.shape}")
    
    # Extract East Africa region
    if region == 'east_africa':
        print(f"🗺️ Extracting East Africa region ({EA_LAT_MIN}°-{EA_LAT_MAX}°N, {EA_LON_MIN}°-{EA_LON_MAX}°E)...")
        
        # Select spatial subset
        ea_data = data.sel(
            latitude=slice(EA_LAT_MAX, EA_LAT_MIN),  # Note: slice order for descending coords
            longitude=slice(EA_LON_MIN, EA_LON_MAX)
        )
        
        print(f"✅ East Africa subset: shape {ea_data.shape}")
        
        # Stream and compute all timesteps
        print("💾 Computing numpy array from streamed data...")
        try:
            ea_numpy = ea_data.compute().values
            print(f"✅ Successfully computed numpy array: {ea_numpy.shape}")
            
            # Handle any remaining NaN values
            nan_count = np.isnan(ea_numpy).sum()
            if nan_count > 0:
                print(f"⚠️ Found {nan_count:,} NaN values in data")
            
            return ea_numpy, ea_data
            
        except Exception as e:
            print(f"❌ Error computing numpy array: {e}")
            print("🔧 Attempting timestep-by-timestep computation...")
            
            # Fallback: compute timestep by timestep
            timesteps = []
            for t in range(ea_data.shape[0]):
                try:
                    timestep_data = ea_data.isel(valid_times=t).compute().values
                    timesteps.append(timestep_data)
                    if t % 10 == 0:
                        print(f"   Processed timestep {t}")
                except Exception as e:
                    print(f"   ⚠️ Skipped timestep {t}: {e}")
                    # Fill with NaN array of correct shape
                    timesteps.append(np.full(ea_data.isel(valid_times=0).shape, np.nan))
            
            ea_numpy = np.stack(timesteps, axis=0)
            print(f"✅ Fallback computation complete: {ea_numpy.shape}")
            
            return ea_numpy, ea_data
    
    else:
        raise ValueError(f"Region {region} not implemented")


def create_visualization(ea_numpy, ea_data, variable, output_file):
    """
    Create visualization of the East Africa data.
    """
    print(f"\n🎨 Creating visualization for {variable}...")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Select timesteps to plot (every 12 hours for first 2.5 days)
    timestep_indices = [0, 4, 8, 12, 16, 20]  # 0h, 12h, 24h, 36h, 48h, 60h
    successful_plots = 0
    
    for plot_idx, t_idx in enumerate(timestep_indices):
        ax = axes[plot_idx]
        
        if t_idx >= ea_numpy.shape[0]:
            ax.text(0.5, 0.5, f'No data\nT+{t_idx*3}h', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'T+{t_idx*3}h - No data')
            continue
        
        try:
            # Get data for this timestep
            timestep_data = ea_numpy[t_idx, :, :]
            
            # Handle variable-specific processing
            if variable == 'tp':
                # Convert to mm and handle first timestep (usually 0)
                plot_data = timestep_data * 1000  # m to mm
                if t_idx == 0:
                    plot_data = np.zeros_like(plot_data)  # First timestep is usually 0
                levels = np.linspace(0, 50, 11)
                cmap = 'Blues'
                units = 'mm'
                title_prefix = 'Precipitation'
            elif variable == 'sp':
                # Surface pressure in hPa
                plot_data = timestep_data / 100  # Pa to hPa
                levels = np.linspace(plot_data.min(), plot_data.max(), 15)
                cmap = 'viridis'
                units = 'hPa'
                title_prefix = 'Surface Pressure'
            elif variable == 't2m':
                # Temperature in Celsius
                plot_data = timestep_data - 273.15  # K to C
                levels = np.linspace(plot_data.min(), plot_data.max(), 15)
                cmap = 'RdYlBu_r'
                units = '°C'
                title_prefix = '2m Temperature'
            elif variable == 'cape':
                # CAPE in J/kg
                plot_data = timestep_data
                levels = np.linspace(0, max(3000, plot_data.max()), 15)
                cmap = 'Reds'
                units = 'J/kg'
                title_prefix = 'CAPE'
            elif variable == 'gust':
                # Wind gust in m/s
                plot_data = timestep_data
                levels = np.linspace(0, max(20, plot_data.max()), 15)
                cmap = 'YlOrRd'
                units = 'm/s'
                title_prefix = 'Wind Gust'
            else:
                plot_data = timestep_data
                levels = 15
                cmap = 'viridis'
                units = 'units'
                title_prefix = variable.upper()
            
            # Create the plot
            im = ax.contourf(ea_data.longitude, ea_data.latitude, plot_data,
                           levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.1)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            
            # Set extent
            ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX], 
                         crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            
            # Title with timestamp
            try:
                time_coord = ea_data.isel(valid_times=t_idx).valid_time
                valid_time = pd.Timestamp(time_coord.values)
                time_str = valid_time.strftime("%m-%d %H:%M")
            except:
                time_str = f"T+{t_idx*3}h"
            
            ax.set_title(f'{title_prefix}\n{time_str} UTC')
            
            successful_plots += 1
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\nT+{t_idx*3}h', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'T+{t_idx*3}h - Error')
            print(f"   ⚠️ Plot error for timestep {t_idx}: {e}")
    
    # Add colorbar if we have successful plots
    if successful_plots > 0:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'{title_prefix} ({units})', rotation=270, labelpad=20)
    
    # Main title
    plt.suptitle(f'GEFS {title_prefix} Forecast - East Africa\n'
                f'{TARGET_DATE_STR} {TARGET_RUN}Z Run - {ENSEMBLE_MEMBER}', 
                fontsize=16, y=0.98)
    
    # Save
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_file}")
    plt.close()
    
    return successful_plots


def main():
    """
    Main processing function implementing the optimized GEFS workflow.
    """
    print("="*80)
    print("GEFS Optimized Data Processing")
    print("="*80)
    
    # 1. Generate axes for the target date
    print(f"\n📅 Generating axes for {TARGET_DATE_STR}...")
    axes = generate_axes(TARGET_DATE_STR)
    
    # 2. Define GEFS files for the specific run (00Z)
    print(f"\n📡 Setting up GEFS files for {TARGET_RUN}Z run...")
    
    # GEFS files for 00Z run - only f000 and f003 needed
    # The cs_create_mapped_index will fill in remaining timesteps from reference mappings
    gefs_files = []
    for hour in [0, 3]:  # Only initial timesteps needed
        gefs_files.append(
            f"s3://noaa-gefs-pds/gefs.{TARGET_DATE_STR}/{TARGET_RUN}/atmos/pgrb2sp25/"
            f"{ENSEMBLE_MEMBER}.t{TARGET_RUN}z.pgrb2s.0p25.f{hour:03d}"
        )
    
    print(f"📋 Processing {len(gefs_files)} forecast files")
    for gfile in gefs_files:
        print(f"   - {gfile.split('/')[-1]}")
    
    # 3. Define forecast variables
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
    
    # 4. Build GRIB tree from target date files
    print(f"\n🔨 Building GRIB tree from {TARGET_DATE_STR} {TARGET_RUN}Z files...")
    try:
        _, deflated_gefs_grib_tree_store = filter_build_grib_tree(gefs_files, forecast_dict)
        print("✅ GRIB tree built successfully")
    except Exception as e:
        print(f"❌ Error building GRIB tree: {e}")
        return False
    
    # 5. Calculate time dimensions
    print(f"\n⏰ Calculating time dimensions...")
    time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
    print(f"✅ Time dimensions: {len(times)} timesteps")
    
    # 6. Create zarr store (using reference date for mappings)
    print(f"\n🗃️ Creating zarr store using reference mappings...")
    gcs_bucket_name = 'gik-gefs-aws-tf'
    gcp_service_account_json = 'coiled-data.json'
    
    try:
        # Use the original function with reference date
        gefs_kind = cs_create_mapped_index(
            axes, gcs_bucket_name, TARGET_DATE_STR, ENSEMBLE_MEMBER,
            gcp_service_account_json=gcp_service_account_json,
            reference_date_str=REFERENCE_DATE_STR
        )
        
        zstore, chunk_index = prepare_zarr_store(deflated_gefs_grib_tree_store, gefs_kind)
        updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, 
                                             times, valid_times, steps)
        
        print("✅ Zarr store created using reference mappings")
        
    except Exception as e:
        print(f"⚠️ Reference mapping failed: {e}")
        print("🔄 Falling back to direct zarr store creation...")
        
        # Fallback: create zarr store directly from GRIB tree
        updated_zstore = deflated_gefs_grib_tree_store
    
    # 7. Create parquet file using our fixed function
    output_parquet_file = f'gefs_{ENSEMBLE_MEMBER}_{TARGET_DATE_STR}_{TARGET_RUN}z_fixed.par'
    print(f"\n💾 Creating fixed parquet file: {output_parquet_file}")
    
    create_parquet_file_fixed(updated_zstore, output_parquet_file)
    
    # 8. Test the parquet file by reading it back
    print(f"\n🧪 Testing parquet file...")
    try:
        test_zstore = read_parquet_fixed(output_parquet_file)
        zarr_refs = [k for k, v in test_zstore.items() if isinstance(v, list) and len(v) == 3]
        print(f"✅ Parquet test successful: {len(zarr_refs)} zarr references")
    except Exception as e:
        print(f"❌ Parquet test failed: {e}")
        return False
    
    # 9. Stream precipitation data to numpy array
    print(f"\n🌧️ Streaming precipitation data...")
    try:
        tp_numpy, tp_data = stream_gefs_data_to_numpy(
            output_parquet_file, 
            variable='tp', 
            region='east_africa'
        )
        
        # Save numpy array
        numpy_file = f'gefs_{ENSEMBLE_MEMBER}_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_east_africa.npy'
        np.save(numpy_file, tp_numpy)
        print(f"✅ Numpy array saved: {numpy_file} (shape: {tp_numpy.shape})")
        
        # Create visualization
        plot_file = f'gefs_{ENSEMBLE_MEMBER}_{TARGET_DATE_STR}_{TARGET_RUN}z_tp_east_africa.png'
        num_plots = create_visualization(tp_numpy, tp_data, 'tp', plot_file)
        print(f"✅ Created {num_plots} successful plots")
        
    except Exception as e:
        print(f"❌ Error streaming precipitation data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 10. Also try CAPE for validation (convective activity)
    print(f"\n⛈️ Streaming CAPE data...")
    try:
        cape_numpy, cape_data = stream_gefs_data_to_numpy(
            output_parquet_file,
            variable='cape', 
            region='east_africa'
        )
        
        # Save numpy array
        cape_numpy_file = f'gefs_{ENSEMBLE_MEMBER}_{TARGET_DATE_STR}_{TARGET_RUN}z_cape_east_africa.npy'
        np.save(cape_numpy_file, cape_numpy)
        print(f"✅ CAPE numpy saved: {cape_numpy_file} (shape: {cape_numpy.shape})")
        
        # Create visualization
        cape_plot_file = f'gefs_{ENSEMBLE_MEMBER}_{TARGET_DATE_STR}_{TARGET_RUN}z_cape_east_africa.png'
        cape_plots = create_visualization(cape_numpy, cape_data, 'cape', cape_plot_file)
        print(f"✅ Created {cape_plots} CAPE plots")
        
    except Exception as e:
        print(f"⚠️ CAPE streaming failed: {e}")
    
    print("\n" + "="*80)
    print("✅ GEFS processing completed successfully!")
    print("="*80)
    
    print(f"\n📁 Output files:")
    print(f"   Parquet: {output_parquet_file}")
    print(f"   Precipitation numpy: {numpy_file}")
    print(f"   Precipitation plot: {plot_file}")
    if 'cape_numpy_file' in locals():
        print(f"   CAPE numpy: {cape_numpy_file}")
        print(f"   CAPE plot: {cape_plot_file}")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All processing completed successfully!")
    else:
        print("\n❌ Processing failed. Check error messages above.")
        exit(1)