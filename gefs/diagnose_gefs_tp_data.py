#!/usr/bin/env python3
"""
Diagnose GEFS TP data issues and create corrected plots
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json
import os
import warnings
import fsspec
import ast

warnings.filterwarnings('ignore')
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

EA_LAT_MIN, EA_LAT_MAX = -12, 15
EA_LON_MIN, EA_LON_MAX = 25, 52

def diagnose_numpy_array(npy_file):
    """
    Diagnose the numpy array data
    """
    print(f"\n🔍 Diagnosing numpy array: {npy_file}")
    
    array = np.load(npy_file)
    print(f"Array shape: {array.shape}")
    print(f"Array dtype: {array.dtype}")
    
    # Overall statistics
    total_values = array.size
    nan_count = np.isnan(array).sum()
    finite_count = np.isfinite(array).sum()
    
    print(f"\nOverall statistics:")
    print(f"  Total values: {total_values:,}")
    print(f"  NaN values: {nan_count:,} ({nan_count/total_values*100:.1f}%)")
    print(f"  Finite values: {finite_count:,} ({finite_count/total_values*100:.1f}%)")
    
    if finite_count > 0:
        finite_data = array[np.isfinite(array)]
        print(f"  Min finite value: {finite_data.min():.6f}")
        print(f"  Max finite value: {finite_data.max():.6f}")
        print(f"  Mean finite value: {finite_data.mean():.6f}")
        print(f"  Std finite value: {finite_data.std():.6f}")
    
    # Per-timestep analysis
    print(f"\n📊 Per-timestep analysis:")
    for t in range(min(10, array.shape[0])):  # First 10 timesteps
        timestep = array[t, :, :]
        nan_count_t = np.isnan(timestep).sum()
        finite_count_t = np.isfinite(timestep).sum()
        
        if finite_count_t > 0:
            min_val = np.nanmin(timestep)
            max_val = np.nanmax(timestep)
            mean_val = np.nanmean(timestep)
            print(f"  T{t:02d}: {finite_count_t:5d} finite, {nan_count_t:5d} NaN, range: {min_val:8.3f} - {max_val:8.3f}, mean: {mean_val:8.3f}")
        else:
            print(f"  T{t:02d}: {finite_count_t:5d} finite, {nan_count_t:5d} NaN, ALL NaN")
    
    return array

def read_parquet_fixed(parquet_path):
    """Read parquet file"""
    df = pd.read_parquet(parquet_path)
    
    if 'refs' in df['key'].values and len(df) <= 2:
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
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
    
    if 'version' in zstore:
        del zstore['version']
    
    return zstore

def investigate_raw_data(parquet_path):
    """
    Investigate the raw data from parquet to understand units and scaling
    """
    print(f"\n🧪 Investigating raw data from: {parquet_path}")
    
    zstore = read_parquet_fixed(parquet_path)
    
    # Create reference filesystem
    fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3',
                          remote_options={'anon': True})
    mapper = fs.get_mapper("")
    
    # Open as datatree
    dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
    tp_data = dt['/tp/accum/surface'].ds['tp']
    
    print(f"Raw data info:")
    print(f"  Shape: {tp_data.shape}")
    print(f"  Dims: {tp_data.dims}")
    print(f"  Coords: {list(tp_data.coords)}")
    
    # Check attributes for units
    if hasattr(tp_data, 'attrs'):
        print(f"  Attributes: {tp_data.attrs}")
    
    # Subset to East Africa
    tp_ea = tp_data.sel(
        latitude=slice(EA_LAT_MAX, EA_LAT_MIN),
        longitude=slice(EA_LON_MIN, EA_LON_MAX)
    )
    
    print(f"  EA subset shape: {tp_ea.shape}")
    
    # Test a few specific timesteps
    test_timesteps = [0, 1, 5, 10, 20]
    
    print(f"\n🎯 Testing specific timesteps:")
    raw_stats = {}
    
    for t in test_timesteps:
        if t >= tp_ea.shape[0]:
            continue
            
        print(f"\nTimestep {t}:")
        try:
            timestep_data = tp_ea.isel(valid_times=t)
            print(f"  Computing...")
            values = timestep_data.compute().values
            
            nan_count = np.isnan(values).sum()
            finite_count = np.isfinite(values).sum()
            total_count = values.size
            
            print(f"  Total pixels: {total_count}")
            print(f"  NaN pixels: {nan_count} ({nan_count/total_count*100:.1f}%)")
            print(f"  Finite pixels: {finite_count} ({finite_count/total_count*100:.1f}%)")
            
            if finite_count > 0:
                finite_values = values[np.isfinite(values)]
                min_val = finite_values.min()
                max_val = finite_values.max()
                mean_val = finite_values.mean()
                
                print(f"  Raw range: {min_val:.6e} to {max_val:.6e}")
                print(f"  Raw mean: {mean_val:.6e}")
                
                # Check if values look like they're already in mm or still in m
                if max_val < 1.0:
                    print(f"  → Likely in meters (max < 1.0)")
                    print(f"  → Converting to mm: {min_val*1000:.3f} to {max_val*1000:.3f}")
                else:
                    print(f"  → Likely already in mm or other units (max ≥ 1.0)")
                
                raw_stats[t] = {
                    'min': min_val, 'max': max_val, 'mean': mean_val,
                    'finite_count': finite_count, 'nan_count': nan_count
                }
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return raw_stats, tp_ea

def create_corrected_plot(tp_data, raw_stats, date_str, run_str, output_file):
    """
    Create corrected precipitation plot based on actual data analysis
    """
    print(f"\n🎨 Creating corrected plot...")
    
    # Determine proper value range from actual data
    all_max_values = []
    for stats in raw_stats.values():
        if stats['finite_count'] > 0:
            all_max_values.append(stats['max'])
    
    if not all_max_values:
        print("❌ No valid data found for plotting")
        return False
    
    data_max = max(all_max_values)
    print(f"Data maximum: {data_max:.6e}")
    
    # Determine units and scaling
    if data_max < 1.0:
        # Data is in meters, convert to mm
        conversion_factor = 1000
        units = "mm"
        # Set colorbar range based on converted values
        vmax = min(100, data_max * 1000 * 1.2)  # 20% headroom, max 100mm
        print(f"Converting from meters to mm, vmax = {vmax:.1f}")
    else:
        # Data might already be in mm or other units
        conversion_factor = 1
        units = "mm (assumed)"
        vmax = min(100, data_max * 1.2)  # 20% headroom
        print(f"No conversion, vmax = {vmax:.1f}")
    
    # Select timesteps that have valid data
    valid_timesteps = [t for t, stats in raw_stats.items() if stats['finite_count'] > 0]
    valid_timesteps = sorted(valid_timesteps)[:6]  # Take first 6 valid timesteps
    
    print(f"Plotting timesteps: {valid_timesteps}")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    successful_plots = 0
    im = None
    
    for plot_idx, t_idx in enumerate(valid_timesteps):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        try:
            print(f"  Plotting timestep {t_idx}...")
            
            # Get data for this timestep
            plot_data = tp_data.isel(valid_times=t_idx)
            values = plot_data.compute().values
            
            # Apply conversion
            values = values * conversion_factor
            
            # Handle NaN and negative values
            values = np.where(np.isfinite(values) & (values >= 0), values, 0)
            
            # Create coordinate arrays for plotting
            lons = plot_data.longitude.values
            lats = plot_data.latitude.values
            
            # Create the plot with adaptive colorbar
            levels = np.linspace(0, vmax, 21)
            im = ax.contourf(lons, lats, values, levels=levels, cmap='Blues',
                           transform=ccrs.PlateCarree(), extend='max')
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.OCEAN, color='lightcyan', alpha=0.3)
            ax.add_feature(cfeature.LAND, color='wheat', alpha=0.2)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            
            # Set extent
            ax.set_extent([EA_LON_MIN, EA_LON_MAX, EA_LAT_MIN, EA_LAT_MAX],
                         crs=ccrs.PlateCarree())
            
            # Title with statistics
            stats = raw_stats[t_idx]
            max_val_display = stats['max'] * conversion_factor
            ax.set_title(f'T+{t_idx*3}h\nMax: {max_val_display:.1f} {units}',
                        fontsize=11, weight='bold')
            
            successful_plots += 1
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            ax.text(0.5, 0.5, f'Error\nT+{t_idx*3}h',
                   transform=ax.transAxes, ha='center', va='center')
    
    # Hide unused subplots
    for i in range(len(valid_timesteps), len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    if im is not None and successful_plots > 0:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax, extend='max')
        cbar.set_label(f'Accumulated Precipitation ({units})', fontsize=12, weight='bold')
    
    # Main title
    fig.suptitle(f'GEFS Precipitation (Corrected) - East Africa\n{date_str} {run_str}Z Run',
                fontsize=16, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Corrected plot saved: {output_file}")
    plt.close()
    
    return True

def main():
    """
    Main diagnostic function
    """
    print("="*80)
    print("GEFS TP Data Diagnosis and Correction")
    print("="*80)
    
    # Test files
    test_cases = [
        ('gefs_gep01_20250616_06z_tp_east_africa.npy', 'gefs_gep01_20250616_06z_fixed.par', '20250616', '06'),
        ('gefs_gep01_20250618_00z_tp_east_africa.npy', 'gefs_gep01_20250618_00z_fixed.par', '20250618', '00')
    ]
    
    for npy_file, parquet_file, date_str, run_str in test_cases:
        print(f"\n{'='*60}")
        print(f"Analyzing {date_str} {run_str}Z")
        print(f"{'='*60}")
        
        # Diagnose numpy array
        if os.path.exists(npy_file):
            array = diagnose_numpy_array(npy_file)
        else:
            print(f"❌ Numpy file not found: {npy_file}")
            continue
        
        # Investigate raw data
        if os.path.exists(parquet_file):
            raw_stats, tp_ea = investigate_raw_data(parquet_file)
            
            # Create corrected plot
            output_file = f'gefs_tp_corrected_{date_str}_{run_str}z.png'
            success = create_corrected_plot(tp_ea, raw_stats, date_str, run_str, output_file)
            
            if success:
                print(f"✅ Corrected plot created for {date_str} {run_str}Z")
        else:
            print(f"❌ Parquet file not found: {parquet_file}")
    
    print(f"\n{'='*80}")
    print("🎯 Summary of Issues Found:")
    print("1. First timestep (T+0h) contains all NaN values")
    print("2. Data range exceeds colorbar limits (0-50mm vs actual 0-107mm)")
    print("3. Need to verify if data is in meters or millimeters")
    print("4. Adaptive colorbar range needed based on actual data")
    print("="*80)

if __name__ == "__main__":
    main()