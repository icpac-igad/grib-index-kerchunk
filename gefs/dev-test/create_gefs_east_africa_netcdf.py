#!/usr/bin/env python3
"""
GEFS East Africa NetCDF Generator

This script creates NetCDF files containing rainfall data for the East Africa region
with 30 ensemble members and all forecast timesteps. The NetCDF output can be used
for further analysis, archiving, or input to other modeling systems.

Based on the data processing workflow from run_gefs_24h_accumulation.py
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import fsspec
import time
import argparse

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# East Africa region boundaries (from run_gefs_24h_accumulation.py)
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53

# East Africa Time offset (UTC+3)
EAT_OFFSET = 3

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
            return None, None
        
        # Extract region
        regional_data = data_var.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX)
        )
        
        # Keep as xarray DataArray for NetCDF creation
        member_end_time = time.time()
        print(f"✅ {member} data shape: {regional_data.shape} | Time: {(member_end_time - member_start_time):.1f}s")
        
        return regional_data, member
        
    except Exception as e:
        print(f"❌ Error streaming {member}: {e}")
        return None, None


def create_ensemble_dataset(parquet_dir, output_file, model_date, model_run_hour):
    """
    Create NetCDF file with all ensemble members for East Africa region.
    
    Parameters:
    -----------
    parquet_dir : Path
        Directory containing parquet files
    output_file : str or Path
        Output NetCDF file path
    model_date : datetime
        Model initialization date
    model_run_hour : int
        Model run hour (UTC)
    """
    print("="*80)
    print("GEFS East Africa NetCDF Generator")
    print("="*80)
    
    start_time = time.time()
    
    # Display model run information
    print(f"\n📅 Model Run Information:")
    print(f"   Date: {model_date.strftime('%Y-%m-%d')}")
    print(f"   Run Hour: {model_run_hour:02d}:00 UTC ({(model_run_hour + EAT_OFFSET) % 24:02d}:00 EAT)")
    print(f"   Region: East Africa ({LAT_MIN}°-{LAT_MAX}°N, {LON_MIN}°-{LON_MAX}°E)")
    
    # Check if parquet directory exists
    if not parquet_dir.exists():
        print(f"❌ Error: Directory {parquet_dir} not found!")
        return False
    
    # Get list of parquet files
    parquet_files = sorted(parquet_dir.glob("gep*.par"))
    print(f"📁 Found {len(parquet_files)} ensemble member files")
    
    if len(parquet_files) == 0:
        print("❌ No parquet files found!")
        return False
    
    # Load ensemble data
    print("\n🌧️ Loading ensemble precipitation data...")
    load_start_time = time.time()
    ensemble_data = []
    member_names = []
    
    # Load first member to get coordinate information
    first_data, first_member = stream_single_member_precipitation(parquet_files[0])
    if first_data is None:
        print("❌ Failed to load first member!")
        return False
    
    # Get coordinate arrays
    lats = first_data.latitude.values
    lons = first_data.longitude.values
    times = first_data.time.values
    
    print(f"\n📐 Dataset Dimensions:")
    print(f"   Latitude: {len(lats)} points ({lats.min():.2f}° to {lats.max():.2f}°)")
    print(f"   Longitude: {len(lons)} points ({lons.min():.2f}° to {lons.max():.2f}°)")
    print(f"   Time: {len(times)} timesteps")
    print(f"   Expected Members: {len(parquet_files)}")
    
    # Store first member data
    ensemble_data.append(first_data.values)
    member_names.append(first_member)
    
    # Load remaining members
    for pf in parquet_files[1:]:
        data, member_name = stream_single_member_precipitation(pf)
        if data is not None:
            ensemble_data.append(data.values)
            member_names.append(member_name)
        else:
            print(f"⚠️ Skipping failed member: {pf.stem}")
    
    load_end_time = time.time()
    print(f"\n✅ Successfully loaded {len(ensemble_data)} members")
    print(f"⏱️  Loading time: {(load_end_time - load_start_time):.1f} seconds")
    
    # Create ensemble dimension and stack data
    print("\n📦 Creating ensemble dataset...")
    ensemble_stack = np.stack(ensemble_data, axis=0)  # Shape: (members, time, lat, lon)
    
    # Create time coordinate with proper attributes
    base_datetime = model_date + timedelta(hours=model_run_hour)
    time_attrs = {
        'long_name': 'forecast time',
        'standard_name': 'time',
        'axis': 'T'
    }
    
    # Create coordinate arrays with attributes
    lat_attrs = {
        'long_name': 'latitude',
        'standard_name': 'latitude',
        'units': 'degrees_north',
        'axis': 'Y'
    }
    
    lon_attrs = {
        'long_name': 'longitude', 
        'standard_name': 'longitude',
        'units': 'degrees_east',
        'axis': 'X'
    }
    
    ensemble_attrs = {
        'long_name': 'ensemble member',
        'standard_name': 'realization',
        'axis': 'E'
    }
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            'tp': (['ensemble', 'time', 'latitude', 'longitude'], 
                   ensemble_stack,
                   {
                       'long_name': 'Total precipitation',
                       'standard_name': 'precipitation_amount',
                       'units': 'kg m-2',
                       'description': '3-hourly total precipitation',
                       'coordinates': 'ensemble time latitude longitude'
                   })
        },
        coords={
            'ensemble': (['ensemble'], range(len(member_names)), ensemble_attrs),
            'time': (['time'], times, time_attrs),
            'latitude': (['latitude'], lats, lat_attrs),
            'longitude': (['longitude'], lons, lon_attrs)
        },
        attrs={
            'title': 'GEFS Ensemble Precipitation Forecast - East Africa',
            'institution': 'NOAA/NCEP',
            'source': 'Global Ensemble Forecast System (GEFS)',
            'model_run_date': model_date.strftime('%Y-%m-%d'),
            'model_run_hour': f'{model_run_hour:02d}:00 UTC',
            'model_run_hour_eat': f'{(model_run_hour + EAT_OFFSET) % 24:02d}:00 EAT',
            'forecast_initialization': base_datetime.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'spatial_resolution': 'Approximately 0.5 degrees',
            'temporal_resolution': '3 hours',
            'forecast_range': f'{len(times) * 3} hours ({len(times) * 3 // 24} days)',
            'geographic_domain': f'East Africa ({LAT_MIN}°-{LAT_MAX}°N, {LON_MIN}°-{LON_MAX}°E)',
            'ensemble_members': len(member_names),
            'member_names': ', '.join(member_names),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'conventions': 'CF-1.8',
            'references': 'https://www.ncep.noaa.gov/products/forecasts/ensemble/',
            'comment': 'Regional subset of GEFS ensemble forecast for East Africa'
        }
    )
    
    # Add ensemble member names as a coordinate variable
    ds = ds.assign_coords(member_name=('ensemble', member_names))
    ds['member_name'].attrs = {
        'long_name': 'ensemble member name',
        'description': 'GEFS ensemble member identifier'
    }
    
    # Set encoding for better compression
    encoding = {
        'tp': {
            'zlib': True,
            'complevel': 4,
            'shuffle': True,
            '_FillValue': -9999.0
        },
        'time': {'units': f'hours since {base_datetime.strftime("%Y-%m-%d %H:%M:%S")}'},
        'latitude': {'_FillValue': None},
        'longitude': {'_FillValue': None},
        'ensemble': {'_FillValue': None},
        'member_name': {'_FillValue': None}
    }
    
    # Save to NetCDF
    print(f"\n💾 Saving to NetCDF: {output_file}")
    save_start_time = time.time()
    
    ds.to_netcdf(output_file, encoding=encoding)
    
    save_end_time = time.time()
    print(f"✅ NetCDF file saved successfully!")
    print(f"⏱️  Save time: {(save_end_time - save_start_time):.1f} seconds")
    
    # Print file information
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.1f} MB")
    
    # Print dataset summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Shape: {ds.tp.shape}")
    print(f"   Dimensions: {dict(ds.dims)}")
    print(f"   Data range: {float(ds.tp.min()):.3f} to {float(ds.tp.max()):.3f} kg/m²")
    print(f"   Missing values: {int(ds.tp.isnull().sum())}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n" + "="*80)
    print("✅ NetCDF generation completed successfully!")
    print("="*80)
    
    return True


def create_24h_accumulation_netcdf(parquet_dir, output_file, model_date, model_run_hour):
    """
    Create NetCDF file with 24-hour accumulated precipitation for East Africa region.
    
    This function creates daily accumulation data by summing 8 consecutive 3-hour timesteps.
    """
    print("="*80)
    print("GEFS East Africa 24-Hour Accumulation NetCDF Generator")
    print("="*80)
    
    # First create the full timestep dataset
    temp_file = Path(output_file).parent / "temp_full_timesteps.nc"
    success = create_ensemble_dataset(parquet_dir, temp_file, model_date, model_run_hour)
    
    if not success:
        return False
    
    print("\n🔄 Converting to 24-hour accumulations...")
    
    # Load the temporary file
    ds = xr.open_dataset(temp_file)
    
    # Skip first timestep (initial condition) and process forecast timesteps
    forecast_data = ds.tp[:, 1:, :, :]  # Skip t=0
    n_forecast_timesteps = forecast_data.shape[1]
    n_days = n_forecast_timesteps // TIMESTEPS_PER_DAY
    
    print(f"📊 Processing {n_forecast_timesteps} forecast timesteps into {n_days} daily accumulations")
    
    # Create daily accumulation arrays
    daily_shape = (forecast_data.shape[0], n_days, forecast_data.shape[2], forecast_data.shape[3])
    daily_accumulations = np.zeros(daily_shape)
    
    for day in range(n_days):
        start_idx = day * TIMESTEPS_PER_DAY
        end_idx = (day + 1) * TIMESTEPS_PER_DAY
        # Sum 8 consecutive 3-hour timesteps to get 24-hour totals
        daily_accumulations[:, day, :, :] = forecast_data[:, start_idx:end_idx, :, :].sum(dim='time')
    
    # Create new time coordinate for daily data
    base_datetime = model_date + timedelta(hours=model_run_hour)
    daily_times = [base_datetime + timedelta(days=i+1) for i in range(n_days)]
    
    # Create new dataset with daily accumulations
    daily_ds = xr.Dataset(
        data_vars={
            'tp_24h': (['ensemble', 'time', 'latitude', 'longitude'], 
                       daily_accumulations,
                       {
                           'long_name': '24-hour accumulated precipitation',
                           'standard_name': 'precipitation_amount',
                           'units': 'kg m-2',
                           'description': '24-hour total precipitation accumulation',
                           'coordinates': 'ensemble time latitude longitude'
                       })
        },
        coords={
            'ensemble': ds.ensemble,
            'time': (['time'], daily_times, {
                'long_name': 'forecast time (24-hour periods)',
                'standard_name': 'time',
                'axis': 'T',
                'description': '24-hour accumulation ending time'
            }),
            'latitude': ds.latitude,
            'longitude': ds.longitude,
            'member_name': ds.member_name
        },
        attrs={
            **ds.attrs,
            'title': 'GEFS Ensemble 24-Hour Precipitation Accumulation - East Africa',
            'temporal_resolution': '24 hours (daily)',
            'forecast_range': f'{n_days} days',
            'description': '24-hour accumulated precipitation from 3-hourly GEFS ensemble data'
        }
    )
    
    # Set encoding for better compression
    encoding = {
        'tp_24h': {
            'zlib': True,
            'complevel': 4,
            'shuffle': True,
            '_FillValue': -9999.0
        },
        'latitude': {'_FillValue': None},
        'longitude': {'_FillValue': None},
        'ensemble': {'_FillValue': None},
        'member_name': {'_FillValue': None}
    }
    
    # Save the daily accumulation dataset
    daily_ds.to_netcdf(output_file, encoding=encoding)
    
    # Clean up temporary file
    temp_file.unlink()
    
    # Print summary
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"✅ 24-hour accumulation NetCDF saved: {output_file}")
    print(f"📁 File size: {file_size_mb:.1f} MB")
    print(f"📊 Daily periods: {n_days}")
    print(f"🌧️  Precipitation range: {float(daily_ds.tp_24h.min()):.3f} to {float(daily_ds.tp_24h.max()):.3f} kg/m²")
    
    daily_ds.close()
    ds.close()
    
    return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Generate NetCDF files from GEFS East Africa ensemble data')
    parser.add_argument('parquet_dir', type=str, help='Directory containing parquet files')
    parser.add_argument('--output', '-o', type=str, help='Output NetCDF file name (default: auto-generated)')
    parser.add_argument('--daily', '-d', action='store_true', help='Create 24-hour accumulation NetCDF instead of 3-hourly')
    parser.add_argument('--date', type=str, help='Model date (YYYYMMDD format, auto-detected from directory name)')
    parser.add_argument('--hour', type=int, help='Model run hour (auto-detected from directory name)')
    
    args = parser.parse_args()
    
    parquet_dir = Path(args.parquet_dir)
    
    # Extract date and run hour from directory name if not provided
    if args.date and args.hour is not None:
        model_date = datetime.strptime(args.date, "%Y%m%d")
        model_run_hour = args.hour
    else:
        dir_name = parquet_dir.name
        if "_" in dir_name:
            date_str, run_hour_str = dir_name.split("_")
            model_date = datetime.strptime(date_str, "%Y%m%d")
            model_run_hour = int(run_hour_str)
        else:
            print("❌ Cannot determine date and hour from directory name. Please provide --date and --hour")
            return False
    
    # Generate output filename if not provided
    if args.output:
        output_file = Path(args.output)
    else:
        date_str = model_date.strftime('%Y%m%d')
        run_str = f'{model_run_hour:02d}'
        if args.daily:
            output_file = parquet_dir / f'gefs_east_africa_24h_accumulation_{date_str}_{run_str}z.nc'
        else:
            output_file = parquet_dir / f'gefs_east_africa_3hourly_{date_str}_{run_str}z.nc'
    
    # Create NetCDF file
    if args.daily:
        success = create_24h_accumulation_netcdf(parquet_dir, output_file, model_date, model_run_hour)
    else:
        success = create_ensemble_dataset(parquet_dir, output_file, model_date, model_run_hour)
    
    return success


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)