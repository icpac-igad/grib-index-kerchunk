#!/usr/bin/env python3
"""
CFS Parquet to xArray Dataset Converter

This script demonstrates how to read CFS parquet files created from GRIB index parsing
and convert them to xarray datasets for analysis.
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import json

def read_cfs_parquet_to_xarray(parquet_path: str) -> xr.Dataset:
    """
    Read a CFS parquet file and convert it to an xarray Dataset.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        xarray.Dataset containing the CFS data
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    print(f"Loaded parquet file: {parquet_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Show data types and unique values for key columns
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print(f"\nUnique variables: {sorted(df['varname'].unique())}")
    print(f"Unique levels: {sorted(df['level'].unique())}")
    print(f"Unique level types: {sorted(df['typeOfLevel'].unique())}")
    
    # Create a simple xarray dataset from the parquet data
    # Group by variable and level to create data arrays
    datasets = {}
    
    for var in df['varname'].unique():
        var_data = df[df['varname'] == var].copy()
        
        # For each variable, create coordinates based on available dimensions
        coords = {}
        dims = []
        
        # Always have step (forecast time) as a dimension
        if 'step' in var_data.columns:
            unique_steps = sorted(var_data['step'].unique())
            coords['step'] = ('step', unique_steps)
            dims.append('step')
        
        # Add level dimension if multiple levels exist for this variable
        unique_levels = sorted(var_data['level'].unique())
        if len(unique_levels) > 1:
            coords['level'] = ('level', unique_levels)
            dims.append('level')
        
        # For now, create a simple data array with the available metadata
        # In a real implementation, you would need actual gridded data values
        # Here we'll use the record information as a proxy
        
        if len(dims) == 1:  # Only step dimension
            data_shape = (len(coords[dims[0]][1]),)
            data_values = np.arange(len(coords[dims[0]][1]), dtype=float)
        elif len(dims) == 2:  # Step and level dimensions
            data_shape = (len(coords[dims[0]][1]), len(coords[dims[1]][1]))
            data_values = np.arange(np.prod(data_shape), dtype=float).reshape(data_shape)
        else:
            # Single value case
            data_shape = ()
            data_values = float(len(var_data))
            dims = []
        
        # Create the data array
        data_array = xr.DataArray(
            data_values,
            dims=dims,
            coords=coords,
            name=var,
            attrs={
                'long_name': f'{var} from CFS',
                'units': 'unknown',  # Would need to parse from GRIB metadata
                'source': 'NOAA CFS',
                'records_in_grib': len(var_data)
            }
        )
        
        datasets[var] = data_array
    
    # Combine all variables into a single dataset
    ds = xr.Dataset(datasets)
    
    # Add global attributes
    ds.attrs.update({
        'title': 'CFS Data from Parquet Index',
        'source': f'Converted from {parquet_path}',
        'total_records': len(df),
        'variables': list(df['varname'].unique()),
        'levels': list(df['level'].unique()),
        'level_types': list(df['typeOfLevel'].unique())
    })
    
    return ds

def demonstrate_parquet_reading():
    """Demonstrate reading both available parquet files."""
    
    parquet_files = [
        "cfs_simple_20250801_00/cfs-simple-20250801-00-f000.parquet",
        "cfs_research_20250802_00_f000/cfs-research-20250802-00-f000.parquet"
    ]
    
    for parquet_file in parquet_files:
        if Path(parquet_file).exists():
            print("=" * 80)
            print(f"Processing: {parquet_file}")
            print("=" * 80)
            
            try:
                # Read and convert to xarray
                ds = read_cfs_parquet_to_xarray(parquet_file)
                
                print(f"\nxarray Dataset summary:")
                print(ds)
                
                print(f"\nDataset info:")
                print(ds.info())
                
                print(f"\nDataset attributes:")
                for key, value in ds.attrs.items():
                    print(f"  {key}: {value}")
                
                print(f"\nData variables:")
                for var in ds.data_vars:
                    da = ds[var]
                    print(f"  {var}: {da.dims} {da.shape} - {da.attrs.get('records_in_grib', 'unknown')} records")
                
                # Show a sample variable in detail
                first_var = list(ds.data_vars)[0]
                print(f"\nSample variable '{first_var}' details:")
                print(ds[first_var])
                
            except Exception as e:
                print(f"Error processing {parquet_file}: {e}")
        else:
            print(f"File not found: {parquet_file}")

def show_raw_parquet_structure():
    """Show the raw structure of the parquet files for understanding."""
    
    parquet_files = [
        "cfs_simple_20250801_00/cfs-simple-20250801-00-f000.parquet",
        "cfs_research_20250802_00_f000/cfs-research-20250802-00-f000.parquet"
    ]
    
    for parquet_file in parquet_files:
        if Path(parquet_file).exists():
            print("=" * 60)
            print(f"Raw structure: {parquet_file}")
            print("=" * 60)
            
            df = pd.read_parquet(parquet_file)
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            print("\nColumn details:")
            for col in df.columns:
                unique_count = df[col].nunique()
                print(f"  {col}: {df[col].dtype}, {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {sorted(df[col].unique())}")
                else:
                    sample_values = sorted(df[col].unique())[:5]
                    print(f"    Sample: {sample_values}... (and {unique_count-5} more)")
            
            print("\nSample records:")
            print(df.head(10))
            print()

if __name__ == "__main__":
    print("CFS Parquet to xArray Demonstration")
    print("==================================")
    
    # First show the raw parquet structure
    show_raw_parquet_structure()
    
    # Then demonstrate conversion to xarray
    demonstrate_parquet_reading()