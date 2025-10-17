# CFS Parquet File Usage Guide

## Overview

This guide demonstrates how to read and work with CFS (Climate Forecast System) parquet files created from GRIB index parsing. The parquet files contain metadata about GRIB records, including variable names, levels, offsets, and record numbers.

## File Structure

The CFS parquet files contain the following columns:

- **record**: GRIB record number (integer)
- **offset**: Byte offset in the GRIB file (integer) 
- **date**: Date string in format 'd=YYYYMMDDHH' (string)
- **varname**: Variable name (e.g., UFLX, VFLX, TMP) (string)
- **level**: Level description (e.g., 'surface', '2 m above ground') (string)
- **forecast**: Forecast type ('anl' for analysis) (string)
- **line_num**: Line number in the original index file (integer)

## Current Zarr Version

We are using Zarr v3.0.9, which is the latest version and compatible with our workflow.

## Basic Usage Examples

### 1. Reading Parquet Files with Pandas

```python
import pandas as pd
import xarray as xr
import numpy as np

# Read CFS parquet file
df = pd.read_parquet('cfs_simple_20250801_00/cfs-simple-20250801-00-f000.parquet')

print(f"Shape: {df.shape}")
print(f"Variables: {len(df['varname'].unique())}")
print(f"Levels: {len(df['level'].unique())}")

# Show first few records
print(df.head())

# Analyze variable distribution
print("\nVariable counts:")
print(df['varname'].value_counts().head(10))

# Analyze level distribution  
print("\nLevel counts:")
print(df['level'].value_counts().head(10))
```

### 2. Converting to xarray Dataset

```python
def parquet_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Convert CFS parquet DataFrame to xarray Dataset."""
    
    data_vars = {}
    
    # Create data arrays for each variable
    for var_name in df['varname'].unique():
        var_data = df[df['varname'] == var_name]
        
        # Create data array with metadata
        data_array = xr.DataArray(
            data=var_data['record'].values,  # Using record numbers as data
            dims=['grib_record'],
            coords={
                'grib_record': np.arange(len(var_data)),
                'level': ('grib_record', var_data['level'].values),
                'offset': ('grib_record', var_data['offset'].values),
                'line_num': ('grib_record', var_data['line_num'].values)
            },
            attrs={
                'long_name': f'{var_name} GRIB record information',
                'variable_name': var_name,
                'grib_records': len(var_data),
                'unique_levels': len(var_data['level'].unique())
            }
        )
        
        data_vars[var_name] = data_array
    
    # Create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={
            'title': 'CFS GRIB Index Metadata',
            'source': 'NOAA Climate Forecast System',
            'total_variables': len(df['varname'].unique()),
            'total_records': len(df),
            'date': df['date'].iloc[0] if len(df) > 0 else 'unknown',
            'forecast_type': df['forecast'].iloc[0] if len(df) > 0 else 'unknown'
        }
    )
    
    return ds

# Convert to xarray
ds = parquet_to_xarray(df)
print(ds)
```

### 3. Analyzing CFS Variables

```python
# Show all unique variables
variables = sorted(df['varname'].unique())
print(f"All {len(variables)} variables:")
for i, var in enumerate(variables):
    count = len(df[df['varname'] == var])
    levels = df[df['varname'] == var]['level'].nunique()
    print(f"  {i+1:2d}. {var:6s} - {count:2d} records, {levels:2d} levels")

# Show all unique levels
levels = sorted(df['level'].unique())
print(f"\nAll {len(levels)} levels:")
for i, level in enumerate(levels):
    count = len(df[df['level'] == level])
    variables = df[df['level'] == level]['varname'].nunique()
    print(f"  {i+1:2d}. {level:30s} - {count:2d} records, {variables:2d} variables")
```

### 4. Working with Multiple Files

```python
def compare_cfs_files(file1: str, file2: str):
    """Compare two CFS parquet files."""
    
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    
    print(f"File 1 ({file1}):")
    print(f"  Date: {df1['date'].iloc[0]}")
    print(f"  Records: {len(df1)}")
    print(f"  Variables: {len(df1['varname'].unique())}")
    
    print(f"\nFile 2 ({file2}):")
    print(f"  Date: {df2['date'].iloc[0]}")
    print(f"  Records: {len(df2)}")
    print(f"  Variables: {len(df2['varname'].unique())}")
    
    # Check if same variables
    vars1 = set(df1['varname'].unique())
    vars2 = set(df2['varname'].unique())
    
    print(f"\nVariable comparison:")
    print(f"  Common variables: {len(vars1 & vars2)}")
    print(f"  Only in file 1: {len(vars1 - vars2)}")
    print(f"  Only in file 2: {len(vars2 - vars1)}")
    
    if vars1 - vars2:
        print(f"  Unique to file 1: {list(vars1 - vars2)}")
    if vars2 - vars1:
        print(f"  Unique to file 2: {list(vars2 - vars1)}")

# Compare files
compare_cfs_files(
    'cfs_simple_20250801_00/cfs-simple-20250801-00-f000.parquet',
    'cfs_research_20250802_00_f000/cfs-research-20250802-00-f000.parquet'
)
```

## Key Findings from Analysis

Based on the current CFS parquet files:

- **97 GRIB records** per file (out of 101 total in index, 4 skipped due to decimal record numbers)
- **64 unique variables** including flux variables (UFLX, VFLX), temperature (TMP), radiation (DSWRF, DLWRF), etc.
- **21 unique levels** including surface, atmospheric layers, and soil depths
- **Single forecast type**: 'anl' (analysis)
- **File sizes**: ~6.8 KB per parquet file, very efficient storage

### Common Variables
- **UFLX, VFLX**: Momentum fluxes  
- **SHTFL, LHTFL**: Sensible and latent heat fluxes
- **TMP**: Temperature at various levels
- **DSWRF, DLWRF**: Downward shortwave/longwave radiation
- **PRATE**: Precipitation rate
- **PWAT**: Precipitable water

### Common Levels
- **surface**: Most variables (surface conditions)
- **2 m above ground**: Temperature, humidity
- **Various soil depths**: 0-0.1m, 0.1-0.4m, 0.4-1m, 1-2m below ground
- **Atmospheric layers**: Boundary layer, cloud layers
- **Top of atmosphere**: Radiation variables

## Integration with Existing Workflow

This parquet approach provides efficient metadata access without needing to:
- Parse GRIB files directly
- Handle async filesystem issues with kerchunk
- Manage large memory requirements for full data loading

The parquet files serve as a lightweight index for accessing specific GRIB records when needed for full data analysis.

## Usage in Production

For production use, these parquet files can be:
1. **Uploaded to GCS** using the existing workflow
2. **Combined across dates** for time series analysis  
3. **Used as indexes** for on-demand GRIB data loading
4. **Analyzed for data availability** and quality control

## Next Steps

The current implementation successfully:
- ✅ Parses CFS index files bypassing kerchunk issues
- ✅ Creates efficient parquet metadata files
- ✅ Supports both local storage and research file downloads
- ✅ Works with current Zarr v3.0.9 installation
- ✅ Provides xarray integration capabilities

No zarr v2 downgrade is needed - the current setup works well with zarr v3.