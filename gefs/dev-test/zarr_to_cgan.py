#!/usr/bin/env python3
"""
Convert GEFS ensemble zarr files to cGAN-ready NetCDF format.

This script reads individual member zarr files created by run_single_gefs_to_zarr_gribberish.py,
combines them into ensemble arrays, and outputs NetCDF files in the format expected by cGAN:
    (time, member, valid_time, latitude, longitude)

Usage:
    python zarr_to_cgan.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00

    # With custom forecast hours
    python zarr_to_cgan.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \
        --start_hour 30 --end_hour 54 --hour_interval 6

    # Process specific variables only
    python zarr_to_cgan.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \
        --variables t2m,tp,cape
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import xarray as xr

warnings.filterwarnings('ignore')

# Variable name mapping: gefs_name -> cgan_name
VARIABLE_MAPPING = {
    'cape': 'cape',     # Convective Available Potential Energy
    'sp': 'pres',       # Surface Pressure -> Pressure
    'mslet': 'msl',     # Mean Sea Level Pressure
    'pwat': 'pwat',     # Precipitable Water
    't2m': 'tmp',       # 2m Temperature -> Temperature
    'u10': 'ugrd',      # 10m U-wind component
    'v10': 'vgrd',      # 10m V-wind component
    'tp': 'apcp',       # Total Precipitation -> Accumulated Precipitation
}


def find_member_zarrs(input_dir: Path) -> dict:
    """
    Find all member zarr files in the input directory.

    Returns dict mapping member number to zarr path.
    """
    members = {}
    pattern = re.compile(r'gep(\d+).*\.zarr$')

    for item in input_dir.iterdir():
        if item.is_dir() and item.suffix == '.zarr':
            match = pattern.match(item.name)
            if match:
                member_num = int(match.group(1))
                members[member_num] = item

    return dict(sorted(members.items()))


def load_and_combine_members(member_paths: dict, variables: list = None) -> xr.Dataset:
    """
    Load all member zarr files and combine along a new 'member' dimension.

    Parameters:
        member_paths: dict mapping member number to zarr path
        variables: list of variables to load (None = all)

    Returns:
        Combined xarray Dataset with dimensions (member, step, latitude, longitude)
    """
    datasets = []
    member_ids = []

    for member_num, zarr_path in member_paths.items():
        print(f"  Loading member {member_num:02d}: {zarr_path.name}")

        ds = xr.open_zarr(zarr_path)

        # Filter variables if specified
        if variables:
            available_vars = [v for v in variables if v in ds.data_vars]
            if not available_vars:
                print(f"    Warning: No requested variables found in {zarr_path.name}")
                continue
            ds = ds[available_vars]

        datasets.append(ds)
        member_ids.append(member_num)

    if not datasets:
        raise ValueError("No valid datasets found")

    # Combine along new 'member' dimension
    print(f"  Combining {len(datasets)} members...")
    combined = xr.concat(datasets, dim='member')
    combined['member'] = member_ids

    return combined


def convert_step_to_hours(ds: xr.Dataset) -> np.ndarray:
    """Convert step coordinate from timedelta64 to hours as float."""
    step_values = ds.step.values

    # Handle timedelta64
    if np.issubdtype(step_values.dtype, np.timedelta64):
        # Convert to hours
        hours = step_values.astype('timedelta64[h]').astype(float)
    else:
        # Assume already in nanoseconds, convert to hours
        hours = step_values / 3.6e12

    return hours


def filter_forecast_hours(ds: xr.Dataset, start_hour: int, end_hour: int,
                         hour_interval: int) -> xr.Dataset:
    """
    Filter dataset to specific forecast hours.

    Parameters:
        ds: Input dataset with 'step' dimension
        start_hour: Starting forecast hour (e.g., 30)
        end_hour: Ending forecast hour (e.g., 54)
        hour_interval: Interval between hours (e.g., 6)

    Returns:
        Filtered dataset
    """
    step_hours = convert_step_to_hours(ds)
    target_hours = np.arange(start_hour, end_hour + 1, hour_interval)

    print(f"  Available forecast hours: {step_hours.min():.0f} to {step_hours.max():.0f}")
    print(f"  Target forecast hours: {target_hours}")

    # Find indices matching target hours
    indices = []
    matched_hours = []
    for hour in target_hours:
        idx = np.argmin(np.abs(step_hours - hour))
        if np.abs(step_hours[idx] - hour) < 0.5:  # Within 30 minutes
            indices.append(idx)
            matched_hours.append(step_hours[idx])
        else:
            print(f"    Warning: No step found near hour {hour}")

    if not indices:
        raise ValueError(f"No matching forecast hours found between {start_hour} and {end_hour}")

    print(f"  Found {len(indices)} matching hours: {matched_hours}")

    # Select the data
    ds_filtered = ds.isel(step=indices)

    return ds_filtered, matched_hours


def restructure_for_cgan(ds: xr.Dataset, init_time: datetime,
                         forecast_hours: list) -> dict:
    """
    Restructure dataset for cGAN format.

    Input format: (member, step, latitude, longitude)
    Output format: (time, member, valid_time, latitude, longitude)

    Returns dict of {cgan_var_name: xr.Dataset}
    """
    output_datasets = {}

    for var_name in ds.data_vars:
        print(f"  Processing variable: {var_name}")

        # Get cGAN variable name
        cgan_var = VARIABLE_MAPPING.get(var_name, var_name)

        # Get the data array
        data = ds[var_name]

        # Current dims: (member, step, latitude, longitude)
        # Add singleton time dimension at the beginning
        data_expanded = data.expand_dims(dim={'time': 1}, axis=0)

        # Rename step -> valid_time
        data_renamed = data_expanded.rename({'step': 'valid_time'})

        # Create output dataset - don't pass coords to avoid reindexing issues
        ds_out = xr.Dataset({cgan_var: data_renamed})

        # Assign new coordinate values
        ds_out = ds_out.assign_coords({
            'time': [init_time],
            'valid_time': np.arange(len(forecast_hours)),  # Simple index
        })

        # Add attributes
        ds_out[cgan_var].attrs.update({
            'long_name': ds[var_name].attrs.get('long_name', cgan_var),
            'units': ds[var_name].attrs.get('units', ''),
            'original_variable': var_name,
            'forecast_hours': str(forecast_hours),
        })

        # Add coordinate attributes
        ds_out['latitude'].attrs = {
            'units': 'degrees_north',
            'standard_name': 'latitude',
        }
        ds_out['longitude'].attrs = {
            'units': 'degrees_east',
            'standard_name': 'longitude',
        }
        ds_out['valid_time'].attrs = {
            'long_name': 'forecast valid time index',
            'forecast_hours': str(forecast_hours),
        }

        output_datasets[cgan_var] = ds_out
        print(f"    -> {cgan_var}: {dict(ds_out.sizes)}")

    return output_datasets


def save_datasets(datasets: dict, output_dir: Path, year: int):
    """Save each variable dataset to NetCDF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for var_name, ds in datasets.items():
        output_file = output_dir / f"{var_name}_{year}.nc"
        print(f"  Saving: {output_file}")

        # Add global attributes
        ds.attrs.update({
            'title': f'GEFS {var_name} for cGAN inference',
            'source': 'GEFS (Global Ensemble Forecast System)',
            'institution': 'NOAA/NCEP',
            'created_by': 'zarr_to_cgan.py',
            'ensemble_size': len(ds.member),
        })

        # Load data from dask to memory before saving
        ds_computed = ds.compute()
        ds_computed.to_netcdf(output_file)
        saved_files.append(output_file)

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert GEFS ensemble zarr files to cGAN-ready NetCDF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python zarr_to_cgan.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00

    # Custom forecast hours (30-54, every 6 hours)
    python zarr_to_cgan.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \\
        --start_hour 30 --end_hour 54 --hour_interval 6

    # Specific variables only
    python zarr_to_cgan.py --input_dir zarr_stores/20250918_00 --output_dir cgan_input/20250918_00 \\
        --variables t2m,tp,cape,sp
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing member zarr files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for cGAN NetCDF files')
    parser.add_argument('--start_hour', type=int, default=30,
                       help='Starting forecast hour (default: 30)')
    parser.add_argument('--end_hour', type=int, default=54,
                       help='Ending forecast hour (default: 54)')
    parser.add_argument('--hour_interval', type=int, default=6,
                       help='Interval between forecast hours (default: 6)')
    parser.add_argument('--variables', type=str, default=None,
                       help='Comma-separated list of variables to process (default: all)')
    parser.add_argument('--date', type=str, default=None,
                       help='Initialization date YYYYMMDD (default: inferred from input_dir)')
    parser.add_argument('--run', type=str, default='00',
                       help='Model run hour (default: 00)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Parse variables
    variables = None
    if args.variables:
        variables = [v.strip() for v in args.variables.split(',')]

    # Infer date from input directory name if not provided
    if args.date:
        date_str = args.date
    else:
        # Try to extract from directory name like "20250918_00"
        dir_name = input_dir.name
        match = re.match(r'(\d{8})_(\d{2})', dir_name)
        if match:
            date_str = match.group(1)
        else:
            date_str = datetime.now().strftime('%Y%m%d')
            print(f"Warning: Could not infer date, using {date_str}")

    year = int(date_str[:4])

    # Create initialization datetime
    init_time = datetime.strptime(f"{date_str}{args.run}", '%Y%m%d%H')

    print("=" * 70)
    print("GEFS Zarr to cGAN Converter")
    print("=" * 70)
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Init time:        {init_time}")
    print(f"  Forecast hours:   {args.start_hour}-{args.end_hour} (every {args.hour_interval}h)")
    print(f"  Variables:        {variables or 'all'}")
    print()

    # Step 1: Find member zarr files
    print("1. Finding member zarr files...")
    member_paths = find_member_zarrs(input_dir)
    print(f"   Found {len(member_paths)} members: {list(member_paths.keys())}")

    if not member_paths:
        print("Error: No member zarr files found!")
        sys.exit(1)

    # Step 2: Load and combine members
    print("\n2. Loading and combining members...")
    combined_ds = load_and_combine_members(member_paths, variables)
    print(f"   Combined shape: {dict(combined_ds.sizes)}")
    print(f"   Variables: {list(combined_ds.data_vars)}")

    # Step 3: Filter forecast hours
    print("\n3. Filtering forecast hours...")
    filtered_ds, forecast_hours = filter_forecast_hours(
        combined_ds,
        args.start_hour,
        args.end_hour,
        args.hour_interval
    )
    print(f"   Filtered shape: {dict(filtered_ds.sizes)}")

    # Step 4: Restructure for cGAN
    print("\n4. Restructuring for cGAN format...")
    cgan_datasets = restructure_for_cgan(filtered_ds, init_time, forecast_hours)

    # Step 5: Save to NetCDF
    print("\n5. Saving NetCDF files...")
    saved_files = save_datasets(cgan_datasets, output_dir, year)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Output files saved to: {output_dir}")
    for f in saved_files:
        print(f"    - {f.name}")

    # Verification
    print("\n6. Verification...")
    for f in saved_files[:2]:  # Check first 2 files
        ds_check = xr.open_dataset(f)
        print(f"  {f.name}:")
        print(f"    Dimensions: {dict(ds_check.sizes)}")
        var_name = list(ds_check.data_vars)[0]
        print(f"    {var_name} range: [{float(ds_check[var_name].min()):.4f}, {float(ds_check[var_name].max()):.4f}]")
        ds_check.close()

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
