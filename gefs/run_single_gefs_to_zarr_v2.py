#!/usr/bin/env python3
"""
GEFS Single Ensemble Member Processing Script V2
Enhanced version with East Africa subsetting and combined zarr output.

This script processes a single GEFS ensemble member from parquet to zarr format
with optional regional subsetting and variable filtering.

Usage:
    python run_single_gefs_to_zarr_v2.py <date> <run> <member> [options]

Example:
    python run_single_gefs_to_zarr_v2.py 20250909 18 gep01 --region east_africa --variables t2m,tp,u10,v10,cape

Features:
    - Single unified zarr output instead of multiple files
    - East Africa regional subsetting (98.2% size reduction)
    - Variable filtering and validation
    - Enhanced compression and chunking
    - CF-compliant metadata
"""

import os
import sys
import argparse
import json
import warnings
import time
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import fsspec
import numcodecs

# Suppress warnings
warnings.filterwarnings('ignore')

def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling - from original script."""
    import ast

    df = pd.read_parquet(parquet_path)
    print(f"📊 Parquet file loaded: {len(df)} rows")

    if 'refs' in df['key'].values and len(df) <= 2:
        # Old format - single refs row
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
        print(f"✅ Extracted {len(zstore)} entries from old format")
    else:
        # New format - each key-value pair is a row
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

# Regional definitions
REGIONS = {
    'global': {
        'lat_min': -90.0, 'lat_max': 90.0,
        'lon_min': 0.0, 'lon_max': 359.75,
        'description': 'Global coverage'
    },
    'east_africa': {
        'lat_min': -12.0, 'lat_max': 23.0,
        'lon_min': 21.0, 'lon_max': 53.0,
        'description': 'East Africa region (Kenya, Tanzania, Uganda, Ethiopia, etc.)'
    }
}

# Variable mapping from requested names to available names
VARIABLE_MAPPING = {
    'pres': 'sp',          # Surface pressure
    'tmp': 't2m',          # 2m temperature
    'ugrd': 'u10',         # 10m U wind component
    'vgrd': 'v10',         # 10m V wind component
    'pwat': 'pwat',        # Precipitable water (check if available)
    'cape': 'cape',        # Convective available potential energy
    'msl': 'mslet',        # Mean sea level pressure
    'apcp': 'tp'           # Total precipitation
}

def test_zarr_access_and_discover_variables(zstore, member_name):
    """Test accessing zarr data and discover all variables - adapted from original."""
    print(f"\n🧪 Testing zarr access for {member_name}...")

    try:
        # Create reference filesystem
        fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3',
                              remote_options={'anon': True})
        mapper = fs.get_mapper("")

        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
        print(f"✅ Successfully opened datatree")

        # List available groups
        print(f"📁 Available groups: {len(dt.groups)} groups")

        # Find all variable groups (those with actual data variables)
        variable_groups = {}
        for group_path in dt.groups:
            if group_path in ['/', '']:
                continue
            try:
                group = dt[group_path]
                if hasattr(group, 'ds') and group.ds.data_vars:
                    for var_name in group.ds.data_vars:
                        variable_groups[group_path] = {
                            'variable': var_name,
                            'shape': group.ds[var_name].shape,
                            'dims': group.ds[var_name].dims
                        }
                        break  # Take the first data variable from each group
            except:
                continue

        print(f"📊 Found {len(variable_groups)} variable groups:")
        for group_path, info in variable_groups.items():
            print(f"   {group_path}: {info['variable']} {info['shape']} {info['dims']}")

        # Test accessing one variable
        if variable_groups:
            test_group = list(variable_groups.keys())[0]
            test_var = variable_groups[test_group]['variable']
            print(f"\n🧪 Testing access to {test_group}/{test_var}...")

            data_var = dt[test_group].ds[test_var]
            # Use actual dimension names from the variable
            dims = data_var.dims
            sample_indexer = {}
            for dim in dims:
                if dim in ['latitude', 'lat']:
                    sample_indexer[dim] = slice(0, 5)
                elif dim in ['longitude', 'lon']:
                    sample_indexer[dim] = slice(0, 5)
                elif dim in ['time', 'valid_time', 'step']:
                    sample_indexer[dim] = 0

            sample_data = data_var.isel(**sample_indexer)
            sample_values = sample_data.compute()
            print(f"✅ Sample data access successful: {sample_values.shape}")

            return True, variable_groups, dt
        else:
            print(f"⚠️ No data variables found")
            return False, {}, None

    except Exception as e:
        print(f"❌ Error accessing zarr data: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, None

def filter_variables(variable_groups, requested_vars=None):
    """Filter variables based on user request or return all."""
    available_var_names = [info['variable'] for info in variable_groups.values()]

    if requested_vars is None:
        print(f"📋 Processing all {len(available_var_names)} available variables: {available_var_names}")
        return variable_groups

    print(f"🔧 Filtering variables: requested {requested_vars}")
    print(f"🔍 Available variables: {available_var_names}")
    print(f"🗺️ Variable mapping: {VARIABLE_MAPPING}")

    # Filter variable groups to include only requested variables
    filtered_groups = {}
    found_vars = []

    for group_path, info in variable_groups.items():
        var_name = info['variable']

        # Direct match
        if var_name in requested_vars:
            filtered_groups[group_path] = info
            found_vars.append(var_name)
            print(f"   ✅ {var_name}: Direct match")
        # Check mapped variables
        else:
            for req_var in requested_vars:
                if req_var in VARIABLE_MAPPING and VARIABLE_MAPPING[req_var] == var_name:
                    filtered_groups[group_path] = info
                    found_vars.append(var_name)
                    print(f"   ✅ {req_var} → {var_name}: Mapped match")
                    break

    # Report missing variables with more detail
    missing_vars = []
    for req_var in requested_vars:
        if req_var not in found_vars:
            mapped_var = VARIABLE_MAPPING.get(req_var)
            if mapped_var and mapped_var not in found_vars:
                missing_vars.append(f"{req_var} → {mapped_var}")
            elif not mapped_var:
                missing_vars.append(f"{req_var} (no mapping)")
            else:
                missing_vars.append(req_var)

    if missing_vars:
        print(f"   ⚠️ Variables not found: {missing_vars}")

    if not filtered_groups:
        raise ValueError("No valid variables found. Check variable names and availability.")

    final_var_names = [info['variable'] for info in filtered_groups.values()]
    print(f"📋 Processing {len(filtered_groups)} filtered variables: {final_var_names}")
    return filtered_groups

def subset_region(ds, region):
    """Subset dataset to specified region."""
    if region == 'global':
        print("🌍 Using global coverage")
        return ds

    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}. Available: {list(REGIONS.keys())}")

    region_info = REGIONS[region]
    print(f"🌍 Subsetting to {region}: {region_info['description']}")

    # Get region bounds
    lat_min, lat_max = region_info['lat_min'], region_info['lat_max']
    lon_min, lon_max = region_info['lon_min'], region_info['lon_max']

    # Subset the dataset
    ds_subset = ds.sel(
        latitude=slice(lat_max, lat_min),  # Note: reversed for decreasing latitude
        longitude=slice(lon_min, lon_max)
    )

    # Log subsetting results
    orig_size = ds.sizes['latitude'] * ds.sizes['longitude']
    new_size = ds_subset.sizes['latitude'] * ds_subset.sizes['longitude']
    reduction = (1 - new_size / orig_size) * 100

    print(f"   Original grid: {ds.sizes['latitude']} × {ds.sizes['longitude']} = {orig_size:,} points")
    print(f"   Subset grid: {ds_subset.sizes['latitude']} × {ds_subset.sizes['longitude']} = {new_size:,} points")
    print(f"   Size reduction: {reduction:.1f}%")

    return ds_subset


def save_combined_zarr_v2(dt, member_name, output_dir, variable_groups, region='global'):
    """Save variables as a single combined zarr store with optional regional subsetting."""
    print(f"\n💾 Saving combined zarr for {member_name} (region: {region})...")

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Collect all datasets to combine
        datasets_to_combine = []
        total_orig_size = 0
        total_new_size = 0

        for group_path, info in variable_groups.items():
            try:
                # Get the dataset for this group
                group_ds = dt[group_path].ds
                var_name = info['variable']

                print(f"   Processing {var_name} from {group_path}...")

                # Apply regional subsetting if requested
                if region != 'global':
                    orig_size = group_ds.sizes.get('latitude', 1) * group_ds.sizes.get('longitude', 1)
                    group_ds = subset_region(group_ds, region)
                    new_size = group_ds.sizes.get('latitude', 1) * group_ds.sizes.get('longitude', 1)
                    total_orig_size += orig_size
                    total_new_size += new_size

                datasets_to_combine.append(group_ds)

            except Exception as e:
                print(f"   ⚠️ Failed to process {group_path}: {e}")
                continue

        if not datasets_to_combine:
            print(f"   ❌ No datasets to combine")
            return None

        # Combine all datasets into one with simplified approach to avoid coordinate conflicts
        print(f"   🔗 Combining {len(datasets_to_combine)} datasets...")

        # Use a simpler approach: start with first dataset and add variables one by one
        if datasets_to_combine:
            combined_ds = datasets_to_combine[0].copy()

            # Add variables from other datasets
            for ds in datasets_to_combine[1:]:
                for var_name in ds.data_vars:
                    if var_name not in combined_ds.data_vars:
                        # Add variable with its essential coordinates only
                        combined_ds[var_name] = ds[var_name]
        else:
            raise ValueError("No datasets to combine")

        # Add global attributes
        combined_ds.attrs.update({
            'title': 'GEFS Ensemble Forecast Data',
            'institution': 'NOAA/NCEP',
            'source': 'GEFS (Global Ensemble Forecast System)',
            'region': region,
            'processing_version': '2.0',
            'created_by': 'run_single_gefs_to_zarr_v2.py'
        })

        # Create output filename
        if region == 'east_africa':
            output_filename = f"{member_name}_combined_east_africa.zarr"
        else:
            output_filename = f"{member_name}_combined_global.zarr"

        output_path = os.path.join(output_dir, output_filename)

        # Configure encoding for better compression
        encoding = {}
        for var_name in combined_ds.data_vars:
            encoding[var_name] = {
                'compressor': numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
            }

        # Save to zarr
        print(f"   💾 Writing to {output_filename}...")
        combined_ds.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)

        # Calculate file size
        import subprocess
        result = subprocess.run(['du', '-sb', output_path], capture_output=True, text=True)
        if result.returncode == 0:
            size_bytes = int(result.stdout.split()[0])
            size_mb = size_bytes / (1024 * 1024)
            print(f"   ✅ Saved: {size_mb:.1f}MB")
        else:
            print(f"   ✅ Saved successfully")

        # Report size reduction if regional subsetting was applied
        if region != 'global' and total_orig_size > 0:
            reduction = (1 - total_new_size / total_orig_size) * 100
            print(f"   📉 Overall size reduction: {reduction:.1f}%")

        return output_path

    except Exception as e:
        print(f"   ❌ Error saving combined zarr: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(date_str: str, run_str: str, member: str, region: str = 'global', variables: str = None, parquet_dir: str = None):
    """
    Process a single GEFS ensemble member from parquet to zarr - V2 with enhancements.

    Args:
        date_str: Date string in YYYYMMDD format
        run_str: Run string (e.g., '00', '06', '12', '18')
        member: Ensemble member (e.g., 'gep01')
        region: Region to subset ('global' or 'east_africa')
        variables: Comma-separated list of variables to process
        parquet_dir: Directory containing GEFS parquet files
    """
    print(f"🚀 Processing GEFS ensemble member: {member}")
    print(f"📅 Date: {date_str}, Run: {run_str}")
    print(f"🌍 Region: {region}")
    print(f"🔧 Version: 2.0 (Enhanced)")

    # Set up parquet directory
    if parquet_dir is None:
        parquet_dir = f"{date_str}_{run_str}"

    parquet_path = Path(parquet_dir)
    if not parquet_path.exists():
        print(f"❌ Parquet directory {parquet_path} does not exist.")
        print(f"💡 Run run_day_gefs_ensemble_full.py first to create parquet files.")
        return False

    # Find parquet file for this member
    parquet_file = parquet_path / f"{member}.par"
    if not parquet_file.exists():
        print(f"❌ Parquet file {parquet_file} not found.")
        return False

    print(f"📂 Using parquet file: {parquet_file}")

    # Step 1: Read parquet file
    print(f"\n📜 Step 1: Reading parquet file...")
    try:
        zstore = read_parquet_fixed(str(parquet_file))
        print(f"✅ Successfully loaded zarr store with {len(zstore)} entries")
    except Exception as e:
        print(f"❌ Failed to read parquet file: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Test zarr access and discover all variables
    print(f"\n🧪 Step 2: Testing zarr data access and discovering variables...")
    access_success, variable_groups, dt = test_zarr_access_and_discover_variables(zstore, member)
    if not access_success:
        print(f"⚠️ Zarr access test failed")
        return False

    # Step 3: Filter variables if requested
    if variables:
        requested_vars = [v.strip() for v in variables.split(',')]
        try:
            variable_groups = filter_variables(variable_groups, requested_vars)
        except ValueError as e:
            print(f"❌ {e}")
            return False
    else:
        variable_groups = filter_variables(variable_groups, None)

    # Step 4: Save combined zarr with enhancements
    print(f"\n💾 Step 3: Saving enhanced combined zarr store...")
    output_dir = f"./zarr_stores/{date_str}_{run_str}"
    output_path = save_combined_zarr_v2(dt, member, output_dir, variable_groups, region)

    # Cleanup
    if dt:
        dt.close()

    if output_path:
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Enhanced zarr saved to: {output_path}")

        # Test reading the saved zarr file
        print(f"\n🔍 Testing saved zarr file: {os.path.basename(output_path)}...")
        try:
            ds = xr.open_dataset(output_path, engine='zarr')
            print(f"📊 Variables: {list(ds.data_vars)}")
            print(f"📐 Coordinates: {list(ds.coords)}")
            print(f"📊 Dataset shape: {dict(ds.sizes)}")
            if ds.data_vars:
                first_var = list(ds.data_vars)[0]
                print(f"🎯 Sample variable '{first_var}' shape: {ds[first_var].shape}")
            ds.close()
            return True
        except Exception as e:
            print(f"⚠️ Could not read saved zarr: {e}")
            return False
    else:
        print(f"❌ Processing failed")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process single GEFS ensemble member to zarr format - V2 Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process gep01 with global coverage
  python run_single_gefs_to_zarr_v2.py 20250909 18 gep01

  # Process with East Africa subsetting
  python run_single_gefs_to_zarr_v2.py 20250909 18 gep01 --region east_africa

  # Process specific variables only
  python run_single_gefs_to_zarr_v2.py 20250909 18 gep01 --variables t2m,tp,u10,v10,cape

  # Combine regional subsetting with variable filtering
  python run_single_gefs_to_zarr_v2.py 20250909 18 gep01 --region east_africa --variables t2m,tp,cape
        """
    )

    parser.add_argument('date', help='Date in YYYYMMDD format (e.g., 20250909)')
    parser.add_argument('run', help='Run hour in HH format (e.g., 18)')
    parser.add_argument('member', help='Ensemble member (e.g., gep01)')
    parser.add_argument('--region', choices=list(REGIONS.keys()), default='east_africa',
                       help='Region to subset (default: global)')
    parser.add_argument('--variables', type=str,
                       help='Comma-separated list of variables to process (default: all available)')
    parser.add_argument('--parquet_dir', type=str,
                       help='Directory containing GEFS parquet files (defaults to {date}_{run})')

    args = parser.parse_args()
    success = main(args.date, args.run, args.member, args.region, args.variables, args.parquet_dir)

    if not success:
        sys.exit(1)