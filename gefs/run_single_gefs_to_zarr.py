#!/usr/bin/env python3
"""
Single GEFS Ensemble Member Zarr Processing (No Dask)
This script processes a single GEFS ensemble parquet file and converts it to zarr 
without using dask cluster, for testing and debugging purposes.
"""

import os
import argparse
import json
from pathlib import Path
import pandas as pd
import fsspec
import xarray as xr
from dotenv import load_dotenv

def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling - from run_gefs_24h_accumulation.py."""
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


def test_zarr_access(zstore, member_name):
    """Test accessing zarr data through fsspec and list all variables."""
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
            
            return True, variable_groups
        else:
            print(f"⚠️ No data variables found")
            return False, {}
            
    except Exception as e:
        print(f"❌ Error accessing zarr data: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def save_zarr_locally(zstore, member_name, output_dir, variable_groups):
    """Save all variables from zarr store locally."""
    print(f"\n💾 Saving all variables locally for {member_name}...")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create reference filesystem
        fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3', 
                              remote_options={'anon': True})
        mapper = fs.get_mapper("")
        
        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
        
        saved_files = []
        total_size = 0
        
        # Save each variable group as a separate zarr store
        for group_path, info in variable_groups.items():
            try:
                # Create a clean filename from group path
                clean_group = group_path.replace('/', '_').strip('_')
                output_path = os.path.join(output_dir, f"{member_name}_{clean_group}.zarr")
                
                # Get the dataset for this group
                group_ds = dt[group_path].ds
                
                # Save to zarr
                group_ds.to_zarr(output_path, mode='w')
                
                # Calculate size
                import subprocess
                result = subprocess.run(['du', '-sb', output_path], capture_output=True, text=True)
                if result.returncode == 0:
                    size_bytes = int(result.stdout.split()[0])
                    size_mb = size_bytes / (1024 * 1024)
                    total_size += size_mb
                    print(f"   ✅ {clean_group}: {info['variable']} → {output_path} ({size_mb:.1f}MB)")
                else:
                    print(f"   ✅ {clean_group}: {info['variable']} → {output_path}")
                
                saved_files.append(output_path)
                
            except Exception as e:
                print(f"   ❌ Failed to save {group_path}: {e}")
                continue
        
        print(f"\n📊 Summary for {member_name}:")
        print(f"   Variables saved: {len(saved_files)}")
        print(f"   Total size: {total_size:.1f}MB")
        print(f"   Output directory: {output_dir}")
        
        return saved_files
            
    except Exception as e:
        print(f"❌ Error saving zarr store: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(date_str: str, run_str: str, member: str, parquet_dir: str = None):
    """
    Process a single GEFS ensemble member from parquet to zarr.
    
    Args:
        date_str: Date string in YYYYMMDD format
        run_str: Run string (e.g., '00', '06', '12', '18')
        member: Ensemble member (e.g., 'gep01')
        parquet_dir: Directory containing GEFS parquet files
    """
    print(f"🚀 Processing single GEFS ensemble member: {member}")
    print(f"📅 Date: {date_str}, Run: {run_str}")
    
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
    print(f"\n📖 Step 1: Reading parquet file...")
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
    access_success, variable_groups = test_zarr_access(zstore, member)
    if not access_success:
        print(f"⚠️ Zarr access test failed, but continuing...")
        return False
    
    # Step 3: Save all variables locally
    print(f"\n💾 Step 3: Saving all variables to local zarr stores...")
    output_dir = f"./zarr_stores/{date_str}_{run_str}"
    saved_files = save_zarr_locally(zstore, member, output_dir, variable_groups)
    
    if saved_files:
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 All variables saved to: {output_dir}")
        
        # Test reading one of the saved zarr files
        if saved_files:
            test_file = saved_files[0]
            print(f"\n🔍 Testing saved zarr file: {os.path.basename(test_file)}...")
            try:
                ds = xr.open_dataset(test_file, engine='zarr')
                print(f"📊 Variables: {list(ds.data_vars)}")
                print(f"📐 Coordinates: {list(ds.coords)}")
                if ds.data_vars:
                    first_var = list(ds.data_vars)[0]
                    print(f"🎯 Sample variable '{first_var}' shape: {ds[first_var].shape}")
                return True
            except Exception as e:
                print(f"⚠️ Could not read saved zarr: {e}")
                return False
    else:
        print(f"❌ Processing failed")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process single GEFS ensemble member to zarr (no dask).")
    parser.add_argument("date_str", type=str, help="Date string in YYYYMMDD format.")
    parser.add_argument("run_str", type=str, help="Run string (e.g., '00', '06', '12', '18').")
    parser.add_argument("member", type=str, help="Ensemble member (e.g., 'gep01').")
    parser.add_argument("--parquet_dir", type=str, help="Directory containing GEFS parquet files (defaults to {date_str}_{run_str})")
    
    args = parser.parse_args()
    success = main(args.date_str, args.run_str, args.member, args.parquet_dir)
    
    if not success:
        exit(1)