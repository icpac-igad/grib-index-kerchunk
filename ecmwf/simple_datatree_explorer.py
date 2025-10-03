#!/usr/bin/env python
"""
Simple ECMWF DataTree Explorer

Simplified version focused on practical usage - exploring your specific DataTree
with 39 groups and only 2 time steps (0h, 3h).

Based on the GEFS processing pattern from run_gefs_24h_accumulation.py
"""

import fsspec
import xarray as xr
import pandas as pd
import numpy as np
import json
import ast


def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling - from GEFS script."""
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


def explore_datatree_simple(dt):
    """Simple exploration of DataTree structure."""
    print("\n" + "="*60)
    print("🌳 DATATREE STRUCTURE")
    print("="*60)

    print(f"DataTree type: {type(dt)}")

    # Check if we can iterate over the tree
    try:
        # Method 1: Try to get groups
        if hasattr(dt, 'groups'):
            groups = dt.groups()
            print(f"Total groups: {len(groups)}")
            print(f"First 10 groups: {list(groups)[:10]}")
    except:
        print("Cannot access groups directly")

    # Method 2: Explore children
    if hasattr(dt, 'children'):
        print(f"\nRoot children: {list(dt.children.keys()) if dt.children else 'None'}")

    # Method 3: Try to iterate
    try:
        print(f"\nIterating through tree:")
        count = 0
        for path, node in dt.items():
            if count < 10:  # Show first 10
                print(f"  {path}: {type(node)}")
                if hasattr(node, 'ds') and node.ds is not None:
                    print(f"    → Dataset with {len(node.ds.data_vars)} variables")
                    print(f"    → Dimensions: {dict(node.ds.dims)}")
            count += 1

        if count > 10:
            print(f"  ... and {count - 10} more nodes")

    except Exception as e:
        print(f"Cannot iterate: {e}")

    return dt


def find_surface_data(dt):
    """Find surface-level data similar to GEFS pattern."""
    print("\n" + "="*60)
    print("🔍 FINDING SURFACE DATA")
    print("="*60)

    surface_datasets = {}

    # Common ECMWF surface variables
    target_vars = ['tp', 't2m', 'u10', 'v10', 'msl', 'tcc', 'sp', 'd2m']

    for path, node in dt.items():
        if hasattr(node, 'ds') and node.ds is not None:
            ds = node.ds

            # Check if this contains surface data
            if 'surface' in path.lower() or any(var in ds.data_vars for var in target_vars):
                surface_datasets[path] = ds

                print(f"📍 Found surface data at: {path}")
                print(f"   Variables: {list(ds.data_vars)}")
                print(f"   Dimensions: {dict(ds.dims)}")

                # Check for time/step dimensions
                for dim, size in ds.dims.items():
                    if 'step' in dim.lower() or 'time' in dim.lower():
                        print(f"   {dim}: {size} (values: {ds.coords[dim].values if dim in ds.coords else 'no coords'})")

    return surface_datasets


def extract_precipitation_like_gefs(dt):
    """Extract precipitation data following GEFS pattern."""
    print("\n" + "="*60)
    print("🌧️ EXTRACTING PRECIPITATION DATA")
    print("="*60)

    # Look for precipitation in various paths
    tp_paths = [
        'tp/accum/surface',
        'tp/instant/surface',
        'tp/surface',
        'precip/surface'
    ]

    for tp_path in tp_paths:
        try:
            print(f"🔍 Trying path: {tp_path}")

            # Navigate to the path
            parts = tp_path.split('/')
            node = dt

            for part in parts:
                if hasattr(node, 'children') and part in node.children:
                    node = node.children[part]
                    print(f"   → Found {part}")
                else:
                    print(f"   ✗ {part} not found")
                    break
            else:
                # We successfully navigated the full path
                if hasattr(node, 'ds') and node.ds is not None:
                    ds = node.ds
                    print(f"✅ Found precipitation dataset!")
                    print(f"   Variables: {list(ds.data_vars)}")
                    print(f"   Dimensions: {dict(ds.dims)}")

                    # Look for tp variable
                    if 'tp' in ds.data_vars:
                        tp_data = ds['tp']
                        print(f"   TP variable shape: {tp_data.shape}")
                        print(f"   TP dimensions: {tp_data.dims}")

                        # Show coordinate values
                        for dim in tp_data.dims:
                            if dim in ds.coords:
                                coord_vals = ds.coords[dim].values
                                print(f"   {dim}: {coord_vals}")

                        return tp_data, ds

        except Exception as e:
            print(f"   Error accessing {tp_path}: {e}")

    print("❌ Could not find precipitation data")
    return None, None


def analyze_time_steps(dt):
    """Analyze the time step issue (only 2 steps: 0h, 3h)."""
    print("\n" + "="*60)
    print("⏰ TIME STEP ANALYSIS")
    print("="*60)

    for path, node in dt.items():
        if hasattr(node, 'ds') and node.ds is not None:
            ds = node.ds

            # Look for step dimension
            if 'step' in ds.dims:
                step_values = ds.coords['step'].values if 'step' in ds.coords else None
                print(f"📍 {path}:")
                print(f"   step dimension size: {ds.dims['step']}")
                if step_values is not None:
                    print(f"   step values: {step_values}")

                # This is the problem: only 2 steps instead of 85
                if ds.dims['step'] == 2:
                    print(f"   ⚠️  WARNING: Only 2 time steps found!")
                    print(f"   💡 Solution: Use index-based processor for all 85 steps")

            # Look for time dimension
            if 'time' in ds.dims:
                time_values = ds.coords['time'].values if 'time' in ds.coords else None
                print(f"   time dimension size: {ds.dims['time']}")
                if time_values is not None:
                    print(f"   time values: {time_values}")


def convert_to_single_dataset(dt):
    """Convert DataTree to single Dataset like GEFS processing."""
    print("\n" + "="*60)
    print("🔄 CONVERTING TO SINGLE DATASET")
    print("="*60)

    all_datasets = []
    path_map = {}

    for path, node in dt.items():
        if hasattr(node, 'ds') and node.ds is not None and len(node.ds.data_vars) > 0:
            ds = node.ds.copy()

            # Rename variables with path prefix
            renamed_vars = {}
            for var in ds.data_vars:
                new_name = f"{path.replace('/', '_')}_{var}"
                renamed_vars[var] = new_name
                path_map[new_name] = path

            ds = ds.rename(renamed_vars)
            all_datasets.append(ds)

            print(f"   Added: {path} → {list(renamed_vars.values())}")

    if all_datasets:
        try:
            merged_ds = xr.merge(all_datasets, compat='override')
            print(f"\n✅ Successfully merged {len(all_datasets)} datasets")
            print(f"📊 Final dataset:")
            print(f"   Variables: {len(merged_ds.data_vars)}")
            print(f"   Dimensions: {dict(merged_ds.dims)}")

            # Show variable mapping
            print(f"\n📋 Variable mapping:")
            for new_var, orig_path in sorted(path_map.items()):
                print(f"   {new_var} ← {orig_path}")

            return merged_ds, path_map

        except Exception as e:
            print(f"❌ Error merging: {e}")
            return None, None
    else:
        print("❌ No datasets found")
        return None, None


def demonstrate_usage():
    """Demonstrate how to use the converted dataset."""
    print("\n" + "="*60)
    print("💡 USAGE EXAMPLES")
    print("="*60)

    print("# Example 1: Load your ECMWF parquet file")
    print("parquet_path = 'ecmwf_20250628_18_efficient/members/ens_01/ens_01.parquet'")
    print("zstore = read_parquet_fixed(parquet_path)")
    print("")

    print("# Example 2: Create DataTree")
    print("fs = fsspec.filesystem('reference', fo=zstore, remote_protocol='s3', remote_options={'anon': True})")
    print("mapper = fs.get_mapper('')")
    print("dt = xr.open_datatree(mapper, engine='zarr', consolidated=False, decode_timedelta=False)")
    print("")

    print("# Example 3: Extract specific data (like GEFS does)")
    print("# For precipitation:")
    print("try:")
    print("    tp_data = dt['tp']['accum']['surface'].ds['tp']")
    print("    print(f'Precipitation shape: {tp_data.shape}')")
    print("except:")
    print("    print('Could not access tp data directly')")
    print("")

    print("# Example 4: Convert to single dataset")
    print("merged_ds, path_map = convert_to_single_dataset(dt)")
    print("if merged_ds:")
    print("    # Access variables with flattened names")
    print("    precip = merged_ds['tp_accum_surface_tp']")
    print("    temp = merged_ds['t_instant_surface_t2m']  # if exists")
    print("")

    print("# Example 5: Work around the 2-timestep limitation")
    print("# Your current data only has steps [0, 3]")
    print("# To get all 85 timesteps, use ecmwf_index_processor.py:")
    print("# python ecmwf_index_processor.py --date 20250628 --member ens01")


def main():
    """Main function to explore your specific ECMWF DataTree."""
    # Your parquet file path
    parquet_path = 'ecmwf_20250628_18_efficient/members/ens_01/ens_01.parquet'

    print("="*80)
    print("🌳 SIMPLE ECMWF DATATREE EXPLORER")
    print("="*80)
    print(f"📁 File: {parquet_path}")

    # Step 1: Read the parquet
    print(f"\n📊 Reading parquet file...")
    zstore = read_parquet_fixed(parquet_path)

    # Step 2: Create DataTree
    print(f"\n🌳 Creating DataTree...")
    fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3',
                          remote_options={'anon': True})
    mapper = fs.get_mapper("")
    dt = xr.open_datatree(mapper, engine="zarr", consolidated=False, decode_timedelta=False)

    print(f"✅ DataTree created: {type(dt)}")

    # Step 3: Explore structure
    explore_datatree_simple(dt)

    # Step 4: Find surface data
    surface_data = find_surface_data(dt)

    # Step 5: Try to extract precipitation
    tp_data, tp_dataset = extract_precipitation_like_gefs(dt)

    # Step 6: Analyze time steps
    analyze_time_steps(dt)

    # Step 7: Convert to single dataset
    merged_ds, path_map = convert_to_single_dataset(dt)

    # Step 8: Show usage examples
    demonstrate_usage()

    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print("✅ Your DataTree has been analyzed")
    print("⚠️  Issue: Only 2 time steps (0h, 3h) instead of 85")
    print("💡 Solution: Use ecmwf_index_processor.py for complete data")
    print("📊 Converted to single dataset for easier analysis")

    if merged_ds is not None:
        print(f"🎯 Use 'merged_ds' variable to access all data:")
        print(f"   Variables: {len(merged_ds.data_vars)}")
        example_var = list(merged_ds.data_vars)[0]
        print(f"   Example: merged_ds['{example_var}']")

    print("="*80)

    return dt, merged_ds, path_map


if __name__ == "__main__":
    dt, merged_ds, path_map = main()