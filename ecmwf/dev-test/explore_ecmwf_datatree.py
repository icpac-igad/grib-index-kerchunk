#!/usr/bin/env python
"""
ECMWF DataTree Explorer

This script explores and analyzes xarray DataTree structures created from ECMWF parquet files.
Based on the GEFS processing approach but adapted for ECMWF's hierarchical structure.

Usage:
    python explore_ecmwf_datatree.py --parquet ecmwf_20250628_18_efficient/members/ens_01/ens_01.parquet
    python explore_ecmwf_datatree.py --parquet ecmwf_20250628_18_efficient/members/ens_01/ens_01.parquet --extract tp
"""

import fsspec
import xarray as xr
import pandas as pd
import numpy as np
import json
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any


def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling - from GEFS script."""
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


def analyze_zarr_references(zstore):
    """Analyze the zarr reference structure."""
    print("\n" + "="*60)
    print("📋 ZARR REFERENCES ANALYSIS")
    print("="*60)

    # Count different types of keys
    metadata_keys = []
    data_keys = []
    array_keys = []

    for key in zstore.keys():
        if key.startswith('.'):
            metadata_keys.append(key)
        elif key.endswith('.0.0') or key.endswith('.0'):
            data_keys.append(key)
        elif '/' in key:
            array_keys.append(key)
        else:
            metadata_keys.append(key)

    print(f"📊 Reference structure:")
    print(f"   Total keys: {len(zstore)}")
    print(f"   Metadata keys: {len(metadata_keys)}")
    print(f"   Data chunk keys: {len(data_keys)}")
    print(f"   Array keys: {len(array_keys)}")

    # Show some examples
    print(f"\n📝 Example keys:")
    if metadata_keys:
        print(f"   Metadata: {metadata_keys[:5]}")
    if data_keys:
        print(f"   Data chunks: {data_keys[:5]}")
    if array_keys:
        print(f"   Arrays: {array_keys[:5]}")

    return {
        'total_keys': len(zstore),
        'metadata_keys': metadata_keys,
        'data_keys': data_keys,
        'array_keys': array_keys
    }


def explore_datatree_structure(dt):
    """Explore and document the DataTree structure."""
    print("\n" + "="*60)
    print("🌳 DATATREE STRUCTURE EXPLORATION")
    print("="*60)

    # Basic information
    try:
        group_count = len(dt.groups())
        print(f"📊 Total groups: {group_count}")
    except:
        print("📊 Groups method not available")

    # Try different ways to access groups
    print(f"\n🔍 DataTree properties:")
    print(f"   Type: {type(dt)}")

    if hasattr(dt, 'children'):
        print(f"   Children: {list(dt.children.keys()) if dt.children else 'None'}")

    if hasattr(dt, 'ds'):
        print(f"   Dataset at root: {dt.ds is not None}")
        if dt.ds is not None:
            print(f"   Root data vars: {list(dt.ds.data_vars)}")
            print(f"   Root dimensions: {dict(dt.ds.dims)}")

    # Explore the tree structure
    tree_structure = explore_tree_recursive(dt, path="", max_depth=5)

    return tree_structure


def explore_tree_recursive(node, path="", depth=0, max_depth=5):
    """Recursively explore DataTree structure."""
    if depth > max_depth:
        return {}

    structure = {
        'path': path,
        'depth': depth,
        'has_dataset': False,
        'dimensions': {},
        'variables': [],
        'attributes': {},
        'children': {}
    }

    # Check if this node has a dataset
    if hasattr(node, 'ds') and node.ds is not None:
        structure['has_dataset'] = True
        structure['dimensions'] = dict(node.ds.dims)
        structure['variables'] = list(node.ds.data_vars)
        structure['attributes'] = dict(node.ds.attrs)

        print(f"{'  ' * depth}📁 {path or 'ROOT'}")
        if structure['dimensions']:
            print(f"{'  ' * depth}   Dims: {structure['dimensions']}")
        if structure['variables']:
            print(f"{'  ' * depth}   Vars: {structure['variables']}")

    # Explore children
    if hasattr(node, 'children') and node.children:
        for child_name, child_node in node.children.items():
            child_path = f"{path}/{child_name}" if path else child_name
            structure['children'][child_name] = explore_tree_recursive(
                child_node, child_path, depth + 1, max_depth
            )

    return structure


def extract_variable_data(dt, variable_path):
    """Extract specific variable data from DataTree."""
    print(f"\n🎯 EXTRACTING VARIABLE: {variable_path}")
    print("="*60)

    try:
        # Try different access methods
        methods = [
            f"dt['{variable_path}']",
            f"dt.{variable_path.replace('/', '.')}",
            "Navigate step by step"
        ]

        data = None
        successful_method = None

        # Method 1: Direct path access
        try:
            parts = variable_path.split('/')
            node = dt
            for part in parts:
                if hasattr(node, 'children') and part in node.children:
                    node = node.children[part]
                else:
                    node = getattr(node, part)

            if hasattr(node, 'ds') and node.ds is not None:
                data = node.ds
                successful_method = "Step-by-step navigation"
        except:
            pass

        # Method 2: Dictionary-style access
        if data is None:
            try:
                node = dt[variable_path]
                if hasattr(node, 'ds'):
                    data = node.ds
                    successful_method = "Dictionary-style access"
            except:
                pass

        if data is not None:
            print(f"✅ Successfully accessed using: {successful_method}")
            print(f"📊 Dataset info:")
            print(f"   Dimensions: {dict(data.dims)}")
            print(f"   Variables: {list(data.data_vars)}")
            print(f"   Coordinates: {list(data.coords)}")

            # Show data for each variable
            for var_name in data.data_vars:
                var_data = data[var_name]
                print(f"\n   Variable '{var_name}':")
                print(f"      Shape: {var_data.shape}")
                print(f"      Dtype: {var_data.dtype}")
                print(f"      Min/Max: {float(var_data.min()):.2f} / {float(var_data.max()):.2f}")

                # Show attributes
                if var_data.attrs:
                    print(f"      Attributes: {var_data.attrs}")

            return data
        else:
            print(f"❌ Could not access variable at path: {variable_path}")
            return None

    except Exception as e:
        print(f"❌ Error extracting variable: {e}")
        return None


def find_variables_in_tree(dt):
    """Find all variables in the DataTree."""
    print(f"\n🔍 FINDING ALL VARIABLES")
    print("="*60)

    variables = {}

    def search_node(node, path=""):
        if hasattr(node, 'ds') and node.ds is not None:
            for var_name in node.ds.data_vars:
                full_path = f"{path}/{var_name}" if path else var_name
                variables[full_path] = {
                    'path': path,
                    'name': var_name,
                    'shape': node.ds[var_name].shape,
                    'dtype': str(node.ds[var_name].dtype),
                    'dims': node.ds[var_name].dims
                }

        if hasattr(node, 'children') and node.children:
            for child_name, child_node in node.children.items():
                child_path = f"{path}/{child_name}" if path else child_name
                search_node(child_node, child_path)

    search_node(dt)

    print(f"📊 Found {len(variables)} variables:")
    for var_path, var_info in variables.items():
        print(f"   {var_path}: {var_info['shape']} ({var_info['dtype']})")

    return variables


def convert_datatree_to_dataset(dt, flatten_names=True):
    """Convert DataTree to a single xarray Dataset."""
    print(f"\n🔄 CONVERTING DATATREE TO DATASET")
    print("="*60)

    datasets = []

    def collect_datasets(node, path=""):
        if hasattr(node, 'ds') and node.ds is not None and len(node.ds.data_vars) > 0:
            ds = node.ds.copy()

            if flatten_names and path:
                # Rename variables with path prefix
                ds = ds.rename({var: f"{path.replace('/', '_')}_{var}" for var in ds.data_vars})

            datasets.append(ds)

        if hasattr(node, 'children'):
            for child_name, child_node in node.children.items():
                child_path = f"{path}/{child_name}" if path else child_name
                collect_datasets(child_node, child_path)

    collect_datasets(dt)

    if datasets:
        try:
            merged_ds = xr.merge(datasets, compat='override')
            print(f"✅ Successfully merged {len(datasets)} datasets")
            print(f"📊 Result:")
            print(f"   Variables: {len(merged_ds.data_vars)}")
            print(f"   Dimensions: {dict(merged_ds.dims)}")
            print(f"   Coordinates: {list(merged_ds.coords)}")

            return merged_ds
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return None
    else:
        print(f"❌ No datasets found to merge")
        return None


def analyze_time_dimensions(dt):
    """Analyze time-related dimensions in the DataTree."""
    print(f"\n⏰ TIME DIMENSIONS ANALYSIS")
    print("="*60)

    time_info = {}

    def analyze_node(node, path=""):
        if hasattr(node, 'ds') and node.ds is not None:
            ds = node.ds

            # Look for time-related dimensions
            time_dims = {}
            for dim_name, dim_size in ds.dims.items():
                if any(t in dim_name.lower() for t in ['time', 'step', 'hour', 'forecast']):
                    time_dims[dim_name] = dim_size

            if time_dims:
                time_info[path or 'ROOT'] = {
                    'dimensions': time_dims,
                    'coords': {}
                }

                # Get coordinate values
                for dim_name in time_dims:
                    if dim_name in ds.coords:
                        coord_values = ds.coords[dim_name].values
                        time_info[path or 'ROOT']['coords'][dim_name] = {
                            'values': coord_values,
                            'first': coord_values[0] if len(coord_values) > 0 else None,
                            'last': coord_values[-1] if len(coord_values) > 0 else None
                        }

        if hasattr(node, 'children'):
            for child_name, child_node in node.children.items():
                child_path = f"{path}/{child_name}" if path else child_name
                analyze_node(child_node, child_path)

    analyze_node(dt)

    print(f"⏰ Time dimension summary:")
    for path, info in time_info.items():
        print(f"   {path}:")
        for dim_name, dim_size in info['dimensions'].items():
            print(f"      {dim_name}: {dim_size} steps")
            if dim_name in info['coords']:
                coord_info = info['coords'][dim_name]
                print(f"         Range: {coord_info['first']} → {coord_info['last']}")

    return time_info


def export_to_netcdf(dt, output_file):
    """Export DataTree structure to NetCDF file."""
    print(f"\n💾 EXPORTING TO NETCDF")
    print("="*60)

    try:
        # Convert to single dataset first
        ds = convert_datatree_to_dataset(dt, flatten_names=True)

        if ds is not None:
            ds.to_netcdf(output_file, engine='netcdf4')
            print(f"✅ Successfully exported to: {output_file}")

            # Show file info
            file_size = Path(output_file).stat().st_size / (1024*1024)
            print(f"📊 File size: {file_size:.1f} MB")

            return True
        else:
            print(f"❌ Failed to convert DataTree to Dataset")
            return False

    except Exception as e:
        print(f"❌ Error exporting to NetCDF: {e}")
        return False


def main():
    """Main function to explore ECMWF DataTree."""
    parser = argparse.ArgumentParser(
        description="Explore ECMWF DataTree structure from parquet files"
    )

    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Path to parquet file"
    )

    parser.add_argument(
        "--extract",
        type=str,
        default=None,
        help="Variable path to extract (e.g., 'tp/accum/surface')"
    )

    parser.add_argument(
        "--export-netcdf",
        type=str,
        default=None,
        help="Export to NetCDF file"
    )

    parser.add_argument(
        "--show-refs",
        action="store_true",
        help="Show zarr references analysis"
    )

    args = parser.parse_args()

    parquet_path = args.parquet

    print("="*80)
    print("🌳 ECMWF DATATREE EXPLORER")
    print("="*80)
    print(f"📁 Parquet file: {parquet_path}")

    if not Path(parquet_path).exists():
        print(f"❌ Error: File not found: {parquet_path}")
        return

    # Step 1: Read parquet and create zarr store
    print(f"\n📊 Step 1: Reading parquet file...")
    zstore = read_parquet_fixed(parquet_path)

    # Step 2: Analyze zarr references (optional)
    if args.show_refs:
        analyze_zarr_references(zstore)

    # Step 3: Create DataTree
    print(f"\n🌳 Step 2: Creating DataTree...")
    try:
        fs = fsspec.filesystem("reference", fo=zstore, remote_protocol='s3',
                              remote_options={'anon': True})
        mapper = fs.get_mapper("")

        # Open as datatree
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False, decode_timedelta=False)

        print(f"✅ DataTree created successfully")
        print(f"📊 Type: {type(dt)}")

    except Exception as e:
        print(f"❌ Error creating DataTree: {e}")
        return

    # Step 4: Explore structure
    structure = explore_datatree_structure(dt)

    # Step 5: Find all variables
    variables = find_variables_in_tree(dt)

    # Step 6: Analyze time dimensions
    time_info = analyze_time_dimensions(dt)

    # Step 7: Extract specific variable if requested
    if args.extract:
        extracted_data = extract_variable_data(dt, args.extract)

    # Step 8: Convert to single dataset
    print(f"\n📊 Step 3: Converting to single dataset...")
    merged_ds = convert_datatree_to_dataset(dt)

    # Step 9: Export to NetCDF if requested
    if args.export_netcdf:
        export_to_netcdf(dt, args.export_netcdf)

    # Step 10: Summary and recommendations
    print(f"\n" + "="*80)
    print("📋 SUMMARY AND RECOMMENDATIONS")
    print("="*80)

    print(f"🌳 DataTree structure:")
    print(f"   - Contains hierarchical ECMWF data")
    print(f"   - {len(variables)} variables found")
    print(f"   - Time steps: Limited to 2 (0h, 3h) - Need index-based expansion")

    print(f"\n💡 Recommendations:")
    print(f"   1. Use the index-based processor to expand to 85 time steps")
    print(f"   2. Convert to single dataset for easier analysis")
    print(f"   3. Focus on specific variables like 'tp' for precipitation")

    print(f"\n🔧 Example usage:")
    print(f"   # Access precipitation data")
    print(f"   surface_tp = dt['tp']['accum']['surface'].ds['tp']")
    print(f"   ")
    print(f"   # Convert to single dataset")
    print(f"   ds = convert_datatree_to_dataset(dt)")
    print(f"   precip = ds['tp_accum_surface_tp']")

    if merged_ds is not None:
        print(f"\n✅ Conversion successful! Use merged_ds for analysis.")

    print("="*80)


if __name__ == "__main__":
    main()