#!/usr/bin/env python3
"""
Rebuild hierarchical zarr structure from Stage 2-only references.

This takes Stage 2-only parquet files and creates proper hierarchical structure
so they can be opened with the simple dt['/tp/accum/surface'].ds['tp'] syntax.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict

os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'


def read_parquet_fixed(parquet_path):
    """Read parquet with proper handling."""
    import ast

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


def build_hierarchical_from_stage2(stage2_refs):
    """
    Convert Stage 2-only references to hierarchical structure.

    Input format:  step_003/tp/surface/3 -> [url, offset, length]
    Output format: tp/accum/surface/tp/0 -> [url, offset, length]
                   tp/accum/surface/.zarray -> {...proper metadata...}
    """
    print("\n🔧 Building hierarchical structure from Stage 2 data...")

    new_refs = {'.zgroup': json.dumps({"zarr_format": 2})}

    # Group data by variable and level
    var_data = defaultdict(lambda: defaultdict(list))  # {var: {level_path: [(step, ref)]}}

    for key, ref in stage2_refs.items():
        if not key.startswith('step_'):
            new_refs[key] = ref
            continue

        # Parse: step_003/tp/surface/3
        parts = key.split('/')
        if len(parts) < 3:
            continue

        step_str = parts[0]  # step_003
        varname = parts[1]   # tp
        level_type = parts[2]  # surface or isobaricInhPa, etc.

        # Extract step number
        step = int(step_str.replace('step_', ''))

        # Determine step type
        step_type = 'accum' if varname in ['tp', 'cp', 'lsp'] else 'instant'

        # Build level path
        if level_type == 'surface':
            level_path = 'surface'
        elif len(parts) > 3 and parts[3].replace('.', '').isdigit():
            # Has level value: tp/isobaricInhPa/500/
            level_val = parts[3]
            level_path = f'{level_type}/{level_val}'
        else:
            level_path = level_type

        # Store reference
        var_level_key = f"{varname}/{step_type}/{level_path}"
        var_data[varname][var_level_key].append((step, ref))

    print(f"   Found {len(var_data)} variables")

    # Build hierarchical structure for each variable
    for varname, levels in var_data.items():
        print(f"   Processing {varname}...")

        # Variable root group
        new_refs[f'{varname}/.zgroup'] = json.dumps({"zarr_format": 2})
        new_refs[f'{varname}/.zattrs'] = json.dumps({})

        for var_level_path, step_refs in levels.items():
            # Sort by step
            step_refs.sort(key=lambda x: x[0])
            steps = [s for s, _ in step_refs]
            refs_only = [r for _, r in step_refs]

            print(f"      {var_level_path}: {len(steps)} timesteps")

            # Create groups along the path
            path_parts = var_level_path.split('/')
            for i in range(1, len(path_parts) + 1):
                partial_path = '/'.join(path_parts[:i])
                group_key = f'{partial_path}/.zgroup'
                if group_key not in new_refs:
                    new_refs[group_key] = json.dumps({"zarr_format": 2})
                    new_refs[f'{partial_path}/.zattrs'] = json.dumps({})

            # Create coordinate arrays
            base_path = var_level_path

            # Time coordinate
            new_refs[f'{base_path}/time/.zarray'] = json.dumps({
                "chunks": [1],
                "compressor": None,
                "dtype": "<i8",
                "fill_value": None,
                "filters": None,
                "order": "C",
                "shape": [len(steps)],
                "zarr_format": 2
            })
            new_refs[f'{base_path}/time/.zattrs'] = json.dumps({
                "units": "seconds since 1970-01-01",
                "calendar": "proleptic_gregorian",
                "_ARRAY_DIMENSIONS": ["time"]
            })

            # Step coordinate
            new_refs[f'{base_path}/step/.zarray'] = json.dumps({
                "chunks": [len(steps)],
                "compressor": None,
                "dtype": "<i8",
                "fill_value": None,
                "filters": None,
                "order": "C",
                "shape": [len(steps)],
                "zarr_format": 2
            })
            step_data = np.array([s * 3600000000000 for s in steps], dtype='<i8')  # hours to nanoseconds
            new_refs[f'{base_path}/step/0'] = step_data.tobytes()
            new_refs[f'{base_path}/step/.zattrs'] = json.dumps({
                "units": "nanoseconds",
                "_ARRAY_DIMENSIONS": ["step"]
            })

            # Main variable array - just store references
            # We'll create a simple structure that points to the GRIB data
            new_refs[f'{base_path}/{varname}/.zarray'] = json.dumps({
                "chunks": [1, 721, 1440],  # Typical ECMWF 0.25 degree grid
                "compressor": None,
                "dtype": "<f4",
                "fill_value": None,
                "filters": None,
                "order": "C",
                "shape": [len(steps), 721, 1440],
                "zarr_format": 2
            })
            new_refs[f'{base_path}/{varname}/.zattrs'] = json.dumps({
                "long_name": varname,
                "_ARRAY_DIMENSIONS": ["step", "latitude", "longitude"]
            })

            # Add references for each timestep
            for idx, (step, ref) in enumerate(step_refs):
                new_refs[f'{base_path}/{varname}/{idx}'] = ref

            # Add lat/lon coordinates (stub - actual values from first GRIB)
            # Latitude
            new_refs[f'{base_path}/latitude/.zarray'] = json.dumps({
                "chunks": [721],
                "compressor": None,
                "dtype": "<f4",
                "fill_value": None,
                "filters": None,
                "order": "C",
                "shape": [721],
                "zarr_format": 2
            })
            lat_data = np.linspace(90, -90, 721, dtype='<f4')
            new_refs[f'{base_path}/latitude/0'] = lat_data.tobytes()
            new_refs[f'{base_path}/latitude/.zattrs'] = json.dumps({
                "units": "degrees_north",
                "long_name": "latitude",
                "_ARRAY_DIMENSIONS": ["latitude"]
            })

            # Longitude
            new_refs[f'{base_path}/longitude/.zarray'] = json.dumps({
                "chunks": [1440],
                "compressor": None,
                "dtype": "<f4",
                "fill_value": None,
                "filters": None,
                "order": "C",
                "shape": [1440],
                "zarr_format": 2
            })
            lon_data = np.linspace(0, 359.75, 1440, dtype='<f4')
            new_refs[f'{base_path}/longitude/0'] = lon_data.tobytes()
            new_refs[f'{base_path}/longitude/.zattrs'] = json.dumps({
                "units": "degrees_east",
                "long_name": "longitude",
                "_ARRAY_DIMENSIONS": ["longitude"]
            })

    print(f"   Created {len(new_refs)} references")
    return new_refs


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Rebuild hierarchical structure')
    parser.add_argument('--input', type=str, required=True,
                       help='Input parquet file (Stage 2-only structure)')
    parser.add_argument('--output', type=str,
                       help='Output parquet file (defaults to input_hierarchical.parquet)')

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_hierarchical.parquet"

    print("="*80)
    print("REBUILD HIERARCHICAL STRUCTURE")
    print("="*80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("="*80)

    # Read Stage 2-only structure
    print("\n📁 Reading Stage 2-only structure...")
    stage2_refs = read_parquet_fixed(input_file)
    print(f"   Loaded {len(stage2_refs)} references")

    # Check if this is Stage 2-only
    has_step = any(k.startswith('step_') for k in stage2_refs.keys())
    if not has_step:
        print("\n⚠️ This file already has hierarchical structure!")
        print("   No conversion needed.")
        return

    # Build hierarchical structure
    hierarchical_refs = build_hierarchical_from_stage2(stage2_refs)

    # Save as parquet
    print(f"\n💾 Saving hierarchical structure...")
    data = []
    for key, value in hierarchical_refs.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        elif isinstance(value, bytes):
            encoded_value = value
        else:
            encoded_value = str(value).encode('utf-8')
        data.append((key, encoded_value))

    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_file)

    print(f"✅ Saved: {output_file}")
    print(f"   {len(df)} references")

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print("\nNow you can use the standard syntax:")
    print(f"   zstore = read_parquet_fixed('{output_file}')")
    print("   dt = xr.open_datatree(...)")
    print("   data = dt['/tp/accum/surface'].ds['tp']")
    print("="*80)


if __name__ == "__main__":
    main()
