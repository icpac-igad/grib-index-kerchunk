#!/usr/bin/env python3
"""
ECMWF Stage 3 Parquet Reader - AIFS-ETL Method (All 85 Timesteps)

This script extracts ALL 85 timesteps by looping through individual step_XXX arrays
using the exact AIFS-ETL extraction method.

Key features:
1. Follows aifs-etl.py extraction flow exactly
2. Loops through all step_XXX/variable/sfc/member/0.0.0 arrays
3. Fetches S3 GRIB2 data and decodes with cfgrib
4. Assembles into 3D numpy array (85, 721, 1440)
5. Supports regional subsetting for memory efficiency

Usage:
    python read_stage3_aifs_all_timesteps.py --member control --variable 2t
    python read_stage3_aifs_all_timesteps.py --member ens_01 --variable tp --output data.pkl
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from pathlib import Path
import base64
import tempfile
import time
import re
from typing import Optional, Tuple, Dict

# Configuration
DEFAULT_INPUT_DIR = Path("/scratch/notebook/test_ecmwf_three_stage_prebuilt_output")

# Set up anonymous S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Predefined regions
PREDEFINED_REGIONS = {
    'east-africa': {
        'name': 'East Africa',
        'lat_min': -12, 'lat_max': 23,
        'lon_min': 21, 'lon_max': 53
    },
    'europe-africa': {
        'name': 'Europe + Africa',
        'lat_min': -12, 'lat_max': 55,
        'lon_min': -25, 'lon_max': 65
    },
    'europe': {
        'name': 'Europe',
        'lat_min': 35, 'lat_max': 70,
        'lon_min': -10, 'lon_max': 40
    },
    'north-america': {
        'name': 'North America',
        'lat_min': 15, 'lat_max': 72,
        'lon_min': -170, 'lon_max': -50
    },
    'south-asia': {
        'name': 'South Asia',
        'lat_min': 5, 'lat_max': 40,
        'lon_min': 60, 'lon_max': 100
    },
    'global': {
        'name': 'Global',
        'lat_min': -90, 'lat_max': 90,
        'lon_min': -180, 'lon_max': 180
    }
}

# Default regional subsetting (will be overridden by command-line args)
USE_REGIONAL_SUBSET = True
LAT_MIN, LAT_MAX = -12, 23  # East Africa (default)
LON_MIN, LON_MAX = 21, 53


def read_parquet_to_refs(parquet_path):
    """Read parquet file and extract zarr references. From aifs-etl.py"""
    print(f"  📊 Reading parquet file: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)

    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        if isinstance(value, bytes):
            value = value.decode('utf-8')

        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            try:
                value = json.loads(value)
            except:
                pass

        zstore[key] = value

    if 'version' in zstore:
        del zstore['version']

    print(f"  ✅ Loaded {len(zstore)} references")
    return zstore


def decode_chunk_reference(chunk_ref):
    """Decode a chunk reference. From aifs-etl.py"""
    if isinstance(chunk_ref, str):
        if chunk_ref.startswith('base64:'):
            base64_str = chunk_ref[7:]
            try:
                decoded = base64.b64decode(base64_str)
                return 'base64', decoded
            except:
                return 'unknown', chunk_ref
        else:
            return 'unknown', chunk_ref

    elif isinstance(chunk_ref, list):
        if len(chunk_ref) >= 3:
            url = chunk_ref[0]
            offset = chunk_ref[1]
            length = chunk_ref[2]

            if isinstance(url, str) and ('s3://' in url or 's3.amazonaws.com' in url):
                return 's3', (url, offset, length)

    return 'unknown', chunk_ref


def fetch_s3_byte_range_fsspec(url, offset, length, max_retries=3, retry_delay=2):
    """Fetch a byte range from S3 using fsspec. From aifs-etl.py"""
    for attempt in range(max_retries):
        try:
            import fsspec

            if url.startswith('s3://'):
                s3_path = url
            else:
                s3_path = f"s3://{url}"

            fs = fsspec.filesystem('s3', anon=True)

            with fs.open(s3_path, 'rb') as f:
                f.seek(offset)
                data = f.read(length)

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"    ❌ S3 fetch failed after {max_retries} attempts: {e}")
                return None

    return None


def fetch_s3_byte_range_obstore(url, offset, length):
    """Fetch a byte range from S3 using obstore. From aifs-etl.py"""
    try:
        import obstore as obs
        from obstore.store import from_url

        if url.startswith('s3://'):
            url_parts = url[5:].split('/', 1)
            bucket = url_parts[0]
            key = url_parts[1] if len(url_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid S3 URL: {url}")

        bucket_regions = {
            'ecmwf-forecasts': 'eu-central-1',
        }
        region = bucket_regions.get(bucket, 'eu-central-1')

        bucket_url = f"s3://{bucket}"
        store = from_url(bucket_url, region=region, skip_signature=True)

        result = obs.get_range(store, key, start=offset, end=offset + length)
        data = bytes(result)

        return data

    except ImportError:
        return fetch_s3_byte_range_fsspec(url, offset, length)
    except Exception as e:
        return fetch_s3_byte_range_fsspec(url, offset, length)


def decode_grib2_data(data: bytes) -> Optional[np.ndarray]:
    """Decode GRIB2 data to numpy array. From aifs-etl.py"""
    if data[:4] != b'GRIB':
        return None

    try:
        import cfgrib
        import xarray as xr

        with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        var_names = list(ds.data_vars)
        if var_names:
            var_data = ds[var_names[0]].values
            ds.close()
            os.unlink(tmp_path)
            return var_data

        ds.close()
        os.unlink(tmp_path)
        return None

    except Exception as e:
        # Try eccodes as fallback
        try:
            import eccodes
            gid = eccodes.codes_new_from_message(data)
            values = eccodes.codes_get_array(gid, 'values')
            eccodes.codes_release(gid)
            # Reshape to ECMWF grid
            if len(values) == 721 * 1440:
                return values.reshape(721, 1440)
            return values
        except:
            print(f"    ⚠️ GRIB2 decode failed: {e}")
            return None


def extract_single_timestep(zstore: Dict, variable: str, step_hour: int,
                           member: str = 'control', use_obstore: bool = False) -> Optional[np.ndarray]:
    """
    Extract a single timestep from step_XXX arrays using AIFS-ETL method.

    Args:
        zstore: Zarr store dictionary
        variable: Variable name (e.g., '2t', 'tp')
        step_hour: Forecast hour (0, 3, 6, ..., 360)
        member: Ensemble member name
        use_obstore: Use obstore for S3 fetching

    Returns:
        numpy array (2D: lat, lon) or None
    """
    # Build the key for this timestep
    # Format: step_XXX/<var>/sfc/<member>/0.0.0
    step_key = f"step_{step_hour:03d}/{variable}/sfc/{member}/0.0.0"

    if step_key not in zstore:
        # Try alternative level names
        for level in ['sf', 'surface']:
            alt_key = f"step_{step_hour:03d}/{variable}/{level}/{member}/0.0.0"
            if alt_key in zstore:
                step_key = alt_key
                break
        else:
            return None

    # Get the chunk reference
    chunk_ref = zstore[step_key]
    ref_type, ref_data = decode_chunk_reference(chunk_ref)

    if ref_type == 'base64':
        # Base64 encoded data
        data = ref_data
        try:
            import numcodecs
            # Check if compressed
            array = np.frombuffer(data, dtype='<f4')
            if array.size == 721 * 1440:
                return array.reshape((721, 1440))
        except:
            pass
        return None

    elif ref_type == 's3':
        # S3 reference - fetch and decode
        url, offset, length = ref_data

        # FIX: Some step_XXX URLs are missing .grib2 extension
        if not url.endswith('.grib2') and not url.endswith('.grb') and not url.endswith('.grb2'):
            url = url + '.grib2'

        # Fetch from S3
        if use_obstore:
            chunk_data = fetch_s3_byte_range_obstore(url, offset, length)
        else:
            chunk_data = fetch_s3_byte_range_fsspec(url, offset, length)

        if chunk_data is None:
            return None

        # Decode GRIB2
        array_2d = decode_grib2_data(chunk_data)
        return array_2d

    return None


def extract_all_timesteps(parquet_file: Path, variable: str = '2t',
                          member: str = 'control', use_obstore: bool = False) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Extract all 85 timesteps from Stage 3 parquet using AIFS-ETL method.

    Args:
        parquet_file: Path to Stage 3 parquet file
        variable: Variable to extract (e.g., '2t', 'tp')
        member: Ensemble member name (e.g., 'control', 'ens_01', 'ens01')
        use_obstore: Use obstore for faster S3 access

    Returns:
        tuple: (data_3d, metadata)
            data_3d: numpy array (timesteps, lat, lon)
            metadata: dict with coordinates and info
    """
    print(f"\n{'='*80}")
    print(f"Extracting ALL timesteps for {variable} - {member}")
    print(f"{'='*80}")

    # Step 1: Read parquet
    zstore = read_parquet_to_refs(parquet_file)

    # Step 2: Find all available timesteps
    # Try both with and without underscore (e.g., 'ens_01' and 'ens01')
    member_normalized = member.replace('_', '')  # Remove underscores

    # Try multiple patterns
    patterns_to_try = [
        rf'step_(\d+)/{variable}/sfc/{member}/0\.0\.0',      # Original
        rf'step_(\d+)/{variable}/sfc/{member_normalized}/0\.0\.0',  # Without underscore
        rf'step_(\d+)/{variable}/sf/{member}/0\.0\.0',        # Alternative level name
        rf'step_(\d+)/{variable}/sf/{member_normalized}/0\.0\.0',
    ]

    available_steps = []
    actual_member_name = member  # Will be updated when we find matches

    for pattern_str in patterns_to_try:
        step_pattern = re.compile(pattern_str)
        for key in zstore.keys():
            match = step_pattern.match(key)
            if match:
                step_hour = int(match.group(1))
                available_steps.append(step_hour)
                # Extract actual member name from the key
                parts = key.split('/')
                if len(parts) >= 4:
                    actual_member_name = parts[3]

        if available_steps:
            print(f"  Using member name: '{actual_member_name}' (from parquet)")
            break

    available_steps = sorted(list(set(available_steps)))  # Remove duplicates
    n_steps = len(available_steps)

    print(f"\n  Found {n_steps} timesteps")
    print(f"  Range: {min(available_steps)}h to {max(available_steps)}h")

    if n_steps == 0:
        print(f"  ❌ No timesteps found for {variable}/{member}")
        return None, None

    # Step 3: Extract each timestep
    print(f"\n  Extracting timesteps...")
    timestep_arrays = []
    failed_steps = []

    for i, step_hour in enumerate(available_steps):
        if i % 10 == 0 or i == n_steps - 1:
            print(f"    Progress: {i+1}/{n_steps} (step {step_hour}h)")

        # Use the actual member name found in the parquet
        array_2d = extract_single_timestep(zstore, variable, step_hour, actual_member_name, use_obstore)

        if array_2d is not None:
            timestep_arrays.append(array_2d)
        else:
            failed_steps.append(step_hour)
            print(f"    ⚠️ Failed: step {step_hour}h")

    if not timestep_arrays:
        print(f"\n  ❌ No timesteps successfully extracted")
        return None, None

    # Step 4: Stack into 3D array
    print(f"\n  Stacking {len(timestep_arrays)} timesteps...")
    data_3d = np.stack(timestep_arrays, axis=0)

    print(f"  ✅ Data shape: {data_3d.shape}")
    print(f"  ✅ Memory: ~{data_3d.nbytes / 1024 / 1024:.1f} MB")

    if failed_steps:
        print(f"  ⚠️ Failed steps ({len(failed_steps)}): {failed_steps[:10]}{'...' if len(failed_steps) > 10 else ''}")

    # Step 5: Extract coordinates
    print(f"\n  Extracting coordinates...")

    # Get lat/lon from aggregated structure (if available)
    # Try t2m structure first
    var_mapping = {
        '2t': 't2m/instant/heightAboveGround',
        't2m': 't2m/instant/heightAboveGround',
        'tp': 'tp/accum/surface',
        '10u': 'u10/instant/heightAboveGround',
        '10v': 'v10/instant/heightAboveGround',
    }

    lats = None
    lons = None

    if variable in var_mapping:
        coord_path = var_mapping[variable]

        # Try to extract latitude
        lat_key = f"{coord_path}/latitude/.zarray"
        if lat_key in zstore:
            lat_data_key = f"{coord_path}/latitude/0"
            if lat_data_key in zstore:
                lat_ref = zstore[lat_data_key]
                if isinstance(lat_ref, str) and lat_ref.startswith('base64:'):
                    try:
                        decoded = base64.b64decode(lat_ref[7:])
                        lats = np.frombuffer(decoded, dtype='<f8')
                    except:
                        pass

        # Try to extract longitude
        lon_key = f"{coord_path}/longitude/.zarray"
        if lon_key in zstore:
            lon_data_key = f"{coord_path}/longitude/0"
            if lon_data_key in zstore:
                lon_ref = zstore[lon_data_key]
                if isinstance(lon_ref, str) and lon_ref.startswith('base64:'):
                    try:
                        decoded = base64.b64decode(lon_ref[7:])
                        lons = np.frombuffer(decoded, dtype='<f8')
                    except:
                        pass

    # Fallback: Create default ECMWF grid
    if lats is None or lons is None:
        print(f"    Using default ECMWF 0.25° grid")
        lats = np.linspace(90, -90, 721)
        lons = np.linspace(0, 360, 1440)
    else:
        print(f"    ✅ Extracted coordinates from parquet")

    # Step 6: Apply regional subset if enabled
    if USE_REGIONAL_SUBSET:
        print(f"\n  Applying regional subset: lat[{LAT_MIN}:{LAT_MAX}], lon[{LON_MIN}:{LON_MAX}]")

        # Convert longitude range to 0-360
        lon_min_360 = LON_MIN % 360
        lon_max_360 = LON_MAX % 360

        # Find indices
        lat_idx = np.where((lats >= LAT_MIN) & (lats <= LAT_MAX))[0]
        lon_idx = np.where((lons >= lon_min_360) & (lons <= lon_max_360))[0]

        if len(lat_idx) > 0 and len(lon_idx) > 0:
            # Subset data
            data_3d = data_3d[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
            lats = lats[lat_idx[0]:lat_idx[-1]+1]
            lons = lons[lon_idx[0]:lon_idx[-1]+1]

            # Convert lons back to -180 to 180
            lons = np.where(lons > 180, lons - 360, lons)

            print(f"    ✅ Subset shape: {data_3d.shape}")
            print(f"    ✅ Memory reduced to: ~{data_3d.nbytes / 1024 / 1024:.1f} MB")

    # Step 7: Create metadata
    metadata = {
        'variable': variable,
        'member': member,
        'n_timesteps': len(timestep_arrays),
        'forecast_hours': available_steps[:len(timestep_arrays)],
        'latitude': lats,
        'longitude': lons,
        'shape': data_3d.shape,
        'failed_steps': failed_steps,
        'regional_subset': USE_REGIONAL_SUBSET,
        'subset_bounds': {
            'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
            'lon_min': LON_MIN, 'lon_max': LON_MAX
        } if USE_REGIONAL_SUBSET else None
    }

    return data_3d, metadata


def save_to_pickle(data, metadata, output_file):
    """Save extracted data to pickle file (like aifs-etl.py)."""
    import pickle

    output_data = {
        'data': data,
        'metadata': metadata
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"\n💾 Saved to: {output_file}")
    print(f"📊 File size: {file_size:.2f} MB")


def save_to_numpy(data, metadata, output_file):
    """Save extracted data to numpy npz file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        data=data,
        forecast_hours=metadata['forecast_hours'],
        latitude=metadata['latitude'],
        longitude=metadata['longitude'],
        variable=metadata['variable'],
        member=metadata['member']
    )

    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"\n💾 Saved to: {output_file}")
    print(f"📊 File size: {file_size:.2f} MB")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract ALL 85 timesteps from ECMWF Stage 3 using AIFS-ETL method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Available predefined regions:
  east-africa      East Africa (-12 to 23°N, 21 to 53°E)
  europe-africa    Europe + Africa (-12 to 55°N, -25 to 65°E)
  europe           Europe (35 to 70°N, -10 to 40°E)
  north-america    North America (15 to 72°N, -170 to -50°E)
  south-asia       South Asia (5 to 40°N, 60 to 100°E)
  global           Global coverage

Examples:
  # Extract East Africa region (default)
  python read_stage3_aifs_all_timesteps.py --member ens_02 --variable tp --output e02_precip.npz

  # Extract with custom region
  python read_stage3_aifs_all_timesteps.py --member control --variable 2t --custom-region -12 23 21 53

  # Extract global data (no regional subset)
  python read_stage3_aifs_all_timesteps.py --member control --variable 2t --no-subset
        '''
    )
    parser.add_argument('--member', type=str, default='control',
                       help='Ensemble member (default: control)')
    parser.add_argument('--variable', type=str, default='2t',
                       help='Variable to extract (default: 2t)')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR,
                       help='Input directory')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output file (.pkl or .npz)')
    parser.add_argument('--use-obstore', action='store_true',
                       help='Use obstore for faster S3 fetching')
    parser.add_argument('--region', type=str, default='east-africa',
                       choices=list(PREDEFINED_REGIONS.keys()),
                       help='Predefined region for extraction (default: east-africa)')
    parser.add_argument('--custom-region', nargs=4, type=float, metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                       help='Custom region bounds (overrides --region)')
    parser.add_argument('--no-subset', action='store_true',
                       help='Extract full global data (no regional subset, overrides --region)')
    args = parser.parse_args()

    # Override regional subset and bounds based on arguments
    global USE_REGIONAL_SUBSET, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    if args.no_subset:
        # No regional subset - extract global
        USE_REGIONAL_SUBSET = False
    elif args.custom_region:
        # Custom region specified
        USE_REGIONAL_SUBSET = True
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = args.custom_region
    else:
        # Use predefined region
        USE_REGIONAL_SUBSET = True
        region_config = PREDEFINED_REGIONS[args.region]
        LAT_MIN = region_config['lat_min']
        LAT_MAX = region_config['lat_max']
        LON_MIN = region_config['lon_min']
        LON_MAX = region_config['lon_max']

    print("="*80)
    print("ECMWF Stage 3 Extraction - ALL 85 TIMESTEPS")
    print("="*80)
    print(f"Member: {args.member}")
    print(f"Variable: {args.variable}")
    print(f"S3 method: {'obstore' if args.use_obstore else 'fsspec'}")
    print(f"Regional subset: {'Yes' if USE_REGIONAL_SUBSET else 'No (global)'}")
    if USE_REGIONAL_SUBSET:
        if args.custom_region:
            print(f"  Region: Custom")
        else:
            region_name = PREDEFINED_REGIONS[args.region]['name']
            print(f"  Region: {region_name} (--region {args.region})")
        print(f"  Bounds: lat[{LAT_MIN}:{LAT_MAX}], lon[{LON_MIN}:{LON_MAX}]")
    print("="*80)

    # Find parquet file
    parquet_file = args.input_dir / f"stage3_{args.member}_final.parquet"

    if not parquet_file.exists():
        print(f"❌ Error: Parquet file not found: {parquet_file}")
        print(f"\nAvailable files:")
        for f in sorted(args.input_dir.glob("stage3_*_final.parquet")):
            print(f"   - {f.name}")
        return False

    # Extract all timesteps
    start_time = time.time()
    data, metadata = extract_all_timesteps(
        parquet_file,
        args.variable,
        args.member,
        args.use_obstore
    )
    elapsed = time.time() - start_time

    if data is None:
        print("\n❌ Failed to extract data")
        return False

    # Save if output specified
    if args.output:
        if args.output.suffix == '.pkl':
            save_to_pickle(data, metadata, args.output)
        elif args.output.suffix == '.npz':
            save_to_numpy(data, metadata, args.output)
        else:
            print(f"⚠️ Unknown output format: {args.output.suffix}")
            print("   Supported: .pkl, .npz")

    # Summary
    print(f"\n{'='*80}")
    print("✅ EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Variable: {metadata['variable']}")
    print(f"Member: {metadata['member']}")
    print(f"Timesteps: {metadata['n_timesteps']}")
    print(f"Forecast hours: {metadata['forecast_hours'][0]}h to {metadata['forecast_hours'][-1]}h")
    print(f"Data shape: {data.shape}")
    print(f"Lat range: {metadata['latitude'].min():.2f} to {metadata['latitude'].max():.2f}")
    print(f"Lon range: {metadata['longitude'].min():.2f} to {metadata['longitude'].max():.2f}")
    print(f"Memory: ~{data.nbytes / 1024 / 1024:.1f} MB")
    print(f"Extraction time: {elapsed:.1f} seconds")

    if metadata['failed_steps']:
        print(f"\n⚠️ Warning: {len(metadata['failed_steps'])} steps failed")
        print(f"   Failed hours: {metadata['failed_steps'][:10]}{'...' if len(metadata['failed_steps']) > 10 else ''}")

    print(f"{'='*80}")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
