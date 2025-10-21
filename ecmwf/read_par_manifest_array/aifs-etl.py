#!/usr/bin/env python3
"""
ECMWF Parquet to PKL Processor

This script processes ECMWF parquet files to create input states for ensemble members.
Phase 1: Uses obstore method to collect variables and save as pkl (following extract_hybrid_refs.py pattern)
Phase 2: Will apply ekr.interpolate and other downstream processing

Usage:
    python aifs-etl.py
"""

import datetime
from collections import defaultdict
import time
import os
import pickle
import json
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


# Configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

# Output configuration
OUTPUT_DIR = "ecmwf_pkl_from_parquet"
SAVE_STATES = True


def read_parquet_to_refs(parquet_path):
    """Read parquet file and extract zarr references (from extract_hybrid_refs.py)."""
    print(f"  📊 Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        if isinstance(value, bytes):
            value = value.decode('utf-8')

        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
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
    """
    Decode a chunk reference. Returns (type, data).
    From extract_hybrid_refs.py
    """
    if isinstance(chunk_ref, str):
        if chunk_ref.startswith('base64:'):
            base64_str = chunk_ref[7:]
            try:
                decoded = base64.b64decode(base64_str)
                return 'base64', decoded
            except Exception as e:
                print(f"    ⚠️ Error decoding base64: {e}")
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


def fetch_s3_byte_range_fsspec(url, offset, length):
    """Fetch a byte range from S3 using fsspec."""
    try:
        import fsspec

        if url.startswith('s3://'):
            s3_path = url[5:]
        else:
            s3_path = url

        fs = fsspec.filesystem('s3', anon=True)

        with fs.open(s3_path, 'rb') as f:
            f.seek(offset)
            data = f.read(length)

        return data

    except Exception as e:
        print(f"    ❌ Error fetching from S3 with fsspec: {e}")
        return None


def fetch_s3_byte_range_obstore(url, offset, length):
    """
    Fetch a byte range from S3 using obstore (if available).
    From extract_hybrid_refs.py
    """
    try:
        import obstore as obs
        from obstore.store import from_url

        # Parse bucket and key from URL
        if url.startswith('s3://'):
            url_parts = url[5:].split('/', 1)
            bucket = url_parts[0]
            key = url_parts[1] if len(url_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid S3 URL: {url}")

        # ECMWF buckets are in EU regions
        bucket_regions = {
            'ecmwf-forecasts': 'eu-central-1',
        }
        region = bucket_regions.get(bucket, 'eu-central-1')

        # Create S3 store
        bucket_url = f"s3://{bucket}"
        store = from_url(bucket_url, region=region, skip_signature=True)

        # Fetch byte range
        result = obs.get_range(store, key, start=offset, end=offset + length)
        data = bytes(result)

        return data

    except ImportError:
        return fetch_s3_byte_range_fsspec(url, offset, length)
    except Exception as e:
        print(f"    ⚠️ obstore error: {e}, falling back to fsspec")
        return fetch_s3_byte_range_fsspec(url, offset, length)


def extract_variable_hybrid(zstore, variable_path, use_obstore=False):
    """
    Extract a variable handling both base64 and S3 references.
    From extract_hybrid_refs.py
    """
    # Get metadata
    zarray_key = f"{variable_path}/.zarray"
    if zarray_key not in zstore:
        print(f"    ⚠️ No metadata found for {variable_path}")
        return None

    metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]

    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['dtype'])
    chunks = tuple(metadata['chunks'])
    compressor = metadata.get('compressor', None)

    # Collect chunks
    chunks_data = {}

    for key in sorted(zstore.keys()):
        if key.startswith(variable_path + "/") and not key.endswith(('.zarray', '.zattrs', '.zgroup')):
            chunk_ref = zstore[key]
            ref_type, ref_data = decode_chunk_reference(chunk_ref)

            if ref_type == 'base64':
                data = ref_data
                if compressor is not None:
                    try:
                        import numcodecs
                        codec = numcodecs.get_codec(compressor)
                        data = codec.decode(data)
                    except:
                        try:
                            import blosc
                            data = blosc.decompress(data)
                        except:
                            pass
                chunks_data[key] = data

            elif ref_type == 's3':
                url, offset, length = ref_data

                # Fetch from S3
                if use_obstore:
                    data = fetch_s3_byte_range_obstore(url, offset, length)
                else:
                    data = fetch_s3_byte_range_fsspec(url, offset, length)

                if data is not None:
                    # Check if it's GRIB2 data
                    if data[:4] == b'GRIB':
                        # Decode GRIB2 message
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
                                chunks_data[key] = var_data

                            os.unlink(tmp_path)
                            ds.close()

                        except ImportError:
                            print(f"      ⚠️ cfgrib not available, trying eccodes")
                            try:
                                import eccodes
                                gid = eccodes.codes_new_from_message(data)
                                values = eccodes.codes_get_array(gid, 'values')
                                eccodes.codes_release(gid)
                                chunks_data[key] = values
                            except:
                                print(f"      ❌ Cannot decode GRIB2 data")
                        except Exception as e:
                            print(f"      ❌ Error decoding GRIB2: {e}")
                    else:
                        # Try decompression if needed
                        if compressor is not None:
                            try:
                                import numcodecs
                                codec = numcodecs.get_codec(compressor)
                                data = codec.decode(data)
                                chunks_data[key] = data
                            except:
                                pass
                        else:
                            chunks_data[key] = data

    if not chunks_data:
        return None

    # Reconstruct array
    try:
        if len(chunks_data) == 1:
            chunk_data = list(chunks_data.values())[0]

            if isinstance(chunk_data, np.ndarray):
                array = chunk_data
            else:
                array = np.frombuffer(chunk_data, dtype=dtype)

            if array.size == np.prod(shape):
                array = array.reshape(shape)

            return array

        else:
            # Multiple chunks - reassemble
            first_chunk = list(chunks_data.values())[0]
            if isinstance(first_chunk, np.ndarray):
                actual_dtype = first_chunk.dtype
            else:
                actual_dtype = dtype

            array = np.zeros(shape, dtype=actual_dtype)

            for chunk_key, chunk_data in chunks_data.items():
                chunk_idx_str = chunk_key.replace(variable_path + "/", "")
                chunk_indices = tuple(int(x) for x in chunk_idx_str.split('.'))

                if isinstance(chunk_data, np.ndarray):
                    chunk_array = chunk_data
                else:
                    chunk_array = np.frombuffer(chunk_data, dtype=actual_dtype)

                # GRIB2 data comes as 2D, but metadata expects 4D
                if chunk_array.ndim == 2 and len(shape) == 4:
                    time_idx = chunk_indices[0] if len(chunk_indices) > 0 else 0
                    step_idx = chunk_indices[1] if len(chunk_indices) > 1 else 0
                    array[time_idx, step_idx, :, :] = chunk_array
                else:
                    # Standard zarr chunk reassembly
                    chunk_shape = []
                    for i, (idx, chunk_size, dim_size) in enumerate(zip(chunk_indices, chunks, shape)):
                        if (idx + 1) * chunk_size <= dim_size:
                            chunk_shape.append(chunk_size)
                        else:
                            chunk_shape.append(dim_size - idx * chunk_size)

                    if chunk_array.size == np.prod(chunk_shape):
                        chunk_array = chunk_array.reshape(tuple(chunk_shape))

                    slices = []
                    for idx, chunk_size, dim_size in zip(chunk_indices, chunks, shape):
                        start = idx * chunk_size
                        end = min(start + chunk_size, dim_size)
                        slices.append(slice(start, end))

                    array[tuple(slices)] = chunk_array

            return array

    except Exception as e:
        print(f"    ❌ Error reconstructing array: {e}")
        return None


def get_variable_path_mapping():
    """
    Map parameter names to their paths in the zarr store.
    Based on ECMWF data structure.
    """
    return {
        # Surface parameters
        '10u': 'u10m/instant/heightAboveGround/u10m',
        '10v': 'v10m/instant/heightAboveGround/v10m',
        '2t': 't2m/instant/heightAboveGround/t2m',
        '2d': 'd2m/instant/heightAboveGround/d2m',
        'msl': 'msl/instant/meanSea/msl',
        'sp': 'sp/instant/surface/sp',
        'skt': 'skt/instant/surface/skt',
        'tcw': 'tcw/instant/atmosphere/tcw',
        # Fixed fields
        'lsm': 'lsm/instant/surface/lsm',
        'z': 'z/instant/surface/z',
        'slor': 'slor/instant/surface/slor',
        'sdor': 'sdor/instant/surface/sdor',
        # Soil parameters
        'sot': 'stl/instant/depthBelowLandLayer/stl',
        # Pressure level parameters
        'gh': 'gh/instant/isobaricInhPa/gh',
        't': 't/instant/isobaricInhPa/t',
        'u': 'u/instant/isobaricInhPa/u',
        'v': 'v/instant/isobaricInhPa/v',
        'w': 'w/instant/isobaricInhPa/w',
        'q': 'q/instant/isobaricInhPa/q',
    }


def get_data_from_parquet_obstore(parquet_path, param, levelist=[], use_obstore=True):
    """
    Retrieve data from parquet files using obstore method (Phase 1).
    NO interpolation or rolling - just raw data collection.

    Args:
        parquet_path: Parquet file path
        param: List of parameters to extract
        levelist: List of pressure/soil levels (optional)
        use_obstore: Use obstore for S3 fetching (faster)

    Returns:
        Dictionary of {param_name: numpy_array}
    """
    fields = {}
    print(f"    Retrieving {param} data" + (f" at levels {levelist}" if levelist else ""))

    # Read parquet to get references
    zstore = read_parquet_to_refs(parquet_path)

    # Get variable path mapping
    var_paths = get_variable_path_mapping()

    # Process each parameter
    for p in param:
        if p not in var_paths:
            print(f"    ⚠️ Parameter {p} not in mapping")
            continue

        base_path = var_paths[p]

        # Extract the variable
        print(f"    Extracting {p} from {base_path}")
        array = extract_variable_hybrid(zstore, base_path, use_obstore=use_obstore)

        if array is not None:
            # Handle multi-dimensional data (time, step, level, lat, lon)
            # For now, take the first time/step if multiple
            if array.ndim > 2:
                # Typically shape is (1, 1, lat, lon) or (levels, lat, lon)
                if levelist and array.ndim >= 3:
                    # Has level dimension
                    for i, level in enumerate(levelist):
                        if i < array.shape[0]:
                            if array.ndim == 4:
                                # Shape: (1, 1, lat, lon)
                                level_data = array[0, 0, :, :]
                            elif array.ndim == 3:
                                # Shape: (level, lat, lon)
                                level_data = array[i, :, :]
                            else:
                                level_data = array

                            name = f"{p}_{level}"
                            fields[name] = level_data
                            print(f"      ✅ {name}: shape={level_data.shape}")
                else:
                    # No level dimension, take first time/step
                    if array.ndim == 4:
                        data = array[0, 0, :, :]
                    elif array.ndim == 3:
                        data = array[0, :, :]
                    else:
                        data = array

                    fields[p] = data
                    print(f"      ✅ {p}: shape={data.shape}")
            else:
                # Already 2D
                if levelist:
                    # This shouldn't happen, but handle it
                    for level in levelist:
                        name = f"{p}_{level}"
                        fields[name] = array
                        print(f"      ✅ {name}: shape={array.shape}")
                else:
                    fields[p] = array
                    print(f"      ✅ {p}: shape={array.shape}")
        else:
            print(f"    ❌ Failed to extract {p}")

    return fields


def create_input_state_from_parquet(parquet_path, member, use_obstore=True):
    """
    Create input state for a specific ensemble member using parquet file.
    Phase 1: Collect raw variables only (no interpolation/rolling).

    Args:
        parquet_path: Path to parquet file
        member: Ensemble member number
        use_obstore: Use obstore for faster S3 access

    Returns:
        Dictionary with 'date' and 'fields'
    """
    print(f"\n{'='*60}")
    print(f"Creating input state for ensemble member {member}")
    print(f"{'='*60}")
    start_time = time.time()

    fields = {}

    # Check if obstore is available
    try:
        import obstore
        if use_obstore:
            print("  ✅ obstore available - using for S3 fetching")
    except ImportError:
        print("  ⚠️ obstore not available - using fsspec for S3 fetching")
        use_obstore = False

    # Add single level fields
    print("\n  Getting surface fields...")
    fields.update(get_data_from_parquet_obstore(parquet_path, param=PARAM_SFC, use_obstore=use_obstore))

    print("\n  Getting constant surface fields...")
    fields.update(get_data_from_parquet_obstore(parquet_path, param=PARAM_SFC_FC, use_obstore=use_obstore))

    # Add soil fields
    print("\n  Getting soil fields...")
    soil = get_data_from_parquet_obstore(parquet_path, param=PARAM_SOIL, levelist=SOIL_LEVELS, use_obstore=use_obstore)

    # Rename soil parameters
    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2'}
    for k, v in mapping.items():
        if k in soil:
            fields[v] = soil[k]

    # Add pressure level fields
    print("\n  Getting pressure level fields...")
    fields.update(get_data_from_parquet_obstore(parquet_path, param=PARAM_PL, levelist=LEVELS, use_obstore=use_obstore))

    # Convert geopotential height to geopotential
    print("\n  Converting geopotential height to geopotential...")
    for level in LEVELS:
        if f"gh_{level}" in fields:
            gh = fields.pop(f"gh_{level}")
            fields[f"z_{level}"] = gh * 9.80665
            print(f"    ✅ z_{level}")

    # Extract date from parquet path
    # Path format: ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet
    path_parts = Path(parquet_path).parts
    for part in path_parts:
        if 'ecmwf_' in part and '_efficient' in part:
            date_parts = part.replace('ecmwf_', '').replace('_efficient', '').split('_')
            if len(date_parts) >= 2:
                date_str = date_parts[0]
                hour_str = date_parts[1]
                date = datetime.datetime.strptime(f"{date_str}_{hour_str}", "%Y%m%d_%H")
                break
    else:
        date = datetime.datetime.now()

    input_state = dict(date=date, fields=fields)

    elapsed_time = time.time() - start_time
    print(f"\n  ✅ Completed in {elapsed_time:.2f} seconds")

    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Total fields: {len(fields)}")
    if fields:
        sample_shape = list(fields.values())[0].shape
        print(f"  Sample field shape: {sample_shape}")

        # Calculate memory usage
        total_elements = sum(field.size for field in fields.values())
        memory_mb = (total_elements * 4) / (1024 * 1024)
        print(f"  Approximate memory: {memory_mb:.2f} MB")

    return input_state


def verify_input_state(input_state, member):
    """Verify the input state has all required fields."""
    fields = input_state['fields']

    # Expected fields
    expected_surface = PARAM_SFC + PARAM_SFC_FC + ['stl1', 'stl2']
    expected_pressure = []
    for param in PARAM_PL:
        if param == 'gh':
            param = 'z'  # Converted to geopotential
        for level in LEVELS:
            expected_pressure.append(f"{param}_{level}")

    expected_total = expected_surface + expected_pressure

    # Check if all fields are present
    missing = [f for f in expected_total if f not in fields]
    extra = [f for f in fields if f not in expected_total]

    print(f"\n  Verification for member {member}:")
    print(f"    Expected fields: {len(expected_total)}")
    print(f"    Actual fields: {len(fields)}")

    if missing:
        print(f"    ⚠️  Missing fields: {missing}")
    if extra:
        print(f"    ⚠️  Extra fields: {extra}")

    if not missing and not extra:
        print(f"    ✅ All fields present and correct!")
        return True
    return False


def main():
    """
    Main function to process ECMWF parquet files to PKL.
    Phase 1: Collect variables and save as pkl using obstore method.
    """

    # Define the parquet file to process
    parquet_file = "ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet"

    print("="*80)
    print("🌳 ECMWF PARQUET TO PKL PROCESSOR (PHASE 1)")
    print("="*80)
    print("Phase 1: Collect raw variables using obstore method")
    print("Phase 2 (future): Apply interpolation and rolling")
    print("="*80)
    print(f"\nProcessing parquet file: {parquet_file}")

    # Create output directory if saving states
    if SAVE_STATES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory: {OUTPUT_DIR}/")

    # Check that file exists
    if not Path(parquet_file).exists():
        print(f"\n❌ File not found: {parquet_file}")
        return

    try:
        # Create input state from parquet file
        input_state = create_input_state_from_parquet(parquet_file, member=1, use_obstore=True)

        # Verify the state
        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}")
        is_valid = verify_input_state(input_state, member=1)

        if SAVE_STATES:
            # Save the state as PKL
            output_file = f"{OUTPUT_DIR}/input_state_member_001_phase1.pkl"
            print(f"\n💾 Saving to: {output_file}")

            with open(output_file, 'wb') as f:
                pickle.dump(input_state, f)

            # Show file size
            file_size = Path(output_file).stat().st_size / (1024 * 1024)
            print(f"✅ Saved successfully!")
            print(f"📊 File size: {file_size:.2f} MB")

        # Show field samples
        if 'fields' in input_state and input_state['fields']:
            print(f"\n{'='*60}")
            print("FIELD SAMPLES")
            print(f"{'='*60}")
            field_names = sorted(input_state['fields'].keys())

            surface_fields = [f for f in field_names if '_' not in f]
            pressure_fields = [f for f in field_names if '_' in f]

            print(f"Surface fields ({len(surface_fields)}): {surface_fields[:10]}")
            print(f"Pressure level fields ({len(pressure_fields)}) sample: {pressure_fields[:5]}")
            print(f"Total fields: {len(field_names)}")

            # Show shapes
            print(f"\nField shapes (first 3):")
            for i, (fname, fdata) in enumerate(list(input_state['fields'].items())[:3]):
                print(f"  {fname}: {fdata.shape}, dtype={fdata.dtype}")

    except Exception as e:
        print(f"\n❌ Error processing parquet file: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE!")
    print("="*80)
    print("\n💡 Next steps (Phase 2):")
    print("  - Apply coordinate rolling (longitude shift)")
    print("  - Apply interpolation (ekr.interpolate)")
    print("  - Create final PKL for AI-FS input")


if __name__ == "__main__":
    main()