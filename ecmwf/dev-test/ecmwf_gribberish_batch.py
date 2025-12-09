#!/usr/bin/env python3
"""
ECMWF Ensemble Batch Processing using gribberish (Rust) with cfgrib fallback.

Processes multiple ensemble members in parallel using ProcessPoolExecutor.
Each member is processed in a separate process to isolate gribberish panics.

Usage:
    python ecmwf_gribberish_batch.py --input-dir ecmwf_20251127_00z_efficient/members \
        --output-dir output_nc/ --workers 5 --variable tp
"""

import pandas as pd
import numpy as np
import json
import time
import tempfile
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import fsspec
import xarray as xr
import argparse

# Config
GRID_SHAPE = (721, 1440)  # ECMWF 0.25° grid

# East Africa bounds
LAT_MIN, LAT_MAX = -12, 23
LON_MIN, LON_MAX = 21, 53


def read_parquet_refs(parquet_path):
    """Read parquet and extract zstore references."""
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
    return zstore


def find_variable_chunks(zstore, variable):
    """Find variable chunks sorted by step."""
    chunks = []
    for key in zstore.keys():
        if f'/{variable}/' in key and key.endswith('/0.0.0'):
            match = re.match(r'step_(\d+)/', key)
            if match:
                step = int(match.group(1))
                chunks.append((step, key))
    return sorted(chunks, key=lambda x: x[0])


def extract_model_run_time(zstore):
    """Extract model run datetime from S3 URL."""
    for key in zstore.keys():
        if key.endswith('/0.0.0'):
            ref = zstore[key]
            if isinstance(ref, list) and len(ref) >= 1:
                url = ref[0]
                match = re.search(r'/(\d{8})/(\d{2})z/', url)
                if match:
                    date_str = match.group(1)
                    hour = int(match.group(2))
                    return datetime(
                        year=int(date_str[:4]),
                        month=int(date_str[4:6]),
                        day=int(date_str[6:8]),
                        hour=hour
                    )
    return None


def compute_valid_times(model_run_time, steps):
    """Compute valid times from model run time and forecast steps."""
    if model_run_time is None:
        return None
    return [model_run_time + timedelta(hours=step) for step in steps]


def decode_with_gribberish(grib_bytes):
    """Try to decode with gribberish (fast path)."""
    try:
        import gribberish
        flat_array = gribberish.parse_grib_array(grib_bytes, 0)
        return flat_array.reshape(GRID_SHAPE), True
    except Exception:
        return None, False


def decode_with_cfgrib(grib_bytes):
    """Fallback: decode using cfgrib (slow but reliable)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name

    try:
        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        array_2d = ds[var_name].values
        ds.close()
    finally:
        os.unlink(tmp_path)

    return array_2d


def fetch_grib_bytes(zstore, chunk_key, fs):
    """Fetch GRIB bytes from S3."""
    ref = zstore[chunk_key]
    url, offset, length = ref[0], ref[1], ref[2]

    if not url.endswith('.grib2'):
        url = url + '.grib2'

    s3_path = url if url.startswith('s3://') else f's3://{url}'

    with fs.open(s3_path, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes


def process_single_member(parquet_file, output_nc, variable='tp', region_bounds=None):
    """
    Process a single ensemble member parquet file to NetCDF.

    This function runs in a separate process to isolate gribberish panics.

    Args:
        parquet_file: Path to parquet file
        output_nc: Output NetCDF path
        variable: Variable to extract
        region_bounds: Optional dict with lat_min, lat_max, lon_min, lon_max

    Returns:
        dict with processing results
    """
    start_time = time.time()
    member_name = Path(parquet_file).stem

    result = {
        'member': member_name,
        'parquet_file': str(parquet_file),
        'output_file': str(output_nc),
        'success': False,
        'error': None,
        'gribberish_count': 0,
        'cfgrib_count': 0,
        'total_chunks': 0,
        'processing_time': 0
    }

    try:
        # Read parquet
        zstore = read_parquet_refs(parquet_file)

        # Extract model run time
        model_run_time = extract_model_run_time(zstore)

        # Find variable chunks
        chunks = find_variable_chunks(zstore, variable)
        result['total_chunks'] = len(chunks)

        if not chunks:
            result['error'] = f"No {variable} chunks found"
            return result

        # Create S3 filesystem
        fs = fsspec.filesystem('s3', anon=True)

        # Process all timesteps
        timestep_data = []
        steps = []
        gribberish_count = 0
        cfgrib_count = 0

        for step, chunk_key in chunks:
            grib_bytes = fetch_grib_bytes(zstore, chunk_key, fs)

            # Try gribberish first
            array_2d, success = decode_with_gribberish(grib_bytes)

            if success:
                gribberish_count += 1
            else:
                # Fallback to cfgrib
                array_2d = decode_with_cfgrib(grib_bytes)
                cfgrib_count += 1

            timestep_data.append(array_2d)
            steps.append(step)

        result['gribberish_count'] = gribberish_count
        result['cfgrib_count'] = cfgrib_count

        # Stack into 3D array
        data_3d = np.stack(timestep_data, axis=0)

        # Compute valid times
        valid_times = compute_valid_times(model_run_time, steps)

        # Create coordinates
        lats = np.linspace(90, -90, 721)
        lons = np.linspace(0, 359.75, 1440)

        # Build coordinates
        coords = {'latitude': lats, 'longitude': lons}

        if valid_times:
            time_values = np.array(valid_times, dtype='datetime64[ns]')
            coords['time'] = time_values
            time_dim = 'time'
        else:
            coords['step'] = steps
            time_dim = 'step'

        # Create Dataset
        ds = xr.Dataset(
            {variable: ([time_dim, 'latitude', 'longitude'], data_3d.astype(np.float32))},
            coords=coords
        )

        # Add step as auxiliary coordinate
        if valid_times:
            ds['step'] = (time_dim, steps)
            ds['step'].attrs = {'long_name': 'forecast step', 'units': 'hours'}

        # Variable attributes
        var_attrs = {
            'long_name': 'Total precipitation' if variable == 'tp' else variable,
            'units': 'm' if variable == 'tp' else 'unknown',
        }
        if variable == 'tp':
            var_attrs['standard_name'] = 'precipitation_amount'
        ds[variable].attrs = var_attrs

        # Global attributes
        ds.attrs['Conventions'] = 'CF-1.8'
        ds.attrs['institution'] = 'ECMWF'
        ds.attrs['source'] = 'IFS Ensemble Forecast'
        ds.attrs['ensemble_member'] = member_name
        ds.attrs['gribberish_chunks'] = gribberish_count
        ds.attrs['cfgrib_chunks'] = cfgrib_count
        ds.attrs['parquet_source'] = str(parquet_file)

        if model_run_time:
            ds.attrs['model_run_time'] = model_run_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            ds.attrs['forecast_reference_time'] = model_run_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Apply regional subset if specified
        if region_bounds:
            ds = ds.sel(
                latitude=slice(region_bounds['lat_max'], region_bounds['lat_min']),
                longitude=slice(region_bounds['lon_min'], region_bounds['lon_max'])
            )

        # Save to NetCDF
        output_nc = Path(output_nc)
        output_nc.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_nc)

        result['success'] = True
        result['processing_time'] = time.time() - start_time

    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time

    return result


def find_parquet_files(input_dir):
    """Find all ensemble member parquet files in directory."""
    input_dir = Path(input_dir)
    parquet_files = []

    # Pattern 1: members/ens_NN/ens_NN.parquet
    members_dir = input_dir / 'members'
    if members_dir.exists():
        for member_dir in sorted(members_dir.iterdir()):
            if member_dir.is_dir():
                for pq in member_dir.glob('*.parquet'):
                    parquet_files.append(pq)

    # Pattern 2: stage3_ens_NN_final.parquet directly in folder
    for pq in sorted(input_dir.glob('stage3_ens_*_final.parquet')):
        parquet_files.append(pq)

    # Pattern 3: ens_NN.parquet directly in folder
    for pq in sorted(input_dir.glob('ens_*.parquet')):
        parquet_files.append(pq)

    return parquet_files


def batch_process_ensemble(input_dir, output_dir, variable='tp', max_workers=5,
                           region='eastafrica', test_members=None):
    """
    Process multiple ensemble members in parallel.

    Args:
        input_dir: Directory containing parquet files
        output_dir: Output directory for NetCDF files
        variable: Variable to extract
        max_workers: Number of parallel workers
        region: Region to subset ('eastafrica', 'global', or custom bounds dict)
        test_members: Optional list of member indices to process (for testing)
    """
    print("=" * 70)
    print("ECMWF ENSEMBLE BATCH PROCESSING (gribberish + cfgrib)")
    print("=" * 70)

    start_total = time.time()

    # Find parquet files
    print(f"\n1. Scanning input directory: {input_dir}")
    parquet_files = find_parquet_files(input_dir)
    print(f"   Found {len(parquet_files)} parquet files")

    if not parquet_files:
        print("   ERROR: No parquet files found!")
        return

    # Apply test filter if specified
    if test_members:
        parquet_files = parquet_files[:test_members]
        print(f"   Testing with first {test_members} members")

    # Set region bounds
    if region == 'eastafrica':
        region_bounds = {
            'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
            'lon_min': LON_MIN, 'lon_max': LON_MAX
        }
        print(f"   Region: East Africa ({LAT_MIN}°-{LAT_MAX}°N, {LON_MIN}°-{LON_MAX}°E)")
    elif region == 'global':
        region_bounds = None
        print("   Region: Global")
    else:
        region_bounds = region
        print(f"   Region: Custom bounds")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Output directory: {output_dir}")

    # Prepare tasks
    tasks = []
    for pq_file in parquet_files:
        member_name = pq_file.stem.replace('_final', '').replace('stage3_', '')
        output_nc = output_dir / f"{member_name}_{variable}.nc"
        tasks.append((pq_file, output_nc, variable, region_bounds))

    print(f"\n2. Processing {len(tasks)} members with {max_workers} parallel workers...")
    print("-" * 70)

    # Process in parallel using ProcessPoolExecutor
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_member, pq, nc, var, bounds): (pq, nc)
            for pq, nc, var, bounds in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            pq_file, output_nc = future_to_task[future]
            completed += 1

            try:
                result = future.result(timeout=600)  # 10 min timeout per member
                results.append(result)

                status = "OK" if result['success'] else f"FAIL: {result['error']}"
                grib_pct = 100 * result['gribberish_count'] / max(1, result['total_chunks'])

                print(f"   [{completed:2d}/{len(tasks)}] {result['member']}: {status} "
                      f"({result['processing_time']:.1f}s, gribberish: {grib_pct:.0f}%)")

            except Exception as e:
                results.append({
                    'member': Path(pq_file).stem,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0,
                    'gribberish_count': 0,
                    'cfgrib_count': 0,
                    'total_chunks': 0
                })
                print(f"   [{completed:2d}/{len(tasks)}] {Path(pq_file).stem}: EXCEPTION: {e}")

    # Summary
    total_time = time.time() - start_total
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_gribberish = sum(r['gribberish_count'] for r in results)
    total_cfgrib = sum(r['cfgrib_count'] for r in results)
    total_chunks = sum(r['total_chunks'] for r in results)

    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Members processed: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Gribberish decoded: {total_gribberish} ({100*total_gribberish/max(1,total_chunks):.1f}%)")
    print(f"   Cfgrib fallback: {total_cfgrib} ({100*total_cfgrib/max(1,total_chunks):.1f}%)")
    print(f"   Output directory: {output_dir}")

    if failed > 0:
        print(f"\n   Failed members:")
        for r in results:
            if not r['success']:
                print(f"     - {r['member']}: {r['error']}")

    # Save processing summary
    summary_file = output_dir / 'processing_summary.json'
    summary = {
        'total_time_seconds': total_time,
        'members_processed': len(results),
        'successful': successful,
        'failed': failed,
        'total_chunks': total_chunks,
        'gribberish_decoded': total_gribberish,
        'cfgrib_fallback': total_cfgrib,
        'results': results
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   Summary saved: {summary_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch process ECMWF ensemble parquet files to NetCDF using gribberish'
    )
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='Input directory containing parquet files')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory for NetCDF files')
    parser.add_argument('--variable', '-v', type=str, default='tp',
                        help='Variable to extract (default: tp)')
    parser.add_argument('--workers', '-w', type=int, default=5,
                        help='Number of parallel workers (default: 5)')
    parser.add_argument('--region', '-r', type=str, default='eastafrica',
                        choices=['eastafrica', 'global'],
                        help='Region to subset (default: eastafrica)')
    parser.add_argument('--test', '-t', type=int, default=None,
                        help='Test with first N members only')

    args = parser.parse_args()

    batch_process_ensemble(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        variable=args.variable,
        max_workers=args.workers,
        region=args.region,
        test_members=args.test
    )
