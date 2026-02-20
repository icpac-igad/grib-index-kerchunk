#!/usr/bin/env python3
"""
ECMWF Data Streaming for cGAN Inference - Coiled Dask Simple Version
=====================================================================

Simplified version that returns data directly from workers without Icechunk.
Reads parquet files from GCS so Coiled workers can access them.

Usage:
    # Test mode with GCS parquets
    python stream_cgan_variables_coiled_simple.py --test \
        --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z

    # Full run
    python stream_cgan_variables_coiled_simple.py \
        --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z \
        --n-workers 20

Author: ICPAC GIK Team
Date: 2026-02-05
"""

import os
import sys
import time
import json
import logging
import warnings
import tempfile
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import re

import numpy as np
import pandas as pd
import xarray as xr
import fsspec

warnings.filterwarnings('ignore')
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecmwf_cgan_coiled_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Can be local path or GCS path (gs://bucket/prefix/date_runz)
DEFAULT_GCS_PARQUET_PATH = os.environ.get('GCS_PARQUET_PATH', 'gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z')
DEFAULT_PARQUET_DIR = Path("ecmwf_three_stage_20260203_00z")
TARGET_STEPS = [36, 39, 42, 45, 48, 51, 54, 57, 60]

# cGAN variables to extract (ECMWF variable name -> output name)
# 10 surface variables + 2 pressure level variables = 12 total
CGAN_SURFACE_VARS = {
    'tp': 'tp',           # Total Precipitation
    '2t': 't2m',          # 2-meter Temperature (ECMWF uses 2t)
    'sp': 'sp',           # Surface Pressure
    'ssr': 'ssr',         # Surface Solar Radiation
    'ssrd': 'ssrd',       # Surface Solar Radiation Downwards
    'sf': 'sf',           # Snowfall (for cp estimation)
    'ro': 'ro',           # Runoff
    'tcw': 'tcw',         # Total Cloud Water
    'tcwv': 'tcwv',       # Total Column Water Vapour
    'tcc': 'tcc',         # Total Cloud Cover (for mcc)
}

CGAN_PRESSURE_VARS = {
    'u': 'u700',          # U-wind at 700 hPa
    'v': 'v700',          # V-wind at 700 hPa
}

TARGET_PRESSURE_LEVEL = 700
ECMWF_GRID_SHAPE = (721, 1440)
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

DEFAULT_N_WORKERS = 5
DEFAULT_MEMBERS_PER_BATCH = 1
OUTPUT_DIR = Path("cgan_output")


def get_icpac_indices():
    lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
    lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    lats = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]
    return lat_indices, lon_indices, lats, lons


LAT_INDICES, LON_INDICES, ICPAC_LATS, ICPAC_LONS = get_icpac_indices()


def get_all_member_parquets_local(parquet_dir: Path) -> List[Tuple[int, str, str]]:
    """Get member parquets from local directory."""
    parquet_files = sorted(parquet_dir.glob("stage3_*_final.parquet"))
    members = []
    for idx, pf in enumerate(parquet_files):
        raw_member = pf.stem.replace('stage3_', '').replace('_final', '')
        member_key = raw_member.replace('_', '')
        members.append((idx, member_key, str(pf)))
    return members


def get_all_member_parquets_gcs(gcs_path: str) -> List[Tuple[int, str, str]]:
    """Get member parquets from GCS path."""
    import gcsfs

    fs = gcsfs.GCSFileSystem()

    # Remove gs:// prefix for gcsfs
    if gcs_path.startswith('gs://'):
        gcs_path_clean = gcs_path[5:]
    else:
        gcs_path_clean = gcs_path

    # List parquet files
    parquet_files = sorted(fs.glob(f"{gcs_path_clean}/*_final.parquet"))

    members = []
    for idx, pf in enumerate(parquet_files):
        filename = pf.split('/')[-1]
        raw_member = filename.replace('stage3_', '').replace('_final.parquet', '')
        member_key = raw_member.replace('_', '')
        # Return full GCS path
        members.append((idx, member_key, f"gs://{pf}"))

    return members


def get_all_member_parquets(parquet_path: str) -> List[Tuple[int, str, str]]:
    """Get member parquets from local or GCS path."""
    if parquet_path.startswith('gs://'):
        return get_all_member_parquets_gcs(parquet_path)
    else:
        return get_all_member_parquets_local(Path(parquet_path))


def create_member_batches(members: List, members_per_batch: int) -> List[List]:
    n_batches = math.ceil(len(members) / members_per_batch)
    batches = []
    for i in range(n_batches):
        start = i * members_per_batch
        end = min(start + members_per_batch, len(members))
        batches.append(members[start:end])
    return batches


# ==============================================================================
# WORKER FUNCTION - Returns data directly (no Icechunk)
# ==============================================================================

def process_member_batch_simple(args: Tuple) -> Dict[str, Any]:
    """
    Worker function that returns processed data directly.
    Reads parquet from GCS or local path.
    No Icechunk storage - data returned to client.
    """
    import os
    import sys
    import json
    import time
    import tempfile
    import warnings
    import numpy as np
    import pandas as pd
    import xarray as xr
    import fsspec

    warnings.filterwarnings('ignore')
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    (batch_id, member_list, parquet_dir_str, surface_vars, pressure_vars,
     target_steps, target_pressure_level, icpac_lat_slice, icpac_lon_slice,
     ecmwf_grid_shape) = args

    try:
        start_time = time.time()

        # Try gribberish
        try:
            import gribberish
            gribberish_available = True
        except ImportError:
            gribberish_available = False

        # S3 filesystem for ECMWF data
        fs_s3 = fsspec.filesystem('s3', anon=True)

        # GCS filesystem for parquet files (if using GCS)
        fs_gcs = None
        if parquet_dir_str.startswith('gs://'):
            import gcsfs
            fs_gcs = gcsfs.GCSFileSystem()

        batch_data = {}
        all_vars = {**surface_vars, **pressure_vars}

        for member_idx, member_name, parquet_path_str in member_list:
            # Read parquet from GCS or local
            if parquet_path_str.startswith('gs://'):
                # GCS path - use gcsfs
                df = pd.read_parquet(parquet_path_str, filesystem=fs_gcs)
            else:
                # Local path
                df = pd.read_parquet(parquet_path_str)
            zstore = {}
            for _, row in df.iterrows():
                key = row['key']
                value = row['value']
                if isinstance(value, bytes):
                    try:
                        decoded = value.decode('utf-8')
                        if decoded.startswith('[') or decoded.startswith('{'):
                            value = json.loads(decoded)
                        else:
                            value = decoded
                    except:
                        pass
                elif isinstance(value, str):
                    if value.startswith('[') or value.startswith('{'):
                        try:
                            value = json.loads(value)
                        except:
                            pass
                zstore[key] = value

            # Process variables
            for ecmwf_var, output_var in all_vars.items():
                is_pressure = ecmwf_var in pressure_vars

                if output_var not in batch_data:
                    batch_data[output_var] = {}

                var_data = []

                for step in target_steps:
                    if is_pressure:
                        patterns = [
                            f'step_{step:03d}/{ecmwf_var}/pl/{member_name}/0.0.0',
                            f'step_{step:03d}/{ecmwf_var}/pl/0.0.0',
                        ]
                    else:
                        patterns = [
                            f'step_{step:03d}/{ecmwf_var}/sfc/{member_name}/0.0.0',
                            f'step_{step:03d}/{ecmwf_var}/sfc/0.0.0',
                            f'step_{step:03d}/{ecmwf_var}/surface/{member_name}/0.0.0',
                        ]

                    ref = None
                    for pattern in patterns:
                        if pattern in zstore:
                            ref = zstore[pattern]
                            if isinstance(ref, list) and len(ref) >= 3:
                                break
                            ref = None

                    if ref is None:
                        var_data.append(np.nan * np.ones(
                            (icpac_lat_slice[1] - icpac_lat_slice[0],
                             icpac_lon_slice[1] - icpac_lon_slice[0]),
                            dtype=np.float32))
                        continue

                    try:
                        url, offset, length = ref[0], ref[1], ref[2]
                        if not url.endswith('.grib2'):
                            url = url + '.grib2'

                        with fs_s3.open(url, 'rb') as f:
                            f.seek(offset)
                            grib_bytes = f.read(length)

                        if gribberish_available:
                            flat_array = gribberish.parse_grib_array(grib_bytes, 0)
                            array_2d = flat_array.reshape(ecmwf_grid_shape)
                        else:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
                                tmp.write(grib_bytes)
                                tmp_path = tmp.name
                            try:
                                ds = xr.open_dataset(tmp_path, engine='cfgrib')
                                vn = list(ds.data_vars)[0]
                                array_2d = ds[vn].values.copy()
                                ds.close()
                            finally:
                                os.unlink(tmp_path)

                        subset = array_2d[icpac_lat_slice[0]:icpac_lat_slice[1],
                                          icpac_lon_slice[0]:icpac_lon_slice[1]]
                        var_data.append(subset.astype(np.float32))

                    except Exception as e:
                        var_data.append(np.nan * np.ones(
                            (icpac_lat_slice[1] - icpac_lat_slice[0],
                             icpac_lon_slice[1] - icpac_lon_slice[0]),
                            dtype=np.float32))

                batch_data[output_var][member_idx] = np.stack(var_data, axis=0)

        processing_time = time.time() - start_time

        return {
            'batch_id': batch_id,
            'status': 'success',
            'n_members': len(member_list),
            'member_indices': [m[0] for m in member_list],
            'n_variables': len(all_vars),
            'n_steps': len(target_steps),
            'processing_time': processing_time,
            'data': batch_data  # Return data directly
        }

    except Exception as e:
        import traceback
        return {
            'batch_id': batch_id,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# ==============================================================================
# MAIN
# ==============================================================================

def stream_cgan_simple(
    parquet_path: str,
    n_workers: int = DEFAULT_N_WORKERS,
    members_per_batch: int = DEFAULT_MEMBERS_PER_BATCH,
    output_dir: Path = OUTPUT_DIR,
    max_members: Optional[int] = None,
    coiled_workspace: str = "gcp-sewaa-nka"
) -> bool:
    """
    Simple Coiled streaming without Icechunk.

    Args:
        parquet_path: Local path or GCS path (gs://bucket/prefix/date_runz)
    """
    import coiled
    from dask.distributed import Client, as_completed

    is_gcs = parquet_path.startswith('gs://')

    logger.info("=" * 70)
    logger.info("ECMWF cGAN Streaming - Coiled Simple (No Icechunk)")
    logger.info("=" * 70)
    logger.info(f"Parquet Path: {parquet_path}")
    logger.info(f"Source: {'GCS' if is_gcs else 'Local'}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Members per batch: {members_per_batch}")
    logger.info("=" * 70)

    start_time = time.time()

    # Validate path
    if not is_gcs:
        parquet_dir = Path(parquet_path)
        if not parquet_dir.exists():
            logger.error(f"Parquet directory {parquet_dir} not found!")
            return False

    # Extract date from path
    match = re.search(r'(\d{8})_(\d{2})z', str(parquet_path))
    if match:
        model_date = datetime.strptime(match.group(1), '%Y%m%d')
        run_hour = int(match.group(2))
    else:
        model_date = datetime.now()
        run_hour = 0

    logger.info(f"Model Date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")

    # Get members from GCS or local
    all_members = get_all_member_parquets(parquet_path)
    if max_members:
        all_members = all_members[:max_members]

    if not all_members:
        logger.error(f"No parquet files found at {parquet_path}")
        return False

    logger.info(f"Total members: {len(all_members)}")

    # Create batches
    member_batches = create_member_batches(all_members, members_per_batch)
    logger.info(f"Created {len(member_batches)} batches")

    # ICPAC slice indices
    icpac_lat_slice = (LAT_INDICES[0], LAT_INDICES[-1] + 1)
    icpac_lon_slice = (LON_INDICES[0], LON_INDICES[-1] + 1)

    # Prepare task args
    task_args = []
    for batch_id, batch in enumerate(member_batches):
        # batch already has (idx, member_key, path_str)
        batch_serializable = [(m[0], m[1], m[2]) for m in batch]
        args = (
            batch_id,
            batch_serializable,
            parquet_path,  # Pass the base path for logging
            CGAN_SURFACE_VARS,
            CGAN_PRESSURE_VARS,
            TARGET_STEPS,
            TARGET_PRESSURE_LEVEL,
            icpac_lat_slice,
            icpac_lon_slice,
            ECMWF_GRID_SHAPE
        )
        task_args.append(args)

    # Create Coiled cluster
    logger.info(f"\nStarting Coiled cluster with {n_workers} workers...")

    cluster = coiled.Cluster(
        name=f"ecmwf-cgan-simple-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types=["n2-standard-4"],
        package_sync=True,
        region="us-east1",
        workspace=coiled_workspace,
        idle_timeout="10 minutes",
    )
    client = Client(cluster)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    # Submit tasks
    logger.info(f"\nSubmitting {len(task_args)} batch tasks...")
    futures = client.map(process_member_batch_simple, task_args)

    # Collect results
    all_batch_data = []
    successful_batches = 0

    for i, future in enumerate(as_completed(futures), 1):
        try:
            result = future.result()

            if result['status'] == 'success':
                successful_batches += 1
                all_batch_data.append(result)
                logger.info(f"  Batch {result['batch_id']}: {result['n_members']} members "
                          f"in {result['processing_time']:.1f}s")
            else:
                logger.error(f"  Batch {result.get('batch_id')}: FAILED - {result.get('error')}")

        except Exception as e:
            logger.error(f"  Future error: {e}")

    # Close cluster
    logger.info("Closing Coiled cluster...")
    client.close()
    cluster.close()

    parallel_time = time.time() - start_time
    logger.info(f"\nParallel phase complete: {parallel_time:.1f}s")
    logger.info(f"  Successful: {successful_batches}/{len(task_args)}")

    if successful_batches == 0:
        logger.error("No batches succeeded!")
        return False

    # Aggregate data
    logger.info("\nAggregating results...")

    all_vars = list(CGAN_SURFACE_VARS.values()) + list(CGAN_PRESSURE_VARS.values())
    aggregated = {var: [] for var in all_vars}
    member_indices = []

    for batch_result in sorted(all_batch_data, key=lambda x: x['batch_id']):
        batch_data = batch_result['data']
        member_indices.extend(batch_result['member_indices'])

        for var in all_vars:
            if var in batch_data:
                for member_idx in batch_result['member_indices']:
                    if member_idx in batch_data[var]:
                        aggregated[var].append(batch_data[var][member_idx])

    # Compute ensemble statistics
    logger.info("Computing ensemble statistics...")

    data_dict = {}
    for var_name in all_vars:
        if aggregated[var_name]:
            data = np.stack(aggregated[var_name], axis=0)  # (n_members, n_steps, lat, lon)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ensemble_mean = np.nanmean(data, axis=0)
                ensemble_std = np.nanstd(data, axis=0)

            data_dict[var_name] = (ensemble_mean, ensemble_std)
            logger.info(f"  {var_name}: shape={data.shape}, "
                       f"mean=[{np.nanmin(ensemble_mean):.4g}, {np.nanmax(ensemble_mean):.4g}]")

    # Create NetCDF
    logger.info("\nCreating output NetCDF...")

    output_dir.mkdir(exist_ok=True)
    date_str = model_date.strftime('%Y%m%d')
    output_file = output_dir / f"IFS_{date_str}_{run_hour:02d}Z_cgan_simple.nc"

    base_time = model_date + timedelta(hours=run_hour)
    valid_times = [base_time + timedelta(hours=h) for h in TARGET_STEPS]

    coords = {
        'time': [base_time],
        'valid_time': valid_times,
        'latitude': ICPAC_LATS,
        'longitude': ICPAC_LONS,
    }

    data_vars = {}
    for var_name, (mean_data, std_data) in data_dict.items():
        data_vars[f'{var_name}_ensemble_mean'] = xr.DataArray(
            mean_data[np.newaxis, :, :, :],
            dims=['time', 'valid_time', 'latitude', 'longitude'],
            attrs={'long_name': f'{var_name} ensemble mean'}
        )
        data_vars[f'{var_name}_ensemble_standard_deviation'] = xr.DataArray(
            std_data[np.newaxis, :, :, :],
            dims=['time', 'valid_time', 'latitude', 'longitude'],
            attrs={'long_name': f'{var_name} ensemble standard deviation'}
        )

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs['title'] = 'ECMWF Ensemble Data for cGAN (Coiled Simple)'
    ds.attrs['n_ensemble_members'] = len(member_indices)
    ds.attrs['history'] = f'Created {datetime.now().isoformat()}'

    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                for var in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved: {output_file} ({file_size_mb:.2f} MB)")

    # Summary
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Members: {len(member_indices)}")
    logger.info(f"Variables: {len(data_dict)}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Output: {output_file}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='ECMWF cGAN Streaming - Coiled Simple',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with GCS parquets (recommended)
    python stream_cgan_variables_coiled_simple.py --test \\
        --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z

    # Full production run
    python stream_cgan_variables_coiled_simple.py \\
        --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z \\
        --n-workers 20
        """
    )
    parser.add_argument('--gcs-parquet-path', type=str, default=DEFAULT_GCS_PARQUET_PATH,
                        help='GCS path to parquets (gs://bucket/prefix/date_runz)')
    parser.add_argument('--parquet-dir', type=str, default=None,
                        help='Local parquet directory (alternative to GCS)')
    parser.add_argument('--n-workers', type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument('--members-per-batch', type=int, default=DEFAULT_MEMBERS_PER_BATCH)
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--max-members', type=int, default=None)
    parser.add_argument('--workspace', type=str, default="gcp-sewaa-nka")
    parser.add_argument('--test', action='store_true', help='Test mode: 3 members, 3 workers')

    args = parser.parse_args()

    # Determine parquet path (GCS takes precedence if both specified)
    if args.parquet_dir:
        parquet_path = args.parquet_dir
    else:
        parquet_path = args.gcs_parquet_path

    if args.test:
        args.max_members = 3
        args.n_workers = 3
        args.members_per_batch = 1
        logger.info("TEST MODE: 3 members, 3 workers")

    success = stream_cgan_simple(
        parquet_path=parquet_path,
        n_workers=args.n_workers,
        members_per_batch=args.members_per_batch,
        output_dir=Path(args.output_dir),
        max_members=args.max_members,
        coiled_workspace=args.workspace
    )

    sys.exit(0 if success else 1)
