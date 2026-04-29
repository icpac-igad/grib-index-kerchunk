#!/usr/bin/env python3
"""
ECMWF Data Streaming for cGAN Inference - Coiled Dask Version
==============================================================

This script streams the 13 variables required for cGAN inference from
ECMWF ensemble data using Coiled Dask workers for parallel processing
and Icechunk for intermediate storage.

Architecture:
1. Coiled cluster with 20 workers processes members in parallel
2. Each worker writes to its own Icechunk branch (no conflicts)
3. Local client aggregates results and computes ensemble statistics
4. Final NetCDF with ensemble mean/std for cGAN input

Based on: CMORPH multi-year processor lessons for avoiding conflicts

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
import gc
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecmwf_cgan_coiled_streaming.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default parquet directory (can be overridden via CLI)
DEFAULT_PARQUET_DIR = Path("ecmwf_three_stage_20260203_00z")

# Target timesteps (36-60 hours in 3-hour intervals)
TARGET_STEPS = [36, 39, 42, 45, 48, 51, 54, 57, 60]

# cGAN variables to extract
CGAN_SURFACE_VARS = {
    'tp': 'tp',           # Total Precipitation
    '2t': 't2m',          # 2-meter Temperature
    'sp': 'sp',           # Surface Pressure
    'ssr': 'ssr',         # Surface Solar Radiation
    'ssrd': 'ssrd',       # Surface Solar Radiation Downwards
    'sf': 'sf',           # Snowfall
    'ro': 'ro',           # Runoff
    'tcw': 'tcw',         # Total Cloud Water
    'tcwv': 'tcwv',       # Total Column Water Vapour
    'tcc': 'tcc',         # Total Cloud Cover
}

CGAN_PRESSURE_VARS = {
    'u': 'u700',          # U-wind at 700 hPa
    'v': 'v700',          # V-wind at 700 hPa
}

TARGET_PRESSURE_LEVEL = 700  # hPa

# ECMWF grid specification
ECMWF_GRID_SHAPE = (721, 1440)
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# ICPAC region subset
LAT_MIN, LAT_MAX = -14, 25
LON_MIN, LON_MAX = 19, 55

# Coiled configuration
DEFAULT_N_WORKERS = 20
DEFAULT_MEMBERS_PER_BATCH = 3
DEFAULT_COILED_REGION = "eu-west-1"  # Close to ECMWF S3

# Icechunk configuration
DEFAULT_GCS_BUCKET = "cpc_awc"
DEFAULT_GCS_PREFIX = "ecmwf_cgan_intermediate"

# Output configuration
OUTPUT_DIR = Path("cgan_output")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_icpac_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get ICPAC region subset indices and coordinates."""
    lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
    lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    lats = ECMWF_LATS[lat_indices[0]:lat_indices[-1]+1]
    lons = ECMWF_LONS[lon_indices[0]:lon_indices[-1]+1]

    return lat_indices, lon_indices, lats, lons


# Pre-compute ICPAC indices
LAT_INDICES, LON_INDICES, ICPAC_LATS, ICPAC_LONS = get_icpac_indices()


def get_all_member_parquets(parquet_dir: Path) -> List[Tuple[int, str, Path]]:
    """
    Get all member parquet files with their indices and names.

    Returns list of (member_idx, member_name, parquet_path).
    """
    parquet_files = sorted(parquet_dir.glob("stage3_*.parquet"))

    members = []
    for idx, pf in enumerate(parquet_files):
        raw_member = pf.stem.replace('stage3_', '').replace('_final', '')
        member_key = raw_member.replace('_', '')  # ens_01 -> ens01
        members.append((idx, member_key, pf))

    return members


def create_member_batches(members: List, members_per_batch: int) -> List[List]:
    """Split members into batches for parallel processing."""
    n_batches = math.ceil(len(members) / members_per_batch)
    batches = []
    for i in range(n_batches):
        start = i * members_per_batch
        end = min(start + members_per_batch, len(members))
        batches.append(members[start:end])
    return batches


# ==============================================================================
# WORKER FUNCTION (Runs on Coiled workers)
# ==============================================================================

def process_member_batch_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to process a batch of ensemble members.

    This function runs on Coiled workers and:
    1. Fetches GRIB data for assigned members
    2. Decodes and subsets to ICPAC region
    3. Writes to unique Icechunk branch (avoids conflicts)

    Args:
        args: Tuple containing all necessary parameters

    Returns:
        Dictionary with batch processing results
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

    # Unpack arguments
    (batch_id, member_list, parquet_dir_str, surface_vars, pressure_vars,
     target_steps, target_pressure_level, icpac_lat_slice, icpac_lon_slice,
     creds_content, gcs_bucket, gcs_prefix, ecmwf_grid_shape) = args

    branch_name = f"batch_{batch_id}"
    creds_file = None

    try:
        start_time = time.time()

        # Try to import gribberish for fast decoding
        try:
            import gribberish
            gribberish_available = True
        except ImportError:
            gribberish_available = False

        # Write credentials to temp file
        creds_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        creds_file.write(creds_content)
        creds_file.close()

        # Initialize S3 filesystem
        fs = fsspec.filesystem('s3', anon=True)

        parquet_dir = Path(parquet_dir_str)

        # Data storage for this batch
        batch_data = {}  # {var_name: {member_idx: array}}

        all_vars = {**surface_vars, **pressure_vars}

        # Process each member in this batch
        for member_idx, member_name, parquet_path_str in member_list:
            parquet_path = Path(parquet_path_str)

            # Read parquet references
            df = pd.read_parquet(parquet_path)
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

            # Process each variable
            for ecmwf_var, output_var in all_vars.items():
                is_pressure = ecmwf_var in pressure_vars

                if output_var not in batch_data:
                    batch_data[output_var] = {}

                var_data = []

                for step in target_steps:
                    # Build key patterns
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
                        var_data.append(np.nan * np.ones((icpac_lat_slice[1] - icpac_lat_slice[0],
                                                          icpac_lon_slice[1] - icpac_lon_slice[0]),
                                                         dtype=np.float32))
                        continue

                    # Fetch GRIB bytes
                    try:
                        url, offset, length = ref[0], ref[1], ref[2]
                        if not url.endswith('.grib2'):
                            url = url + '.grib2'

                        with fs.open(url, 'rb') as f:
                            f.seek(offset)
                            grib_bytes = f.read(length)

                        # Decode GRIB
                        if gribberish_available:
                            flat_array = gribberish.parse_grib_array(grib_bytes, 0)
                            array_2d = flat_array.reshape(ecmwf_grid_shape)
                        else:
                            # Fallback to cfgrib
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

                        # Subset to ICPAC
                        subset = array_2d[icpac_lat_slice[0]:icpac_lat_slice[1],
                                          icpac_lon_slice[0]:icpac_lon_slice[1]]
                        var_data.append(subset.astype(np.float32))

                    except Exception as e:
                        var_data.append(np.nan * np.ones((icpac_lat_slice[1] - icpac_lat_slice[0],
                                                          icpac_lon_slice[1] - icpac_lon_slice[0]),
                                                         dtype=np.float32))

                # Stack timesteps: (n_steps, lat, lon)
                batch_data[output_var][member_idx] = np.stack(var_data, axis=0)

        # Write to Icechunk with unique batch branch
        import icechunk

        config = icechunk.RepositoryConfig.default()
        container = icechunk.VirtualChunkContainer(
            "s3://ecmwf-forecasts/",
            store=icechunk.s3_store(region="eu-west-2", anonymous=True),
        )
        config.set_virtual_chunk_container(container)

        storage = icechunk.gcs_storage(
            bucket=gcs_bucket,
            prefix=gcs_prefix,
            service_account_file=creds_file.name
        )

        repo = icechunk.Repository.open(storage, config=config)

        # Create unique branch for this batch
        try:
            repo.create_branch(branch_name, repo.lookup_branch("main"))
        except Exception as e:
            if "already exists" in str(e).lower():
                try:
                    repo.delete_branch(branch_name)
                    repo.create_branch(branch_name, repo.lookup_branch("main"))
                except:
                    pass

        session = repo.writable_session(branch_name)

        # Create xarray Dataset for this batch
        member_indices = [m[0] for m in member_list]
        n_members = len(member_indices)
        n_steps = len(target_steps)
        lat_size = icpac_lat_slice[1] - icpac_lat_slice[0]
        lon_size = icpac_lon_slice[1] - icpac_lon_slice[0]

        data_vars = {}
        for var_name, member_data in batch_data.items():
            # Stack member data: (n_members, n_steps, lat, lon)
            arrays = [member_data.get(idx, np.nan * np.ones((n_steps, lat_size, lon_size)))
                      for idx in member_indices]
            stacked = np.stack(arrays, axis=0)

            data_vars[var_name] = xr.DataArray(
                stacked,
                dims=['member', 'step', 'lat', 'lon'],
                attrs={'long_name': var_name}
            )

        ds = xr.Dataset(
            data_vars,
            coords={
                'member': member_indices,
                'step': target_steps,
            }
        )

        # Write to Icechunk
        ds.to_zarr(session.store, group=f"batch_{batch_id}", mode='w')

        # Commit
        snapshot_id = session.commit(f"Batch {batch_id}: {len(member_list)} members")

        processing_time = time.time() - start_time

        return {
            'batch_id': batch_id,
            'branch_name': branch_name,
            'status': 'success',
            'n_members': len(member_list),
            'member_indices': member_indices,
            'n_variables': len(all_vars),
            'n_steps': len(target_steps),
            'snapshot_id': str(snapshot_id),
            'processing_time': processing_time
        }

    except Exception as e:
        import traceback
        return {
            'batch_id': batch_id,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        if creds_file is not None:
            try:
                os.unlink(creds_file.name)
            except:
                pass


# ==============================================================================
# ICECHUNK SETUP
# ==============================================================================

def create_icechunk_repository(gcs_bucket: str, gcs_prefix: str,
                                service_account_file: str) -> str:
    """Create or verify Icechunk repository exists."""
    import icechunk

    logger.info(f"Setting up Icechunk repository at gs://{gcs_bucket}/{gcs_prefix}")

    config = icechunk.RepositoryConfig.default()
    container = icechunk.VirtualChunkContainer(
        "s3://ecmwf-forecasts/",
        store=icechunk.s3_store(region="eu-west-2", anonymous=True),
    )
    config.set_virtual_chunk_container(container)

    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_prefix,
        service_account_file=service_account_file
    )

    try:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("Opened existing Icechunk repository")
        return "opened"
    except Exception:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("Created new Icechunk repository")
        return "created"


def aggregate_from_icechunk(
    gcs_bucket: str,
    gcs_prefix: str,
    service_account_file: str,
    batch_branches: List[str],
    variables: List[str]
) -> Dict[str, np.ndarray]:
    """
    Aggregate data from all batch branches.

    Returns dict of {var_name: array of shape (n_members, n_steps, lat, lon)}
    """
    import icechunk

    logger.info(f"Aggregating from {len(batch_branches)} batch branches...")

    config = icechunk.RepositoryConfig.default()
    container = icechunk.VirtualChunkContainer(
        "s3://ecmwf-forecasts/",
        store=icechunk.s3_store(region="eu-west-2", anonymous=True),
    )
    config.set_virtual_chunk_container(container)

    storage = icechunk.gcs_storage(
        bucket=gcs_bucket,
        prefix=gcs_prefix,
        service_account_file=service_account_file
    )

    repo = icechunk.Repository.open(storage, config=config)

    all_data = {var: [] for var in variables}
    all_member_indices = []

    for branch_name in sorted(batch_branches):
        try:
            session = repo.readonly_session(branch=branch_name)
            batch_id = branch_name.replace('batch_', '')

            ds = xr.open_zarr(session.store, group=f"batch_{batch_id}")

            for var in variables:
                if var in ds:
                    all_data[var].append(ds[var].values)

            all_member_indices.extend(ds.coords['member'].values.tolist())
            ds.close()

            logger.info(f"  Read {branch_name}: {len(ds.coords['member'])} members")

        except Exception as e:
            logger.warning(f"  Failed to read {branch_name}: {e}")

    # Concatenate across batches
    result = {}
    for var in variables:
        if all_data[var]:
            result[var] = np.concatenate(all_data[var], axis=0)
            logger.info(f"  {var}: {result[var].shape}")
        else:
            result[var] = None

    return result, all_member_indices


# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================

def stream_cgan_variables_coiled(
    parquet_dir: Path,
    n_workers: int = DEFAULT_N_WORKERS,
    members_per_batch: int = DEFAULT_MEMBERS_PER_BATCH,
    service_account_file: Optional[str] = None,
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
    coiled_region: str = DEFAULT_COILED_REGION,
    output_dir: Path = OUTPUT_DIR,
    max_members: Optional[int] = None,
    use_icechunk: bool = True,
    coiled_workspace: str = "gcp-sewaa-nka"
) -> bool:
    """
    Stream ECMWF data for cGAN using Coiled parallel processing.

    Args:
        parquet_dir: Directory with stage3 parquet files
        n_workers: Number of Coiled workers
        members_per_batch: Members per worker batch
        service_account_file: GCS service account JSON (optional - uses ADC if None)
        gcs_bucket: GCS bucket for Icechunk
        gcs_prefix: GCS prefix for Icechunk
        coiled_region: AWS region for Coiled cluster
        output_dir: Output directory for final NetCDF
        max_members: Limit number of members (for testing)
        use_icechunk: If False, return data directly without Icechunk storage
        coiled_workspace: Coiled workspace name

    Returns:
        True if successful
    """
    import coiled
    from dask.distributed import Client, as_completed

    logger.info("=" * 70)
    logger.info("ECMWF cGAN Streaming - Coiled Dask Version")
    logger.info("=" * 70)
    logger.info(f"Parquet Directory: {parquet_dir}")
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Members per batch: {members_per_batch}")
    logger.info(f"Icechunk: gs://{gcs_bucket}/{gcs_prefix}")
    logger.info("=" * 70)

    start_time = time.time()

    if not parquet_dir.exists():
        logger.error(f"Parquet directory {parquet_dir} not found!")
        return False

    # Extract date from directory name
    match = re.search(r'(\d{8})_(\d{2})z', str(parquet_dir))
    if match:
        model_date = datetime.strptime(match.group(1), '%Y%m%d')
        run_hour = int(match.group(2))
    else:
        model_date = datetime.now()
        run_hour = 0

    logger.info(f"Model Date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")

    # Get all members
    all_members = get_all_member_parquets(parquet_dir)
    if max_members:
        all_members = all_members[:max_members]

    logger.info(f"Total members: {len(all_members)}")

    # Load credentials
    with open(service_account_file, 'r') as f:
        creds_content = f.read()

    # Create Icechunk repository
    create_icechunk_repository(gcs_bucket, gcs_prefix, service_account_file)

    # Create member batches
    member_batches = create_member_batches(all_members, members_per_batch)
    logger.info(f"Created {len(member_batches)} batches")

    # Prepare ICPAC slice indices
    icpac_lat_slice = (LAT_INDICES[0], LAT_INDICES[-1] + 1)
    icpac_lon_slice = (LON_INDICES[0], LON_INDICES[-1] + 1)

    # Prepare task arguments
    task_args = []
    for batch_id, batch in enumerate(member_batches):
        # Convert paths to strings for serialization
        batch_serializable = [(m[0], m[1], str(m[2])) for m in batch]

        args = (
            batch_id,
            batch_serializable,
            str(parquet_dir),
            CGAN_SURFACE_VARS,
            CGAN_PRESSURE_VARS,
            TARGET_STEPS,
            TARGET_PRESSURE_LEVEL,
            icpac_lat_slice,
            icpac_lon_slice,
            creds_content,
            gcs_bucket,
            gcs_prefix,
            ECMWF_GRID_SHAPE
        )
        task_args.append(args)

    # Create Coiled cluster
    logger.info(f"\nStarting Coiled cluster with {n_workers} workers...")

    cluster = coiled.Cluster(
        name=f"ecmwf-cgan-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types="n2-standard-4",  # 4 vCPU, 16GB RAM
        package_sync=True,
        region=coiled_region,
        workspace="e4drr",
        idle_timeout="15 minutes",
    )
    client = Client(cluster)
    logger.info(f"Cluster ready: {client.dashboard_link}")

    # Phase 1: Submit batch tasks
    logger.info(f"\n[Phase 1] Submitting {len(task_args)} batch tasks...")
    futures = client.map(process_member_batch_worker, task_args)

    # Collect results
    results = {
        'successful_batches': 0,
        'failed_batches': 0,
        'batch_branches': [],
        'errors': []
    }

    for i, future in enumerate(as_completed(futures), 1):
        try:
            result = future.result()

            if result['status'] == 'success':
                results['successful_batches'] += 1
                results['batch_branches'].append(result['branch_name'])
                logger.info(f"  Batch {result['batch_id']}: {result['n_members']} members "
                          f"in {result['processing_time']:.1f}s -> {result['branch_name']}")
            else:
                results['failed_batches'] += 1
                results['errors'].append(result.get('error', 'Unknown'))
                logger.error(f"  Batch {result.get('batch_id')}: FAILED - {result.get('error')}")

        except Exception as e:
            logger.error(f"  Future error: {e}")
            results['failed_batches'] += 1
            results['errors'].append(str(e))

        if i % 5 == 0:
            logger.info(f"  Progress: {i}/{len(task_args)} batches")

    # Close cluster
    logger.info("Closing Coiled cluster...")
    client.close()
    cluster.close()

    parallel_time = time.time() - start_time
    logger.info(f"\nPhase 1 complete: {parallel_time:.1f}s")
    logger.info(f"  Successful: {results['successful_batches']}/{len(task_args)}")

    if results['successful_batches'] == 0:
        logger.error("No batches succeeded!")
        return False

    # Phase 2: Aggregate from Icechunk
    logger.info("\n[Phase 2] Aggregating from Icechunk branches...")

    all_vars = list(CGAN_SURFACE_VARS.values()) + list(CGAN_PRESSURE_VARS.values())
    aggregated_data, member_indices = aggregate_from_icechunk(
        gcs_bucket, gcs_prefix, service_account_file,
        results['batch_branches'], all_vars
    )

    # Phase 3: Compute ensemble statistics
    logger.info("\n[Phase 3] Computing ensemble statistics...")

    data_dict = {}
    for var_name in all_vars:
        if aggregated_data.get(var_name) is not None:
            data = aggregated_data[var_name]  # (n_members, n_steps, lat, lon)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ensemble_mean = np.nanmean(data, axis=0)  # (n_steps, lat, lon)
                ensemble_std = np.nanstd(data, axis=0)

            data_dict[var_name] = (ensemble_mean, ensemble_std)
            logger.info(f"  {var_name}: mean=[{np.nanmin(ensemble_mean):.4g}, {np.nanmax(ensemble_mean):.4g}]")

    # Phase 4: Create NetCDF
    logger.info("\n[Phase 4] Creating output NetCDF...")

    output_dir.mkdir(exist_ok=True)
    date_str = model_date.strftime('%Y%m%d')
    output_file = output_dir / f"IFS_{date_str}_{run_hour:02d}Z_cgan.nc"

    # Create coordinates
    n_steps = len(TARGET_STEPS)
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
            attrs={'long_name': f'{var_name} ensemble mean', 'units': 'varies'}
        )
        data_vars[f'{var_name}_ensemble_standard_deviation'] = xr.DataArray(
            std_data[np.newaxis, :, :, :],
            dims=['time', 'valid_time', 'latitude', 'longitude'],
            attrs={'long_name': f'{var_name} ensemble standard deviation', 'units': 'varies'}
        )

    ds = xr.Dataset(data_vars, coords=coords)

    # Add global attributes
    ds.attrs['title'] = 'ECMWF Ensemble Data for cGAN Inference (Coiled Parallel)'
    ds.attrs['institution'] = 'ICPAC'
    ds.attrs['source'] = 'ECMWF IFS Ensemble'
    ds.attrs['model_date'] = model_date.strftime('%Y-%m-%d')
    ds.attrs['model_run'] = f'{run_hour:02d}Z'
    ds.attrs['forecast_hours'] = str(TARGET_STEPS)
    ds.attrs['n_ensemble_members'] = len(member_indices)
    ds.attrs['history'] = f'Created {datetime.now().isoformat()} using Coiled Dask'

    # Save
    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                for var in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved: {output_file} ({file_size_mb:.2f} MB)")

    # Final summary
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("STREAMING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Model Date: {model_date.strftime('%Y-%m-%d')} {run_hour:02d}Z")
    logger.info(f"Members processed: {len(member_indices)}")
    logger.info(f"Variables extracted: {len(data_dict)}")
    logger.info(f"Timesteps: {TARGET_STEPS}")
    logger.info(f"Parallel phase: {parallel_time:.1f}s ({parallel_time/60:.1f} min)")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 70)

    return True


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Stream ECMWF data for cGAN using Coiled Dask'
    )
    parser.add_argument(
        '--parquet-dir', type=str, default=str(DEFAULT_PARQUET_DIR),
        help='Directory containing stage3 parquet files'
    )
    parser.add_argument(
        '--n-workers', type=int, default=DEFAULT_N_WORKERS,
        help=f'Number of Coiled workers (default: {DEFAULT_N_WORKERS})'
    )
    parser.add_argument(
        '--members-per-batch', type=int, default=DEFAULT_MEMBERS_PER_BATCH,
        help=f'Members per batch (default: {DEFAULT_MEMBERS_PER_BATCH})'
    )
    parser.add_argument(
        '--service-account', type=str, default='coiled-data-e4drr.json',
        help='GCS service account JSON file'
    )
    parser.add_argument(
        '--gcs-bucket', type=str, default=DEFAULT_GCS_BUCKET,
        help=f'GCS bucket for Icechunk (default: {DEFAULT_GCS_BUCKET})'
    )
    parser.add_argument(
        '--gcs-prefix', type=str, default=DEFAULT_GCS_PREFIX,
        help=f'GCS prefix for Icechunk (default: {DEFAULT_GCS_PREFIX})'
    )
    parser.add_argument(
        '--coiled-region', type=str, default=DEFAULT_COILED_REGION,
        help=f'AWS region for Coiled (default: {DEFAULT_COILED_REGION})'
    )
    parser.add_argument(
        '--output-dir', type=str, default=str(OUTPUT_DIR),
        help='Output directory for NetCDF'
    )
    parser.add_argument(
        '--max-members', type=int, default=None,
        help='Limit number of members (for testing)'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Test mode: 5 members, 5 workers'
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.max_members = 5
        args.n_workers = 5
        args.members_per_batch = 1
        logger.info("TEST MODE: 5 members, 5 workers")

    success = stream_cgan_variables_coiled(
        parquet_dir=Path(args.parquet_dir),
        n_workers=args.n_workers,
        members_per_batch=args.members_per_batch,
        service_account_file=args.service_account,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        coiled_region=args.coiled_region,
        output_dir=Path(args.output_dir),
        max_members=args.max_members
    )

    if success:
        logger.info("\nCoiled streaming completed successfully!")
    else:
        logger.error("\nCoiled streaming failed.")
        sys.exit(1)
