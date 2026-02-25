#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "dask",
#     "distributed",
#     "pandas",
#     "numpy",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "requests",
#     "google-cloud-storage",
#     "kerchunk",
#     "h5py",
#     "zarr",
# ]
# ///
"""
CFS Template Creation — One-time Preprocessing
================================================

Creates CFS mapping parquets and zarr store templates for HuggingFace.
This is the CFS equivalent of gefs/dev-test/gefs_index_preprocessing_fixed.py
combined with the zarr store template building from gefs/gefs_util.py.

Architecture:
    Local orchestrator (this script)
        |
        v
    Coiled Cluster (50 workers, Dask distributed)  [optional]
        |
        +-- Step A: dask.delayed(build_idx_grib_mapping) x N files
        |   -> mapping parquets uploaded to GCS per file
        |
        +-- Step B: Sequential post-processing (after Step A completes)
            -> scan_grib + grib_tree + map_from_index -> zarr store parquets
            -> Package into tar.gz

Step A — Per-timestep mapping parquets:
    For each GRIB file, call build_idx_grib_mapping, deduplicate, save
    to GCS at: gs://{bucket}/cfs/time_idx/{year}/{init_date}/{run}/

Step B — Zarr store templates:
    For representative init_date+run combinations, scan_grib on first 2
    GRIB files, build zarr skeleton, merge with Step A parquets via
    parse_grib_idx + map_from_index.

Step C — Package into tar.gz for HuggingFace upload.

Scale:
    Each init_date+run: ~861 GRIB files (215 days x 4 timesteps/day)
    Full month (124 members): 31 days x 4 runs = ~106,764 files
    Sequential: ~150-300 hours -> Coiled cluster: ~3-6 hours with 50 workers

Usage:
    # Step A: single init_date+run (local test, ~8 files for 48h)
    uv run run_cfs_template_creation.py --step-a --init-date 20251101 --run 00 \\
        --max-forecast-hours 48 --bucket gik-fmrc

    # Step A: full month with Coiled cluster
    uv run run_cfs_template_creation.py --step-a --init-month 202511 \\
        --max-forecast-hours 5160 --bucket gik-fmrc --use-coiled --coiled-workers 50

    # Step B: build zarr store templates from mapping parquets
    uv run run_cfs_template_creation.py --step-b --init-month 202511 \\
        --bucket gik-fmrc --mapping-bucket gik-fmrc

    # Step C: package into tar.gz for HuggingFace
    uv run run_cfs_template_creation.py --package --output gik-fmrc-v2cfs_fmrc.tar.gz

Author: ICPAC GIK Team
"""

import argparse
import copy
import io
import logging
import os
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile
import time
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import fsspec
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

S3_BUCKET = "noaa-cfs-pds"
CFS_MEMBER = "01"  # 6hrly_grib_01
DEFAULT_MAX_FORECAST_HOURS = 5160  # ~215 days

# CFS variable dictionary — all 8 target variables are in flux files
CFS_VAR_DICT = {
    "Precipitation rate": "PRATE:surface",
    "Temperature": "TMP:2 m above ground",
    "Downward short-wave radiation flux": "DSWRF:surface",
    "Upward short-wave radiation flux": "USWRF:surface",
    "Downward long-wave radiation flux": "DLWRF:surface",
    "Upward long-wave radiation flux": "ULWRF:surface",
    "U-component of wind": "UGRD:10 m above ground",
    "V-component of wind": "VGRD:10 m above ground",
}

# GCS layout for mapping parquets
# gs://{bucket}/cfs/time_idx/{year}/{init_date}/{run}/cfs-time-{init_date}-{run}-rt{hour:04d}.parquet
GCS_TIME_IDX_PREFIX = "cfs/time_idx"

# GCS layout for final zarr store parquets (Step B output)
# gs://{bucket}/cfs/zarr_templates/{init_date}/{run}/cfs-{init_date}{run}-member01-rt000.par
GCS_ZARR_TEMPLATE_PREFIX = "cfs/zarr_templates"


# ==============================================================================
# CFS URL GENERATION
# ==============================================================================

def cfs_s3_url_maker(date_str: str, run: str = "00",
                     max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS,
                     member: str = CFS_MEMBER) -> List[str]:
    """
    Generate S3 URLs for all CFS flux files for a given init_date and run.

    Parameters:
        date_str: Init date in YYYYMMDD format
        run: Run hour (00, 06, 12, 18)
        max_forecast_hours: Maximum forecast hours to include
        member: CFS member (default: 01)

    Returns:
        Sorted list of S3 URLs for CFS flux GRIB files
    """
    init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
    init_time_str = init_dt.strftime("%Y%m%d%H")

    urls = []
    for step_hours in range(0, max_forecast_hours + 1, 6):
        forecast_dt = init_dt + timedelta(hours=step_hours)
        forecast_time_str = forecast_dt.strftime("%Y%m%d%H")

        url = (
            f"s3://{S3_BUCKET}/cfs.{date_str}/{run}/6hrly_grib_{member}/"
            f"flxf{forecast_time_str}.{member}.{init_time_str}.grb2"
        )
        urls.append(url)

    logger.info(f"Generated {len(urls)} CFS URLs for {date_str} run {run} "
                f"(0-{max_forecast_hours}h, 6-hourly)")
    return urls


def get_cfs_details(url: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Extract date, run, forecast hour, and init_time from CFS flux URL.

    Returns:
        (date, run, forecast_hour, init_time) or (None, None, None, None) on failure
    """
    pattern = (
        r"s3://noaa-cfs-pds/cfs\.(\d{8})/(\d{2})/6hrly_grib_\d{2}/"
        r"flxf(\d{10})\.\d{2}\.(\d{10})\.grb2"
    )
    match = re.match(pattern, url)
    if not match:
        return None, None, None, None

    date = match.group(1)
    run = match.group(2)
    forecast_time = match.group(3)
    init_time = match.group(4)

    init_dt = datetime.strptime(init_time, "%Y%m%d%H")
    forecast_dt = datetime.strptime(forecast_time, "%Y%m%d%H")
    forecast_hour = int((forecast_dt - init_dt).total_seconds() / 3600)

    return date, run, forecast_hour, init_time


def generate_init_dates(init_month: str) -> List[Tuple[str, str]]:
    """
    Generate all (init_date, run) pairs for a given month.

    Parameters:
        init_month: Month in YYYYMM format

    Returns:
        List of (date_str, run) tuples — 31 days x 4 runs = up to 124 members
    """
    year = int(init_month[:4])
    month = int(init_month[4:6])
    _, last_day = monthrange(year, month)

    members = []
    for day in range(1, last_day + 1):
        date_str = f"{year}{month:02d}{day:02d}"
        for run in ["00", "06", "12", "18"]:
            members.append((date_str, run))

    logger.info(f"Generated {len(members)} init_date+run pairs for {init_month}")
    return members


# ==============================================================================
# STEP A: PER-TIMESTEP MAPPING PARQUETS
# ==============================================================================

def upload_to_gcs_worker(bucket_name: str, source_file: str,
                         destination_blob: str) -> bool:
    """
    Upload a file to GCS using worker credentials (Dask or local).

    Tries Dask worker credentials first, falls back to Application Default
    Credentials or a local service account file.
    """
    try:
        from google.cloud import storage as gcs_storage

        # Try Dask worker credentials
        try:
            from distributed import get_worker
            worker = get_worker()
            creds_path = str(pathlib.Path(worker.local_directory) / 'coiled-data-e4drr_202505.json')
            if os.path.exists(creds_path):
                client = gcs_storage.Client.from_service_account_json(creds_path)
            else:
                client = gcs_storage.Client()
        except (ValueError, ImportError):
            # Not running on a Dask worker — use ADC or local creds
            local_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
            if local_creds and os.path.exists(local_creds):
                client = gcs_storage.Client.from_service_account_json(local_creds)
            else:
                client = gcs_storage.Client()

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(source_file)
        return True

    except Exception as e:
        logger.error(f"GCS upload failed ({destination_blob}): {e}")
        return False


@dask.delayed
def process_cfs_time_idx_data(s3url: str, bucket_name: str) -> bool:
    """
    Build index mapping for a single CFS GRIB file and upload to GCS.

    This is the CFS equivalent of process_gefs_time_idx_data from
    gefs_index_preprocessing_fixed.py.

    Parameters:
        s3url: S3 URL to CFS flux GRIB file
        bucket_name: GCS bucket for storing mapping parquets

    Returns:
        True on success, raises on failure
    """
    from kerchunk._grib_idx import build_idx_grib_mapping

    date_str, run, forecast_hour, init_time = get_cfs_details(s3url)
    if not all([date_str, run, forecast_hour is not None, init_time]):
        logger.error(f"Invalid CFS URL format: {s3url}")
        return False

    logger.info(f"Processing CFS: {date_str} run {run} f{forecast_hour:04d}")

    # Build mapping with anonymous S3 access
    storage_options = {"anon": True, "asynchronous": False}
    mapping = build_idx_grib_mapping(s3url, storage_options=storage_options)

    # Deduplicate on attrs
    deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
    deduped_mapping = deduped_mapping.copy()
    deduped_mapping.set_index('attrs', inplace=True)

    # Save as parquet locally
    output_dir = f"cfs_mapping_{date_str}_{run}"
    os.makedirs(output_dir, exist_ok=True)
    parquet_filename = f"cfs-time-{date_str}-{run}-rt{forecast_hour:04d}.parquet"
    parquet_path = os.path.join(output_dir, parquet_filename)
    deduped_mapping.to_parquet(parquet_path, index=True)

    # Upload to GCS
    year = date_str[:4]
    gcs_blob = f"{GCS_TIME_IDX_PREFIX}/{year}/{date_str}/{run}/{parquet_filename}"
    upload_to_gcs_worker(bucket_name, parquet_path, gcs_blob)

    # Cleanup
    os.remove(parquet_path)
    try:
        os.rmdir(output_dir)
    except OSError:
        pass

    logger.info(f"CFS mapping uploaded: {date_str} run {run} f{forecast_hour:04d}")
    return True


def run_step_a_single(date_str: str, run: str, bucket_name: str,
                      max_forecast_hours: int, use_dask: bool = True) -> Dict:
    """
    Run Step A for a single init_date+run combination.

    Parameters:
        date_str: Init date YYYYMMDD
        run: Run hour (00, 06, 12, 18)
        bucket_name: GCS bucket name
        max_forecast_hours: Maximum forecast hours to process
        use_dask: Whether to use Dask for parallelism

    Returns:
        Dictionary with results summary
    """
    logger.info(f"Step A: Creating mapping parquets for {date_str} run {run}")
    start_time = time.time()

    # Generate all URLs for this init_date+run
    urls = cfs_s3_url_maker(date_str, run, max_forecast_hours)

    if use_dask:
        # Use Dask delayed for parallel processing
        tasks = [process_cfs_time_idx_data(url, bucket_name) for url in urls]
        results = dask.compute(*tasks)
    else:
        # Process sequentially (for testing without Dask cluster)
        results = []
        for url in urls:
            try:
                result = process_cfs_time_idx_data.compute(url, bucket_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                results.append(False)

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r)

    logger.info(f"Step A complete for {date_str} run {run}: "
                f"{success_count}/{len(urls)} files in {elapsed:.1f}s")

    return {
        "date": date_str,
        "run": run,
        "total_files": len(urls),
        "successful": success_count,
        "failed": len(urls) - success_count,
        "elapsed_seconds": round(elapsed, 1),
    }


def run_step_a_month(init_month: str, bucket_name: str,
                     max_forecast_hours: int, use_coiled: bool = False,
                     coiled_workers: int = 50) -> List[Dict]:
    """
    Run Step A for an entire month of init dates (all runs).

    Parameters:
        init_month: Month in YYYYMM format
        bucket_name: GCS bucket
        max_forecast_hours: Max forecast hours
        use_coiled: Whether to spin up a Coiled cluster
        coiled_workers: Number of Coiled workers

    Returns:
        List of result dicts per init_date+run
    """
    members = generate_init_dates(init_month)

    if use_coiled:
        try:
            import coiled
            from distributed import Client

            logger.info(f"Starting Coiled cluster with {coiled_workers} workers")
            cluster = coiled.Cluster(
                name=f"cfs-template-{init_month}",
                n_workers=coiled_workers,
                worker_memory="4 GiB",
                worker_cpu=2,
                software="cfs-gik-env",
                shutdown_on_close=True,
            )
            client = Client(cluster)
            logger.info(f"Coiled cluster ready: {client.dashboard_link}")
        except ImportError:
            logger.warning("Coiled not installed, falling back to local Dask")
            use_coiled = False

    all_results = []
    for date_str, run in members:
        result = run_step_a_single(date_str, run, bucket_name,
                                   max_forecast_hours, use_dask=True)
        all_results.append(result)

    if use_coiled:
        try:
            client.close()
            cluster.close()
        except Exception:
            pass

    # Summary
    total_files = sum(r["total_files"] for r in all_results)
    total_success = sum(r["successful"] for r in all_results)
    logger.info(f"Step A month complete: {total_success}/{total_files} files "
                f"across {len(members)} init_date+run combinations")

    return all_results


# ==============================================================================
# STEP B: ZARR STORE TEMPLATE BUILDING
# ==============================================================================

def generate_cfs_axes(date_str: str, run: str = "00",
                      max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS) -> List[pd.Index]:
    """
    Generate temporal axes for CFS (6-hourly intervals).

    Adapted from gefs_util.py:generate_axes — uses 6-hourly frequency
    and extends to the full CFS forecast range.

    Parameters:
        date_str: Init date YYYYMMDD
        run: Run hour
        max_forecast_hours: Max forecast hours

    Returns:
        [valid_time_index, time_index]
    """
    start_date = pd.Timestamp(f"{date_str}{run}", tz=None)
    end_date = start_date + pd.Timedelta(hours=max_forecast_hours)

    valid_time_index = pd.date_range(
        start_date, end_date, freq="360min", name="valid_time"
    )
    time_index = pd.Index([start_date], name="time")

    logger.info(f"CFS axes: {len(valid_time_index)} timesteps "
                f"({start_date} to {end_date}, 6-hourly)")
    return [valid_time_index, time_index]


def filter_cfs_scan_grib(grib_url: str, var_dict: Dict = None) -> List:
    """
    Scan a CFS GRIB file and filter messages by variable dictionary.

    Parameters:
        grib_url: S3 URL to CFS GRIB file
        var_dict: Variable filter dictionary (default: CFS_VAR_DICT)

    Returns:
        List of filtered scan_grib groups
    """
    from kerchunk._grib_idx import parse_grib_idx
    from kerchunk.grib2 import scan_grib

    if var_dict is None:
        var_dict = CFS_VAR_DICT

    storage_options = {"anon": True}
    gsc = scan_grib(grib_url, storage_options=storage_options)

    # Parse the index to identify which messages match our variables
    idx_df = parse_grib_idx(
        basename=grib_url, suffix="idx", storage_options=storage_options
    )

    # Match variables from the dictionary to index entries
    selected_indices = set()
    for var_desc, var_filter in var_dict.items():
        parts = var_filter.split(":")
        var_name = parts[0]
        var_level = parts[1] if len(parts) > 1 else ""

        for i, row in idx_df.iterrows():
            attrs_str = str(row.get("attrs", ""))
            if var_name in attrs_str and var_level in attrs_str:
                selected_indices.add(i)

    # Return only the matching groups
    filtered = [gsc[i] for i in sorted(selected_indices) if i < len(gsc)]
    logger.info(f"CFS scan_grib: {len(filtered)}/{len(gsc)} messages match filter")
    return filtered


def build_cfs_grib_tree(grib_files: List[str],
                        var_dict: Dict = None) -> Tuple[dict, dict]:
    """
    Scan CFS GRIB files, build hierarchical zarr tree, create deflated copy.

    Adapted from gefs_util.py:filter_build_grib_tree.

    Parameters:
        grib_files: List of CFS GRIB file URLs (typically f000 + f006)
        var_dict: Variable filter dictionary

    Returns:
        (grib_tree_store, deflated_grib_tree_store)
    """
    from kerchunk._grib_idx import strip_datavar_chunks
    from kerchunk.grib2 import grib_tree

    logger.info(f"Building CFS grib tree from {len(grib_files)} files")
    all_groups = []
    for gurl in grib_files:
        groups = filter_cfs_scan_grib(gurl, var_dict)
        all_groups.extend(groups)

    tree_store = grib_tree(all_groups)
    deflated_store = copy.deepcopy(tree_store)
    strip_datavar_chunks(deflated_store)

    logger.info(f"CFS grib tree: {len(tree_store['refs'])} refs "
                f"(deflated: {len(deflated_store['refs'])})")
    return tree_store, deflated_store


def create_cfs_mapped_index(axes: List[pd.Index], mapping_bucket: str,
                            date_str: str, run: str,
                            max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS
                            ) -> pd.DataFrame:
    """
    Create mapped index from CFS index files using pre-built Step A parquets.

    Adapted from gefs_util.py:create_gefs_mapped_index.

    Parameters:
        axes: Temporal axes [valid_time_index, time_index]
        mapping_bucket: GCS bucket containing Step A parquets
        date_str: Init date YYYYMMDD
        run: Run hour
        max_forecast_hours: Max forecast hours

    Returns:
        DataFrame with merged index for all timesteps
    """
    from kerchunk._grib_idx import map_from_index, parse_grib_idx

    logger.info(f"Creating CFS mapped index for {date_str} run {run}")

    import gcsfs
    gcs_fs = gcsfs.GCSFileSystem()

    dtaxes = axes[0]
    init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
    init_time_str = init_dt.strftime("%Y%m%d%H")
    year = date_str[:4]

    mapped_index_list = []
    for idx, datestr in enumerate(dtaxes):
        forecast_hour = idx * 6
        if forecast_hour > max_forecast_hours:
            break

        try:
            forecast_dt = init_dt + timedelta(hours=forecast_hour)
            forecast_time_str = forecast_dt.strftime("%Y%m%d%H")

            # S3 URL for this timestep
            fname = (
                f"s3://{S3_BUCKET}/cfs.{date_str}/{run}/6hrly_grib_{CFS_MEMBER}/"
                f"flxf{forecast_time_str}.{CFS_MEMBER}.{init_time_str}.grb2"
            )

            # Fresh index from S3
            idxdf = parse_grib_idx(basename=fname)

            # Pre-built mapping from GCS (Step A output)
            mapping_path = (
                f"gs://{mapping_bucket}/{GCS_TIME_IDX_PREFIX}/{year}/{date_str}/{run}/"
                f"cfs-time-{date_str}-{run}-rt{forecast_hour:04d}.parquet"
            )
            deduped_mapping = pd.read_parquet(mapping_path, filesystem=gcs_fs)

            # Merge fresh index with template mapping
            mapped_index = map_from_index(
                datestr,
                deduped_mapping,
                idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
            )
            mapped_index_list.append(mapped_index)

        except Exception as e:
            logger.warning(f"Error at f{forecast_hour:04d}: {e}")

    if not mapped_index_list:
        logger.error("No mapped indices created")
        return pd.DataFrame()

    result = pd.concat(mapped_index_list)
    logger.info(f"CFS mapped index: {len(result)} entries, "
                f"{len(result.drop_duplicates('varname'))} unique variables")
    return result


def process_cfs_unique_groups(zstore: dict, chunk_index: pd.DataFrame,
                              time_dims: Dict, time_coords: Dict,
                              times: np.ndarray, valid_times: np.ndarray,
                              steps: np.ndarray) -> dict:
    """
    Process unique variable groups and update zarr store.

    Adapted from gefs_util.py:process_unique_groups for CFS variables.

    Parameters:
        zstore: Deflated zarr store references
        chunk_index: DataFrame with mapped index data
        time_dims: Time dimension dict
        time_coords: Time coordinate dict
        times, valid_times, steps: Time arrays

    Returns:
        Updated zstore dict
    """
    from kerchunk._grib_idx import store_coord_var, store_data_var

    logger.info("Processing CFS unique groups")
    unique_groups = chunk_index.set_index(
        ["varname", "stepType", "typeOfLevel"]
    ).index.unique()

    logger.info(f"Chunk index: {len(chunk_index)} rows, "
                f"{len(unique_groups)} unique groups")

    # Remove keys not in our variable set
    keys_to_delete = []
    for key in list(zstore.keys()):
        lookup = tuple(
            [val for val in os.path.dirname(key).split("/")[:3] if val != ""]
        )
        if lookup not in unique_groups:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del zstore[key]

    logger.info(f"Removed {len(keys_to_delete)} non-matching keys from zstore")

    # Process each variable group
    for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
        try:
            base_path = "/".join(key)
            lvals = group.level.unique()
            dims = time_dims.copy()
            coords = time_coords.copy()

            # Check required zarr structure exists
            required_keys = [
                f"{base_path}/time/.zattrs",
                f"{base_path}/valid_time/.zattrs",
                f"{base_path}/step/.zattrs",
            ]
            missing = [k for k in required_keys if k not in zstore]
            if missing:
                logger.debug(f"Skipping {key}: missing zarr keys")
                continue

            if len(lvals) == 1:
                lvals = lvals.squeeze()
                dims[key[2]] = 0
            elif len(lvals) > 1:
                lvals = np.sort(lvals)
                dims[key[2]] = len(lvals)
                coords["datavar"] += (key[2],)
            else:
                continue

            store_coord_var(
                key=f"{base_path}/time", zstore=zstore,
                coords=time_coords["time"],
                data=times.astype("datetime64[s]")
            )
            store_coord_var(
                key=f"{base_path}/valid_time", zstore=zstore,
                coords=time_coords["valid_time"],
                data=valid_times.astype("datetime64[s]")
            )
            store_coord_var(
                key=f"{base_path}/step", zstore=zstore,
                coords=time_coords["step"],
                data=steps.astype("timedelta64[s]").astype("float64") / 3600.0
            )
            store_coord_var(
                key=f"{base_path}/{key[2]}", zstore=zstore,
                coords=(key[2],) if lvals.shape else (),
                data=lvals
            )
            store_data_var(
                key=f"{base_path}/{key[0]}", zstore=zstore,
                dims=dims, coords=coords, data=group,
                steps=steps, times=times,
                lvals=lvals if lvals.shape else None
            )

            logger.debug(f"Processed group: {key}")

        except Exception as e:
            logger.warning(f"Skipping CFS group {key}: {e}")

    return zstore


def calculate_cfs_time_dimensions(
    axes: List[pd.Index]
) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time dimensions and coordinates for CFS.

    Adapted from gefs_util.py:calculate_time_dimensions.

    Returns:
        (time_dims, time_coords, times, valid_times, steps)
    """
    axes_by_name = {pdi.name: pdi for pdi in axes}

    time_dims = {"valid_time": len(axes_by_name["valid_time"])}

    assert len(axes_by_name["time"]) == 1, \
        "Time axis must describe a single init date"
    reference_time = axes_by_name["time"].to_numpy()[0]

    time_coords = {
        "step": ("valid_time",),
        "valid_time": ("valid_time",),
        "time": ("valid_time",),
        "datavar": ("valid_time",),
    }

    valid_times = axes_by_name["valid_time"].to_numpy()
    times = np.where(valid_times <= reference_time, valid_times, reference_time)
    steps = valid_times - times

    # Override times to match valid_times (same as GEFS pattern)
    times = valid_times
    return time_dims, time_coords, times, valid_times, steps


def run_step_b_single(date_str: str, run: str, mapping_bucket: str,
                      output_bucket: str,
                      max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS
                      ) -> Optional[str]:
    """
    Run Step B for a single init_date+run: build zarr store template.

    1. scan_grib on first 2 GRIB files (f000 + f006) -> grib_tree -> skeleton
    2. strip_datavar_chunks -> deflated store
    3. For each timestep: parse_grib_idx + map_from_index using Step A parquets
    4. process_unique_groups -> complete zarr store
    5. Save as final parquet

    Parameters:
        date_str: Init date YYYYMMDD
        run: Run hour
        mapping_bucket: GCS bucket with Step A parquets
        output_bucket: GCS bucket for Step B output
        max_forecast_hours: Max forecast hours

    Returns:
        GCS path of output parquet, or None on failure
    """
    logger.info(f"Step B: Building zarr template for {date_str} run {run}")
    start_time = time.time()

    try:
        # 1. Scan first 2 GRIB files to build skeleton
        init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
        init_time_str = init_dt.strftime("%Y%m%d%H")

        f000_dt = init_dt
        f006_dt = init_dt + timedelta(hours=6)

        grib_files = [
            (f"s3://{S3_BUCKET}/cfs.{date_str}/{run}/6hrly_grib_{CFS_MEMBER}/"
             f"flxf{f000_dt.strftime('%Y%m%d%H')}.{CFS_MEMBER}.{init_time_str}.grb2"),
            (f"s3://{S3_BUCKET}/cfs.{date_str}/{run}/6hrly_grib_{CFS_MEMBER}/"
             f"flxf{f006_dt.strftime('%Y%m%d%H')}.{CFS_MEMBER}.{init_time_str}.grb2"),
        ]

        tree_store, deflated_store = build_cfs_grib_tree(grib_files, CFS_VAR_DICT)

        # 2. Generate axes and time dimensions
        axes = generate_cfs_axes(date_str, run, max_forecast_hours)
        time_dims, time_coords, times, valid_times, steps = \
            calculate_cfs_time_dimensions(axes)

        # 3. Create mapped index from Step A parquets
        cfs_kind = create_cfs_mapped_index(
            axes, mapping_bucket, date_str, run, max_forecast_hours
        )
        if cfs_kind.empty:
            logger.error(f"No mapped index for {date_str} run {run}")
            return None

        # 4. Process unique groups
        zstore = copy.deepcopy(deflated_store["refs"])
        zstore = process_cfs_unique_groups(
            zstore, cfs_kind, time_dims, time_coords,
            times, valid_times, steps
        )

        # 5. Save as parquet
        output_dir = Path(tempfile.mkdtemp(prefix="cfs_template_"))
        parquet_filename = f"cfs-{date_str}{run}-member{CFS_MEMBER}-rt000.par"
        parquet_path = output_dir / parquet_filename

        # Convert zstore to DataFrame
        data = []
        for key, value in zstore.items():
            if isinstance(value, (dict, list, int, float)):
                value = str(value).encode('utf-8')
            elif isinstance(value, str):
                value = value.encode('utf-8')
            data.append((key, value))

        df = pd.DataFrame(data, columns=['key', 'value'])
        df.to_parquet(parquet_path)

        # Upload to GCS
        import gcsfs
        gcs_fs = gcsfs.GCSFileSystem()
        gcs_blob_path = (
            f"{output_bucket}/{GCS_ZARR_TEMPLATE_PREFIX}/"
            f"{date_str}/{run}/{parquet_filename}"
        )
        gcs_fs.put(str(parquet_path), gcs_blob_path)

        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)

        elapsed = time.time() - start_time
        logger.info(f"Step B complete for {date_str} run {run}: "
                    f"gs://{gcs_blob_path} ({elapsed:.1f}s)")
        return f"gs://{gcs_blob_path}"

    except Exception as e:
        logger.error(f"Step B failed for {date_str} run {run}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_step_b_month(init_month: str, mapping_bucket: str,
                     output_bucket: str,
                     max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS
                     ) -> List[Dict]:
    """
    Run Step B for all init_date+run combinations in a month.

    Parameters:
        init_month: Month YYYYMM
        mapping_bucket: GCS bucket with Step A parquets
        output_bucket: GCS bucket for Step B output
        max_forecast_hours: Max forecast hours

    Returns:
        List of result dicts
    """
    members = generate_init_dates(init_month)
    results = []

    for i, (date_str, run) in enumerate(members, 1):
        logger.info(f"[{i}/{len(members)}] Processing {date_str} run {run}")
        gcs_path = run_step_b_single(
            date_str, run, mapping_bucket, output_bucket, max_forecast_hours
        )
        results.append({
            "date": date_str,
            "run": run,
            "success": gcs_path is not None,
            "gcs_path": gcs_path,
        })

    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Step B month complete: {success_count}/{len(members)} templates")
    return results


# ==============================================================================
# STEP C: PACKAGE INTO TAR.GZ
# ==============================================================================

def run_step_c_package(init_month: str, output_bucket: str,
                       output_file: str = "gik-fmrc-v2cfs_fmrc.tar.gz") -> str:
    """
    Package Step B zarr store parquets into a tar.gz for HuggingFace.

    Structure inside tar.gz:
        gik-fmrc/v2cfs_fmrc/{init_date}_{run}/cfs-{init_date}{run}-member01-rt000.par

    Parameters:
        init_month: Month YYYYMM
        output_bucket: GCS bucket containing Step B output
        output_file: Local output tar.gz filename

    Returns:
        Path to output tar.gz
    """
    import gcsfs
    gcs_fs = gcsfs.GCSFileSystem()

    members = generate_init_dates(init_month)

    logger.info(f"Step C: Packaging {len(members)} templates into {output_file}")

    with tarfile.open(output_file, "w:gz") as tar:
        added = 0
        for date_str, run in members:
            parquet_filename = f"cfs-{date_str}{run}-member{CFS_MEMBER}-rt000.par"
            gcs_path = (
                f"{output_bucket}/{GCS_ZARR_TEMPLATE_PREFIX}/"
                f"{date_str}/{run}/{parquet_filename}"
            )

            try:
                if not gcs_fs.exists(gcs_path):
                    logger.warning(f"Template not found: gs://{gcs_path}")
                    continue

                # Download to temp and add to tar
                with gcs_fs.open(gcs_path, 'rb') as gcs_f:
                    data = gcs_f.read()

                tar_path = f"gik-fmrc/v2cfs_fmrc/{date_str}_{run}/{parquet_filename}"
                info = tarfile.TarInfo(name=tar_path)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                added += 1

            except Exception as e:
                logger.warning(f"Error packaging {date_str} run {run}: {e}")

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"Step C complete: {output_file} ({size_mb:.1f} MB, "
                f"{added}/{len(members)} templates)")

    return output_file


# ==============================================================================
# VERIFICATION UTILITIES
# ==============================================================================

def verify_step_a_output(bucket_name: str, date_str: str, run: str,
                         max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS
                         ) -> Dict:
    """Verify Step A mapping parquets exist in GCS for a single init_date+run."""
    import gcsfs
    gcs_fs = gcsfs.GCSFileSystem()

    year = date_str[:4]
    prefix = f"{bucket_name}/{GCS_TIME_IDX_PREFIX}/{year}/{date_str}/{run}/"

    try:
        files = gcs_fs.ls(prefix)
        parquets = [f for f in files if f.endswith('.parquet')]
        expected = (max_forecast_hours // 6) + 1

        return {
            "date": date_str,
            "run": run,
            "found": len(parquets),
            "expected": expected,
            "complete": len(parquets) >= expected,
            "prefix": f"gs://{prefix}",
        }
    except Exception as e:
        return {
            "date": date_str,
            "run": run,
            "found": 0,
            "expected": 0,
            "complete": False,
            "error": str(e),
        }


# ==============================================================================
# SUMMARY PRINTER
# ==============================================================================

def print_summary(results: List[Dict], step_name: str, total_time: float):
    """Print execution summary."""
    success_count = sum(1 for r in results if r.get("success", r.get("complete", False)))

    print("\n" + "=" * 70)
    print(f"EXECUTION SUMMARY — {step_name}")
    print("=" * 70)
    print(f"Total items: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    for r in results:
        date = r.get("date", "?")
        run = r.get("run", "?")
        status = "OK" if r.get("success", r.get("complete", False)) else "FAIL"
        msg = r.get("gcs_path", r.get("message", ""))
        print(f"  {status} {date} {run}z: {msg}")


# ==============================================================================
# CLI + MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CFS Template Creation — One-time Preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Step selection
    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument('--step-a', action='store_true',
                            help='Step A: Create per-timestep mapping parquets')
    step_group.add_argument('--step-b', action='store_true',
                            help='Step B: Build zarr store templates')
    step_group.add_argument('--package', action='store_true',
                            help='Step C: Package templates into tar.gz')
    step_group.add_argument('--verify', action='store_true',
                            help='Verify Step A output in GCS')

    # Date selection
    parser.add_argument('--init-date', type=str,
                        help='Single init date (YYYYMMDD)')
    parser.add_argument('--init-month', type=str,
                        help='Full month of init dates (YYYYMM)')
    parser.add_argument('--run', type=str, default='00',
                        choices=['00', '06', '12', '18'],
                        help='Run hour (default: 00, used with --init-date)')

    # Processing options
    parser.add_argument('--max-forecast-hours', type=int,
                        default=DEFAULT_MAX_FORECAST_HOURS,
                        help=f'Max forecast hours (default: {DEFAULT_MAX_FORECAST_HOURS})')
    parser.add_argument('--bucket', type=str, default='gik-fmrc',
                        help='GCS bucket for mapping parquets (default: gik-fmrc)')
    parser.add_argument('--mapping-bucket', type=str,
                        help='GCS bucket with Step A parquets (default: same as --bucket)')
    parser.add_argument('--output', type=str,
                        default='gik-fmrc-v2cfs_fmrc.tar.gz',
                        help='Output tar.gz filename for --package')

    # Cluster options
    parser.add_argument('--use-coiled', action='store_true',
                        help='Use Coiled cluster for Step A (recommended for full month)')
    parser.add_argument('--coiled-workers', type=int, default=50,
                        help='Number of Coiled workers (default: 50)')

    args = parser.parse_args()

    if not args.mapping_bucket:
        args.mapping_bucket = args.bucket

    # Validate arguments
    if args.step_a or args.step_b or args.verify:
        if not args.init_date and not args.init_month:
            parser.error("--init-date or --init-month required")
    if args.package and not args.init_month:
        parser.error("--init-month required for --package")

    # Print configuration
    print("=" * 70)
    print("CFS TEMPLATE CREATION")
    print("=" * 70)

    start_time = time.time()

    # Execute the requested step
    if args.step_a:
        print("Step A: Creating per-timestep mapping parquets")
        print(f"  Bucket: gs://{args.bucket}/")
        print(f"  Max forecast hours: {args.max_forecast_hours}")

        if args.init_date:
            result = run_step_a_single(
                args.init_date, args.run, args.bucket,
                args.max_forecast_hours, use_dask=True
            )
            results = [result]
        else:
            results = run_step_a_month(
                args.init_month, args.bucket, args.max_forecast_hours,
                args.use_coiled, args.coiled_workers
            )

        total_time = time.time() - start_time
        print_summary(results, "Step A", total_time)

    elif args.step_b:
        print("Step B: Building zarr store templates")
        print(f"  Mapping bucket: gs://{args.mapping_bucket}/")
        print(f"  Output bucket: gs://{args.bucket}/")

        if args.init_date:
            gcs_path = run_step_b_single(
                args.init_date, args.run, args.mapping_bucket,
                args.bucket, args.max_forecast_hours
            )
            results = [{
                "date": args.init_date, "run": args.run,
                "success": gcs_path is not None, "gcs_path": gcs_path,
            }]
        else:
            results = run_step_b_month(
                args.init_month, args.mapping_bucket,
                args.bucket, args.max_forecast_hours
            )

        total_time = time.time() - start_time
        print_summary(results, "Step B", total_time)

    elif args.package:
        print(f"Step C: Packaging templates into {args.output}")
        output = run_step_c_package(args.init_month, args.bucket, args.output)
        total_time = time.time() - start_time
        print(f"\nPackaged: {output} in {total_time:.1f}s")
        print(f"\nUpload to HuggingFace:")
        print(f"  huggingface-cli upload Nishadhka/gfs_s3_gik_refs {output}")

    elif args.verify:
        print("Verifying Step A output in GCS")

        if args.init_date:
            result = verify_step_a_output(
                args.bucket, args.init_date, args.run, args.max_forecast_hours
            )
            results = [result]
        else:
            members = generate_init_dates(args.init_month)
            results = []
            for date_str, run in members:
                result = verify_step_a_output(
                    args.bucket, date_str, run, args.max_forecast_hours
                )
                results.append(result)

        total_time = time.time() - start_time
        print_summary(results, "Verification", total_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
