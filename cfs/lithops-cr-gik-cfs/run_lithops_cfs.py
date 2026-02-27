#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lithops",
#     "google-cloud-storage",
#     "google-cloud-pubsub",
#     "google-api-python-client",
#     "google-auth",
#     "httplib2",
#     "pandas",
#     "numpy",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "requests",
# ]
# ///
"""
Lithops-native CFS Parquet Generation (Cloud Run)
==================================================

Processes CFS seasonal forecast data through the GIK three-stage pipeline
using Lithops to execute Python functions on Cloud Run workers.

No kerchunk dependency on Cloud Run workers — uses pre-built templates from
HuggingFace. Parses CFS .idx files directly (colon-separated text format).

CFS ensemble: Multi-date ensemble — 124 members (31 init dates x 4 runs
from the preceding month). Each Lithops worker = 1 member (init_date+run).

Architecture:
    run_lithops_cfs.py (this script, runs locally via uv run)
         |
         v
    [Lithops FunctionExecutor(backend='gcp_cloudrun')]
      /    |    \\
     /     |     \\  (cloudpickle-serialized functions via GCS)
    v      v      v
  [CR 1]  [CR 2]  ... [CR N]  (Lithops proxy containers on Cloud Run)
    |      |           |
    v      v           v
  process_cfs_member() runs directly inside each container:
    Stage 1: Load zarr from HuggingFace template (~5s)
    Stage 2: Fetch S3 .idx files + merge (~5-15 min)
    Stage 3: Create final parquet files (~1-2 min)
    Upload to GCS

Usage:
    # Single member (local testing)
    uv run run_lithops_cfs.py --init-date 20251201 --run 00 \\
        --max-forecast-hours 48 --sequential

    # Full ensemble for a target month (Cloud Run)
    uv run run_lithops_cfs.py --target-month 202512 --max-workers 20

    # Dry run
    uv run run_lithops_cfs.py --target-month 202512 --dry-run

    # Custom parallelism
    uv run run_lithops_cfs.py --target-month 202512 --max-workers 50 \\
        --max-forecast-hours 4320

Author: ICPAC GIK Team
"""

import os
import io
import sys
import json
import time
import argparse
import tarfile
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import fsspec
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION (serialized to Cloud Run workers via cloudpickle)
# ==============================================================================

GCS_BUCKET = os.environ.get('GCS_BUCKET', 'gik-cfs-aws-tf')
GCS_PARQUET_PREFIX = os.environ.get('GCS_PARQUET_PREFIX', 'run_par_cfs')
PARALLEL_WORKERS = int(os.environ.get('PARALLEL_WORKERS', '8'))

TEMPLATE_URL = os.environ.get(
    'TEMPLATE_URL',
    'https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-v2cfs_fmrc.tar.gz'
)

# CFS uses 6-hourly intervals
DEFAULT_MAX_FORECAST_HOURS = 4320  # 6 months = ~180 days
FORECAST_INTERVAL = 6  # hours

S3_BUCKET = "noaa-cfs-pds"
CFS_MEMBER = "01"  # 6hrly_grib_01

# Reference init date used when building the template
REFERENCE_INIT_DATE = '20251101'

# Use baked-in template from Docker image if available, fallback to /tmp download
TEMPLATE_CACHE_PATH = os.environ.get(
    'CFS_TEMPLATE_PATH',
    '/tmp/gik-fmrc-v2cfs_fmrc.tar.gz'
)

# CFS variable dictionary — 8 target variables from flux files
CFS_VAR_FILTER = {
    "PRATE": "surface",
    "TMP": "2 m above ground",
    "DSWRF": "surface",
    "USWRF": "surface",
    "DLWRF": "surface",
    "ULWRF": "surface",
    "UGRD": "10 m above ground",
    "VGRD": "10 m above ground",
}


# ==============================================================================
# STAGE 1: LOAD ZARR STRUCTURE FROM TEMPLATE
# ==============================================================================

def ensure_template(local_template: Optional[str] = None) -> str:
    """Get template path (local file, pre-baked in image, or download to /tmp)."""
    # Use local template if provided (for sequential testing)
    if local_template and os.path.exists(local_template):
        logger.info(f"Using local template: {local_template}")
        return local_template

    # Check if template is baked into the Docker image (Cloud Run workers)
    baked_path = os.environ.get('CFS_TEMPLATE_PATH')
    if baked_path and os.path.exists(baked_path):
        logger.info(f"Using pre-baked template: {baked_path}")
        return baked_path

    # Fallback: download to /tmp (local sequential runs)
    if os.path.exists(TEMPLATE_CACHE_PATH):
        logger.info(f"Template cached at {TEMPLATE_CACHE_PATH}")
        return TEMPLATE_CACHE_PATH

    logger.info(f"Downloading template from {TEMPLATE_URL}")
    response = requests.get(TEMPLATE_URL, stream=True, timeout=300)
    response.raise_for_status()

    with open(TEMPLATE_CACHE_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(TEMPLATE_CACHE_PATH) / (1024 * 1024)
    logger.info(f"Template downloaded: {size_mb:.1f} MB")
    return TEMPLATE_CACHE_PATH


def _load_parquet_as_zstore(parquet_path: str) -> Dict:
    """Load a parquet file with ['key', 'value'] columns into a zarr store dict."""
    template_df = pd.read_parquet(parquet_path)

    zstore = {}
    for _, row in template_df.iterrows():
        key = row['key']
        value = row['value']
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if isinstance(value, str):
            if value.startswith('[') or value.startswith('{'):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
        zstore[key] = value

    if 'version' in zstore:
        del zstore['version']

    return zstore


def build_deflated_store_from_template(
    template_path: str,
    init_date: str,
    run: str,
) -> Optional[Dict]:
    """
    Load zarr store from template.

    Supports two formats:
    1. Local parquet file (from run_cfs_template_creation.py --zarr-template)
    2. tar.gz archive (HuggingFace template for Cloud Run)

    Parameters:
        template_path: Path to template parquet or tar.gz
        init_date: Init date YYYYMMDD
        run: Run hour (00, 06, 12, 18)

    Returns:
        Dict of zarr store key-value pairs, or None on failure
    """
    logger.info(f"Stage 1: Loading zarr structure for {init_date} {run}z")
    start_time = time.time()

    try:
        # Local parquet file (from --zarr-template)
        if template_path.endswith('.parquet'):
            zstore = _load_parquet_as_zstore(template_path)
            elapsed = time.time() - start_time
            logger.info(f"Stage 1 complete: {len(zstore)} keys from local parquet in {elapsed:.1f}s")
            return zstore

        # tar.gz archive (HuggingFace template)
        tar_member_path = (
            f"gik-fmrc/v2cfs_fmrc/{init_date}_{run}/"
            f"cfs-{init_date}{run}-member{CFS_MEMBER}-rt000.par"
        )

        with tarfile.open(template_path, 'r:gz') as tar:
            try:
                member_info = tar.getmember(tar_member_path)
            except KeyError:
                logger.warning(f"Template not found for {init_date} {run}z: {tar_member_path}")
                return None

            f = tar.extractfile(member_info)
            if f is None:
                return None

            parquet_bytes = f.read()
            template_df = pd.read_parquet(io.BytesIO(parquet_bytes))

            zstore = {}
            for _, row in template_df.iterrows():
                key = row['key']
                value = row['value']
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if isinstance(value, str):
                    if value.startswith('[') or value.startswith('{'):
                        try:
                            value = json.loads(value)
                        except Exception:
                            pass
                zstore[key] = value

            if 'version' in zstore:
                del zstore['version']

        elapsed = time.time() - start_time
        logger.info(f"Stage 1 complete: {len(zstore)} keys in {elapsed:.1f}s")
        return zstore

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 2: IDX PARSING + TEMPLATE MERGE
# ==============================================================================

def parse_cfs_idx(idx_url: str) -> List[Dict]:
    """
    Parse CFS .idx file (colon-separated text format).

    CFS .idx format:
        1:0:d=2025110100:PRATE:surface:6 hour fcst:
        2:2784924:d=2025110100:TMP:2 m above ground:6 hour fcst:
      36.1:1898885:d=2025110100:UGRD:10 m above ground:6 hour fcst:
      36.2:1898885:d=2025110100:VGRD:10 m above ground:6 hour fcst:

    Fields: record:offset:d=YYYYMMDDHH:variable:level:forecast_info:

    Note: CFS uses fractional record numbers (e.g. 36.1, 36.2) for
    variables packed in a single GRIB message (e.g. UGRD+VGRD wind).
    Both entries share the same byte offset.

    Byte length is calculated as offset[i+1] - offset[i]. For entries
    sharing an offset, they get the same byte_length (same GRIB message).
    For the last entry, we use the GRIB file size from fs.info().

    Parameters:
        idx_url: S3 URL to the .idx file

    Returns:
        List of dicts with byte_offset, byte_length, variable, level, etc.
    """
    try:
        fs = fsspec.filesystem("s3", anon=True)
        idx_path = idx_url.replace("s3://", "")

        with fs.open(idx_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return []

        # Parse all entries
        raw_entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(':')
            if len(parts) < 6:
                continue

            try:
                record = float(parts[0])  # float handles "36.1", "79.2" etc.
                offset = int(parts[1])
                date_info = parts[2]  # d=YYYYMMDDHH
                variable = parts[3]
                level = parts[4]
                forecast_info = ':'.join(parts[5:]).strip(':')

                raw_entries.append({
                    'record': record,
                    'byte_offset': offset,
                    'variable': variable,
                    'level': level,
                    'date': date_info,
                    'forecast': forecast_info,
                })
            except (ValueError, IndexError):
                continue

        if not raw_entries:
            return []

        # Calculate byte_length from consecutive offsets.
        # For entries sharing an offset (e.g. UGRD 36.1 + VGRD 36.2),
        # find the next DIFFERENT offset to compute length.
        # Get unique offsets in order for length calculation
        unique_offsets = []
        seen = set()
        for e in raw_entries:
            if e['byte_offset'] not in seen:
                unique_offsets.append(e['byte_offset'])
                seen.add(e['byte_offset'])

        # Get file size for the last entry's length
        grib_url = idx_url.replace('.idx', '')
        grib_path = grib_url.replace("s3://", "")
        try:
            file_info = fs.info(grib_path)
            file_size = file_info.get('size', file_info.get('Size', 0))
        except Exception:
            file_size = None

        # Build offset -> length map
        offset_to_length = {}
        for i in range(len(unique_offsets) - 1):
            offset_to_length[unique_offsets[i]] = unique_offsets[i + 1] - unique_offsets[i]
        # Last unique offset
        if file_size and unique_offsets:
            offset_to_length[unique_offsets[-1]] = file_size - unique_offsets[-1]
        elif unique_offsets:
            # Fallback: estimate from average
            if len(offset_to_length) > 0:
                avg_len = sum(offset_to_length.values()) / len(offset_to_length)
                offset_to_length[unique_offsets[-1]] = int(avg_len * 2)
            else:
                offset_to_length[unique_offsets[-1]] = 1000000

        # Assign byte_length to all entries (shared-offset entries get same length)
        for entry in raw_entries:
            entry['byte_length'] = offset_to_length.get(entry['byte_offset'], 0)

        return raw_entries

    except Exception as e:
        logger.error(f"Error parsing index {idx_url}: {e}")
        return []


def generate_cfs_forecast_urls(
    init_date: str,
    run: str,
    max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS,
) -> List[Tuple[str, str, int]]:
    """
    Generate all (grib_url, idx_url, forecast_hour) tuples for a CFS member.

    Parameters:
        init_date: Init date YYYYMMDD
        run: Run hour
        max_forecast_hours: Max forecast hours

    Returns:
        List of (grib_url, idx_url, forecast_hour) tuples
    """
    init_dt = datetime.strptime(f"{init_date}{run}", "%Y%m%d%H")
    init_time_str = init_dt.strftime("%Y%m%d%H")

    urls = []
    for step_hours in range(0, max_forecast_hours + 1, FORECAST_INTERVAL):
        forecast_dt = init_dt + timedelta(hours=step_hours)
        forecast_time_str = forecast_dt.strftime("%Y%m%d%H")

        grib_url = (
            f"s3://{S3_BUCKET}/cfs.{init_date}/{run}/6hrly_grib_{CFS_MEMBER}/"
            f"flxf{forecast_time_str}.{CFS_MEMBER}.{init_time_str}.grb2"
        )
        idx_url = grib_url + ".idx"

        urls.append((grib_url, idx_url, step_hours))

    return urls


def create_references_from_idx(grib_url: str, idx_entries: List[Dict]) -> Dict[str, Any]:
    """
    Create kerchunk-style references from parsed .idx entries.

    Filters to only CFS_VAR_FILTER variables.

    Parameters:
        grib_url: S3 URL to the GRIB file
        idx_entries: Parsed index entries from parse_cfs_idx

    Returns:
        Dict mapping reference keys to [url, offset, length] triplets
    """
    references = {}

    for entry in idx_entries:
        variable = entry['variable']
        level = entry['level']
        offset = entry['byte_offset']
        length = entry.get('byte_length', -1)

        if length <= 0:
            continue

        # Filter: only include variables in our target set
        if variable not in CFS_VAR_FILTER:
            continue

        expected_level = CFS_VAR_FILTER[variable]
        if expected_level not in level:
            continue

        # Build reference key matching zarr convention
        var_lower = variable.lower()
        level_clean = level.replace(' ', '_')
        key = f"{var_lower}/{level_clean}/0.0.0"

        references[key] = [grib_url, offset, length]

    return references


def process_single_timestep(args: Tuple) -> Tuple[int, Dict]:
    """
    Process a single timestep: parse .idx and create references.

    Worker function for ThreadPoolExecutor.

    Parameters:
        args: (grib_url, idx_url, forecast_hour)

    Returns:
        (forecast_hour, references_dict)
    """
    grib_url, idx_url, forecast_hour = args

    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    try:
        idx_entries = parse_cfs_idx(idx_url)
        if not idx_entries:
            return (forecast_hour, {})

        refs = create_references_from_idx(grib_url, idx_entries)
        return (forecast_hour, refs)

    except Exception as e:
        logger.warning(f"Error at f{forecast_hour:04d}: {e}")
        return (forecast_hour, {})


def build_refs_from_indices(
    init_date: str,
    run: str,
    max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS,
    parallel_workers: int = PARALLEL_WORKERS,
) -> Dict[str, Any]:
    """
    Build references for all timesteps by parsing .idx files from S3.

    Uses ThreadPoolExecutor for parallel index reads (I/O-bound).

    Parameters:
        init_date: Init date YYYYMMDD
        run: Run hour
        max_forecast_hours: Max forecast hours
        parallel_workers: Number of parallel workers

    Returns:
        Dict mapping timestep-qualified keys to [url, offset, length] references
    """
    forecast_urls = generate_cfs_forecast_urls(init_date, run, max_forecast_hours)
    logger.info(f"Stage 2: Parsing {len(forecast_urls)} .idx files for "
                f"{init_date} {run}z")

    all_refs = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_hour = {
            executor.submit(process_single_timestep, url_tuple): url_tuple[2]
            for url_tuple in forecast_urls
        }

        for future in as_completed(future_to_hour):
            hour = future_to_hour[future]
            completed += 1

            try:
                forecast_hour, refs = future.result()

                for key, ref in refs.items():
                    if not key.startswith('.'):
                        timestep_key = f"step_{forecast_hour:04d}/{key}"
                    else:
                        timestep_key = key
                    all_refs[timestep_key] = ref

                if completed % 100 == 0 or completed == len(forecast_urls):
                    logger.info(f"  [{completed}/{len(forecast_urls)}] "
                                f"Parsed f{forecast_hour:04d}, "
                                f"total refs: {len(all_refs)}")

            except Exception as e:
                logger.warning(f"  Worker error at f{hour:04d}: {e}")

    logger.info(f"Stage 2 complete: {len(all_refs)} total references "
                f"from {len(forecast_urls)} files")
    return all_refs


def merge_with_template(
    index_refs: Dict,
    template_store: Dict,
) -> Dict:
    """
    Merge fresh byte-range references (Stage 2) into template zarr store (Stage 1).

    Template keys that are metadata (.zattrs, .zarray, etc.) are preserved.
    Data reference keys are updated with fresh byte-range values.

    Parameters:
        index_refs: Fresh references from build_refs_from_indices
        template_store: Zarr store dict from build_deflated_store_from_template

    Returns:
        Merged zarr store dict
    """
    merged = template_store.copy()

    for key, ref in index_refs.items():
        if not key.startswith('_'):
            merged[key] = ref

    logger.info(f"Merged {len(index_refs)} fresh refs into template "
                f"({len(template_store)} keys -> {len(merged)} keys)")
    return merged


# ==============================================================================
# STAGE 3: CREATE FINAL PARQUET + UPLOAD
# ==============================================================================

def create_parquet_simple(zstore: Dict, output_file: Path):
    """
    Save zarr store as parquet file.

    Same format as ECMWF: DataFrame with columns ['key', 'value'] where
    values are UTF-8 encoded bytes.

    Parameters:
        zstore: Zarr store dict
        output_file: Output parquet path

    Returns:
        DataFrame written to parquet
    """
    data = []
    for key, value in zstore.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        else:
            encoded_value = str(value).encode('utf-8')
        data.append((key, encoded_value))

    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_file)
    return df


def upload_to_gcs(output_dir: Path, init_date: str, run: str,
                  target_month: str) -> Optional[str]:
    """
    Upload final parquet files to GCS.

    GCS path: {bucket}/{prefix}/{target_month}/{init_date}/{run}z/

    Parameters:
        output_dir: Local directory containing parquet files
        init_date: Init date YYYYMMDD
        run: Run hour
        target_month: Target forecast month YYYYMM

    Returns:
        GCS path string, or None on failure
    """
    try:
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        gcs_path = (
            f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{target_month}/"
            f"{init_date}/{run}z"
        )

        parquet_files = list(output_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No final parquet files to upload")
            return None

        logger.info(f"Uploading {len(parquet_files)} files to gs://{gcs_path}")

        uploaded = 0
        for pf in parquet_files:
            gcs_file_path = f"{gcs_path}/{pf.name}"
            fs.put(str(pf), gcs_file_path)
            uploaded += 1

        logger.info(f"Uploaded {uploaded}/{len(parquet_files)} files")
        return f"gs://{gcs_path}"

    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        return None


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_cfs_availability(init_date: str, run: str) -> Tuple[bool, int, int]:
    """
    Pre-flight check: verify .idx file exists for the first timestep.

    Parameters:
        init_date: Init date YYYYMMDD
        run: Run hour

    Returns:
        (is_valid, n_messages, n_variables)
    """
    init_dt = datetime.strptime(f"{init_date}{run}", "%Y%m%d%H")
    init_time_str = init_dt.strftime("%Y%m%d%H")
    forecast_time_str = init_time_str  # f000

    idx_url = (
        f"s3://{S3_BUCKET}/cfs.{init_date}/{run}/6hrly_grib_{CFS_MEMBER}/"
        f"flxf{forecast_time_str}.{CFS_MEMBER}.{init_time_str}.grb2.idx"
    )

    try:
        fs = fsspec.filesystem("s3", anon=True)
        idx_path = idx_url.replace("s3://", "")

        if not fs.exists(idx_path):
            logger.warning(f"Index file not found: {idx_url}")
            return False, 0, 0

        # Parse to count messages and variables
        entries = parse_cfs_idx(f"s3://{idx_path}")
        n_messages = len(entries)

        variables = set()
        for entry in entries:
            variables.add(entry['variable'])

        n_variables = len(variables)

        # Check that we have our target variables
        target_vars = set(CFS_VAR_FILTER.keys())
        found_vars = target_vars.intersection(variables)

        if len(found_vars) < len(target_vars):
            missing = target_vars - found_vars
            logger.warning(f"Missing variables: {missing}")

        logger.info(f"Index validation: {n_messages} messages, "
                    f"{n_variables} variables ({len(found_vars)}/{len(target_vars)} targets)")
        return True, n_messages, n_variables

    except Exception as e:
        logger.error(f"Index validation failed: {e}")
        return False, 0, 0


# ==============================================================================
# MAIN PROCESSING FUNCTION (runs on Cloud Run workers via Lithops)
# ==============================================================================

def process_cfs_member(
    init_date: str,
    run: str,
    target_month: str,
    max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS,
    local_template: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> Dict:
    """
    Process a single CFS member (init_date+run) through the three-stage pipeline.

    This function is serialized by Lithops (cloudpickle) and executed on
    Cloud Run workers. Can also run locally with --sequential.

    Parameters:
        init_date: Init date YYYYMMDD
        run: Run hour (00, 06, 12, 18)
        target_month: Target forecast month YYYYMM (for GCS path)
        max_forecast_hours: Max forecast hours to process
        local_template: Path to local template parquet (for sequential testing)
        output_dir_override: Save output locally instead of GCS upload

    Returns:
        Dict with processing results
    """
    # Ensure anonymous S3 access on the worker
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    logger.info(f"Processing CFS member: {init_date} {run}z")
    pipeline_start = time.time()

    result = {
        "init_date": init_date,
        "run": run,
        "target_month": target_month,
        "success": False,
        "message": "",
        "gcs_path": None,
        "output_path": None,
        "total_time_seconds": 0,
    }

    try:
        # Validate index availability
        idx_valid, n_msgs, n_vars = validate_cfs_availability(init_date, run)
        if not idx_valid:
            result["message"] = f"Index validation failed ({n_vars} variables found)"
            return result

        # Ensure template is available
        template_path = ensure_template(local_template)

        # Stage 1: Load zarr structure from template
        deflated_store = build_deflated_store_from_template(
            template_path, init_date, run
        )
        if not deflated_store:
            result["message"] = "Stage 1 failed: template loading error"
            return result

        # Stage 2: Parse .idx files + build references
        stage2_refs = build_refs_from_indices(
            init_date, run, max_forecast_hours
        )
        if not stage2_refs:
            result["message"] = "Stage 2 failed: no references built"
            return result

        # Merge with template
        final_store = merge_with_template(stage2_refs, deflated_store)

        # Stage 3: Create final parquet
        if output_dir_override:
            output_dir = Path(output_dir_override)
        else:
            output_dir = Path(f"/tmp/cfs_{init_date}_{run}z")
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_file = output_dir / f"{init_date}{run}z.parquet"
        create_parquet_simple(final_store, parquet_file)

        # Upload to GCS or keep locally
        gcs_path = None
        if not output_dir_override:
            gcs_path = upload_to_gcs(output_dir, init_date, run, target_month)
            shutil.rmtree(output_dir, ignore_errors=True)

        total_time = time.time() - pipeline_start

        result.update({
            "success": True,
            "message": (f"Processed {init_date} {run}z in {total_time:.1f}s "
                        f"({len(stage2_refs)} refs)"),
            "gcs_path": gcs_path,
            "output_path": str(parquet_file) if output_dir_override else None,
            "total_time_seconds": round(total_time, 1),
            "stage2_refs": len(stage2_refs),
            "final_keys": len(final_store),
        })

        logger.info(f"Pipeline complete: {result['message']}")
        if output_dir_override:
            logger.info(f"Output saved locally: {parquet_file}")
        return result

    except Exception as e:
        total_time = time.time() - pipeline_start
        result["message"] = f"Pipeline error: {str(e)}"
        result["total_time_seconds"] = round(total_time, 1)
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return result


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def generate_ensemble_members(target_month: str) -> List[Tuple[str, str, str]]:
    """
    Generate all ensemble members for a target month.

    CFS uses multi-date ensemble: init dates from the preceding month.
    31 days x 4 runs = up to 124 members.

    Parameters:
        target_month: Target forecast month YYYYMM

    Returns:
        List of (init_date, run, target_month) tuples
    """
    target_year = int(target_month[:4])
    target_mon = int(target_month[4:6])

    # Preceding month
    if target_mon == 1:
        init_year = target_year - 1
        init_mon = 12
    else:
        init_year = target_year
        init_mon = target_mon - 1

    _, last_day = monthrange(init_year, init_mon)

    members = []
    for day in range(1, last_day + 1):
        init_date = f"{init_year}{init_mon:02d}{day:02d}"
        for run in ["00", "06", "12", "18"]:
            members.append((init_date, run, target_month))

    logger.info(f"Generated {len(members)} ensemble members for target {target_month} "
                f"(init dates: {init_year}-{init_mon:02d}-01 to "
                f"{init_year}-{init_mon:02d}-{last_day})")
    return members


def run_sequential(
    members: List[Tuple[str, str, str]],
    max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS,
    local_template: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """Run processing sequentially (no Lithops, for local testing)."""
    print(f"\nRunning sequential processing for {len(members)} members")
    if local_template:
        print(f"Using local template: {local_template}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print("=" * 70)

    results = []

    for i, (init_date, run, target_month) in enumerate(members, 1):
        print(f"\n[{i}/{len(members)}] Processing {init_date} {run}z...")
        start_time = time.time()

        result = process_cfs_member(
            init_date, run, target_month, max_forecast_hours,
            local_template=local_template,
            output_dir_override=output_dir,
        )

        elapsed = time.time() - start_time
        status = "OK" if result.get('success') else "FAIL"

        print(f"{status} {init_date} {run}z: {result.get('message', 'Unknown')}")
        print(f"   Time: {elapsed:.1f}s")

        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")

        results.append(result)

    return results


def run_with_lithops(
    members: List[Tuple[str, str, str]],
    max_forecast_hours: int = DEFAULT_MAX_FORECAST_HOURS,
    max_workers: int = 10,
) -> List[Dict]:
    """
    Run processing in parallel using Lithops Cloud Run backend, with retry.

    Retry strategy handles two failure modes:
      1. Invoke failure — Cloud Run HTTP 500 during cold-start race
      2. Exec failure — Worker ran but process_cfs_member() returned success=False

    Up to MAX_RETRIES attempts with RETRY_DELAY seconds between retries.
    """
    try:
        import lithops
    except ImportError:
        print("Error: lithops not installed. Install with: pip install lithops")
        sys.exit(1)

    MAX_RETRIES = 3
    WAIT_TIMEOUT = 1200   # 20 min — CFS has many more timesteps than ECMWF
    RETRY_DELAY = 30      # seconds between retries

    print(f"\nRunning Lithops parallel processing for {len(members)} members")
    print(f"Max workers: {max_workers}  |  Max retries: {MAX_RETRIES}")
    print("=" * 70)

    config_path = Path(__file__).parent / 'lithops_config.yaml'
    if config_path.exists():
        print(f"Config: {config_path}")
        fexec = lithops.FunctionExecutor(config_file=str(config_path))
    else:
        print("Config: using default lithops config")
        fexec = lithops.FunctionExecutor(backend='gcp_cloudrun')

    overall_start = time.time()

    # Keyed by (init_date, run) so retries overwrite earlier failed entries
    all_results: Dict[Tuple[str, str], Dict] = {}
    remaining = list(members)

    for attempt in range(1, MAX_RETRIES + 1):
        if not remaining:
            break

        if attempt > 1:
            print(f"\n--- Retry {attempt}/{MAX_RETRIES}: {len(remaining)} members "
                  f"— pausing {RETRY_DELAY}s for Cloud Run warm-up ---")
            time.sleep(RETRY_DELAY)

        print(f"\nAttempt {attempt}: dispatching {len(remaining)} worker(s)...")

        # Lithops map expects a function with a single iterable arg
        # We pass tuples and unpack in the function
        def _worker(member_tuple):
            init_date, run, target_month = member_tuple
            return process_cfs_member(
                init_date, run, target_month, max_forecast_hours
            )

        futures = fexec.map(_worker, remaining)

        try:
            fexec.wait(futures, throw_except=False, timeout=WAIT_TIMEOUT)
        except TimeoutError:
            logger.warning(
                f"Attempt {attempt}: wait timeout ({WAIT_TIMEOUT}s) — "
                f"collecting completed results, remaining will be retried"
            )

        still_failing = []
        for i, f in enumerate(futures):
            init_date, run, target_month = remaining[i]
            member_key = (init_date, run)

            if f.status == 'success':
                try:
                    result = f.result()
                    if isinstance(result, dict) and result.get('success'):
                        all_results[member_key] = result
                        logger.info(f"  OK   {init_date} {run}z: "
                                    f"{result.get('message', '')}")
                    else:
                        msg = (result.get('message', 'unknown')
                               if isinstance(result, dict) else str(result))
                        all_results[member_key] = result if isinstance(result, dict) else {
                            'init_date': init_date, 'run': run,
                            'success': False, 'message': msg
                        }
                        still_failing.append(remaining[i])
                        logger.warning(f"  FAIL {init_date} {run}z: "
                                       f"exec error ({msg}), will retry")
                except Exception as exc:
                    all_results[member_key] = {
                        'init_date': init_date, 'run': run,
                        'success': False, 'message': str(exc)
                    }
                    still_failing.append(remaining[i])
                    logger.warning(f"  FAIL {init_date} {run}z: "
                                   f"result error ({exc}), will retry")
            else:
                msg = f"invoke failed (status={f.status})"
                all_results[member_key] = {
                    'init_date': init_date, 'run': run,
                    'success': False, 'message': msg
                }
                still_failing.append(remaining[i])
                logger.warning(f"  FAIL {init_date} {run}z: {msg}, will retry")

        # GCS verification: rescue members whose worker completed and uploaded
        # to GCS but whose f.status was not updated (network interruption)
        if still_failing:
            try:
                import gcsfs
                _fs = gcsfs.GCSFileSystem()
                gcs_rescued = []
                for member_tuple in still_failing:
                    init_date, run, target_month = member_tuple
                    gcs_prefix = (
                        f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{target_month}/"
                        f"{init_date}/{run}z"
                    )
                    try:
                        items = _fs.ls(gcs_prefix)
                        if items:
                            all_results[(init_date, run)] = {
                                'init_date': init_date, 'run': run,
                                'success': True,
                                'gcs_path': f"gs://{gcs_prefix}",
                                'message': ('GCS-verified '
                                            '(f.status stale — monitor interrupted)'),
                            }
                            gcs_rescued.append(member_tuple)
                            logger.info(f"  GCS-OK {init_date} {run}z: "
                                        f"found {len(items)} file(s)")
                    except Exception:
                        pass
                if gcs_rescued:
                    still_failing = [
                        m for m in still_failing if m not in set(gcs_rescued)
                    ]
                    print(f"  GCS verification rescued {len(gcs_rescued)} member(s)")
            except Exception as gcs_exc:
                logger.warning(f"  GCS verification skipped: {gcs_exc}")

        succeeded = len(remaining) - len(still_failing)
        print(f"Attempt {attempt} done: {succeeded}/{len(remaining)} succeeded, "
              f"{len(still_failing)} queued for retry")
        remaining = still_failing

    # Any members that exhausted all retries
    for member_tuple in remaining:
        init_date, run, target_month = member_tuple
        all_results[(init_date, run)] = {
            'init_date': init_date, 'run': run,
            'success': False, 'message': 'Max retries exceeded'
        }
        logger.error(f"  GIVE UP {init_date} {run}z: "
                     f"max retries ({MAX_RETRIES}) exceeded")

    total_time = time.time() - overall_start
    success_count = sum(1 for r in all_results.values() if r.get('success'))
    print(f"\nCompleted in {total_time:.1f}s ({total_time/60:.1f} min) — "
          f"{success_count}/{len(members)} successful")

    # Return in original member order
    return [
        all_results.get(
            (m[0], m[1]),
            {'init_date': m[0], 'run': m[1], 'success': False, 'message': 'No result'}
        )
        for m in members
    ]


# ==============================================================================
# SUMMARY
# ==============================================================================

def print_summary(members: List[Tuple], results: List[Dict], total_time: float):
    """Print execution summary."""
    success_count = sum(1 for r in results if r.get('success'))
    failed_count = len(results) - success_count

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal members: {len(members)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if success_count > 0:
        avg_time = (
            sum(r.get('total_time_seconds', 0) for r in results if r.get('success'))
            / success_count
        )
        print(f"Average processing time: {avg_time:.1f}s ({avg_time/60:.1f} min)")

    # Show first 20 results
    print("\nResults (first 20):")
    print("-" * 70)

    for member, result in zip(members[:20], results[:20]):
        init_date, run = member[0], member[1]
        status = "OK" if result.get('success') else "FAIL"
        message = result.get('message', 'No message')

        print(f"{status} {init_date} {run}z: {message}")

        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")

    if len(members) > 20:
        print(f"   ... and {len(members) - 20} more")

    # GCS locations
    gcs_paths = [r.get('gcs_path') for r in results if r.get('gcs_path')]
    if gcs_paths:
        print(f"\nGCS OUTPUT ({len(gcs_paths)} locations):")
        for path in gcs_paths[:5]:
            print(f"  {path}")
        if len(gcs_paths) > 5:
            print(f"  ... and {len(gcs_paths) - 5} more")
        print(f"\nList all parquets:")
        print(f"  gsutil ls gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")


# ==============================================================================
# CLI + MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run CFS parquet generation via Lithops Cloud Run workers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Target selection (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        '--target-month', type=str,
        help='Target forecast month YYYYMM (generates 124-member ensemble '
             'from preceding month init dates)'
    )
    target_group.add_argument(
        '--init-date', type=str,
        help='Single init date YYYYMMDD (for testing)'
    )

    parser.add_argument('--run', type=str, default='00',
                        choices=['00', '06', '12', '18'],
                        help='Run hour (default: 00, used with --init-date)')
    parser.add_argument('--max-forecast-hours', type=int,
                        default=DEFAULT_MAX_FORECAST_HOURS,
                        help=f'Max forecast hours (default: {DEFAULT_MAX_FORECAST_HOURS})')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Max parallel workers for Lithops (default: 10)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially without Lithops (local testing)')
    parser.add_argument('--local-template',
                        help='Path to local zarr template parquet (from '
                             'run_cfs_template_creation.py --zarr-template)')
    parser.add_argument('--output-dir',
                        help='Save output locally instead of GCS upload')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show members without executing')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip interactive confirmation')

    args = parser.parse_args()

    # Build member list
    if args.target_month:
        members = generate_ensemble_members(args.target_month)
    else:
        # Single member — target_month is inferred from init_date + 1 month
        init_dt = datetime.strptime(args.init_date, "%Y%m%d")
        if init_dt.month == 12:
            target_month = f"{init_dt.year + 1}01"
        else:
            target_month = f"{init_dt.year}{init_dt.month + 1:02d}"
        members = [(args.init_date, args.run, target_month)]

    # Print configuration
    print("=" * 70)
    print("CFS PARQUET GENERATION - LITHOPS CLOUD RUN")
    print("=" * 70)
    print(f"\nMembers to process: {len(members)}")
    print(f"  First: {members[0][0]} {members[0][1]}z")
    print(f"  Last:  {members[-1][0]} {members[-1][1]}z")
    print(f"Max forecast hours: {args.max_forecast_hours}")
    print(f"Execution mode: "
          f"{'Sequential (local)' if args.sequential else f'Lithops Cloud Run (max {args.max_workers} workers)'}")
    if args.local_template:
        print(f"Local template: {args.local_template}")
    if args.output_dir:
        print(f"Local output: {args.output_dir}")
    else:
        print(f"GCS output: gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")

    if args.dry_run:
        print(f"\nDRY RUN — Members that would be processed:")
        for init_date, run, target_month in members[:20]:
            print(f"  {init_date} {run}z -> {target_month}")
        if len(members) > 20:
            print(f"  ... and {len(members) - 20} more")
        return

    # Confirm before processing large batches
    if len(members) > 5 and not args.yes:
        response = input(f"\nProcess {len(members)} members? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Execute
    start_time = time.time()

    if args.sequential:
        results = run_sequential(
            members, args.max_forecast_hours,
            local_template=args.local_template,
            output_dir=args.output_dir,
        )
    else:
        results = run_with_lithops(
            members, args.max_forecast_hours, args.max_workers
        )

    total_time = time.time() - start_time

    # Print summary
    print_summary(members, results, total_time)


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
