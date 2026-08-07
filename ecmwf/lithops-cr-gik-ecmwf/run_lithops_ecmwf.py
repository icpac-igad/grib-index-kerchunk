#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lithops==3.6.4",  # MUST match the deployed Cloud Run runtime (Dockerfile pins ==3.6.4)
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
Lithops-native ECMWF Parquet Generation (Cloud Run)
====================================================

Processes ECMWF weather data through the GIK three-stage pipeline using
Lithops to directly execute Python functions on Cloud Run workers.

Lithops serializes the processing function via cloudpickle, uploads it to
GCS, and Cloud Run workers download and execute it. No Flask app, no
separate worker service - Lithops manages the entire lifecycle.

Architecture:
    run_lithops_ecmwf.py (this script, runs locally via uv run)
         |
         v
    [Lithops FunctionExecutor(backend='gcp_cloudrun')]
      /    |    \\
     /     |     \\  (cloudpickle-serialized functions via GCS)
    v      v      v
  [CR 1]  [CR 2]  ... [CR N]  (Lithops proxy containers on Cloud Run)
    |      |           |
    v      v           v
  process_ecmwf_date() runs directly inside each container:
    Stage 1: Load zarr from HuggingFace template (~5s)
    Stage 2: Fetch S3 .index files + merge (~5-15 min)
    Stage 3: Create final parquet files (~1-2 min)
    Upload to GCS

Usage:
    # Build custom runtime (one-time setup)
    lithops runtime build -b gcp_cloudrun -f Dockerfile ecmwf-runtime

    # Single date
    uv run run_lithops_ecmwf.py --date 20260206

    # Last 7 days
    uv run run_lithops_ecmwf.py --days-back 7

    # Date range
    uv run run_lithops_ecmwf.py --start-date 20260201 --end-date 20260207

    # Custom parallelism
    uv run run_lithops_ecmwf.py --days-back 30 --max-workers 20

    # Dry run (show dates without executing)
    uv run run_lithops_ecmwf.py --days-back 7 --dry-run

    # Sequential (no Lithops, for local testing)
    uv run run_lithops_ecmwf.py --date 20260206 --sequential

Configuration:
    Uses lithops_config.yaml in current directory, or set LITHOPS_CONFIG_FILE.

Author: ICPAC GIK Team
Date: 2026-02-12
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

GCS_BUCKET = os.environ.get('GCS_BUCKET', 'gik-ecmwf-aws-tf')
# v20260623 = versioned catalog of the per-level-keys-FIXED reprocess (4ca1c21).
# The legacy 'run_par_ecmwf' path holds pre-fix (collapsed pressure-level) data.
GCS_PARQUET_PREFIX = os.environ.get('GCS_PARQUET_PREFIX', 'v20260623_run_par_ecmwf')
PARALLEL_WORKERS = int(os.environ.get('PARALLEL_WORKERS', '4'))

TEMPLATE_URL = os.environ.get(
    'TEMPLATE_URL',
    'https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main/gik-fmrc-v2ecmwf_fmrc.tar.gz'
)

# ECMWF forecast hours
HOURS_3H = list(range(0, 145, 3))       # 0-144h at 3h intervals (49 steps)
HOURS_6H = list(range(150, 361, 6))     # 150-360h at 6h intervals (36 steps)
ALL_FORECAST_HOURS = HOURS_3H + HOURS_6H  # Total: 85 steps


def forecast_hours_for_run(run: str) -> List[int]:
    """Forecast hours published for a given cycle.

    00z/12z are full-length ensembles (0-360h, 85 steps); 06z/18z are short
    (0-144h, 49 steps) -- S3-verified: +150h and beyond are 404 for those cycles.
    Asking for the missing 36 steps still costs 36 x 51 = 1836 futile ranged GETs
    per date against a bucket that throttles, so scope the loop to the cycle.
    """
    return ALL_FORECAST_HOURS if str(run).zfill(2) in ('00', '12') else HOURS_3H

REFERENCE_DATE = os.environ.get('ECMWF_REFERENCE_DATE', '20240529')
S3_BUCKET = "ecmwf-forecasts"

# Source-path resolution, selected at deploy time (same per-era image
# strategy as the 49r1/50r1 cutover -- one env var, no per-date logic):
#   0p25 (default): ifs/0p25/...   grid 721x1440, >= 2024-02-29
#   0p4           : 0p4-beta/...    grid 451x900, 2023-01-18 .. 2024-02-28
# A 0p4 deployment also sets ECMWF_REFERENCE_DATE to a 0.4deg date and
# TEMPLATE_URL / ECMWF_TEMPLATE_PATH to the 0.4deg template artifact.
ECMWF_RESOLUTION = os.environ.get('ECMWF_RESOLUTION', '0p25').lower()
STREAM_PATH = '0p4-beta' if ECMWF_RESOLUTION in ('0p4', '0p40', '0p4-beta') else 'ifs/0p25'

# Which stream carries the CONTROL member.
#   0p4-beta / 49r1: control is bundled in enfo/ef as number=0 -> 'enfo'.
#   50r1 (>= 2026-05-12 06z): control moved to oper/fc -> set 'oper'.
# Perturbed members 1..50 always come from enfo/ef. Deploy-time, no per-date
# logic (matches the per-era image strategy).
CONTROL_STREAM = os.environ.get('ECMWF_CONTROL_STREAM', 'enfo').lower()
logger.info(f"Source resolution: {ECMWF_RESOLUTION} (path segment '{STREAM_PATH}'), "
            f"reference date {REFERENCE_DATE}, control stream '{CONTROL_STREAM}'")


def ecmwf_index_url(date_str: str, run: str, hour: int, stream: str = 'enfo') -> str:
    """Build the S3 .index URL for one (date, run, hour), resolution-aware.

    0.25deg -> s3://ecmwf-forecasts/{date}/{run}z/ifs/0p25/{stream}/...
    0.4deg  -> s3://ecmwf-forecasts/{date}/{run}z/0p4-beta/{stream}/...
    stream 'enfo' -> '-enfo-ef' (perturbed + bundled control for 0p4/49r1);
    stream 'oper' -> '-oper-fc' (50r1 control). The companion GRIB URL is
    this with '.index' stripped (consumer appends '.grib2' by convention).
    """
    suffix = 'oper-fc' if stream == 'oper' else 'enfo-ef'
    return (
        f"s3://{S3_BUCKET}/{date_str}/{run}z/{STREAM_PATH}/{stream}/"
        f"{date_str}{run}0000-{hour}h-{suffix}.index"
    )


# Use baked-in template from Docker image if available, fallback to /tmp download
TEMPLATE_CACHE_PATH = os.environ.get(
    'ECMWF_TEMPLATE_PATH',
    '/tmp/gik-fmrc-v2ecmwf_fmrc.tar.gz'
)


# ==============================================================================
# STAGE 1: LOAD ZARR STRUCTURE FROM TEMPLATE
# ==============================================================================

def ensure_template() -> str:
    """Get template path (pre-baked in image or download to /tmp)."""
    # Check if template is baked into the image (Cloud Run workers)
    baked_path = os.environ.get('ECMWF_TEMPLATE_PATH')
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


def build_deflated_stores_from_template(
    template_tar_path: str,
    template_date: str = REFERENCE_DATE,
    max_members: Optional[int] = None
) -> Optional[Dict]:
    """
    Build deflated_stores from the HuggingFace template archive.
    Replaces slow scan_grib (~73 min) with direct template loading (~2-5s).
    """
    logger.info("Stage 1: Loading zarr structure from template")
    start_time = time.time()

    all_members = ['ens_control'] + [f'ens_{i:02d}' for i in range(1, 51)]
    if max_members:
        all_members = all_members[:max_members]

    logger.info(f"Loading {len(all_members)} members from template")
    deflated_stores = {}

    try:
        with tarfile.open(template_tar_path, 'r:gz') as tar:
            for member_dir in all_members:
                if member_dir == 'ens_control':
                    member_key = 'control'
                    filename_member = 'control'
                else:
                    num = int(member_dir.replace('ens_', ''))
                    member_key = f'ens_{num:02d}'
                    filename_member = f'ens{num:02d}'

                tar_member_path = (
                    f"gik-fmrc/v2ecmwf_fmrc/{member_dir}/"
                    f"ecmwf-{template_date}00-{filename_member}-rt000.par"
                )

                try:
                    member_info = tar.getmember(tar_member_path)
                except KeyError:
                    logger.warning(f"Template not found for {member_key}")
                    continue

                f = tar.extractfile(member_info)
                if f is None:
                    continue

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

                deflated_stores[member_key] = zstore

        elapsed = time.time() - start_time
        logger.info(f"Stage 1 complete: {len(deflated_stores)} members in {elapsed:.1f}s")
        return deflated_stores

    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 2: INDEX FILES + TEMPLATE MERGE
# ==============================================================================

def parse_grib_index(idx_url: str, member_filter: Optional[str] = None) -> List[Dict]:
    """Parse ECMWF GRIB index file to extract byte ranges and metadata."""
    try:
        fs = fsspec.filesystem("s3", anon=True)
        entries = []

        with fs.open(idx_url, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                entry_data = json.loads(line.strip())

                member_num = int(entry_data.get('number', 0))
                if member_num == 0:
                    member = 'control'
                else:
                    member = f'ens{member_num:02d}'

                if member_filter and member != member_filter:
                    continue

                entry = {
                    'byte_offset': entry_data['_offset'],
                    'byte_length': entry_data['_length'],
                    'variable': entry_data.get('param', ''),
                    'level': entry_data.get('levtype', ''),
                    # `levelist` is the actual level value (e.g. '250' for pl,
                    # layer string for sol). Pre-fix this was dropped and the
                    # key collapsed all 13/14 isobaric levels into one per
                    # (var, step, member) -- the cGAN per-level bug. See
                    # cGAN_tutorial GIK_PARQUET_PER_LEVEL_KEYS_NEEDED.md and
                    # 2026-05-29-per-level-keys-fix-and-template-implications.md.
                    'level_value': str(entry_data.get('levelist', '')),
                    'step': entry_data.get('step', '0'),
                    'member': member,
                    'date': entry_data.get('date', ''),
                    'time': entry_data.get('time', ''),
                    'line_num': line_num
                }
                entries.append(entry)

        return entries

    except Exception as e:
        logger.error(f"Error parsing index {idx_url}: {e}")
        return []


def create_references_from_index(grib_url: str, idx_entries: List[Dict]) -> Dict[str, Any]:
    """Create kerchunk references using index byte ranges."""
    references = {}

    for entry in idx_entries:
        start = entry['byte_offset']
        length = entry['byte_length']

        if length == -1:
            continue

        var_name = entry['variable'].lower().replace(' ', '_')
        level_name = entry['level'].replace(' ', '_')
        member_name = entry['member']

        # Per-pressure-level fix: pl messages get the hPa value embedded in
        # the key so each isobaric level becomes its own reference; sfc/sol
        # keep their old shape. See cGAN_tutorial GIK_MAINTAINER_REQUEST.md.
        # This mirrors the same fix landed in ecmwf/ecmwf_index_processor.py
        # via 4ca1c21 -- the inline copy here is what the Cloud Run runtime
        # actually executes (run_lithops_ecmwf.py is self-contained per
        # CLAUDE.md design), so the ecmwf_index_processor.py fix alone did
        # NOT reach production. This edit closes that gap.
        if level_name == 'pl' and entry.get('level_value'):
            key = f"{var_name}/pl/{entry['level_value']}/{member_name}/0.0.0"
        else:
            key = f"{var_name}/{level_name}/{member_name}/0.0.0"
        references[key] = [grib_url, start, length]

    references['.zgroup'] = json.dumps({"zarr_format": 2})
    return references


def build_refs_from_indices(
    date_str: str,
    run: str,
    member_name: str,
    hours: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Build references for a member using S3 index files."""
    if hours is None:
        hours = forecast_hours_for_run(run)

    all_refs = {}

    for hour in hours:
        try:
            # Include run hour in timestamp: 00z→000000, 18z→180000.
            # Resolution-aware path (ifs/0p25 vs 0p4-beta) via ECMWF_RESOLUTION.
            # Control reads from CONTROL_STREAM (oper for 50r1, else enfo);
            # perturbed members always from enfo/ef.
            stream = CONTROL_STREAM if member_name == 'control' else 'enfo'
            idx_url = ecmwf_index_url(date_str, run, hour, stream=stream)
            grib_url = idx_url.replace('.index', '')

            idx_entries = parse_grib_index(idx_url, member_filter=member_name)
            if not idx_entries:
                continue

            hour_refs = create_references_from_index(grib_url, idx_entries)

            for key, ref in hour_refs.items():
                if not key.startswith('.'):
                    timestep_key = f"step_{hour:03d}/{key}"
                else:
                    timestep_key = key
                all_refs[timestep_key] = ref

        except Exception as e:
            logger.warning(f"Error at hour {hour}: {e}")

    return all_refs


def merge_with_local_template(
    index_refs: Dict,
    template_date: str,
    member_name: str,
    template_tar_path: str
) -> Dict:
    """Merge index-based references with template structure from tar.gz."""
    try:
        if member_name == 'control':
            member_dir = 'ens_control'
            filename_member = 'control'
        else:
            if member_name.startswith('ens'):
                member_num_str = member_name.replace('ens', '')
            else:
                member_num_str = member_name
            member_dir = f'ens_{int(member_num_str):02d}'
            filename_member = f'ens{int(member_num_str):02d}'

        tar_member_path = (
            f"gik-fmrc/v2ecmwf_fmrc/{member_dir}/"
            f"ecmwf-{template_date}00-{filename_member}-rt000.par"
        )

        extract_dir = tempfile.mkdtemp(prefix="ecmwf_template_")
        try:
            with tarfile.open(template_tar_path, 'r:gz') as tar:
                try:
                    member_info = tar.getmember(tar_member_path)
                except KeyError:
                    logger.warning(f"Template not found: {tar_member_path}")
                    return index_refs

                tar.extract(member_info, path=extract_dir)

            extracted_parquet_path = Path(extract_dir) / tar_member_path
            template_df = pd.read_parquet(extracted_parquet_path)

            template_refs = {}
            for _, row in template_df.iterrows():
                key = row['key']
                value = row['value']
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                template_refs[key] = value

            merged_refs = template_refs.copy()
            for key, value in index_refs.items():
                if not key.startswith('_'):
                    merged_refs[key] = value

            return merged_refs
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    except Exception as e:
        logger.warning(f"Template merge failed: {e}")
        return index_refs


def process_single_member_stage2(args: tuple) -> Tuple:
    """Worker function for parallel Stage 2 processing."""
    member, member_normalized, date_str, run, template_tar_path = args

    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    try:
        refs = build_refs_from_indices(
            date_str=date_str,
            run=run,
            member_name=member_normalized,
            hours=forecast_hours_for_run(run)
        )

        if refs:
            refs = merge_with_local_template(
                refs, REFERENCE_DATE, member_normalized, template_tar_path
            )
            return (member, refs)

        return (member, None)

    except Exception as e:
        return (member, None, str(e))


def run_stage2(
    date_str: str,
    run: str,
    test_members: List[str],
    template_tar_path: str,
    parallel_workers: int = PARALLEL_WORKERS
) -> Optional[Dict]:
    """Stage 2: Index + template merge for all members."""
    logger.info(f"Stage 2: Processing {len(test_members)} members (workers={parallel_workers})")
    start_time = time.time()

    task_args = []
    for member in test_members:
        if member == 'control':
            member_normalized = 'control'
        else:
            member_normalized = member.replace('_', '')
            if member_normalized.startswith('ens'):
                member_num_str = member_normalized.replace('ens', '')
                member_normalized = f'ens{int(member_num_str):02d}'

        task_args.append((member, member_normalized, date_str, run, template_tar_path))

    member_results = {}

    if parallel_workers > 1 and len(test_members) > 1:
        n_workers = min(parallel_workers, len(test_members))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_member = {
                executor.submit(process_single_member_stage2, args): args[0]
                for args in task_args
            }

            completed = 0
            for future in as_completed(future_to_member):
                member_name = future_to_member[future]
                completed += 1
                try:
                    result = future.result()
                    if len(result) == 2:
                        member, refs = result
                        if refs:
                            member_results[member] = refs
                            logger.info(f"[{completed}/{len(test_members)}] {member}: {len(refs)} refs")
                    else:
                        member, _, error = result
                        logger.warning(f"[{completed}/{len(test_members)}] {member}: {error}")
                except Exception as e:
                    logger.error(f"Worker error {member_name}: {e}")
    else:
        for args in task_args:
            result = process_single_member_stage2(args)
            if len(result) == 2:
                member, refs = result
                if refs:
                    member_results[member] = refs
                    logger.info(f"{member}: {len(refs)} refs")

    elapsed = time.time() - start_time
    logger.info(f"Stage 2 complete: {len(member_results)}/{len(test_members)} members in {elapsed:.1f}s")
    return member_results


# ==============================================================================
# STAGE 3: CREATE FINAL ZARR STORE
# ==============================================================================

def create_parquet_simple(zstore: Dict, output_file: Path):
    """Save zarr store as parquet."""
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


def run_stage3(
    deflated_stores: Dict,
    stage2_refs: Dict,
    date_str: str,
    run: str,
    output_dir: Path
) -> Optional[Dict]:
    """Stage 3: Merge deflated stores with Stage 2 refs, create final parquets."""
    logger.info("Stage 3: Creating final zarr stores")
    start_time = time.time()

    results = {}

    for member in deflated_stores.keys():
        if member not in stage2_refs:
            continue

        deflated_store = deflated_stores[member]
        complete_refs = stage2_refs[member]

        if isinstance(deflated_store, dict) and 'refs' in deflated_store:
            final_store = deflated_store.get('refs', {}).copy()
        else:
            final_store = deflated_store.copy()

        for key, ref in complete_refs.items():
            if not key.startswith('_'):
                final_store[key] = ref

        # Filename format: {date_str}{run}z-{member}.parquet (e.g., 2026021000z-control.parquet)
        stage3_output = output_dir / f"{date_str}{run}z-{member}.parquet"
        create_parquet_simple(final_store, stage3_output)

        results[member] = (final_store, stage3_output)

    elapsed = time.time() - start_time
    logger.info(f"Stage 3 complete: {len(results)} members in {elapsed:.1f}s")
    return results


# ==============================================================================
# GCS UPLOAD
# ==============================================================================

def upload_to_gcs(output_dir: Path, date_str: str, run: str) -> Optional[str]:
    """Upload final parquet files to GCS."""
    try:
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        # GCS path: {bucket}/{prefix}/YYYY/MM/YYYYMMDD/{run}z/ (e.g., .../2026/02/20260210/00z/)
        year = date_str[:4]
        month = date_str[4:6]
        gcs_path = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{year}/{month}/{date_str}/{run}z"

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

def validate_index_availability(date_str: str, run: str) -> Tuple[bool, int, int]:
    """Check that .index files are available on S3 for the target date."""
    # Include run hour in timestamp: 00z→000000, 18z→180000.
    # Resolution-aware path (ifs/0p25 vs 0p4-beta) via ECMWF_RESOLUTION.
    idx_url = ecmwf_index_url(date_str, run, 0, stream='enfo')

    try:
        fs = fsspec.filesystem("s3", anon=True)

        if not fs.exists(idx_url):
            logger.warning(f"Index file not found: {idx_url}")
            return False, 0, 0

        with fs.open(idx_url, 'r') as f:
            lines = f.readlines()

        members = set()
        for line in lines:
            data = json.loads(line.strip().rstrip(','))
            members.add(int(data.get('number', -1)))

        n_messages = len(lines)
        n_members = len(members)

        logger.info(f"Index validation: {n_messages} messages, {n_members} members")

        if n_members < 50:
            logger.warning(f"Expected 51 members, found {n_members}")
            return False, n_messages, n_members

        return True, n_messages, n_members

    except Exception as e:
        logger.error(f"Index validation failed: {e}")
        return False, 0, 0


# ==============================================================================
# MAIN PROCESSING FUNCTION (runs on Cloud Run workers via Lithops)
# ==============================================================================

def process_ecmwf_date(date_str: str, run: str = "00") -> Dict:
    """
    Process a single ECMWF date through the three-stage GIK pipeline.

    This function is serialized by Lithops (cloudpickle) and executed on
    Cloud Run workers. All helper functions and module-level constants are
    captured in the serialization.

    Args:
        date_str: Date in YYYYMMDD format
        run: Model run hour (00 or 12)

    Returns:
        Dictionary with processing results
    """
    # Ensure anonymous S3 access on the worker
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    logger.info(f"Processing {date_str} {run}z")
    pipeline_start = time.time()

    result = {
        "date": date_str,
        "run": run,
        "success": False,
        "message": "",
        "gcs_path": None,
        "files_count": 0,
        "total_time_seconds": 0
    }

    try:
        # Validate index availability
        idx_valid, n_msgs, n_members = validate_index_availability(date_str, run)
        if not idx_valid:
            result["message"] = f"Index validation failed ({n_members} members found)"
            return result

        # Ensure template is available
        template_path = ensure_template()

        # Stage 1: Load from template
        deflated_stores = build_deflated_stores_from_template(
            template_path, REFERENCE_DATE
        )
        if not deflated_stores:
            result["message"] = "Stage 1 failed: template loading error"
            return result

        test_members = sorted(deflated_stores.keys())

        # Stage 2: Index + template merge
        stage2_refs = run_stage2(date_str, run, test_members, template_path)
        if not stage2_refs:
            result["message"] = "Stage 2 failed: no members processed"
            return result

        # Stage 3: Create final parquets
        output_dir = Path(f"/tmp/ecmwf_{date_str}_{run}z")
        output_dir.mkdir(parents=True, exist_ok=True)

        stage3_results = run_stage3(deflated_stores, stage2_refs, date_str, run, output_dir)
        if not stage3_results:
            result["message"] = "Stage 3 failed"
            return result

        # Upload to GCS
        gcs_path = upload_to_gcs(output_dir, date_str, run)

        # Cleanup /tmp
        shutil.rmtree(output_dir, ignore_errors=True)

        total_time = time.time() - pipeline_start

        result.update({
            "success": True,
            "message": f"Processed {len(stage3_results)} members in {total_time:.1f}s",
            "gcs_path": gcs_path,
            "files_count": len(stage3_results),
            "total_time_seconds": round(total_time, 1),
            "members_processed": len(stage3_results),
            "stage1_members": len(deflated_stores),
            "stage2_members": len(stage2_refs),
            "stage3_members": len(stage3_results)
        })

        logger.info(f"Pipeline complete: {result['message']}")
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

def generate_date_list(args) -> List[str]:
    """Generate list of dates based on command-line arguments."""
    dates = []

    if args.date:
        dates = [args.date]
    elif args.days_back:
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).strftime("%Y%m%d")
                 for i in range(args.days_back)]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y%m%d")
        end = datetime.strptime(args.end_date, "%Y%m%d")
        dates = [(start + timedelta(days=i)).strftime("%Y%m%d")
                 for i in range((end - start).days + 1)]
    else:
        raise ValueError("Specify --date, --days-back, or --start-date/--end-date")

    return dates


def run_sequential(dates: List[str], run: str = "00") -> List[Dict]:
    """Run processing sequentially (no Lithops, for local testing)."""
    print(f"\nRunning sequential processing for {len(dates)} dates")
    print("=" * 70)

    results = []

    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] Processing {date}...")
        start_time = time.time()

        result = process_ecmwf_date(date, run)

        elapsed = time.time() - start_time
        status = "OK" if result.get('success') else "FAIL"

        print(f"{status} {date}: {result.get('message', result.get('error', 'Unknown'))}")
        print(f"   Time: {elapsed:.1f}s")

        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")

        results.append(result)

    return results


def run_with_lithops(dates: List[str], run: str = "00", max_workers: int = 10) -> List[Dict]:
    """Run processing in parallel using Lithops Cloud Run backend, with retry.

    Retry strategy handles two failure modes:
      1. Invoke failure  — Cloud Run HTTP 500 during cold-start race (future never runs,
                           status stays 'invoked'/'running' after timeout).
      2. Exec failure    — Worker ran but process_ecmwf_date() returned success=False.

    Up to MAX_RETRIES attempts. A 30-second delay before each retry lets Cloud Run
    warm-up instances so subsequent cold-start races are avoided.
    """
    try:
        import lithops
    except ImportError:
        print("Error: lithops not installed. Install with: pip install lithops")
        sys.exit(1)

    MAX_RETRIES  = 3
    WAIT_TIMEOUT = 600    # 10 min — above the ~8.5 min processing time; lets good workers
                          # finish before timing out 429'd ones that never started
    RETRY_DELAY  = 30     # seconds between retries (Cloud Run instances warm by then)

    print(f"\nRunning Lithops parallel processing for {len(dates)} dates")
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

    # Keyed by date so retries overwrite earlier failed entries
    all_results: Dict[str, Dict] = {}
    remaining = list(dates)

    for attempt in range(1, MAX_RETRIES + 1):
        if not remaining:
            break

        if attempt > 1:
            print(f"\n--- Retry {attempt}/{MAX_RETRIES}: {len(remaining)} dates "
                  f"— pausing {RETRY_DELAY}s for Cloud Run warm-up ---")
            time.sleep(RETRY_DELAY)

        print(f"\nAttempt {attempt}: dispatching {len(remaining)} worker(s)...")
        futures = fexec.map(process_ecmwf_date, remaining, extra_args=(run,))

        # throw_except=False — don't raise on function errors
        # TimeoutError is raised unconditionally by Lithops regardless of throw_except,
        # so we catch it explicitly and continue to collect whatever completed.
        try:
            fexec.wait(futures, throw_except=False, timeout=WAIT_TIMEOUT)
        except TimeoutError:
            logger.warning(
                f"Attempt {attempt}: wait timeout ({WAIT_TIMEOUT}s) — collecting "
                f"completed results, remaining will be retried"
            )

        still_failing = []
        for i, f in enumerate(futures):
            date = remaining[i]
            if f.status == 'success':
                try:
                    result = f.result()
                    if isinstance(result, dict) and result.get('success'):
                        all_results[date] = result
                        logger.info(f"  OK   {date}: {result.get('message', '')}")
                    else:
                        # Worker ran but reported failure — queue for retry
                        msg = result.get('message', 'unknown') if isinstance(result, dict) else str(result)
                        all_results[date] = result if isinstance(result, dict) else {
                            'date': date, 'run': run, 'success': False, 'message': msg}
                        still_failing.append(date)
                        logger.warning(f"  FAIL {date}: exec error ({msg}), will retry")
                except Exception as exc:
                    all_results[date] = {'date': date, 'run': run, 'success': False,
                                         'message': str(exc)}
                    still_failing.append(date)
                    logger.warning(f"  FAIL {date}: result error ({exc}), will retry")
            else:
                # 'error', 'invoked', 'running' — invoke never completed (HTTP 500 race)
                msg = f"invoke failed (status={f.status})"
                all_results[date] = {'date': date, 'run': run, 'success': False, 'message': msg}
                still_failing.append(date)
                logger.warning(f"  FAIL {date}: {msg}, will retry")

        # GCS verification: rescue dates whose Cloud Run worker completed and wrote
        # to GCS but whose f.status was not updated to 'success' because the Lithops
        # monitor thread was interrupted (e.g. a transient DNS/network failure).
        # GCS is the authoritative ground truth for completion.
        if still_failing:
            try:
                import gcsfs
                _fs = gcsfs.GCSFileSystem()
                gcs_rescued = []
                for date in still_failing:
                    year, month = date[:4], date[4:6]
                    gcs_prefix = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{year}/{month}/{date}/{run}z"
                    try:
                        items = _fs.ls(gcs_prefix)
                        if items:  # directory exists with parquet files
                            all_results[date] = {
                                'date': date, 'run': run, 'success': True,
                                'gcs_path': f"gs://{gcs_prefix}",
                                'message': 'GCS-verified (f.status stale — monitor interrupted)',
                            }
                            gcs_rescued.append(date)
                            logger.info(f"  GCS-OK {date}: found {len(items)} file(s)")
                    except Exception:
                        pass  # not in GCS — genuine failure, keep in still_failing
                if gcs_rescued:
                    still_failing = [d for d in still_failing if d not in set(gcs_rescued)]
                    print(f"  GCS verification rescued {len(gcs_rescued)} date(s) "
                          f"(Lithops monitor was interrupted)")
            except Exception as gcs_exc:
                logger.warning(f"  GCS verification skipped: {gcs_exc}")

        succeeded = len(remaining) - len(still_failing)
        print(f"Attempt {attempt} done: {succeeded}/{len(remaining)} succeeded, "
              f"{len(still_failing)} queued for retry")
        remaining = still_failing

    # Any dates that exhausted all retries
    for date in remaining:
        all_results[date] = {'date': date, 'run': run, 'success': False,
                              'message': 'Max retries exceeded'}
        logger.error(f"  GIVE UP {date}: max retries ({MAX_RETRIES}) exceeded")

    total_time = time.time() - overall_start
    success_count = sum(1 for r in all_results.values() if r.get('success'))
    print(f"\nCompleted in {total_time:.1f}s ({total_time/60:.1f} min) — "
          f"{success_count}/{len(dates)} successful")

    # Return in original date order
    return [all_results.get(d, {'date': d, 'run': run, 'success': False,
                                'message': 'No result'}) for d in dates]


def print_summary(dates: List[str], results: List[Dict], total_time: float):
    """Print execution summary."""
    success_count = sum(1 for r in results if r.get('success'))
    failed_count = len(results) - success_count

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal dates: {len(dates)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if success_count > 0:
        avg_time = sum(r.get('total_time_seconds', 0) for r in results if r.get('success')) / success_count
        print(f"Average processing time: {avg_time:.1f}s ({avg_time/60:.1f} min)")

    print("\nResults by date:")
    print("-" * 70)

    for date, result in zip(dates, results):
        status = "OK" if result.get('success') else "FAIL"
        message = result.get('message', result.get('error', 'No message'))

        print(f"{status} {date}: {message}")

        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")
        if result.get('files_count'):
            print(f"   Files: {result['files_count']}")
        if result.get('total_time_seconds'):
            print(f"   Time: {result['total_time_seconds']:.1f}s")

    # GCS locations for successful uploads
    gcs_paths = [r.get('gcs_path') for r in results if r.get('gcs_path')]

    if gcs_paths:
        print("\n" + "=" * 70)
        print("GCS OUTPUT LOCATIONS")
        print("=" * 70)

        for path in gcs_paths:
            print(f"  {path}")

        print(f"\nList all parquets:")
        print(f"  gsutil ls gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")


def main():
    parser = argparse.ArgumentParser(
        description='Run ECMWF parquet generation via Lithops Cloud Run workers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Date selection (mutually exclusive group)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--date', type=str,
                            help='Single date to process (YYYYMMDD)')
    date_group.add_argument('--start-date', type=str,
                            help='Start date for range (YYYYMMDD)')
    date_group.add_argument('--days-back', type=int,
                            help='Process last N days')

    parser.add_argument('--end-date', type=str,
                        help='End date for range (YYYYMMDD), used with --start-date')
    parser.add_argument('--run', type=str, default='00', choices=['00', '06', '12', '18'],
                        help='Model run hour (default: 00)')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Max parallel workers for Lithops (default: 10)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially without Lithops (local testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show dates without executing')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip interactive confirmation (for batch/automated runs)')

    args = parser.parse_args()

    # Validate start-date/end-date combination
    if args.start_date and not args.end_date:
        parser.error("--end-date is required when using --start-date")
    if args.end_date and not args.start_date:
        parser.error("--start-date is required when using --end-date")

    # Generate date list
    try:
        dates = generate_date_list(args)
    except ValueError as e:
        parser.error(str(e))

    # Print configuration
    print("=" * 70)
    print("ECMWF PARQUET GENERATION - LITHOPS CLOUD RUN")
    print("=" * 70)
    print(f"\nRun hour: {args.run}Z")
    print(f"Dates to process: {len(dates)}")
    print(f"  First: {dates[0]}")
    print(f"  Last: {dates[-1]}")
    print(f"Execution mode: {'Sequential (local)' if args.sequential else f'Lithops Cloud Run (max {args.max_workers} workers)'}")
    print(f"GCS output: gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")

    if args.dry_run:
        print(f"\nDRY RUN - Dates that would be processed:")
        for date in dates:
            print(f"  {date}")
        return

    # Confirm before processing
    if len(dates) > 5 and not args.yes:
        response = input(f"\nProcess {len(dates)} dates? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Execute
    start_time = time.time()

    if args.sequential:
        results = run_sequential(dates, args.run)
    else:
        results = run_with_lithops(dates, args.run, args.max_workers)

    total_time = time.time() - start_time

    # Print summary
    print_summary(dates, results, total_time)


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
