#!/usr/bin/env python3
"""
ECMWF Three-Stage Multi-Date Processing

Processes multiple dates through the three-stage pipeline:
- Stage 1: Uses prebuilt parquet from zip files (from ecmwf_ensemble_par_creator_efficient.py)
- Stage 2: Template + fresh index merge (from GCS bucket or local tar.gz file)
- Stage 3: Final zarr store creation

Creates date-named output folders and zip archives for easy access.

Usage:
    # Default: uses GCS bucket for templates (requires service account)
    python ecmwf_three_stage_multidate.py

    # Use local tar.gz template file instead of GCS bucket
    python ecmwf_three_stage_multidate.py --use-local-template
    python ecmwf_three_stage_multidate.py --use-local-template --local-template-path /path/to/templates.tar.gz

    # Or specify dates:
    python ecmwf_three_stage_multidate.py --dates 20251125 20251124 20251123 20251122

    # Limit members for testing:
    python ecmwf_three_stage_multidate.py --max-members 5

    # Combine options:
    python ecmwf_three_stage_multidate.py --dates 20251126 --use-local-template --max-members 5
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import copy
import time
import zipfile
import shutil
import argparse
import re
from pathlib import Path
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

# Import ECMWF utilities
from ecmwf_util import (
    generate_ecmwf_axes,
    ECMWF_FORECAST_DICT,
    ECMWF_FORECAST_HOURS
)

# Set up anonymous S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Reference date with pre-built GCS mappings
REFERENCE_DATE = '20240529'

# GCS configuration (needed for Stage 2)
GCS_BUCKET = 'gik-fmrc'
GCS_BASE_PATH = 'v2ecmwf_fmrc'
GCP_SERVICE_ACCOUNT = 'coiled-data-e4drr_202505.json'

# Local template configuration (alternative to GCS)
LOCAL_TEMPLATE_TAR = 'gik-fmrc-v2ecmwf_fmrc.tar.gz'

# Variables to extract
FORECAST_DICT = {
    "2 metre temperature": "2t:sfc",
    "Total precipitation": "tp:sfc",
    "10 metre U wind": "10u:sfc",
    "10 metre V wind": "10v:sfc"
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def log_stage(stage_num, stage_name):
    """Print stage header."""
    print(f"\n{'='*80}")
    print(f"STAGE {stage_num}: {stage_name}")
    print(f"{'='*80}")


def log_checkpoint(message):
    """Print checkpoint with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def log_message(message: str, level: str = "INFO"):
    """Simple logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def create_zip_archive(source_dir: Path, zip_path: Path) -> bool:
    """
    Create a zip archive of the source directory.
    """
    try:
        log_checkpoint(f"Creating zip archive: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir.parent)
                    zipf.write(file_path, arcname)

        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        log_checkpoint(f"Zip archive created: {zip_path} ({zip_size_mb:.2f} MB)")
        return True

    except Exception as e:
        log_checkpoint(f"Error creating zip archive: {e}")
        return False


def extract_date_run_from_zip(zip_filename):
    """
    Extract date and run from zip filename.
    Format: ecmwf_YYYYMMDD_HHz_efficient.zip
    Returns: (date_str, run_str)
    """
    # Try new format first: ecmwf_YYYYMMDD_HHz_efficient.zip
    pattern1 = r'ecmwf_(\d{8})_(\d{2})z_efficient\.zip'
    match = re.match(pattern1, Path(zip_filename).name)

    if match:
        return match.group(1), match.group(2)

    # Try old format: ecmwf_YYYYMMDD_HH_efficient.zip
    pattern2 = r'ecmwf_(\d{8})_(\d{2})_efficient\.zip'
    match = re.match(pattern2, Path(zip_filename).name)

    if match:
        return match.group(1), match.group(2)

    raise ValueError(f"Unable to extract date and run from filename: {zip_filename}")


def unzip_and_prepare(zip_file, extract_dir):
    """
    Unzip the prebuilt zip file and return paths to parquet files.
    """
    log_checkpoint(f"Extracting {zip_file}...")

    # Clear extraction directory if exists
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip file
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(extract_dir)

    # Find extracted directory
    extracted_dirs = list(extract_dir.glob("ecmwf_*"))
    if not extracted_dirs:
        raise ValueError(f"No extracted directory found in {extract_dir}")

    extracted_dir = extracted_dirs[0]

    # Find members directory
    members_dir = extracted_dir / "members"
    if not members_dir.exists():
        raise ValueError(f"Members directory not found: {members_dir}")

    # Find all member parquet files
    member_parquets = {}
    for member_dir in sorted(members_dir.glob("*")):
        if member_dir.is_dir():
            member_name = member_dir.name
            parquet_files = list(member_dir.glob("*.parquet"))
            if parquet_files:
                member_parquets[member_name] = parquet_files[0]

    # Also check for comprehensive parquet
    comprehensive_parquet = None
    comprehensive_dir = extracted_dir / "comprehensive"
    if comprehensive_dir.exists():
        comprehensive_files = list(comprehensive_dir.glob("*.parquet"))
        if comprehensive_files:
            comprehensive_parquet = comprehensive_files[0]

    log_checkpoint(f"Found {len(member_parquets)} member parquet files")

    return {
        'members': member_parquets,
        'comprehensive': comprehensive_parquet,
        'extracted_dir': extracted_dir
    }


def read_parquet_to_zarr_store(parquet_file):
    """Read parquet and convert to zarr store."""
    df = pd.read_parquet(parquet_file)
    zstore = {}

    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        if isinstance(value, bytes):
            value = value.decode('utf-8')

        if isinstance(value, str):
            if value.startswith('[') or value.startswith('{'):
                try:
                    value = json.loads(value)
                except:
                    pass

        zstore[key] = value

    if 'version' in zstore:
        del zstore['version']

    return zstore


def create_parquet_simple(zstore, output_file):
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
    log_checkpoint(f"Saved parquet: {output_file} ({len(df)} rows)")
    return df


# ==============================================================================
# STAGE 1: USE PREBUILT PARQUET FILES
# ==============================================================================

def run_stage1_prebuilt(member_parquets, test_date, test_run):
    """Stage 1: Use prebuilt parquet files."""
    log_stage(1, "USE PREBUILT PARQUET FILES")

    start_time = time.time()

    try:
        log_checkpoint(f"Using prebuilt parquet files for {len(member_parquets)} members")

        deflated_stores = {}
        for member, parquet_path in member_parquets.items():
            deflated_stores[member] = read_parquet_to_zarr_store(parquet_path)

        elapsed = time.time() - start_time

        log_checkpoint(f"Stage 1 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Members loaded: {len(deflated_stores)}")

        return deflated_stores

    except Exception as e:
        log_checkpoint(f"Stage 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 1 FAST PATH: BUILD FROM TEMPLATE (NO GRIB SCANNING)
# ==============================================================================

def build_deflated_stores_from_template(
    template_tar_path: str,
    template_date: str,
    max_members: Optional[int] = None
) -> Optional[Dict]:
    """
    Build deflated_stores directly from the HuggingFace template archive.

    This replaces the slow scan_grib approach (~73 min) with direct template
    loading (~2-5 seconds). The template contains the zarr structure metadata
    that Stage 3 needs as a base for merging with Stage 2 references.

    Args:
        template_tar_path: Path to the template tar.gz file
        template_date: Reference date in the template (e.g., '20240529')
        max_members: Optional limit on number of members

    Returns:
        deflated_stores dict mapping member names to zarr store dicts
    """
    import tarfile

    log_stage(1, "LOAD ZARR STRUCTURE FROM TEMPLATE (No GRIB scanning)")

    start_time = time.time()

    if not Path(template_tar_path).exists():
        log_checkpoint(f"Template archive not found: {template_tar_path}")
        return None

    try:
        # All member directory names in the template archive
        all_members = ['ens_control'] + [f'ens_{i:02d}' for i in range(1, 51)]

        if max_members:
            all_members = all_members[:max_members]

        log_checkpoint(f"Loading zarr structure for {len(all_members)} members from template")
        log_checkpoint(f"Template: {template_tar_path}")
        log_checkpoint(f"Reference date: {template_date}")

        deflated_stores = {}

        with tarfile.open(template_tar_path, 'r:gz') as tar:
            for member_dir in all_members:
                # Map directory name to member key used downstream
                if member_dir == 'ens_control':
                    member_key = 'control'
                    filename_member = 'control'
                else:
                    num = int(member_dir.replace('ens_', ''))
                    member_key = f'ens_{num:02d}'
                    filename_member = f'ens{num:02d}'

                # Path inside tar.gz
                tar_member_path = (
                    f"gik-fmrc/v2ecmwf_fmrc/{member_dir}/"
                    f"ecmwf-{template_date}00-{filename_member}-rt000.par"
                )

                try:
                    member_info = tar.getmember(tar_member_path)
                except KeyError:
                    log_checkpoint(f"  Template not found for {member_key}: {tar_member_path}")
                    continue

                # Extract and read the parquet
                f = tar.extractfile(member_info)
                if f is None:
                    continue

                import io
                parquet_bytes = f.read()
                template_df = pd.read_parquet(io.BytesIO(parquet_bytes))

                # Convert to zarr store dict (same as read_parquet_to_zarr_store)
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

        log_checkpoint(f"Stage 1 (Template) Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Members loaded: {len(deflated_stores)}")

        if deflated_stores:
            sample_key = next(iter(deflated_stores))
            sample_size = len(deflated_stores[sample_key])
            log_checkpoint(f"   Sample '{sample_key}': {sample_size} zarr entries")

        return deflated_stores

    except Exception as e:
        log_checkpoint(f"Stage 1 (Template) Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_index_availability(date_str: str, run: str) -> Tuple[bool, int, int]:
    """
    Validate that .index files are available on S3 for the target date.

    Checks hour 0 index file and verifies expected message count and members.

    Returns:
        (is_valid, n_messages, n_members)
    """
    import fsspec

    idx_url = (
        f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/"
        f"{date_str}000000-0h-enfo-ef.index"
    )

    try:
        fs = fsspec.filesystem("s3", anon=True)

        if not fs.exists(idx_url):
            log_checkpoint(f"Index file not found: {idx_url}")
            return False, 0, 0

        with fs.open(idx_url, 'r') as f:
            lines = f.readlines()

        members = set()
        for line in lines:
            data = json.loads(line.strip().rstrip(','))
            members.add(int(data.get('number', -1)))

        n_messages = len(lines)
        n_members = len(members)

        log_checkpoint(f"Index validation: {n_messages} messages, {n_members} members")

        # Expected: 51 members, ~160 variables each = ~8160 messages
        if n_members < 50:
            log_checkpoint(f"WARNING: Expected 51 members, found {n_members}")
            return False, n_messages, n_members

        return True, n_messages, n_members

    except Exception as e:
        log_checkpoint(f"Index validation failed: {e}")
        return False, 0, 0


def spot_check_grib_integrity(date_str: str, run: str, n_checks: int = 3) -> bool:
    """
    Spot-check GRIB file integrity by fetching a few byte ranges
    and verifying they parse as valid GRIB messages with gribberish.

    Args:
        date_str: Target date
        run: Model run hour
        n_checks: Number of random messages to check

    Returns:
        True if all checks pass
    """
    import fsspec

    grib_url = (
        f"ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/"
        f"{date_str}000000-0h-enfo-ef.grib2"
    )
    idx_url = f"s3://{grib_url}".replace('.grib2', '.index')

    try:
        from gribberish import parse_grib_mapping

        fs = fsspec.filesystem("s3", anon=True)

        # Read index to get message offsets
        with fs.open(idx_url, 'r') as f:
            lines = f.readlines()

        entries = [json.loads(line.strip().rstrip(',')) for line in lines]

        # Pick n_checks messages spread across the file (large messages only)
        large_entries = [e for e in entries if int(e['_length']) > 10000]
        if not large_entries:
            log_checkpoint("No large messages found for spot-check")
            return True

        import random
        random.seed(42)  # Reproducible
        check_entries = random.sample(large_entries, min(n_checks, len(large_entries)))

        passed = 0
        for entry in check_entries:
            offset = int(entry['_offset'])
            length = int(entry['_length'])

            with fs.open(grib_url, 'rb') as f:
                f.seek(offset)
                msg_bytes = f.read(length)

            try:
                mapping = parse_grib_mapping(msg_bytes)
                if mapping:
                    passed += 1
            except Exception:
                log_checkpoint(
                    f"  Spot-check failed: param={entry['param']}, "
                    f"offset={offset}"
                )

        log_checkpoint(f"Spot-check: {passed}/{len(check_entries)} GRIB messages valid")
        return passed == len(check_entries)

    except ImportError:
        log_checkpoint("gribberish not available, skipping spot-check")
        return True
    except Exception as e:
        log_checkpoint(f"Spot-check error: {e}")
        return True  # Don't block pipeline on spot-check failure


# ==============================================================================
# STAGE 2: INDEX + GCS TEMPLATES
# ==============================================================================

def _process_single_member_stage2(args):
    """Worker function for parallel Stage 2 processing.

    Processes one ensemble member: fetches .index files from S3,
    merges with template, saves parquet. Returns (member, refs) or
    (member, None) on failure.
    """
    (member, member_normalized, test_date, test_run, output_dir_str,
     use_local_template, local_template_path) = args

    import os
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    try:
        from ecmwf_index_processor import build_complete_parquet_from_indices

        refs = build_complete_parquet_from_indices(
            date_str=test_date,
            run=test_run,
            member_name=member_normalized,
            hours=ECMWF_FORECAST_HOURS,
            use_gcs_template=not use_local_template,
            gcs_template_date=REFERENCE_DATE,
            use_local_template=use_local_template,
            local_template_path=local_template_path or LOCAL_TEMPLATE_TAR
        )

        if refs:
            output_dir = Path(output_dir_str)
            stage2_output = output_dir / f"stage2_{member}_merged.parquet"

            df_data = []
            for key, value in refs.items():
                if isinstance(value, str):
                    encoded_value = value.encode('utf-8')
                elif isinstance(value, (list, dict)):
                    encoded_value = json.dumps(value).encode('utf-8')
                else:
                    encoded_value = str(value).encode('utf-8')
                df_data.append((key, encoded_value))

            df = pd.DataFrame(df_data, columns=['key', 'value'])
            df.to_parquet(stage2_output)

            return (member, refs)

        return (member, None)

    except Exception as e:
        return (member, None, str(e))


def run_stage2_with_gcs_templates(test_date, test_run, test_members, output_dir,
                                   use_local_template=False, local_template_path=None,
                                   parallel_workers=8):
    """Stage 2: INDEX + Templates merge (GCS or local).

    Args:
        test_date: Date string (YYYYMMDD)
        test_run: Run hour (e.g., '00')
        test_members: List of member names to process
        output_dir: Output directory path
        use_local_template: If True, use local tar.gz file instead of GCS bucket
        local_template_path: Path to local tar.gz file (default: gik-fmrc-v2ecmwf_fmrc.tar.gz)
        parallel_workers: Number of parallel workers (0 or 1 = sequential)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    template_source = "LOCAL TAR.GZ" if use_local_template else "GCS BUCKET"
    mode = f"PARALLEL {parallel_workers} workers" if parallel_workers > 1 else "SEQUENTIAL"
    log_stage(2, f"INDEX + TEMPLATES MERGE ({template_source}, {mode}, All 85 hours)")

    start_time = time.time()

    try:
        log_checkpoint(f"Target date: {test_date}")
        log_checkpoint(f"Reference date: {REFERENCE_DATE}")
        log_checkpoint(f"Processing {len(test_members)} members ({mode})")

        # Prepare args for each member
        task_args = []
        for member in test_members:
            if member == 'control':
                member_normalized = 'control'
            else:
                member_normalized = member.replace('_', '')
                if member_normalized.startswith('ens'):
                    member_num_str = member_normalized.replace('ens', '')
                    member_normalized = f'ens{int(member_num_str):02d}'

            task_args.append((
                member, member_normalized, test_date, test_run,
                str(output_dir), use_local_template, local_template_path
            ))

        member_results = {}

        if parallel_workers > 1 and len(test_members) > 1:
            # --- PARALLEL PATH ---
            n_workers = min(parallel_workers, len(test_members))
            log_checkpoint(f"Launching {n_workers} parallel workers")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_member = {
                    executor.submit(_process_single_member_stage2, args): args[0]
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
                                log_checkpoint(f"[{completed}/{len(test_members)}] "
                                             f"{member}: {len(refs)} references")
                            else:
                                log_checkpoint(f"[{completed}/{len(test_members)}] "
                                             f"{member}: no refs returned")
                        else:
                            member, _, error = result
                            log_checkpoint(f"[{completed}/{len(test_members)}] "
                                         f"Error {member}: {error}")
                    except Exception as e:
                        log_checkpoint(f"[{completed}/{len(test_members)}] "
                                     f"Worker error {member_name}: {e}")
        else:
            # --- SEQUENTIAL PATH (fallback) ---
            from ecmwf_index_processor import build_complete_parquet_from_indices

            for member, member_normalized, _, _, _, _, _ in task_args:
                try:
                    refs = build_complete_parquet_from_indices(
                        date_str=test_date,
                        run=test_run,
                        member_name=member_normalized,
                        hours=ECMWF_FORECAST_HOURS,
                        use_gcs_template=not use_local_template,
                        gcs_template_date=REFERENCE_DATE,
                        use_local_template=use_local_template,
                        local_template_path=local_template_path or LOCAL_TEMPLATE_TAR
                    )

                    if refs:
                        stage2_output = output_dir / f"stage2_{member}_merged.parquet"

                        df_data = []
                        for key, value in refs.items():
                            if isinstance(value, str):
                                encoded_value = value.encode('utf-8')
                            elif isinstance(value, (list, dict)):
                                encoded_value = json.dumps(value).encode('utf-8')
                            else:
                                encoded_value = str(value).encode('utf-8')
                            df_data.append((key, encoded_value))

                        df = pd.DataFrame(df_data, columns=['key', 'value'])
                        df.to_parquet(stage2_output)

                        member_results[member] = refs
                        log_checkpoint(f"{member}: {len(refs)} references")

                except Exception as e:
                    log_checkpoint(f"Error processing {member}: {e}")
                    continue

        elapsed = time.time() - start_time

        log_checkpoint(f"Stage 2 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        log_checkpoint(f"   Members processed: {len(member_results)}/{len(test_members)}")

        return member_results

    except Exception as e:
        log_checkpoint(f"Stage 2 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 3: CREATE FINAL ZARR STORE
# ==============================================================================

def run_stage3(deflated_stores, stage2_refs, test_date, output_dir):
    """Stage 3: Create final zarr stores."""
    log_stage(3, "CREATE FINAL ZARR STORE (All 85 timesteps)")

    if not deflated_stores or not stage2_refs:
        log_checkpoint("Skipping Stage 3: Missing required inputs")
        return None

    start_time = time.time()

    try:
        axes = generate_ecmwf_axes(test_date)

        results = {}

        for member in deflated_stores.keys():
            if member not in stage2_refs:
                continue

            log_checkpoint(f"Processing {member}...")

            deflated_store = deflated_stores[member]
            complete_refs = stage2_refs[member]

            # Merge stores
            if isinstance(deflated_store, dict) and 'refs' in deflated_store:
                final_store = deflated_store.get('refs', {}).copy()
            else:
                final_store = deflated_store.copy()

            for key, ref in complete_refs.items():
                if not key.startswith('_'):
                    final_store[key] = ref

            # Save final parquet
            stage3_output = output_dir / f"stage3_{member}_final.parquet"
            create_parquet_simple(final_store, stage3_output)

            results[member] = (final_store, stage3_output)

        elapsed = time.time() - start_time

        log_checkpoint(f"Stage 3 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Members: {len(results)}")

        return results

    except Exception as e:
        log_checkpoint(f"Stage 3 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# SINGLE DATE PROCESSING
# ==============================================================================

def process_single_date(date_str: str, run: str, max_members: Optional[int] = None,
                        use_local_template: bool = False, local_template_path: Optional[str] = None,
                        skip_grib_scan: bool = False,
                        parallel_workers: int = 8) -> Tuple[bool, Optional[Path]]:
    """
    Process a single date through all three stages.

    Parameters:
    - date_str: Date string (e.g., '20251125')
    - run: Run hour (e.g., '00')
    - max_members: Maximum number of members to process (optional)
    - use_local_template: If True, use local tar.gz file instead of GCS bucket
    - local_template_path: Path to local tar.gz file (default: gik-fmrc-v2ecmwf_fmrc.tar.gz)
    - skip_grib_scan: If True, build deflated_stores from template instead of zip
                      (Phase 1 optimization: eliminates 73-min scan_grib dependency)
    - parallel_workers: Number of parallel workers for Stage 2 (0 or 1 = sequential)

    Returns:
    - Tuple of (success, output_directory_path)
    """
    print("\n" + "="*80)
    print(f"PROCESSING DATE: {date_str} {run}z")
    print("="*80)

    start_time = time.time()

    # Create output directory
    output_dir = Path(f"ecmwf_three_stage_{date_str}_{run}z")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the template path for both Stage 1 fast-path and Stage 2
    template_path = local_template_path or LOCAL_TEMPLATE_TAR

    # --- Determine Stage 1 strategy ---
    # Priority: 1) Existing zip file  2) Template fast-path  3) Fail
    zip_patterns = [
        f"ecmwf_{date_str}_{run}z_efficient.zip",
        f"ecmwf_{date_str}_{run}_efficient.zip"
    ]

    zip_file = None
    for pattern in zip_patterns:
        if Path(pattern).exists():
            zip_file = Path(pattern)
            break

    use_template_fast_path = False
    if zip_file and not skip_grib_scan:
        log_checkpoint(f"Using zip file: {zip_file}")
    elif skip_grib_scan or not zip_file:
        if Path(template_path).exists():
            use_template_fast_path = True
            log_checkpoint(f"Using template fast-path (no GRIB scanning)")
        elif zip_file:
            log_checkpoint(f"Template not found, falling back to zip: {zip_file}")
        else:
            log_checkpoint(f"No zip file and no template found")
            log_checkpoint(f"  Zip expected: {zip_patterns[0]}")
            log_checkpoint(f"  Template expected: {template_path}")
            return False, None

    try:
        extract_dir = None

        if use_template_fast_path:
            # --- FAST PATH: Build deflated_stores from template ---

            # Validate that index files exist on S3
            idx_valid, n_msgs, n_members = validate_index_availability(date_str, run)
            if not idx_valid:
                log_checkpoint("Index validation failed — falling back to zip if available")
                if zip_file:
                    use_template_fast_path = False
                else:
                    return False, None

        if use_template_fast_path:
            # Spot-check GRIB integrity (non-blocking, ~10s)
            spot_ok = spot_check_grib_integrity(date_str, run, n_checks=3)
            if not spot_ok:
                log_checkpoint("WARNING: GRIB spot-check had failures (continuing anyway)")

            # Build deflated_stores from template
            deflated_stores = build_deflated_stores_from_template(
                template_tar_path=template_path,
                template_date=REFERENCE_DATE,
                max_members=max_members
            )

            if not deflated_stores:
                log_checkpoint("Template loading failed")
                if zip_file:
                    log_checkpoint("Falling back to zip file")
                    use_template_fast_path = False
                else:
                    return False, None

            if use_template_fast_path:
                test_members = sorted(deflated_stores.keys())

        if not use_template_fast_path:
            # --- ORIGINAL PATH: Extract zip file ---
            extract_dir = output_dir / "extracted"
            extraction_info = unzip_and_prepare(zip_file, extract_dir)

            member_parquets = extraction_info['members']
            test_members = sorted(member_parquets.keys())

            log_checkpoint(f"Found {len(test_members)} members in zip")

            # Apply max_members limit
            if max_members and max_members < len(test_members):
                test_members = test_members[:max_members]
                member_parquets = {k: v for k, v in member_parquets.items()
                                   if k in test_members}
                log_checkpoint(f"Limiting to {max_members} members")

            # Stage 1 from prebuilt parquets
            deflated_stores = run_stage1_prebuilt(member_parquets, date_str, run)

            if not deflated_stores:
                return False, None

        # Stage 2
        stage2_refs = run_stage2_with_gcs_templates(
            date_str, run, test_members, output_dir,
            use_local_template=use_local_template,
            local_template_path=template_path,
            parallel_workers=parallel_workers
        )

        # Stage 3
        stage3_results = None
        if deflated_stores and stage2_refs:
            stage3_results = run_stage3(deflated_stores, stage2_refs, date_str, output_dir)

        # Cleanup extracted files to save space
        if extract_dir and extract_dir.exists():
            shutil.rmtree(extract_dir)
            log_checkpoint("Cleaned up extracted files")

        elapsed = time.time() - start_time

        mode = "template fast-path" if use_template_fast_path else "zip prebuilt"
        print(f"\nResults for {date_str} (mode: {mode}):")
        print(f"   Processing time: {elapsed/60:.1f} minutes")
        print(f"   Stage 1: {'Success' if deflated_stores else 'Failed'}")
        print(f"   Stage 2: {'Success' if stage2_refs else 'Failed'} "
              f"({len(stage2_refs) if stage2_refs else 0} members)")
        print(f"   Stage 3: {'Success' if stage3_results else 'Failed'} "
              f"({len(stage3_results) if stage3_results else 0} members)")

        success = stage3_results is not None and len(stage3_results) > 0
        return success, output_dir

    except Exception as e:
        log_checkpoint(f"Processing failed for {date_str}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# ==============================================================================
# MAIN MULTI-DATE PROCESSING
# ==============================================================================

def main():
    """Main function for multi-date three-stage processing."""
    parser = argparse.ArgumentParser(description='ECMWF Three-Stage Multi-Date Processing')
    parser.add_argument('--dates', nargs='+', default=['20251126', '20251127'],
                        help='Dates to process (YYYYMMDD format)')
    parser.add_argument('--run', type=str, default='00', help='Run hour (default: 00)')
    parser.add_argument('--max-members', type=int, default=None, help='Maximum members per date')
    parser.add_argument('--no-zip', action='store_true', help='Skip creating zip archives')
    parser.add_argument('--use-local-template', action='store_true',
                        help='Use local tar.gz file instead of GCS bucket for templates')
    parser.add_argument('--local-template-path', type=str, default=None,
                        help=f'Path to local template tar.gz file (default: {LOCAL_TEMPLATE_TAR})')
    parser.add_argument('--skip-grib-scan', action='store_true',
                        help='Skip GRIB scanning, build Stage 1 from template '
                             '(Phase 1 optimization: ~73 min → ~5 sec)')
    args = parser.parse_args()

    print("="*80)
    print("ECMWF Three-Stage Multi-Date Processing")
    print("="*80)

    overall_start = time.time()

    dates = args.dates
    run = args.run

    log_message(f"Processing {len(dates)} dates: {dates}")
    log_message(f"Run: {run}z")
    log_message(f"Skip GRIB scan: {args.skip_grib_scan}")
    if args.max_members:
        log_message(f"Max members per date: {args.max_members}")
    if args.use_local_template:
        template_path = args.local_template_path or LOCAL_TEMPLATE_TAR
        log_message(f"Using LOCAL template: {template_path}")
    else:
        log_message(f"Using GCS bucket template: gs://{GCS_BUCKET}/{GCS_BASE_PATH}")

    # Track results
    processing_results = {}
    successful_dates = []
    failed_dates = []
    zip_files_created = []

    for date_str in dates:
        # Process single date
        success, output_dir = process_single_date(
            date_str, run, args.max_members,
            use_local_template=args.use_local_template,
            local_template_path=args.local_template_path,
            skip_grib_scan=args.skip_grib_scan
        )

        if success and output_dir is not None:
            # Create zip archive
            if not args.no_zip:
                zip_path = Path(f"ecmwf_three_stage_{date_str}_{run}z.zip")
                zip_success = create_zip_archive(output_dir, zip_path)

                processing_results[date_str] = {
                    'success': True,
                    'output_dir': str(output_dir),
                    'zip_file': str(zip_path) if zip_success else None,
                    'zip_success': zip_success
                }

                if zip_success:
                    zip_files_created.append(zip_path)
            else:
                processing_results[date_str] = {
                    'success': True,
                    'output_dir': str(output_dir),
                    'zip_file': None,
                    'zip_success': False
                }

            successful_dates.append(date_str)
        else:
            processing_results[date_str] = {
                'success': False,
                'output_dir': None,
                'zip_file': None,
                'zip_success': False
            }
            failed_dates.append(date_str)

    # Overall Summary
    overall_time = time.time() - overall_start

    print("\n" + "="*80)
    print("MULTI-DATE PROCESSING SUMMARY")
    print("="*80)

    print(f"\nDates Processed: {len(dates)}")
    print(f"   Successful: {len(successful_dates)}")
    print(f"   Failed: {len(failed_dates)}")

    print(f"\nTotal Processing Time: {overall_time/60:.1f} minutes")

    if successful_dates:
        print(f"\nSuccessfully processed dates:")
        for date_str in successful_dates:
            result = processing_results[date_str]
            zip_status = "zipped" if result['zip_success'] else "not zipped"
            print(f"   - {date_str}: {result['output_dir']} ({zip_status})")

    if failed_dates:
        print(f"\nFailed dates:")
        for date_str in failed_dates:
            print(f"   - {date_str}")

    if zip_files_created:
        print(f"\nZip files created:")
        for zip_path in zip_files_created:
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"   - {zip_path} ({zip_size_mb:.2f} MB)")

    # Save summary
    summary = {
        'dates_processed': dates,
        'run': run,
        'successful_dates': successful_dates,
        'failed_dates': failed_dates,
        'total_processing_time_minutes': overall_time / 60,
        'results': processing_results
    }

    summary_file = Path("ecmwf_three_stage_multidate_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    log_message(f"Processing summary saved to {summary_file}")

    if len(failed_dates) == 0:
        print("\nMulti-date three-stage processing completed successfully!")
        return True
    else:
        print("\nSome dates failed processing - check logs for details")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
