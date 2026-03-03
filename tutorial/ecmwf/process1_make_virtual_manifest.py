#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "numpy",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "requests",
# ]
# ///
"""
Process 1: Create Virtual Manifest (Parquet Reference Files) for ECMWF
=======================================================================

Self-contained three-stage pipeline that creates parquet reference files
pointing into ECMWF GRIB files on S3. Each parquet contains [url, offset,
length] triplets — no GRIB data is downloaded, only lightweight .index files.

The three stages:
  Stage 1: Load zarr metadata structure from HuggingFace template (~5s)
  Stage 2: Read .index files from S3, extract byte ranges, merge with
           template to get full zarr references (~5-15 min)
  Stage 3: Write final parquet files to disk (~2s)

Output:
  output_parquet/{date}{run}z-control.parquet
  output_parquet/{date}{run}z-ens_01.parquet
  ...

Usage:
    # Quick demo (3 members, yesterday)
    uv run process1_make_virtual_manifest.py

    # Specific date, all 51 members
    uv run process1_make_virtual_manifest.py --date 20260301 --max-members 51

    # Custom run hour
    uv run process1_make_virtual_manifest.py --date 20260301 --run 12

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
# CONFIGURATION
# ==============================================================================

TEMPLATE_URL = (
    'https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates'
    '/resolve/main/gik-fmrc-v2ecmwf_fmrc.tar.gz'
)
TEMPLATE_FILENAME = 'gik-fmrc-v2ecmwf_fmrc.tar.gz'

# ECMWF forecast hours
HOURS_3H = list(range(0, 145, 3))       # 0-144h at 3h intervals (49 steps)
HOURS_6H = list(range(150, 361, 6))     # 150-360h at 6h intervals (36 steps)
ALL_FORECAST_HOURS = HOURS_3H + HOURS_6H  # Total: 85 steps

REFERENCE_DATE = '20240529'
S3_BUCKET = "ecmwf-forecasts"
PARALLEL_WORKERS = 4


# ==============================================================================
# STAGE 1: LOAD ZARR STRUCTURE FROM TEMPLATE
# ==============================================================================

def ensure_template() -> str:
    """Download template to CWD if not already present, return path."""
    local_path = Path(TEMPLATE_FILENAME)

    if local_path.exists():
        logger.info(f"Template cached at {local_path}")
        return str(local_path)

    logger.info(f"Downloading template from {TEMPLATE_URL}")
    response = requests.get(TEMPLATE_URL, stream=True, timeout=300)
    response.raise_for_status()

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(f"Template downloaded: {size_mb:.1f} MB")
    return str(local_path)


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
        hours = ALL_FORECAST_HOURS

    all_refs = {}

    for hour in hours:
        try:
            idx_url = (
                f"s3://{S3_BUCKET}/{date_str}/{run}z/ifs/0p25/enfo/"
                f"{date_str}{run}0000-{hour}h-enfo-ef.index"
            )
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
            hours=ALL_FORECAST_HOURS
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
# STAGE 3: CREATE FINAL PARQUET FILES
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

    output_dir.mkdir(parents=True, exist_ok=True)
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

        stage3_output = output_dir / f"{date_str}{run}z-{member}.parquet"
        create_parquet_simple(final_store, stage3_output)

        results[member] = (final_store, stage3_output)

    elapsed = time.time() - start_time
    logger.info(f"Stage 3 complete: {len(results)} members in {elapsed:.1f}s")
    return results


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_index_availability(date_str: str, run: str) -> Tuple[bool, int, int]:
    """Check that .index files are available on S3 for the target date."""
    idx_url = (
        f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/"
        f"{date_str}{run}0000-0h-enfo-ef.index"
    )

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
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create ECMWF virtual manifest parquet files (three-stage GIK pipeline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick demo (3 members, yesterday's date)
    uv run process1_make_virtual_manifest.py

    # Specific date, all 51 members
    uv run process1_make_virtual_manifest.py --date 20260301 --max-members 51

    # Custom run hour and output directory
    uv run process1_make_virtual_manifest.py --date 20260301 --run 12 --output-dir my_parquets
"""
    )
    parser.add_argument('--date', type=str, default=None,
                        help='Date to process (YYYYMMDD). Default: yesterday')
    parser.add_argument('--run', type=str, default='00', choices=['00', '06', '12', '18'],
                        help='Model run hour (default: 00)')
    parser.add_argument('--max-members', type=int, default=3,
                        help='Max ensemble members to process (default: 3 for quick demo, max: 51)')
    parser.add_argument('--output-dir', type=str, default='output_parquet',
                        help='Output directory for parquet files (default: output_parquet)')

    args = parser.parse_args()

    # Default to yesterday
    if args.date is None:
        args.date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    date_str = args.date
    run = args.run
    max_members = min(args.max_members, 51)
    output_dir = Path(args.output_dir)

    # Ensure anonymous S3 access
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    print("=" * 70)
    print("ECMWF Virtual Manifest Creation (Three-Stage GIK Pipeline)")
    print("=" * 70)
    print(f"Date:        {date_str}")
    print(f"Run:         {run}Z")
    print(f"Members:     {max_members}")
    print(f"Output:      {output_dir}/")
    print("=" * 70)

    pipeline_start = time.time()

    # Pre-flight: validate index availability
    print("\nValidating index file availability on S3...")
    idx_valid, n_msgs, n_members = validate_index_availability(date_str, run)
    if not idx_valid:
        print(f"\nIndex validation failed for {date_str} {run}z")
        print("The ECMWF forecast data may not be available yet for this date.")
        sys.exit(1)
    print(f"  Found {n_msgs} messages across {n_members} members\n")

    # Stage 1: Load zarr structure from template
    template_path = ensure_template()
    deflated_stores = build_deflated_stores_from_template(
        template_path, REFERENCE_DATE, max_members=max_members
    )
    if not deflated_stores:
        print("Stage 1 failed: could not load template")
        sys.exit(1)

    test_members = sorted(deflated_stores.keys())

    # Stage 2: Index + template merge
    stage2_refs = run_stage2(date_str, run, test_members, template_path)
    if not stage2_refs:
        print("Stage 2 failed: no members processed")
        sys.exit(1)

    # Stage 3: Create final parquets
    stage3_results = run_stage3(deflated_stores, stage2_refs, date_str, run, output_dir)
    if not stage3_results:
        print("Stage 3 failed")
        sys.exit(1)

    # Summary
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(stage3_results)} members in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nOutput parquet files:")
    for member, (store, path) in sorted(stage3_results.items()):
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name}  ({size_kb:.0f} KB, {len(store)} keys)")

    print(f"\nNext steps:")
    print(f"  # Open as lazy xarray (no S3 reads until .load())")
    print(f"  uv run process2_open_virtual_dataset.py --date {date_str}")
    print(f"  # Open as materialized xarray (eagerly fetches data)")
    print(f"  uv run process3_open_materialized_dataset.py --date {date_str}")


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
