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
#     "kerchunk",
#     "xarray",
#     "cfgrib",
#     "eccodes",
#     "h5py",
#     "dask",
#     "distributed",
#     "zarr",
# ]
# ///
"""
Lithops-native GEFS Parquet Generation (Cloud Run)
===================================================

Processes NOAA GEFS weather data through the GIK three-stage pipeline using
Lithops to directly execute Python functions on Cloud Run workers.

Unlike the ECMWF Lithops script (which is fully self-contained), this script
imports from gefs_util.py — which must be baked into the Docker image for
cloudpickle deserialization to work on remote workers.

Architecture:
    run_lithops_gefs.py (this script, runs locally via uv run)
         |
         v
    [Lithops FunctionExecutor(backend='gcp_cloudrun')]
      /    |    \\
     /     |     \\  (cloudpickle-serialized functions via GCS)
    v      v      v
  [CR 1]  [CR 2]  ... [CR N]  (Lithops proxy containers on Cloud Run)
    |      |           |
    v      v           v
  process_gefs_date() runs directly inside each container:
    Stage 1: Load deflated store from template parquet (~1s)
    Stage 2: Read .idx files from S3 + merge with mapping parquets (~3-5 min)
    Stage 3: Build zarr store + create final parquets (~1 min)
    Upload to GCS

Usage:
    # Build custom runtime (one-time setup)
    gcloud builds submit \\
      --config=cloudbuild.yaml \\
      --project=e4drr-crafd \\
      --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com

    # Single date
    uv run run_lithops_gefs.py --date 20250106

    # Last 7 days
    uv run run_lithops_gefs.py --days-back 7

    # Date range
    uv run run_lithops_gefs.py --start-date 20250101 --end-date 20250107

    # Custom parallelism
    uv run run_lithops_gefs.py --days-back 30 --max-workers 20

    # Dry run (show dates without executing)
    uv run run_lithops_gefs.py --days-back 7 --dry-run

    # Sequential (no Lithops, for local testing)
    uv run run_lithops_gefs.py --date 20250106 --sequential

    # Limit members (for testing)
    uv run run_lithops_gefs.py --date 20250106 --sequential --max-members 3

Configuration:
    Uses lithops_config.yaml in current directory, or set LITHOPS_CONFIG_FILE.

Author: ICPAC GIK Team
"""

import os
import sys
import json
import time
import argparse
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
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

GCS_BUCKET = os.environ.get('GCS_BUCKET', 'gik-gefs-aws-tf')
GCS_PARQUET_PREFIX = os.environ.get('GCS_PARQUET_PREFIX', 'run_par_gefs')

TEMPLATE_TAR_GZ_URL = (
    "https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main/"
    "gik-fmrc-gefs-20241112.tar.gz"
)
TEMPLATE_PARQUET_URL = (
    "https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main/"
    "gefs-deflated-store-template-20241112.parquet"
)

# Use baked-in templates from Docker image if available, fallback to /tmp
TEMPLATE_TAR_GZ_PATH = os.environ.get(
    'GEFS_TEMPLATE_TAR_GZ',
    '/tmp/gik-fmrc-gefs-20241112.tar.gz'
)
TEMPLATE_PARQUET_PATH = os.environ.get(
    'GEFS_TEMPLATE_PARQUET',
    '/tmp/gefs-deflated-store-template-20241112.parquet'
)

# GEFS ensemble members
ALL_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # gep01-gep30


# ==============================================================================
# TEMPLATE MANAGEMENT
# ==============================================================================

def _download_if_missing(url: str, local_path: str) -> str:
    """Download file from URL if not already present locally."""
    if os.path.exists(local_path):
        logger.info(f"Template cached: {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
        return local_path
    logger.info(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded: {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
    return local_path


def ensure_templates() -> Tuple[str, str]:
    """Ensure both GEFS templates are available (baked-in or downloaded)."""
    tar_gz = _download_if_missing(TEMPLATE_TAR_GZ_URL, TEMPLATE_TAR_GZ_PATH)
    parquet = _download_if_missing(TEMPLATE_PARQUET_URL, TEMPLATE_PARQUET_PATH)
    return tar_gz, parquet


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_idx_availability(date_str: str, run: str = "00") -> Tuple[bool, int]:
    """Check that .idx files are available on S3 for the target date."""
    import fsspec
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    # Check f003 (first non-zero timestep)
    idx_url = (
        f"s3://noaa-gefs-pds/gefs.{date_str}/{run}/atmos/pgrb2sp25/"
        f"gep01.t{run}z.pgrb2s.0p25.f003.idx"
    )
    try:
        fs = fsspec.filesystem("s3", anon=True)
        if not fs.exists(idx_url):
            logger.warning(f"IDX file not found: {idx_url}")
            return False, 0

        # Count lines to estimate message count
        with fs.open(idx_url, 'r') as f:
            lines = f.readlines()
        logger.info(f"IDX validation OK: {len(lines)} messages in f003")
        return True, len(lines)
    except Exception as e:
        logger.error(f"IDX validation failed: {e}")
        return False, 0


# ==============================================================================
# MAIN PROCESSING FUNCTION (runs on Cloud Run workers via Lithops)
# ==============================================================================

def process_gefs_date(
    date_str: str,
    run: str = "00",
    max_members: Optional[int] = None,
) -> Dict:
    """
    Process a single GEFS date through the three-stage GIK pipeline.

    This function is serialized by Lithops (cloudpickle) and executed on
    Cloud Run workers. gefs_util.py must be importable on the worker
    (baked into the Docker image).

    Args:
        date_str: Date in YYYYMMDD format
        run: Model run hour (default "00")
        max_members: Limit number of members (default: all 30)

    Returns:
        Dictionary with processing results
    """
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    logger.info(f"Processing GEFS {date_str} {run}z")
    pipeline_start = time.time()

    result = {
        "date": date_str,
        "run": run,
        "success": False,
        "message": "",
        "gcs_path": None,
        "files_count": 0,
        "total_time_seconds": 0,
    }

    try:
        # Import gefs_util (available on worker via Docker image)
        from gefs_util import (
            build_gefs_deflated_store_from_template,
            LocalTarGzMappingManager,
            generate_axes,
            calculate_time_dimensions,
            cs_create_mapped_index_local,
            prepare_zarr_store,
            process_unique_groups,
            create_parquet_file,
        )

        # Validate .idx availability
        idx_valid, n_msgs = validate_idx_availability(date_str, run)
        if not idx_valid:
            result["message"] = f"IDX validation failed for {date_str}"
            return result

        # Ensure templates
        tar_gz_path, parquet_path = ensure_templates()

        # Select members
        members = ALL_MEMBERS[:max_members] if max_members else ALL_MEMBERS

        # ── Stage 1: Load deflated store from template ──
        stage1_start = time.time()
        deflated_store = build_gefs_deflated_store_from_template(parquet_path)
        stage1_time = time.time() - stage1_start
        logger.info(f"Stage 1: {len(deflated_store['refs'])} refs in {stage1_time:.1f}s")

        # Initialize mapping manager (reused across members)
        mapping_manager = LocalTarGzMappingManager(tar_gz_path)

        # ── Stage 2 + 3: Process each member ──
        output_dir = Path(f"/tmp/gefs_{date_str}_{run}z")
        output_dir.mkdir(parents=True, exist_ok=True)

        stage2_total = 0
        stage3_total = 0
        members_ok = 0

        for i, member in enumerate(members, 1):
            try:
                # Stage 2: Read .idx files, create mapped index
                s2_start = time.time()
                axes = generate_axes(date_str)
                gefs_kind = cs_create_mapped_index_local(
                    axes, date_str, member,
                    tar_gz_path=tar_gz_path,
                    mapping_manager=mapping_manager,
                )
                s2_time = time.time() - s2_start
                stage2_total += s2_time

                # Stage 3: Build zarr store, create parquet
                s3_start = time.time()
                time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
                zstore, chunk_index = prepare_zarr_store(deflated_store, gefs_kind)
                updated_zstore = process_unique_groups(
                    zstore, chunk_index, time_dims, time_coords,
                    times, valid_times, steps,
                )

                # Save parquet
                out_file = str(output_dir / f"{date_str}{run}z-{member}.parquet")
                create_parquet_file(updated_zstore, out_file)
                s3_time = time.time() - s3_start
                stage3_total += s3_time

                members_ok += 1
                if i % 5 == 0 or i == len(members):
                    logger.info(f"  [{i}/{len(members)}] {member} done "
                                f"(s2={s2_time:.1f}s s3={s3_time:.1f}s)")

            except Exception as e:
                logger.warning(f"  [{i}/{len(members)}] {member} FAILED: {e}")

        # Cleanup mapping manager
        mapping_manager.cleanup()

        if members_ok == 0:
            result["message"] = "No members processed successfully"
            return result

        # ── Upload to GCS ──
        gcs_path = _upload_to_gcs(output_dir, date_str, run)

        # Cleanup /tmp
        shutil.rmtree(output_dir, ignore_errors=True)

        total_time = time.time() - pipeline_start

        result.update({
            "success": True,
            "message": (f"{members_ok}/{len(members)} members in {total_time:.1f}s "
                        f"(s1={stage1_time:.1f}s s2={stage2_total:.1f}s s3={stage3_total:.1f}s)"),
            "gcs_path": gcs_path,
            "files_count": members_ok,
            "total_time_seconds": round(total_time, 1),
            "members_processed": members_ok,
            "stage1_time": round(stage1_time, 1),
            "stage2_time": round(stage2_total, 1),
            "stage3_time": round(stage3_total, 1),
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
# GCS UPLOAD
# ==============================================================================

def _upload_to_gcs(output_dir: Path, date_str: str, run: str) -> Optional[str]:
    """Upload final parquet files to GCS."""
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()

        year = date_str[:4]
        month = date_str[4:6]
        gcs_path = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{year}/{month}/{date_str}/{run}z"

        parquet_files = list(output_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No parquet files to upload")
            return None

        logger.info(f"Uploading {len(parquet_files)} files to gs://{gcs_path}")
        for pf in parquet_files:
            fs.put(str(pf), f"{gcs_path}/{pf.name}")

        logger.info(f"Uploaded {len(parquet_files)} files")
        return f"gs://{gcs_path}"

    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        return None


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def generate_date_list(args) -> List[str]:
    """Generate list of dates based on command-line arguments."""
    if args.date:
        return [args.date]
    elif args.days_back:
        end_date = datetime.now()
        return [(end_date - timedelta(days=i)).strftime("%Y%m%d")
                for i in range(args.days_back)]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y%m%d")
        end = datetime.strptime(args.end_date, "%Y%m%d")
        return [(start + timedelta(days=i)).strftime("%Y%m%d")
                for i in range((end - start).days + 1)]
    else:
        raise ValueError("Specify --date, --days-back, or --start-date/--end-date")


def run_sequential(dates: List[str], run: str = "00",
                   max_members: Optional[int] = None) -> List[Dict]:
    """Run processing sequentially (no Lithops, for local testing)."""
    print(f"\nRunning sequential processing for {len(dates)} dates")
    print("=" * 70)

    results = []
    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] Processing {date}...")
        result = process_gefs_date(date, run, max_members)

        status = "OK" if result.get('success') else "FAIL"
        print(f"{status} {date}: {result.get('message', 'Unknown')}")
        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")

        results.append(result)

    return results


def run_with_lithops(dates: List[str], run: str = "00",
                     max_workers: int = 10,
                     max_members: Optional[int] = None) -> List[Dict]:
    """Run processing in parallel using Lithops Cloud Run backend, with retry."""
    try:
        import lithops
    except ImportError:
        print("Error: lithops not installed. Install with: pip install lithops")
        sys.exit(1)

    MAX_RETRIES = 3
    WAIT_TIMEOUT = 600
    RETRY_DELAY = 30

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
        futures = fexec.map(
            process_gefs_date, remaining,
            extra_args=(run, max_members),
        )

        try:
            fexec.wait(futures, throw_except=False, timeout=WAIT_TIMEOUT)
        except TimeoutError:
            logger.warning(
                f"Attempt {attempt}: wait timeout ({WAIT_TIMEOUT}s) — "
                f"collecting completed results"
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
                        msg = result.get('message', 'unknown') if isinstance(result, dict) else str(result)
                        all_results[date] = result if isinstance(result, dict) else {
                            'date': date, 'run': run, 'success': False, 'message': msg}
                        still_failing.append(date)
                        logger.warning(f"  FAIL {date}: {msg}, will retry")
                except Exception as exc:
                    all_results[date] = {'date': date, 'run': run, 'success': False,
                                         'message': str(exc)}
                    still_failing.append(date)
                    logger.warning(f"  FAIL {date}: result error ({exc}), will retry")
            else:
                msg = f"invoke failed (status={f.status})"
                all_results[date] = {'date': date, 'run': run, 'success': False, 'message': msg}
                still_failing.append(date)
                logger.warning(f"  FAIL {date}: {msg}, will retry")

        # GCS verification for dates that may have succeeded but status is stale
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
                        if items:
                            all_results[date] = {
                                'date': date, 'run': run, 'success': True,
                                'gcs_path': f"gs://{gcs_prefix}",
                                'message': 'GCS-verified (f.status stale)',
                            }
                            gcs_rescued.append(date)
                            logger.info(f"  GCS-OK {date}: found {len(items)} file(s)")
                    except Exception:
                        pass
                if gcs_rescued:
                    still_failing = [d for d in still_failing if d not in set(gcs_rescued)]
                    print(f"  GCS verification rescued {len(gcs_rescued)} date(s)")
            except Exception as gcs_exc:
                logger.warning(f"  GCS verification skipped: {gcs_exc}")

        succeeded = len(remaining) - len(still_failing)
        print(f"Attempt {attempt}: {succeeded}/{len(remaining)} succeeded, "
              f"{len(still_failing)} queued for retry")
        remaining = still_failing

    for date in remaining:
        all_results[date] = {'date': date, 'run': run, 'success': False,
                             'message': 'Max retries exceeded'}
        logger.error(f"  GIVE UP {date}: max retries ({MAX_RETRIES}) exceeded")

    total_time = time.time() - overall_start
    success_count = sum(1 for r in all_results.values() if r.get('success'))
    print(f"\nCompleted in {total_time:.1f}s ({total_time / 60:.1f} min) — "
          f"{success_count}/{len(dates)} successful")

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
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    if success_count > 0:
        avg_time = sum(r.get('total_time_seconds', 0) for r in results
                       if r.get('success')) / success_count
        print(f"Average processing time: {avg_time:.1f}s ({avg_time / 60:.1f} min)")

    print("\nResults by date:")
    print("-" * 70)

    for date, result in zip(dates, results):
        status = "OK" if result.get('success') else "FAIL"
        message = result.get('message', 'No message')
        print(f"{status} {date}: {message}")

        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")
        if result.get('files_count'):
            print(f"   Files: {result['files_count']}")

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
        description='Run GEFS parquet generation via Lithops Cloud Run workers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Date selection
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--date', type=str,
                            help='Single date to process (YYYYMMDD)')
    date_group.add_argument('--start-date', type=str,
                            help='Start date for range (YYYYMMDD)')
    date_group.add_argument('--days-back', type=int,
                            help='Process last N days')

    parser.add_argument('--end-date', type=str,
                        help='End date for range (YYYYMMDD), used with --start-date')
    parser.add_argument('--run', type=str, default='00',
                        choices=['00', '06', '12', '18'],
                        help='Model run hour (default: 00)')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Max parallel workers for Lithops (default: 10)')
    parser.add_argument('--max-members', type=int, default=None,
                        help='Limit ensemble members (default: all 30)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially without Lithops (local testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show dates without executing')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip interactive confirmation')

    args = parser.parse_args()

    if args.start_date and not args.end_date:
        parser.error("--end-date is required when using --start-date")
    if args.end_date and not args.start_date:
        parser.error("--start-date is required when using --end-date")

    try:
        dates = generate_date_list(args)
    except ValueError as e:
        parser.error(str(e))

    # Print configuration
    print("=" * 70)
    print("GEFS PARQUET GENERATION - LITHOPS CLOUD RUN")
    print("=" * 70)
    print(f"\nRun hour: {args.run}Z")
    print(f"Dates to process: {len(dates)}")
    print(f"  First: {dates[0]}")
    print(f"  Last: {dates[-1]}")
    print(f"Members: {args.max_members or 30}")
    mode = 'Sequential (local)' if args.sequential else f'Lithops Cloud Run (max {args.max_workers} workers)'
    print(f"Execution mode: {mode}")
    print(f"GCS output: gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")

    if args.dry_run:
        print(f"\nDRY RUN - Dates that would be processed:")
        for date in dates:
            print(f"  {date}")
        return

    if len(dates) > 5 and not args.yes:
        response = input(f"\nProcess {len(dates)} dates? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    start_time = time.time()

    if args.sequential:
        results = run_sequential(dates, args.run, args.max_members)
    else:
        results = run_with_lithops(dates, args.run, args.max_workers, args.max_members)

    total_time = time.time() - start_time
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
