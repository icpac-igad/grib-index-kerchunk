#!/usr/bin/env python3
"""
ECMWF Grib-Index-Kerchunk (GIK) Tutorial - Full Three-Stage Pipeline
=====================================================================

This tutorial demonstrates the complete GIK method for ECMWF data:
- Stage 1: Scan GRIB files to create deflated parquet (OPTIONAL - ~30 min)
- Stage 2: Merge fresh index with template metadata (fast)
- Stage 3: Create final zarr-compatible parquet files
- Stage 4: Upload parquet files to GCS (for Coiled parallel processing)

The tutorial uses:
- process_single_date from ecmwf_three_stage_multidate.py
- Stage 1 processing from ecmwf_ensemble_par_creator_efficient.py

Prerequisites:
    pip install kerchunk zarr xarray pandas numpy fsspec s3fs requests python-dotenv gcsfs

Environment Configuration:
    Copy .env.example to .env and configure:
    - GCS_BUCKET: GCS bucket name (default: gik-ecmwf-aws-tf)
    - GCS_PARQUET_PREFIX: Prefix path (default: run_par_ecmwf)
    - GCS_SERVICE_ACCOUNT_FILE: Optional path to service account JSON

Usage:
    # Full pipeline (includes Stage 1 - takes ~30 minutes)
    python run_ecmwf_tutorial.py --run-stage1

    # Skip Stage 1 if zip file already exists (fast)
    python run_ecmwf_tutorial.py

    # Process specific date
    python run_ecmwf_tutorial.py --date 20260106 --run-stage1

    # Limit members for faster Stage 1 testing
    python run_ecmwf_tutorial.py --run-stage1 --max-members 3

    # Upload to GCS after processing (for Coiled workers)
    python run_ecmwf_tutorial.py --upload-gcs

Template Source:
    Hugging Face: https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-v2ecmwf_fmrc.tar.gz

Author: ICPAC GIK Team
"""

import os
import sys
import time
import json
import zipfile
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Set up anonymous S3 access for ECMWF data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Add local gik_ecmwf directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
GIK_ECMWF_DIR = SCRIPT_DIR / "gik_ecmwf"
sys.path.insert(0, str(GIK_ECMWF_DIR))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Hugging Face URL for pre-built ECMWF templates (used in Stage 2)
TEMPLATE_URL = "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-v2ecmwf_fmrc.tar.gz"
LOCAL_TEMPLATE_FILE = "gik-fmrc-v2ecmwf_fmrc.tar.gz"

# Default target date and run
DEFAULT_TARGET_DATE = '20260106'  # YYYYMMDD format
DEFAULT_TARGET_RUN = '00'         # Model run time (00 or 12)

# ECMWF forecast hours for Stage 1 (subset for tutorial speed)
# Full: 85 timesteps (0-144h at 3h, 150-360h at 6h)
# Tutorial: First 2 timesteps for fast demo
TUTORIAL_HOURS = [0, 3]  # Just 2 hours for fast Stage 1 demo

# Output directory
OUTPUT_DIR = Path("output_parquet")

# GCS Configuration (loaded from .env or environment variables)
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'gik-ecmwf-aws-tf')
GCS_PARQUET_PREFIX = os.environ.get('GCS_PARQUET_PREFIX', 'run_par_ecmwf')
GCS_SERVICE_ACCOUNT_FILE = os.environ.get('GCS_SERVICE_ACCOUNT_FILE', 'coiled-data.json')


def log_message(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}")


# ==============================================================================
# STEP 1: Download Template File from Hugging Face
# ==============================================================================

def download_template_file(url: str, local_path: str) -> bool:
    """Download the pre-built template tar.gz from Hugging Face."""
    print(f"\n{'='*70}")
    print("[Step 1] Downloading template file from Hugging Face")
    print(f"{'='*70}")

    if os.path.exists(local_path):
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        log_message(f"Template file already exists: {local_path} ({file_size_mb:.1f} MB)")
        return True

    try:
        import requests

        log_message(f"URL: {url}")
        log_message("Downloading... (this may take a few minutes)")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {pct:.1f}%", end='', flush=True)

        print()
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        log_message(f"Downloaded: {local_path} ({file_size_mb:.1f} MB)")
        return True

    except Exception as e:
        log_message(f"Error downloading template: {e}", "ERROR")
        log_message(f"Please download manually from: {url}", "ERROR")
        return False


# ==============================================================================
# STEP 1.5: Upload Parquet Files to GCS (for Coiled parallel processing)
# ==============================================================================

def upload_parquets_to_gcs(
    output_dir: Path,
    date_str: str,
    run: str,
    gcs_bucket: str = GCS_BUCKET,
    gcs_prefix: str = GCS_PARQUET_PREFIX,
    service_account_file: str = GCS_SERVICE_ACCOUNT_FILE
) -> str:
    """
    Upload final parquet files to GCS for Coiled workers to access.

    Files are uploaded to: gs://{bucket}/{prefix}/{date}_{run}z/

    Returns the GCS path where files were uploaded.
    """
    print(f"\n{'='*70}")
    print("[Step 4] Uploading Parquet Files to GCS")
    print(f"{'='*70}")

    try:
        import gcsfs

        # Setup GCS filesystem
        if service_account_file and os.path.exists(service_account_file):
            log_message(f"Using service account: {service_account_file}")
            fs = gcsfs.GCSFileSystem(token=service_account_file)
        else:
            log_message("Using Application Default Credentials (ADC)")
            fs = gcsfs.GCSFileSystem()

        # Build GCS path: gs://bucket/prefix/YYYYMMDD_runz/
        gcs_path = f"{gcs_bucket}/{gcs_prefix}/{date_str}_{run}z"
        log_message(f"GCS destination: gs://{gcs_path}")

        # Find all *_final.parquet files
        parquet_files = list(output_dir.glob("*_final.parquet"))

        if not parquet_files:
            log_message(f"No *_final.parquet files found in {output_dir}", "WARNING")
            return None

        log_message(f"Found {len(parquet_files)} parquet files to upload")

        # Upload each file
        uploaded_count = 0
        for pf in parquet_files:
            gcs_file_path = f"{gcs_path}/{pf.name}"
            try:
                fs.put(str(pf), gcs_file_path)
                size_kb = pf.stat().st_size / 1024
                log_message(f"  Uploaded: {pf.name} ({size_kb:.1f} KB)")
                uploaded_count += 1
            except Exception as e:
                log_message(f"  Failed to upload {pf.name}: {e}", "ERROR")

        log_message(f"Uploaded {uploaded_count}/{len(parquet_files)} files to GCS")

        return f"gs://{gcs_path}"

    except ImportError:
        log_message("gcsfs not installed. Install with: pip install gcsfs", "ERROR")
        return None
    except Exception as e:
        log_message(f"GCS upload failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


def check_gcs_parquets(
    date_str: str,
    run: str,
    gcs_bucket: str = GCS_BUCKET,
    gcs_prefix: str = GCS_PARQUET_PREFIX,
    service_account_file: str = GCS_SERVICE_ACCOUNT_FILE
) -> list:
    """
    Check if parquet files exist on GCS for a given date/run.

    Returns list of GCS paths if found, empty list otherwise.
    """
    try:
        import gcsfs

        if service_account_file and os.path.exists(service_account_file):
            fs = gcsfs.GCSFileSystem(token=service_account_file)
        else:
            fs = gcsfs.GCSFileSystem()

        gcs_path = f"{gcs_bucket}/{gcs_prefix}/{date_str}_{run}z"

        if fs.exists(gcs_path):
            files = fs.glob(f"{gcs_path}/*_final.parquet")
            return [f"gs://{f}" for f in files]

        return []

    except Exception:
        return []


# ==============================================================================
# STEP 2: Run Stage 1 - GRIB Scanning (Optional, ~30 minutes)
# ==============================================================================

def run_stage1_processing(
    date_str: str,
    run: str,
    max_members: int = None,
    hours: list = None
) -> Path:
    """
    Run Stage 1 processing using ecmwf_ensemble_par_creator_efficient.py

    This scans GRIB files to create the deflated parquet files needed
    for the three-stage pipeline.

    Returns the path to the created zip file, or None if failed.
    """
    print(f"\n{'='*70}")
    print("[Step 2] Running Stage 1 - GRIB Scanning")
    print(f"{'='*70}")
    log_message("This step takes ~30 minutes for full processing")
    log_message(f"Date: {date_str}, Run: {run}z")

    if hours is None:
        hours = TUTORIAL_HOURS

    try:
        # Import Stage 1 functions
        from ecmwf_ensemble_par_creator_efficient import (
            process_ecmwf_files_efficiently,
            extract_individual_member_parquets,
            save_processing_metadata
        )
        import fsspec

        log_message("Imported functions from ecmwf_ensemble_par_creator_efficient.py")

        # Create output directory for Stage 1
        stage1_output_dir = Path(f"ecmwf_{date_str}_{run}_efficient")
        stage1_output_dir.mkdir(exist_ok=True)
        log_message(f"Stage 1 output directory: {stage1_output_dir}")

        # Build list of GRIB files to process
        ecmwf_files = []
        for hour in hours:
            url = f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-{hour}h-enfo-ef.grib2"
            ecmwf_files.append(url)

        log_message(f"Processing {len(ecmwf_files)} GRIB files for hours: {hours}")

        # Check file availability
        fs = fsspec.filesystem("s3", anon=True)
        available_files = []
        for f in ecmwf_files:
            try:
                if fs.exists(f):
                    available_files.append(f)
                    log_message(f"  Found: {f.split('/')[-1]}")
                else:
                    log_message(f"  Not found: {f.split('/')[-1]}", "WARNING")
            except Exception as e:
                log_message(f"  Error checking {f}: {e}", "WARNING")

        if not available_files:
            log_message("No GRIB files found on S3!", "ERROR")
            return None

        # Define target members
        if max_members:
            # Control (-1) + first N-1 ensemble members
            target_members = [-1] + list(range(1, max_members))
        else:
            # All 51 members: control + ens01-ens50
            target_members = [-1] + list(range(1, 51))

        log_message(f"Target members: {len(target_members)} (control + {len(target_members)-1} ensemble)")

        # Run Stage 1 processing
        stage1_start = time.time()

        results = process_ecmwf_files_efficiently(
            available_files,
            date_str,
            run,
            stage1_output_dir
        )

        # Extract individual member parquets
        extraction_results = extract_individual_member_parquets(
            results['ensemble_tree'],
            stage1_output_dir,
            target_members
        )

        # Save metadata
        save_processing_metadata(
            results,
            extraction_results,
            stage1_output_dir,
            date_str,
            run
        )

        stage1_time = time.time() - stage1_start
        log_message(f"Stage 1 completed in {stage1_time/60:.1f} minutes")

        # Create zip file for process_single_date
        zip_path = create_stage1_zip(stage1_output_dir, date_str, run)

        return zip_path

    except ImportError as e:
        log_message(f"Import error: {e}", "ERROR")
        log_message(f"Make sure ecmwf_ensemble_par_creator_efficient.py is in: {ECMWF_DIR}", "ERROR")
        return None
    except Exception as e:
        log_message(f"Stage 1 failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None


def create_stage1_zip(stage1_dir: Path, date_str: str, run: str) -> Path:
    """Create zip file from Stage 1 output for process_single_date."""
    zip_filename = f"ecmwf_{date_str}_{run}z_efficient.zip"
    zip_path = Path(zip_filename)

    log_message(f"Creating Stage 1 zip file: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in stage1_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(stage1_dir.parent)
                    zipf.write(file_path, arcname)

        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        log_message(f"Created zip: {zip_path} ({zip_size_mb:.2f} MB)")
        return zip_path

    except Exception as e:
        log_message(f"Error creating zip: {e}", "ERROR")
        return None


# ==============================================================================
# STEP 3: Run Three-Stage Pipeline using process_single_date
# ==============================================================================

def run_three_stage_pipeline(
    date_str: str,
    run: str,
    template_path: str,
    max_members: int = None,
    skip_grib_scan: bool = False,
    parallel_workers: int = 8
) -> bool:
    """
    Run the complete three-stage pipeline using process_single_date
    from ecmwf_three_stage_multidate.py.

    When skip_grib_scan=True, builds Stage 1 from template (no zip needed).
    Otherwise requires the Stage 1 zip file to exist.
    """
    print(f"\n{'='*70}")
    mode = "template fast-path" if skip_grib_scan else "process_single_date"
    print(f"[Step 3] Running Three-Stage Pipeline ({mode})")
    print(f"{'='*70}")

    try:
        # Import the main processing function
        from ecmwf_three_stage_multidate import process_single_date

        log_message("Imported process_single_date from ecmwf_three_stage_multidate.py")
        log_message(f"Date: {date_str}, Run: {run}z")
        log_message(f"Template: {template_path}")
        log_message(f"Max members: {max_members if max_members else 'all'}")
        log_message(f"Skip GRIB scan: {skip_grib_scan}")
        log_message(f"Parallel workers: {parallel_workers}")

        # Run the three-stage pipeline with local template
        success, output_dir = process_single_date(
            date_str=date_str,
            run=run,
            max_members=max_members,
            use_local_template=True,
            local_template_path=template_path,
            skip_grib_scan=skip_grib_scan,
            parallel_workers=parallel_workers
        )

        if success:
            log_message(f"Three-stage pipeline completed successfully!")
            log_message(f"Output directory: {output_dir}")
            return True
        else:
            log_message("Three-stage pipeline failed!", "ERROR")
            return False

    except ImportError as e:
        log_message(f"Import error: {e}", "ERROR")
        log_message(f"Make sure ecmwf_three_stage_multidate.py is in: {ECMWF_DIR}", "ERROR")
        return False
    except Exception as e:
        log_message(f"Pipeline failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# MAIN TUTORIAL ROUTINE
# ==============================================================================

def main():
    """Run the complete ECMWF GIK tutorial."""
    parser = argparse.ArgumentParser(
        description='ECMWF GIK Tutorial - Full Three-Stage Pipeline'
    )
    parser.add_argument('--date', type=str, default=DEFAULT_TARGET_DATE,
                        help=f'Target date (YYYYMMDD, default: {DEFAULT_TARGET_DATE})')
    parser.add_argument('--run', type=str, default=DEFAULT_TARGET_RUN,
                        help=f'Model run hour (default: {DEFAULT_TARGET_RUN})')
    parser.add_argument('--run-stage1', action='store_true',
                        help='Run Stage 1 GRIB scanning (takes ~30 min, skip if zip exists)')
    parser.add_argument('--max-members', type=int, default=None,
                        help='Maximum number of members to process')
    parser.add_argument('--hours', type=str, default=None,
                        help='Forecast hours for Stage 1 (comma-separated, e.g., "0,3,6")')
    parser.add_argument('--upload-gcs', action='store_true',
                        help='Upload final parquet files to GCS (for Coiled parallel processing)')
    parser.add_argument('--gcs-bucket', type=str, default=GCS_BUCKET,
                        help=f'GCS bucket name (default: {GCS_BUCKET})')
    parser.add_argument('--gcs-prefix', type=str, default=GCS_PARQUET_PREFIX,
                        help=f'GCS prefix path (default: {GCS_PARQUET_PREFIX})')
    parser.add_argument('--skip-grib-scan', action='store_true',
                        help='Skip Stage 1 GRIB scanning entirely — build from template '
                             '(Phase 1 fast-path: ~73 min → ~5 sec, no zip needed)')
    parser.add_argument('--parallel-workers', type=int, default=8,
                        help='Number of parallel workers for Stage 2 (default: 8, use 1 for sequential)')
    args = parser.parse_args()

    target_date = args.date
    target_run = args.run

    # Parse hours if provided
    if args.hours:
        stage1_hours = [int(h.strip()) for h in args.hours.split(',')]
    else:
        stage1_hours = TUTORIAL_HOURS

    # Print configuration
    print("="*70)
    print("ECMWF Grib-Index-Kerchunk Tutorial")
    print("="*70)
    print(f"Target Date: {target_date}")
    print(f"Model Run: {target_run}Z")
    print(f"Run Stage 1: {'Yes' if args.run_stage1 else 'No (use existing zip)'}")
    print(f"Max Members: {args.max_members if args.max_members else 'all'}")
    print(f"Stage 1 Hours: {stage1_hours}")
    print(f"Skip GRIB Scan: {'Yes (template fast-path)' if args.skip_grib_scan else 'No'}")
    print(f"Parallel Workers: {args.parallel_workers}")
    print(f"Upload to GCS: {'Yes' if args.upload_gcs else 'No'}")
    if args.upload_gcs:
        print(f"GCS Destination: gs://{args.gcs_bucket}/{args.gcs_prefix}/{target_date}_{target_run}z/")
    print(f"ECMWF Module Path: {GIK_ECMWF_DIR}")
    print("="*70)

    start_time = time.time()

    # Step 1: Download template file (always needed for Stage 2)
    if not download_template_file(TEMPLATE_URL, LOCAL_TEMPLATE_FILE):
        log_message("Failed to download template file. Exiting.", "ERROR")
        return False

    # Step 2: Check for Stage 1 zip or run Stage 1
    if args.skip_grib_scan:
        log_message("GRIB scanning SKIPPED — using template fast-path (Phase 1)")
        log_message(f"Template: {LOCAL_TEMPLATE_FILE}")
    else:
        zip_patterns = [
            f"ecmwf_{target_date}_{target_run}z_efficient.zip",
            f"ecmwf_{target_date}_{target_run}_efficient.zip"
        ]

        zip_file = None
        for pattern in zip_patterns:
            if Path(pattern).exists():
                zip_file = Path(pattern)
                break

        if zip_file:
            log_message(f"Found existing Stage 1 zip: {zip_file}")
            if args.run_stage1:
                log_message("--run-stage1 specified, but zip exists. Using existing zip.")
        else:
            if args.run_stage1:
                # Run Stage 1 to create the zip
                zip_file = run_stage1_processing(
                    target_date,
                    target_run,
                    max_members=args.max_members,
                    hours=stage1_hours
                )
                if not zip_file:
                    log_message("Stage 1 failed. Cannot continue.", "ERROR")
                    return False
            else:
                log_message(f"Stage 1 zip file not found: {zip_patterns[0]}", "ERROR")
                log_message("Options:", "ERROR")
                log_message("  1. Run with --run-stage1 to create it (~30 min)", "ERROR")
                log_message("  2. Run with --skip-grib-scan for template fast-path (~5 sec)", "ERROR")
                log_message("  3. Download pre-built zip if available", "ERROR")
                return False

    # Step 3: Run three-stage pipeline
    success = run_three_stage_pipeline(
        date_str=target_date,
        run=target_run,
        template_path=LOCAL_TEMPLATE_FILE,
        max_members=args.max_members,
        skip_grib_scan=args.skip_grib_scan,
        parallel_workers=args.parallel_workers
    )

    # Step 4: Upload to GCS if requested
    output_dir = Path(f"ecmwf_three_stage_{target_date}_{target_run}z")
    gcs_path = None

    if success and args.upload_gcs:
        gcs_path = upload_parquets_to_gcs(
            output_dir=output_dir,
            date_str=target_date,
            run=target_run,
            gcs_bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix
        )

    # Summary
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("TUTORIAL COMPLETE!")
    print(f"{'='*70}")
    print(f"\nProcessing Summary:")
    print(f"  Target Date: {target_date} {target_run}Z")
    print(f"  Pipeline Success: {'Yes' if success else 'No'}")
    print(f"  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    # List output files
    if output_dir.exists():
        print(f"\nLocal Output Files in {output_dir}:")
        for pf in sorted(output_dir.glob("*.parquet")):
            size_kb = pf.stat().st_size / 1024
            print(f"  - {pf.name} ({size_kb:.1f} KB)")

    # GCS upload summary
    if gcs_path:
        print(f"\nGCS Upload:")
        print(f"  Location: {gcs_path}")
        print(f"  Use for Coiled: --gcs-parquet-path {gcs_path}")

    print(f"\nNext Steps:")
    print(f"  1. Run stream_cgan_variables.py to stream data locally")
    print(f"  2. Run stream_cgan_variables_coiled_simple.py for parallel processing (requires GCS upload)")
    print(f"  3. Process different date: python run_ecmwf_tutorial.py --date YYYYMMDD --run-stage1")
    if not args.upload_gcs:
        print(f"  4. Upload to GCS: python run_ecmwf_tutorial.py --date {target_date} --upload-gcs")

    return success


if __name__ == "__main__":
    success = main()
    if success:
        print("\nTutorial completed successfully!")
    else:
        print("\nTutorial failed. Check error messages above.")
        sys.exit(1)
