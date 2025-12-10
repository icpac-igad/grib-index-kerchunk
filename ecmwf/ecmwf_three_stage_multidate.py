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
# STAGE 2: INDEX + GCS TEMPLATES
# ==============================================================================

def run_stage2_with_gcs_templates(test_date, test_run, test_members, output_dir,
                                   use_local_template=False, local_template_path=None):
    """Stage 2: INDEX + Templates merge (GCS or local).

    Args:
        test_date: Date string (YYYYMMDD)
        test_run: Run hour (e.g., '00')
        test_members: List of member names to process
        output_dir: Output directory path
        use_local_template: If True, use local tar.gz file instead of GCS bucket
        local_template_path: Path to local tar.gz file (default: gik-fmrc-v2ecmwf_fmrc.tar.gz)
    """
    template_source = "LOCAL TAR.GZ" if use_local_template else "GCS BUCKET"
    log_stage(2, f"INDEX + TEMPLATES MERGE ({template_source}, All 85 hours)")

    start_time = time.time()

    try:
        from ecmwf_index_processor import build_complete_parquet_from_indices

        log_checkpoint(f"Target date: {test_date}")
        log_checkpoint(f"Reference date: {REFERENCE_DATE}")
        log_checkpoint(f"Processing {len(test_members)} members")

        member_results = {}

        for member in test_members:
            # Normalize member format
            if member == 'control':
                member_normalized = 'control'
            else:
                member_normalized = member.replace('_', '')
                if member_normalized.startswith('ens'):
                    member_num_str = member_normalized.replace('ens', '')
                    member_normalized = f'ens{int(member_num_str):02d}'

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
                        use_local_template: bool = False, local_template_path: Optional[str] = None) -> Tuple[bool, Optional[Path]]:
    """
    Process a single date through all three stages.

    Parameters:
    - date_str: Date string (e.g., '20251125')
    - run: Run hour (e.g., '00')
    - max_members: Maximum number of members to process (optional)
    - use_local_template: If True, use local tar.gz file instead of GCS bucket
    - local_template_path: Path to local tar.gz file (default: gik-fmrc-v2ecmwf_fmrc.tar.gz)

    Returns:
    - Tuple of (success, output_directory_path)
    """
    print("\n" + "="*80)
    print(f"PROCESSING DATE: {date_str} {run}z")
    print("="*80)

    start_time = time.time()

    # Find the zip file for this date
    zip_patterns = [
        f"ecmwf_{date_str}_{run}z_efficient.zip",
        f"ecmwf_{date_str}_{run}_efficient.zip"
    ]

    zip_file = None
    for pattern in zip_patterns:
        if Path(pattern).exists():
            zip_file = Path(pattern)
            break

    if not zip_file:
        log_checkpoint(f"Zip file not found for {date_str}_{run}z")
        log_checkpoint(f"Expected: {zip_patterns[0]} or {zip_patterns[1]}")
        return False, None

    log_checkpoint(f"Using zip file: {zip_file}")

    # Create output directory
    output_dir = Path(f"ecmwf_three_stage_{date_str}_{run}z")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract zip file
        extract_dir = output_dir / "extracted"
        extraction_info = unzip_and_prepare(zip_file, extract_dir)

        member_parquets = extraction_info['members']
        test_members = sorted(member_parquets.keys())

        log_checkpoint(f"Found {len(test_members)} members in zip")

        # Apply max_members limit
        if max_members and max_members < len(test_members):
            test_members = test_members[:max_members]
            member_parquets = {k: v for k, v in member_parquets.items() if k in test_members}
            log_checkpoint(f"Limiting to {max_members} members")

        # Stage 1
        deflated_stores = run_stage1_prebuilt(member_parquets, date_str, run)

        if not deflated_stores:
            return False, None

        # Stage 2
        stage2_refs = run_stage2_with_gcs_templates(
            date_str, run, test_members, output_dir,
            use_local_template=use_local_template,
            local_template_path=local_template_path
        )

        # Stage 3
        stage3_results = None
        if deflated_stores and stage2_refs:
            stage3_results = run_stage3(deflated_stores, stage2_refs, date_str, output_dir)

        # Cleanup extracted files to save space
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            log_checkpoint("Cleaned up extracted files")

        elapsed = time.time() - start_time

        print(f"\nResults for {date_str}:")
        print(f"   Processing time: {elapsed/60:.1f} minutes")
        print(f"   Stage 1: {'Success' if deflated_stores else 'Failed'}")
        print(f"   Stage 2: {'Success' if stage2_refs else 'Failed'} ({len(stage2_refs) if stage2_refs else 0} members)")
        print(f"   Stage 3: {'Success' if stage3_results else 'Failed'} ({len(stage3_results) if stage3_results else 0} members)")

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
    args = parser.parse_args()

    print("="*80)
    print("ECMWF Three-Stage Multi-Date Processing")
    print("="*80)

    overall_start = time.time()

    dates = args.dates
    run = args.run

    log_message(f"Processing {len(dates)} dates: {dates}")
    log_message(f"Run: {run}z")
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
            local_template_path=args.local_template_path
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
