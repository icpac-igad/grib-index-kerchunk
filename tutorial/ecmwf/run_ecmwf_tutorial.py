#!/usr/bin/env python3
"""
ECMWF Grib-Index-Kerchunk (GIK) Tutorial
=========================================

This tutorial demonstrates how to use the GIK method to efficiently access
ECMWF IFS (Integrated Forecasting System) ensemble forecast data from AWS S3
without downloading full GRIB files.

The GIK method:
1. Downloads small .index files from S3 to get byte offsets
2. Merges with template metadata from tar.gz
3. Creates parquet files with step_XXX format for data streaming

Prerequisites:
    pip install kerchunk zarr xarray pandas numpy fsspec s3fs requests

Usage:
    python run_ecmwf_tutorial.py

Template Source:
    Hugging Face: https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/blob/main/gik-fmrc-v2ecmwf_fmrc.tar.gz

Author: ICPAC GIK Team
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
import tarfile
import tempfile
import shutil
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import fsspec

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up anonymous S3 access for ECMWF data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Hugging Face URL for pre-built ECMWF templates
TEMPLATE_URL = "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-v2ecmwf_fmrc.tar.gz"
LOCAL_TEMPLATE_FILE = "gik-fmrc-v2ecmwf_fmrc.tar.gz"

# Reference date (date the templates were built from)
REFERENCE_DATE = '20240529'

# Target date to process (change this to process different dates)
TARGET_DATE = '20241201'  # YYYYMMDD format - use a recent date with available data
TARGET_RUN = '00'         # Model run time (00 or 12)

# Ensemble members to process
# control = deterministic control run
# ens01-ens50 = perturbed ensemble members
ENSEMBLE_MEMBERS = ['control', 'ens01', 'ens02', 'ens03']  # Process 4 members for demo

# Output directory for parquet files
OUTPUT_DIR = Path("output_parquet")

# S3 configuration
S3_BUCKET = "ecmwf-forecasts"

# ECMWF forecast hours (85 total)
# 3-hourly from 0-144h, then 6-hourly from 150-360h
HOURS_3H = list(range(0, 145, 3))     # 0-144h at 3h intervals (49 steps)
HOURS_6H = list(range(150, 361, 6))   # 150-360h at 6h intervals (36 steps)
ECMWF_FORECAST_HOURS = HOURS_3H + HOURS_6H  # Total: 85 steps

# For tutorial, limit to first 25 timesteps (0-72h) for faster processing
TUTORIAL_FORECAST_HOURS = list(range(0, 73, 3))  # 0, 3, 6, ..., 72h (25 steps)

print("="*70)
print("ECMWF Grib-Index-Kerchunk Tutorial")
print("="*70)
print(f"Reference Date (template): {REFERENCE_DATE}")
print(f"Target Date: {TARGET_DATE}")
print(f"Model Run: {TARGET_RUN}Z")
print(f"Ensemble Members: {', '.join(ENSEMBLE_MEMBERS)}")
print(f"Forecast Hours: {len(TUTORIAL_FORECAST_HOURS)} timesteps (T+0 to T+{TUTORIAL_FORECAST_HOURS[-1]}h)")
print("="*70)


# ==============================================================================
# STEP 1: Download Template File from Hugging Face
# ==============================================================================

def download_template_file(url: str, local_path: str) -> bool:
    """Download the pre-built template tar.gz from Hugging Face."""
    print(f"\n[Step 1] Downloading template file from Hugging Face...")

    if os.path.exists(local_path):
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Template file already exists: {local_path} ({file_size_mb:.1f} MB)")
        return True

    try:
        import requests

        print(f"  URL: {url}")
        print(f"  Downloading... (this may take a few minutes)")

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
        print(f"  Downloaded: {local_path} ({file_size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  Error downloading template: {e}")
        print(f"  Please download manually from: {url}")
        return False


# ==============================================================================
# STEP 2: Parse ECMWF Index Files
# ==============================================================================

def parse_grib_index(idx_url: str, member_filter: str) -> List[Dict]:
    """
    Parse ECMWF GRIB index file (JSON format) to extract byte ranges and metadata.

    The .index files on S3 contain JSON entries with:
    - _offset: byte offset in GRIB file
    - _length: byte length of message
    - param: variable name (e.g., '2t', 'tp')
    - levtype: level type (e.g., 'sfc', 'pl')
    - number: ensemble member number (0=control, 1-50=perturbed)
    """
    try:
        fs = fsspec.filesystem("s3", anon=True)

        entries = []
        with fs.open(idx_url, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                # Parse JSON entry
                entry_data = json.loads(line.strip())

                # Parse ensemble member number
                member_num = int(entry_data.get('number', 0))
                if member_num == 0:
                    member = 'control'
                else:
                    member = f'ens{member_num:02d}'

                # Filter by member if specified
                if member_filter and member != member_filter:
                    continue

                # Extract metadata
                entry = {
                    'byte_offset': entry_data['_offset'],
                    'byte_length': entry_data['_length'],
                    'variable': entry_data.get('param', ''),
                    'level': entry_data.get('levtype', ''),
                    'step': entry_data.get('step', '0'),
                    'member': member,
                    'date': entry_data.get('date', ''),
                    'time': entry_data.get('time', ''),
                }

                entries.append(entry)

        return entries

    except Exception as e:
        print(f"      Warning: Error parsing index {idx_url}: {e}")
        return []


def create_references_from_index(grib_url: str, idx_entries: List[Dict]) -> Dict[str, Any]:
    """
    Create kerchunk-style references using index byte ranges.

    Creates references in the format: [url, offset, length]
    """
    references = {}

    for entry in idx_entries:
        start = entry['byte_offset']
        length = entry['byte_length']
        variable = entry['variable']
        level = entry['level']
        member = entry['member']

        # Create reference key: varname/level/member/0.0.0
        if level == 'sfc':
            level_str = 'sfc'
        elif level == 'pl':
            level_str = 'pl'
        else:
            level_str = level

        # Normalize member name
        if member == 'control':
            member_str = 'control'
        else:
            member_str = member.replace('ens', 'ens_')

        key = f"{variable}/{level_str}/{member_str}/0.0.0"

        # Create reference [url, offset, length]
        references[key] = [grib_url, start, length]

    return references


# ==============================================================================
# STEP 3: Build Complete Parquet from Index Files
# ==============================================================================

def build_parquet_from_indices(
    date_str: str,
    run: str,
    member_name: str,
    hours: List[int],
    template_metadata: Dict
) -> Dict[str, Any]:
    """
    Build complete parquet with all time steps using S3 index files.

    For each forecast hour:
    1. Download the .index file from S3
    2. Parse it to get byte offsets for this member
    3. Create references with step_XXX prefix

    Then merge with template metadata.
    """
    all_refs = {}

    # Normalize member name for filtering
    if member_name == 'control':
        filter_member = 'control'
    else:
        filter_member = member_name.replace('ens', 'ens').replace('_', '')

    print(f"    Processing {len(hours)} forecast hours...")

    successful_hours = 0
    for i, hour in enumerate(hours):
        try:
            # Build URLs
            idx_url = f"s3://{S3_BUCKET}/{date_str}/{run}z/ifs/0p25/enfo/{date_str}000000-{hour}h-enfo-ef.index"
            grib_url = f"s3://{S3_BUCKET}/{date_str}/{run}z/ifs/0p25/enfo/{date_str}000000-{hour}h-enfo-ef"

            # Parse index for this member
            idx_entries = parse_grib_index(idx_url, member_filter=filter_member)

            if not idx_entries:
                continue

            # Create references
            hour_refs = create_references_from_index(grib_url, idx_entries)

            # Add to combined references with step_XXX prefix
            for key, ref in hour_refs.items():
                timestep_key = f"step_{hour:03d}/{key}"
                all_refs[timestep_key] = ref

            successful_hours += 1

            # Progress logging
            if i == 0 or (i + 1) % 10 == 0 or i == len(hours) - 1:
                print(f"      T+{hour:03d}h: {len(hour_refs)} variables ({successful_hours}/{i+1} hours)")

        except Exception as e:
            print(f"      T+{hour:03d}h: Error - {str(e)[:50]}")

    # Merge with template metadata
    merged_refs = {**template_metadata, **all_refs}

    print(f"    Completed: {successful_hours}/{len(hours)} hours, {len(all_refs)} data chunks")
    return merged_refs


# ==============================================================================
# STEP 4: Load Template Metadata
# ==============================================================================

def load_template_metadata(tar_gz_path: str, member_name: str) -> Dict:
    """
    Load template metadata (.zarray, .zattrs) from tar.gz archive.

    The templates contain zarr structure metadata that defines:
    - Grid dimensions and data types
    - Variable attributes and names
    - Coordinate information
    """
    # Build member path pattern
    if member_name == 'control':
        member_dir = 'ens_control'
    else:
        member_num_str = member_name.replace('ens', '')
        member_dir = f'ens_{int(member_num_str):02d}'

    print(f"    Loading template metadata: {member_dir}")

    metadata_refs = {}
    temp_dir = tempfile.mkdtemp(prefix='ecmwf_template_')

    try:
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            # Find first parquet file for this member (they all have same metadata)
            member_files = [m for m in tar.getnames()
                           if f'/{member_dir}/' in m and m.endswith('.par')]

            if not member_files:
                print(f"      No template files found for {member_dir}")
                return {}

            # Extract and read first template file
            tar.extract(tar.getmember(member_files[0]), path=temp_dir)
            extracted_path = Path(temp_dir) / member_files[0]
            template_df = pd.read_parquet(extracted_path)

            # Extract only metadata keys (.zarray, .zattrs, etc.)
            for _, row in template_df.iterrows():
                key = row['key']
                value = row['value']

                # Only keep metadata (not data chunks)
                if '.z' in key or key.startswith('zarr') or key == 'metadata':
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            pass
                    metadata_refs[key] = value

            print(f"      Loaded {len(metadata_refs)} metadata entries")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return metadata_refs


# ==============================================================================
# STEP 5: Save Parquet File
# ==============================================================================

def save_parquet_file(refs: Dict, output_path: str):
    """Save references as parquet file."""
    data = []

    for key, value in refs.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        elif isinstance(value, bytes):
            encoded_value = value
        else:
            encoded_value = str(value).encode('utf-8')

        data.append((key, encoded_value))

    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_path)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"    Saved: {output_path} ({file_size_kb:.1f} KB, {len(df)} rows)")


# ==============================================================================
# STEP 6: Validate Output
# ==============================================================================

def validate_parquet(parquet_path: str) -> bool:
    """Validate that the parquet file was created successfully."""
    print(f"\n  Validating: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)

        # Count entries
        print(f"    Total entries: {len(df)}")

        # Count data chunks (step_XXX format)
        data_chunks = 0
        step_hours = set()
        variables = set()

        for _, row in df.iterrows():
            key = row['key']

            # Check for step_XXX format
            match = re.match(r'^step_(\d+)/([^/]+)/', key)
            if match:
                data_chunks += 1
                step_hours.add(int(match.group(1)))
                variables.add(match.group(2))

        print(f"    Data chunks (step_XXX format): {data_chunks}")
        print(f"    Forecast hours covered: {len(step_hours)}")
        if step_hours:
            print(f"      Range: T+{min(step_hours)}h to T+{max(step_hours)}h")
        print(f"    Variables found: {len(variables)}")
        if variables:
            sample_vars = sorted(variables)[:10]
            print(f"      Sample: {', '.join(sample_vars)}")

        # Show sample data chunk keys
        sample_keys = [k for k in df['key'].tolist() if k.startswith('step_')][:3]
        if sample_keys:
            print(f"    Sample data chunk keys:")
            for key in sample_keys:
                print(f"      - {key}")

        if data_chunks > 0:
            print(f"    Validation PASSED")
            return True
        else:
            print(f"    Validation FAILED: No data chunks found")
            return False

    except Exception as e:
        print(f"    Validation failed: {e}")
        return False


# ==============================================================================
# MAIN TUTORIAL ROUTINE
# ==============================================================================

def main():
    """Run the complete ECMWF GIK tutorial."""
    print("\nStarting ECMWF Grib-Index-Kerchunk Tutorial\n")

    start_time = time.time()

    # Step 1: Download template file
    print("="*70)
    if not download_template_file(TEMPLATE_URL, LOCAL_TEMPLATE_FILE):
        print("Failed to download template file. Exiting.")
        return False

    # Step 2: Create output directory
    print("\n[Step 2] Creating output directory...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")

    # Step 3: Process ensemble members
    print("\n[Step 3] Processing ensemble members...")
    print("="*70)

    successful = []
    failed = []

    for member in ENSEMBLE_MEMBERS:
        print(f"\n  Processing member: {member}")

        try:
            # Load template metadata
            template_metadata = load_template_metadata(LOCAL_TEMPLATE_FILE, member)

            if not template_metadata:
                print(f"    Failed to load template for {member}")
                failed.append(member)
                continue

            # Build parquet from index files
            refs = build_parquet_from_indices(
                date_str=TARGET_DATE,
                run=TARGET_RUN,
                member_name=member,
                hours=TUTORIAL_FORECAST_HOURS,
                template_metadata=template_metadata
            )

            if not refs:
                print(f"    Failed to build parquet for {member}")
                failed.append(member)
                continue

            # Save parquet file
            if member == 'control':
                output_filename = "stage3_control_final.parquet"
            else:
                member_num = member.replace('ens', '')
                output_filename = f"stage3_ens_{int(member_num):02d}_final.parquet"

            output_path = OUTPUT_DIR / output_filename
            save_parquet_file(refs, str(output_path))
            successful.append(member)

        except Exception as e:
            print(f"    Error processing {member}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(member)

    # Step 4: Validate outputs
    print("\n[Step 4] Validating output files...")
    print("="*70)

    if successful:
        # Validate first successful member
        if 'control' in successful:
            sample_parquet = OUTPUT_DIR / "stage3_control_final.parquet"
        else:
            member_num = successful[0].replace('ens', '')
            sample_parquet = OUTPUT_DIR / f"stage3_ens_{int(member_num):02d}_final.parquet"
        validate_parquet(str(sample_parquet))

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("TUTORIAL COMPLETE!")
    print("="*70)
    print(f"\nProcessing Summary:")
    print(f"  Target Date: {TARGET_DATE} {TARGET_RUN}Z")
    print(f"  Forecast Hours: T+{TUTORIAL_FORECAST_HOURS[0]}h to T+{TUTORIAL_FORECAST_HOURS[-1]}h")
    print(f"  Members Processed: {len(successful)}")
    print(f"  Members Failed: {len(failed)}")
    print(f"  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    print(f"\nOutput Files:")
    for pf in sorted(OUTPUT_DIR.glob("*.parquet")):
        size_kb = pf.stat().st_size / 1024
        print(f"  - {pf.name} ({size_kb:.1f} KB)")

    print(f"\nNext Steps:")
    print(f"  1. Run run_ecmwf_data_streaming.py to stream data and create plots")
    print(f"  2. Process more members by adding to ENSEMBLE_MEMBERS list")
    print(f"  3. Change TARGET_DATE to process different forecast dates")
    print(f"  4. Set TUTORIAL_FORECAST_HOURS = ECMWF_FORECAST_HOURS for all 85 timesteps")

    return len(successful) > 0


if __name__ == "__main__":
    success = main()
    if success:
        print("\nTutorial completed successfully!")
    else:
        print("\nTutorial failed. Check error messages above.")
        sys.exit(1)
