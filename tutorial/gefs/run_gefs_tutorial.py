#!/usr/bin/env python3
"""
GEFS Grib-Index-Kerchunk (GIK) Tutorial
========================================

This tutorial demonstrates how to use the GIK method to efficiently access
GEFS (Global Ensemble Forecast System) ensemble forecast data from AWS S3
without downloading full GRIB files.

The GIK method uses pre-built parquet mapping templates to enable fast
data streaming directly from cloud storage.

Prerequisites:
    pip install kerchunk zarr xarray pandas numpy fsspec s3fs requests

Usage:
    python run_gefs_tutorial.py

Template Source:
    Hugging Face: https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/blob/main/gik-fmrc-gefs-20241112.tar.gz

Author: ICPAC GIK Team
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import sys
import warnings
import tarfile
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Hugging Face URL for pre-built GEFS templates
TEMPLATE_URL = "https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-gefs-20241112.tar.gz"
LOCAL_TEMPLATE_FILE = "gik-fmrc-gefs-20241112.tar.gz"

# Target date to process (change this to process different dates)
TARGET_DATE = '20250106'  # YYYYMMDD format
TARGET_RUN = '00'         # Model run time (00, 06, 12, 18)

# Ensemble members to process (gep01-gep30 available)
# Process all 30 ensemble members
ENSEMBLE_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # gep01 to gep30

# Variables to extract
FORECAST_VARIABLES = {
    "2 metre temperature": "TMP:2 m above ground",
    "Total Precipitation": "APCP:surface",
    "10m U wind": "UGRD:10 m above ground",
    "10m V wind": "VGRD:10 m above ground",
}

# Output directory for parquet files
OUTPUT_DIR = Path("output_parquet")

print("="*70)
print("GEFS Grib-Index-Kerchunk Tutorial")
print("="*70)
print(f"Target Date: {TARGET_DATE}")
print(f"Model Run: {TARGET_RUN}Z")
print(f"Ensemble Members: {', '.join(ENSEMBLE_MEMBERS)}")
print(f"Variables: {len(FORECAST_VARIABLES)}")
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
# STEP 2: Import Core GIK Utilities (includes LocalTarGzMappingManager)
# ==============================================================================

def setup_gefs_imports():
    """Add parent gefs folder to path and import utilities."""
    # Add parent folder to path
    parent_gefs = Path(__file__).parent.parent.parent / 'gefs'
    if str(parent_gefs) not in sys.path:
        sys.path.insert(0, str(parent_gefs))

    # Import required utilities
    try:
        from gefs_util import (
            generate_axes,
            filter_build_grib_tree,
            calculate_time_dimensions,
            cs_create_mapped_index_local,
            prepare_zarr_store,
            process_unique_groups,
            LocalTarGzMappingManager
        )
        return {
            'generate_axes': generate_axes,
            'filter_build_grib_tree': filter_build_grib_tree,
            'calculate_time_dimensions': calculate_time_dimensions,
            'cs_create_mapped_index_local': cs_create_mapped_index_local,
            'prepare_zarr_store': prepare_zarr_store,
            'process_unique_groups': process_unique_groups,
            'LocalTarGzMappingManager': LocalTarGzMappingManager
        }
    except ImportError as e:
        print(f"Error importing gefs utilities: {e}")
        print("Make sure you're running from the tutorial folder.")
        return None


# ==============================================================================
# STEP 4: Process Single Ensemble Member
# ==============================================================================

def process_ensemble_member(
    member: str,
    target_date: str,
    target_run: str,
    mapping_manager,  # LocalTarGzMappingManager instance
    utils: Dict,
    forecast_dict: Dict
) -> Optional[Dict]:
    """Process a single GEFS ensemble member using local templates."""

    print(f"\n  Processing ensemble member: {member}")

    # Generate time axes
    axes = utils['generate_axes'](target_date)

    # Build GRIB URLs (only need first 2 files for structure)
    gefs_files = []
    for hour in [0, 3]:
        url = (f"s3://noaa-gefs-pds/gefs.{target_date}/{target_run}/atmos/pgrb2sp25/"
               f"{member}.t{target_run}z.pgrb2s.0p25.f{hour:03d}")
        gefs_files.append(url)

    try:
        # Stage 1: Scan GRIB files to build tree structure
        print(f"    Stage 1: Scanning GRIB structure...")
        _, deflated_store = utils['filter_build_grib_tree'](gefs_files, forecast_dict)

        # Stage 2: Create mapped index using local templates
        print(f"    Stage 2: Creating mapped index from templates...")
        gefs_kind = utils['cs_create_mapped_index_local'](
            axes,
            target_date,
            member,
            tar_gz_path=mapping_manager.tar_gz_path,
            mapping_manager=mapping_manager
        )

        # Stage 3: Prepare final zarr store
        print(f"    Stage 3: Building final zarr store...")
        time_dims, time_coords, times, valid_times, steps = utils['calculate_time_dimensions'](axes)

        zstore, chunk_index = utils['prepare_zarr_store'](deflated_store, gefs_kind)
        updated_zstore = utils['process_unique_groups'](
            zstore, chunk_index, time_dims, time_coords,
            times, valid_times, steps
        )

        print(f"    Completed: {len(updated_zstore)} zarr references created")
        return updated_zstore

    except Exception as e:
        print(f"    Error processing {member}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STEP 5: Save Parquet File
# ==============================================================================

def save_parquet_file(zstore: Dict, output_path: str):
    """Save zarr store as parquet file."""
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
    df.to_parquet(output_path)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"    Saved: {output_path} ({file_size_kb:.1f} KB, {len(df)} rows)")


# ==============================================================================
# STEP 6: Validate Output with xarray
# ==============================================================================

def validate_parquet(parquet_path: str) -> bool:
    """Validate that the parquet file can be opened with xarray."""
    print(f"\n  Validating: {parquet_path}")

    try:
        import fsspec

        # Read parquet to zarr store
        df = pd.read_parquet(parquet_path)
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

        # Create fsspec filesystem
        fs = fsspec.filesystem(
            "reference",
            fo={'refs': zstore, 'version': 1},
            remote_protocol='s3',
            remote_options={'anon': True}
        )
        mapper = fs.get_mapper("")

        # Open with xarray
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

        # List available variables
        variables = list(dt.keys())
        print(f"    Variables found: {len(variables)}")
        for var in variables[:5]:
            print(f"      - {var}")
        if len(variables) > 5:
            print(f"      ... and {len(variables) - 5} more")

        print(f"    Validation PASSED")
        return True

    except Exception as e:
        print(f"    Validation failed: {e}")
        return False


# ==============================================================================
# MAIN TUTORIAL ROUTINE
# ==============================================================================

def main():
    """Run the complete GEFS GIK tutorial."""
    print("\nStarting GEFS Grib-Index-Kerchunk Tutorial\n")

    start_time = time.time()

    # Step 1: Download template file
    print("="*70)
    if not download_template_file(TEMPLATE_URL, LOCAL_TEMPLATE_FILE):
        print("Failed to download template file. Exiting.")
        return False

    # Step 2: Import utilities (do this first to get LocalTarGzMappingManager)
    print("\n[Step 2] Importing GIK utilities...")
    utils = setup_gefs_imports()
    if utils is None:
        print("Failed to import utilities. Exiting.")
        return False
    print("  Utilities imported successfully")

    # Step 3: Initialize mapping manager (using imported class)
    print("\n[Step 3] Initializing template mapping manager...")
    LocalTarGzMappingManager = utils['LocalTarGzMappingManager']
    mapping_manager = LocalTarGzMappingManager(LOCAL_TEMPLATE_FILE)
    available_members = mapping_manager.list_ensemble_members()
    print(f"  Available members: {', '.join(available_members[:5])}...")

    # Verify requested members are available
    for member in ENSEMBLE_MEMBERS:
        if member not in available_members:
            print(f"  Warning: {member} not found in templates")

    # Step 4: Create output directory
    print("\n[Step 4] Creating output directory...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")

    # Step 5: Process ensemble members
    print("\n[Step 5] Processing ensemble members...")
    print("="*70)

    successful = []
    failed = []

    for member in ENSEMBLE_MEMBERS:
        try:
            zstore = process_ensemble_member(
                member=member,
                target_date=TARGET_DATE,
                target_run=TARGET_RUN,
                mapping_manager=mapping_manager,
                utils=utils,
                forecast_dict=FORECAST_VARIABLES
            )

            if zstore:
                # Save parquet file
                output_path = OUTPUT_DIR / f"{member}_{TARGET_DATE}_{TARGET_RUN}z.parquet"
                save_parquet_file(zstore, str(output_path))
                successful.append(member)
            else:
                failed.append(member)

        except Exception as e:
            print(f"  Error processing {member}: {e}")
            failed.append(member)

    # Step 6: Cleanup
    print("\n[Step 6] Cleaning up temporary files...")
    mapping_manager.cleanup()
    print("  Cleanup complete")

    # Step 7: Validate outputs
    print("\n[Step 7] Validating output files...")
    print("="*70)

    if successful:
        sample_parquet = OUTPUT_DIR / f"{successful[0]}_{TARGET_DATE}_{TARGET_RUN}z.parquet"
        validate_parquet(str(sample_parquet))

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("TUTORIAL COMPLETE!")
    print("="*70)
    print(f"\nProcessing Summary:")
    print(f"  Target Date: {TARGET_DATE}")
    print(f"  Members Processed: {len(successful)}")
    print(f"  Members Failed: {len(failed)}")
    print(f"  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    print(f"\nOutput Files:")
    for pf in sorted(OUTPUT_DIR.glob("*.parquet")):
        size_kb = pf.stat().st_size / 1024
        print(f"  - {pf.name} ({size_kb:.1f} KB)")

    print(f"\nNext Steps:")
    print(f"  1. Use run_gefs_data_streaming.py to extract actual data")
    print(f"  2. Process more members by adding to ENSEMBLE_MEMBERS list")
    print(f"  3. Change TARGET_DATE to process different forecast dates")

    return len(successful) > 0


if __name__ == "__main__":
    success = main()
    if success:
        print("\nTutorial completed successfully!")
    else:
        print("\nTutorial failed. Check error messages above.")
        sys.exit(1)
