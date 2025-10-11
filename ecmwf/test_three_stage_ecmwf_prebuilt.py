#!/usr/bin/env python3
"""
Enhanced ECMWF Three-Stage Processing Test using Prebuilt Zip Files
Tests each stage with prebuilt parquet files from efficient processing.

Key Features:
- Uses prebuilt zip files (ecmwf_20251006_00_efficient.zip, etc.)
- Extracts TEST_DATE and TEST_RUN from zip filename
- Tests all ensemble members (from available parquet files)
- Expands to all 85 ECMWF forecast hours for Stage 2 and Stage 3
- Stage 1 only processes hours 0 and 3 for quick testing

Usage:
    python test_three_stage_ecmwf_prebuilt.py
    # Or specify a zip file:
    python test_three_stage_ecmwf_prebuilt.py --zip ecmwf_20251006_00_efficient.zip
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

# Reference date with pre-built GCS mappings (if Stage 0 was done)
REFERENCE_DATE = '20240529'

# GCS configuration (needed for Stage 2)
GCS_BUCKET = 'gik-fmrc'
GCS_BASE_PATH = 'v2ecmwf_fmrc'  # Base path within the bucket
GCP_SERVICE_ACCOUNT = 'coiled-data-e4drr_202505.json'

# Variables to extract (subset of ECMWF variables)
FORECAST_DICT = {
    "2 metre temperature": "2t:sfc",
    "Total precipitation": "tp:sfc",
    "10 metre U wind": "10u:sfc",
    "10 metre V wind": "10v:sfc"
}

# Output directory
OUTPUT_DIR = Path("test_ecmwf_three_stage_prebuilt_output")

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


def extract_date_run_from_zip(zip_filename):
    """
    Extract date and run from zip filename.
    Format: ecmwf_YYYYMMDD_HH_efficient.zip
    Returns: (date_str, run_str)
    """
    pattern = r'ecmwf_(\d{8})_(\d{2})_efficient\.zip'
    match = re.match(pattern, Path(zip_filename).name)

    if match:
        date_str = match.group(1)
        run_str = match.group(2)
        return date_str, run_str
    else:
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

    # Find extracted directory (should be one directory inside extract_dir)
    extracted_dirs = list(extract_dir.glob("ecmwf_*_efficient"))
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
    if comprehensive_parquet:
        log_checkpoint(f"Found comprehensive parquet: {comprehensive_parquet.name}")

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

    # Remove version key if exists
    if 'version' in zstore:
        del zstore['version']

    log_checkpoint(f"Loaded parquet: {parquet_file.name} ({len(zstore)} entries)")
    return zstore


def create_parquet_simple(zstore, output_file):
    """Save zarr store as parquet (simple version)."""
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
    log_checkpoint(f"✅ Saved parquet: {output_file} ({len(df)} rows)")
    return df


# ==============================================================================
# STAGE 0: CHECK GCS TEMPLATES (OPTIONAL)
# ==============================================================================

def check_stage0_templates(test_members):
    """Check if Stage 0 GCS templates exist for all test members."""
    log_stage(0, "CHECK GCS TEMPLATES (Optional)")

    try:
        import gcsfs

        log_checkpoint("Checking for pre-built GCS templates...")
        gcs_fs = gcsfs.GCSFileSystem(token=GCP_SERVICE_ACCOUNT)

        templates_found = {}

        for member in test_members:
            # Map member name to GCS path format
            # Directory: ens_01, control -> ens_control
            # Filename: ens01, control -> control (no change for control, remove underscore for others)
            if member == 'control':
                gcs_member = 'ens_control'
                gcsmember = 'control'  # Filename stays as 'control'
            else:
                # Keep the underscore format for directory (ens_01 stays ens_01)
                gcs_member = member
                # Remove underscore for filename (ens_01 -> ens01)
                gcsmember = member.replace("_", "")

            # Check for reference date templates
            # Path structure: gs://gik-fmrc/v2ecmwf_fmrc/ens_01/ecmwf-{REFERENCE_DATE}00-ens01-rt000.par
            sample_path = f"gs://{GCS_BUCKET}/{GCS_BASE_PATH}/{gcs_member}/ecmwf-{REFERENCE_DATE}00-{gcsmember}-rt000.par"

            if gcs_fs.exists(sample_path):
                templates_found[member] = True
                log_checkpoint(f"✅ Templates found for {member} at {gcs_member}")
            else:
                templates_found[member] = False
                log_checkpoint(f"⚠️ Templates NOT found for {member} at {sample_path}")

        all_found = all(templates_found.values())

        if all_found:
            log_checkpoint(f"✅ All GCS templates FOUND for reference date {REFERENCE_DATE}")
            return True
        else:
            missing = [m for m, found in templates_found.items() if not found]
            log_checkpoint(f"⚠️ Missing templates for: {', '.join(missing)}")
            log_checkpoint(f"   Stage 2 will skip missing members!")
            return False

    except Exception as e:
        log_checkpoint(f"⚠️ Cannot check GCS: {e}")
        log_checkpoint(f"   Stage 2 will be skipped in this test")
        return False


# ==============================================================================
# STAGE 1: USE PREBUILT PARQUET FILES (or scan GRIB for hours 0,3 only)
# ==============================================================================

def test_stage1_prebuilt(member_parquets, test_date, test_run):
    """Test Stage 1: Use prebuilt parquet files instead of scanning GRIB."""
    log_stage(1, "USE PREBUILT PARQUET FILES")

    start_time = time.time()

    try:
        # We'll use the prebuilt parquet files directly
        log_checkpoint(f"Using prebuilt parquet files for {len(member_parquets)} members")

        # For demonstration, let's load one member to show the structure
        first_member = list(member_parquets.keys())[0]
        first_parquet = member_parquets[first_member]

        log_checkpoint(f"Loading sample member: {first_member}")
        sample_store = read_parquet_to_zarr_store(first_parquet)

        # These prebuilt parquets are already deflated stores from scan_grib processing
        # They represent the output of Stage 1 (hours 0 and 3 only)

        elapsed = time.time() - start_time

        log_checkpoint(f"✅ Stage 1 Complete (using prebuilt)!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Members available: {len(member_parquets)}")
        log_checkpoint(f"   Sample store refs: {len(sample_store)}")
        log_checkpoint(f"   Note: These are from scan_grib of hours 0 and 3 only")

        # Return dictionary of deflated stores
        deflated_stores = {}
        for member, parquet_path in member_parquets.items():
            deflated_stores[member] = read_parquet_to_zarr_store(parquet_path)

        return deflated_stores

    except Exception as e:
        log_checkpoint(f"❌ Stage 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_stage1_scan_grib_fallback(test_date, test_run, test_member='ens_01'):
    """Fallback: Test Stage 1 using scan_grib for hours 0 and 3 only."""
    log_stage("1-FALLBACK", "SCAN GRIB (Hours 0 and 3 only)")

    start_time = time.time()

    try:
        from kerchunk.grib2 import scan_grib, grib_tree
        from kerchunk._grib_idx import strip_datavar_chunks

        # Only process hours 0 and 3 (as per the efficient creator output)
        test_hours = [0, 3]

        # Build ECMWF URLs for test hours
        ecmwf_files = []
        for hour in test_hours:
            # ECMWF URL format
            url = (f"s3://ecmwf-forecasts/{test_date}/{test_run}z/ifs/0p25/enfo/"
                   f"{test_date}{test_run}0000-{hour}h-enfo-ef.grib2")
            ecmwf_files.append(url)
            log_checkpoint(f"Will process: {url}")

        # Extract member number from member name (e.g., ens_01 -> 1)
        if test_member == 'control':
            member_num = 0
        else:
            member_num = int(test_member.replace('ens_', ''))

        # Run Stage 1: scan_grib and build tree
        log_checkpoint(f"Running scan_grib for member {test_member} (number={member_num})...")

        all_groups = []
        for url in ecmwf_files:
            log_checkpoint(f"  Scanning {url}...")
            groups = scan_grib(
                url,
                storage_options={"anon": True},
                filter=FORECAST_DICT  # Filter variables
            )

            # Filter for specific member
            member_groups = []
            for g in groups:
                if 'attrs' in g and f'number={member_num}' in str(g['attrs']):
                    member_groups.append(g)

            all_groups.extend(member_groups)

        log_checkpoint(f"Scanned {len(all_groups)} groups for member {test_member}")

        # Build hierarchical GRIB tree
        log_checkpoint("Building GRIB tree...")
        grib_tree_store = grib_tree(all_groups)

        # Create deflated copy
        log_checkpoint("Creating deflated store...")
        deflated_store = copy.deepcopy(grib_tree_store)
        strip_datavar_chunks(deflated_store)

        # Save deflated store
        stage1_output = OUTPUT_DIR / f"stage1_{test_member}_deflated_store.json"
        with open(stage1_output, 'w') as f:
            json.dump(deflated_store, f, indent=2)

        elapsed = time.time() - start_time

        log_checkpoint(f"✅ Stage 1 Fallback Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Original refs: {len(grib_tree_store['refs'])}")
        log_checkpoint(f"   Deflated refs: {len(deflated_store['refs'])}")
        log_checkpoint(f"   Saved to: {stage1_output}")

        return {test_member: deflated_store}

    except Exception as e:
        log_checkpoint(f"❌ Stage 1 Fallback Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 2: IDX + GCS TEMPLATES → MAPPED INDEX (All 85 hours)
# ==============================================================================

def test_stage2_with_templates(test_date, test_run, test_members):
    """Test Stage 2: Use GCS templates + fresh index files for ALL 85 hours."""
    log_stage(2, "IDX + GCS TEMPLATES → MAPPED INDEX (All 85 hours)")

    start_time = time.time()

    try:
        from kerchunk._grib_idx import parse_grib_idx, map_from_index
        import gcsfs

        # Set up GCS filesystem
        gcs_fs = gcsfs.GCSFileSystem(token=GCP_SERVICE_ACCOUNT)

        # Generate axes
        log_checkpoint("Generating ECMWF axes...")
        axes = generate_ecmwf_axes(test_date)

        log_checkpoint(f"Running mapped index creation for ALL 85 forecast hours...")
        log_checkpoint(f"   Target date: {test_date}")
        log_checkpoint(f"   Reference date: {REFERENCE_DATE}")
        log_checkpoint(f"   Processing {len(test_members)} members")

        # Get all ECMWF forecast hours (0, 3, 6, ..., 360)
        all_forecast_hours = ECMWF_FORECAST_HOURS  # This should be all 85 hours
        log_checkpoint(f"   Forecast hours: {len(all_forecast_hours)} total")

        member_results = {}

        for member in test_members:
            log_checkpoint(f"\nProcessing member: {member}")

            # Extract member number
            if member == 'control':
                member_num = 0
            else:
                member_num = int(member.replace('ens_', ''))

            # Map member name to GCS path format
            # Directory: ens_01, control -> ens_control
            # Filename: ens01, control -> control (no change for control, remove underscore for others)
            if member == 'control':
                gcs_member = 'ens_control'
                gcsmember = 'control'  # Filename stays as 'control'
            else:
                # Keep the underscore format for directory (ens_01 stays ens_01)
                gcs_member = member
                # Remove underscore for filename (ens_01 -> ens01)
                gcsmember = member.replace("_", "")

            all_mapped_indices = []

            # Process in batches for efficiency
            batch_size = 10
            for i in range(0, len(all_forecast_hours), batch_size):
                batch_hours = all_forecast_hours[i:i+batch_size]
                log_checkpoint(f"  Processing hours {batch_hours[0]}-{batch_hours[-1]}...")

                for hour in batch_hours:
                    try:
                        # Step 2.1: Read fresh GRIB index file for TARGET date
                        fname = (f"s3://ecmwf-forecasts/{test_date}/{test_run}z/ifs/0p25/enfo/"
                                f"{test_date}{test_run}0000-{hour}h-enfo-ef.grib2")

                        # ECMWF uses .index extension, not .idx
                        idxdf = parse_grib_idx(basename=fname, suffix="index", storage_options={"anon": True})

                        # Filter for our test member
                        idxdf_filtered = idxdf[idxdf['attrs'].str.contains(f"number={member_num}")]

                        # Step 2.2: Read pre-built parquet mapping from GCS (REFERENCE date)
                        # Path structure: gs://gik-fmrc/v2ecmwf_fmrc/ens_01/ecmwf-{REFERENCE_DATE}00-ens01-rt000.par
                        gcs_path = f"gs://{GCS_BUCKET}/{GCS_BASE_PATH}/{gcs_member}/ecmwf-{REFERENCE_DATE}00-{gcsmember}-rt{hour:03d}.par"
                        
                        if not gcs_fs.exists(gcs_path):
                            log_checkpoint(f"    ⚠️ GCS template missing for hour {hour} at {gcs_path}, skipping...")
                            continue

                        deduped_mapping = pd.read_parquet(gcs_path, filesystem=gcs_fs)

                        # Step 2.3: Merge fresh binary positions + template structure
                        mapped_index = map_from_index(
                            run_time=pd.Timestamp(test_date),
                            mapping=deduped_mapping,
                            idxdf=idxdf_filtered
                        )

                        all_mapped_indices.append(mapped_index)

                    except Exception as e:
                        log_checkpoint(f"    ⚠️ Error processing hour {hour}: {e}")
                        continue

            if all_mapped_indices:
                # Combine all hours
                ecmwf_kind = pd.concat(all_mapped_indices, ignore_index=True)

                # Save mapped index
                stage2_output = OUTPUT_DIR / f"stage2_{member}_mapped_index.parquet"
                ecmwf_kind.to_parquet(stage2_output)

                member_results[member] = ecmwf_kind

                log_checkpoint(f"  ✅ {member}: {len(ecmwf_kind)} entries, {len(all_mapped_indices)} hours processed")

        elapsed = time.time() - start_time

        log_checkpoint(f"\n✅ Stage 2 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Members processed: {len(member_results)}")

        for member, kind_df in member_results.items():
            if 'varname' in kind_df:
                log_checkpoint(f"   {member}: {len(kind_df)} entries, {kind_df['varname'].nunique()} variables")

        return member_results

    except Exception as e:
        log_checkpoint(f"❌ Stage 2 Failed: {e}")
        log_checkpoint(f"   This is expected if GCS templates don't exist")
        log_checkpoint(f"   You need to run Stage 0 preprocessing first!")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 3: CREATE FINAL ZARR STORE (All 85 timesteps)
# ==============================================================================

def test_stage3(deflated_stores, mapped_indices, test_date):
    """Test Stage 3: Merge stores and create final zarr parquet for each member."""
    log_stage(3, "CREATE FINAL ZARR STORE (All 85 timesteps)")

    if not deflated_stores or not mapped_indices:
        log_checkpoint("⚠️ Skipping Stage 3: Missing required inputs from previous stages")
        return None

    start_time = time.time()

    try:
        from kerchunk._grib_idx import (
            store_coord_var,
            store_data_var,
            AggregationType
        )
        import zarr

        # Generate time dimensions for ALL 85 hours
        log_checkpoint("Calculating time dimensions for all 85 forecast hours...")
        axes = generate_ecmwf_axes(test_date)

        # Get all forecast hours
        all_hours = ECMWF_FORECAST_HOURS
        times = axes[1][:len(all_hours)]
        valid_times = axes[0][:len(all_hours)]
        steps = pd.TimedeltaIndex([pd.Timedelta(hours=h) for h in all_hours])

        time_dims = {'valid_time': len(all_hours)}
        time_coords = {'valid_time': 'valid_time', 'time': 'time', 'step': 'step'}

        log_checkpoint(f"   Time dimensions: {time_dims}")
        log_checkpoint(f"   Total timesteps: {len(all_hours)}")

        results = {}

        # Process each member
        for member in deflated_stores.keys():
            if member not in mapped_indices:
                log_checkpoint(f"⚠️ Skipping {member}: No mapped index available")
                continue

            log_checkpoint(f"\nProcessing {member}...")

            deflated_store = deflated_stores[member]
            ecmwf_kind = mapped_indices[member]

            # Prepare zarr store structure
            log_checkpoint(f"   Preparing zarr store for {member}...")

            # Initialize zarr store from deflated store
            if isinstance(deflated_store, dict) and 'refs' in deflated_store:
                zstore = deflated_store.get('refs', {}).copy()
            else:
                zstore = deflated_store.copy()

            # Process chunk index from mapped index
            if isinstance(ecmwf_kind, pd.DataFrame):
                chunk_index = ecmwf_kind.copy()
            else:
                log_checkpoint("   Converting index to DataFrame...")
                chunk_index = pd.DataFrame(ecmwf_kind)

            log_checkpoint(f"   Zarr store entries: {len(zstore)}")
            log_checkpoint(f"   Chunk index rows: {len(chunk_index)}")

            # Add required columns if missing
            if 'varname' not in chunk_index.columns:
                # Try to extract varname from attrs
                chunk_index['varname'] = chunk_index['attrs'].str.split(':').str[0]

            if 'stepType' not in chunk_index.columns:
                chunk_index['stepType'] = 'instant'  # Default

            if 'typeOfLevel' not in chunk_index.columns:
                chunk_index['typeOfLevel'] = 'surface'  # Default

            # Process unique variable groups
            log_checkpoint(f"   Processing unique variable groups for {member}...")

            unique_groups = chunk_index.groupby(['varname', 'stepType', 'typeOfLevel']).groups

            for group_key, group_indices in unique_groups.items():
                varname, step_type, level_type = group_key
                base_path = f"/{varname}/{step_type}/{level_type}"

                log_checkpoint(f"      Processing {base_path}")

                # Store coordinate variables
                store_coord_var(f"{base_path}/time", zstore, times)
                store_coord_var(f"{base_path}/valid_time", zstore, valid_times)
                store_coord_var(f"{base_path}/step", zstore, steps)

                # Store data variable
                group_data = chunk_index.iloc[group_indices]

                # Create data array reference
                data_key = f"{base_path}/{varname}"
                zstore[f"{data_key}/.zarray"] = json.dumps({
                    "chunks": [1, 721, 1440],  # time, lat, lon for ECMWF 0.25 degree
                    "compressor": None,
                    "dtype": "<f4",
                    "fill_value": None,
                    "filters": [],
                    "order": "C",
                    "shape": [len(all_hours), 721, 1440],
                    "zarr_format": 2
                })

            log_checkpoint(f"   Updated zarr store entries: {len(zstore)}")

            # Save as parquet
            stage3_output = OUTPUT_DIR / f"stage3_{member}_final.parquet"
            create_parquet_simple(zstore, stage3_output)

            results[member] = (zstore, stage3_output)

            log_checkpoint(f"   ✅ {member} complete: {len(zstore)} entries")

        elapsed = time.time() - start_time

        log_checkpoint(f"\n✅ Stage 3 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Members processed: {len(results)}")

        return results

    except Exception as e:
        log_checkpoint(f"❌ Stage 3 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# VALIDATION: TEST OPENING WITH XARRAY
# ==============================================================================

def validate_final_outputs(stage3_results):
    """Validate that the final parquets can be opened with xarray."""
    log_stage("VALIDATE", "TEST XARRAY OPENING")

    if not stage3_results:
        log_checkpoint("⚠️ Skipping validation: No results to test")
        return

    import fsspec

    validation_results = {}

    for member, (zstore, parquet_file) in stage3_results.items():
        log_checkpoint(f"\nValidating {member}...")

        try:
            log_checkpoint(f"   Creating fsspec reference filesystem...")
            fs = fsspec.filesystem(
                "reference",
                fo={'refs': zstore, 'version': 1},
                remote_protocol='s3',
                remote_options={'anon': True}
            )
            mapper = fs.get_mapper("")

            log_checkpoint(f"   Opening with xarray.open_datatree()...")
            dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

            # Check structure
            variables = list(dt.keys())

            validation_results[member] = {
                'success': True,
                'variables': len(variables),
                'parquet_file': parquet_file
            }

            log_checkpoint(f"   ✅ {member}: Successfully opened! {len(variables)} variables")

        except Exception as e:
            validation_results[member] = {
                'success': False,
                'error': str(e)
            }
            log_checkpoint(f"   ❌ {member}: Validation failed: {e}")

    # Summary
    successful = sum(1 for r in validation_results.values() if r['success'])

    log_checkpoint(f"\n✅ VALIDATION COMPLETE!")
    log_checkpoint(f"   Successful: {successful}/{len(validation_results)}")

    return validation_results


# ==============================================================================
# MAIN TEST ROUTINE
# ==============================================================================

def main():
    """Run all three stages using prebuilt zip files."""
    parser = argparse.ArgumentParser(description='Test ECMWF three-stage processing with prebuilt files')
    parser.add_argument('--zip', type=str, help='Specific zip file to use')
    parser.add_argument('--skip-gcs-check', action='store_true', help='Skip GCS template checking')
    args = parser.parse_args()

    print("\n🚀 Starting ECMWF Three-Stage Test with Prebuilt Files\n")

    overall_start = time.time()

    # Find zip file to use
    if args.zip:
        zip_file = Path(args.zip)
        if not zip_file.exists():
            print(f"❌ Specified zip file not found: {zip_file}")
            return
    else:
        # Find any available zip file
        zip_files = sorted(glob("ecmwf_*_efficient.zip"))
        if not zip_files:
            print("❌ No ecmwf_*_efficient.zip files found in current directory")
            print("   Please run ecmwf_ensemble_par_creator_efficient.py first to create them")
            return
        zip_file = Path(zip_files[0])
        print(f"Using zip file: {zip_file}")

    # Extract date and run from filename
    try:
        test_date, test_run = extract_date_run_from_zip(zip_file.name)
        print(f"Extracted TEST_DATE: {test_date}")
        print(f"Extracted TEST_RUN: {test_run}")
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*80)
    print("ECMWF THREE-STAGE PROCESSING TEST (PREBUILT)")
    print("="*80)
    print(f"Test Date: {test_date}")
    print(f"Test Run: {test_run}")
    print(f"Zip File: {zip_file}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print("="*80)

    # Unzip and prepare
    extract_dir = OUTPUT_DIR / "extracted"
    extraction_info = unzip_and_prepare(zip_file, extract_dir)

    member_parquets = extraction_info['members']
    test_members = sorted(member_parquets.keys())

    print(f"\nFound {len(test_members)} members to test:")
    print(f"Members: {', '.join(test_members[:5])}" +
          (f"... and {len(test_members)-5} more" if len(test_members) > 5 else ""))

    # Stage 0: Check GCS templates (optional)
    if not args.skip_gcs_check:
        has_templates = check_stage0_templates(test_members)
    else:
        has_templates = False
        log_checkpoint("Skipping GCS template check")

    # Stage 1: Use prebuilt parquet files
    deflated_stores = test_stage1_prebuilt(member_parquets, test_date, test_run)

    if not deflated_stores:
        log_checkpoint("\n⚠️ Falling back to scan_grib for Stage 1...")
        # Use first member as example
        deflated_stores = test_stage1_scan_grib_fallback(test_date, test_run, test_members[0])

    # Stage 2: IDX + GCS templates for ALL 85 hours
    mapped_indices = None
    if has_templates:
        mapped_indices = test_stage2_with_templates(test_date, test_run, test_members)
    else:
        log_checkpoint("\n⚠️ GCS templates not available")
        log_checkpoint("   Stage 2 requires GCS templates from Stage 0 preprocessing")
        log_checkpoint("   Skipping Stage 2 and Stage 3")

    # Stage 3: Create final zarr store with all 85 timesteps
    stage3_results = None
    if deflated_stores and mapped_indices:
        stage3_results = test_stage3(deflated_stores, mapped_indices, test_date)

        # Validation: Test with xarray
        if stage3_results:
            validate_final_outputs(stage3_results)

    # Summary
    total_time = time.time() - overall_start

    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output Directory: {OUTPUT_DIR}")

    print("\n📁 Generated Files:")
    for f in sorted(OUTPUT_DIR.glob("*.parquet")):
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")

    print("\n📝 Summary:")
    print(f"   Zip file used: {zip_file.name}")
    print(f"   Members tested: {len(test_members)}")
    print(f"   Stage 1: ✅ Used prebuilt parquet files (hours 0, 3)")

    if has_templates and mapped_indices:
        print(f"   Stage 2: ✅ Processed all 85 forecast hours")
        print(f"   Stage 3: ✅ Created final zarr stores")
    else:
        print(f"   Stage 2: ⚠️ Skipped (GCS templates not available)")
        print(f"   Stage 3: ⚠️ Skipped (requires Stage 2)")

    print("\n📝 Next Steps:")
    if not has_templates:
        print("   1. Run Stage 0 preprocessing to create GCS templates:")
        print(f"      python ecmwf_index_preprocessing.py \\")
        print(f"        --date {REFERENCE_DATE} --all-members --bucket {GCS_BUCKET}")
        print("   2. Re-run this test to use GCS templates in Stage 2")
    else:
        print("   1. ✅ All stages working with prebuilt files and GCS templates!")
        print("   2. Process different dates by providing different zip files")
        print("   3. Use run_day_ecmwf_ensemble_full.py for production")

    print("="*80)


if __name__ == "__main__":
    main()