#!/usr/bin/env python3
"""
Simple GEFS Three-Stage Processing Test Routine
Tests each stage independently to learn and understand the workflow quickly.

NO async, NO complex setup - just straightforward sequential processing.

Usage:
    python test_three_stage_gefs_simple.py
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import copy
import time
from pathlib import Path
from datetime import datetime

# Import GEFS utilities
from gefs_util import (
    generate_axes,
    filter_build_grib_tree,
    calculate_time_dimensions,
    prepare_zarr_store,
    process_unique_groups
)

# Set up anonymous S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Test with a single ensemble member for learning
TEST_MEMBER = 'gep01'
TEST_DATE = '20250918'
TEST_RUN = '00'

# Reference date with pre-built GCS mappings (if Stage 0 was done)
REFERENCE_DATE = '20241112'

# GCS configuration (needed for Stage 2)
GCS_BUCKET = 'gik-fmrc'  # or 'gik-gefs-aws-tf'
GCP_SERVICE_ACCOUNT = 'coiled-data-e4drr_202505.json'

# Test with minimal forecast hours for speed
TEST_HOURS = [0, 3]  # Just 2 hours for quick testing

# Variables to extract
FORECAST_DICT = {
    "2 metre temperature": "TMP:2 m above ground",
    "Total Precipitation": "APCP:surface",
}

# Output directory
OUTPUT_DIR = Path("test_three_stage_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("GEFS THREE-STAGE PROCESSING TEST")
print("="*80)
print(f"Test Member: {TEST_MEMBER}")
print(f"Test Date: {TEST_DATE}")
print(f"Test Hours: {TEST_HOURS}")
print(f"Output Dir: {OUTPUT_DIR}")
print("="*80)


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


def read_parquet_simple(parquet_file):
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

    log_checkpoint(f"✅ Loaded parquet: {parquet_file} ({len(zstore)} entries)")
    return zstore


# ==============================================================================
# STAGE 0: CHECK GCS TEMPLATES (OPTIONAL)
# ==============================================================================

def check_stage0_templates():
    """Check if Stage 0 GCS templates exist (optional)."""
    log_stage(0, "CHECK GCS TEMPLATES (Optional)")

    try:
        import gcsfs

        log_checkpoint("Checking for pre-built GCS templates...")
        gcs_fs = gcsfs.GCSFileSystem(token=GCP_SERVICE_ACCOUNT)

        # Check for reference date templates
        sample_path = f"gs://{GCS_BUCKET}/gefs/{TEST_MEMBER}/gefs-time-{REFERENCE_DATE}-{TEST_MEMBER}-rt000.parquet"

        if gcs_fs.exists(sample_path):
            log_checkpoint(f"✅ GCS templates FOUND for reference date {REFERENCE_DATE}")
            log_checkpoint(f"   Path: {sample_path}")

            # List available templates
            template_dir = f"gs://{GCS_BUCKET}/gefs/{TEST_MEMBER}/"
            templates = gcs_fs.ls(template_dir)
            log_checkpoint(f"   Found {len(templates)} template files")

            return True
        else:
            log_checkpoint(f"⚠️ GCS templates NOT FOUND for reference date {REFERENCE_DATE}")
            log_checkpoint(f"   Expected path: {sample_path}")
            log_checkpoint(f"   Stage 2 will need to run Stage 0 preprocessing first!")
            return False

    except Exception as e:
        log_checkpoint(f"⚠️ Cannot check GCS: {e}")
        log_checkpoint(f"   Stage 2 will be skipped in this test")
        return False


# ==============================================================================
# STAGE 1: SCAN GRIB TO DEFLATED STORE
# ==============================================================================

def test_stage1():
    """Test Stage 1: Scan GRIB files and create deflated store."""
    log_stage(1, "SCAN GRIB TO DEFLATED STORE")

    start_time = time.time()

    # Build GRIB URLs for test hours
    gefs_files = []
    for hour in TEST_HOURS:
        url = (f"s3://noaa-gefs-pds/gefs.{TEST_DATE}/{TEST_RUN}/atmos/pgrb2sp25/"
               f"{TEST_MEMBER}.t{TEST_RUN}z.pgrb2s.0p25.f{hour:03d}")
        gefs_files.append(url)
        log_checkpoint(f"Will process: {url}")

    # Run Stage 1: scan_grib and build tree
    log_checkpoint("Running filter_build_grib_tree()...")
    grib_tree_store, deflated_store = filter_build_grib_tree(gefs_files, FORECAST_DICT)

    # Save deflated store
    stage1_output = OUTPUT_DIR / f"stage1_{TEST_MEMBER}_deflated_store.json"
    with open(stage1_output, 'w') as f:
        json.dump(deflated_store, f, indent=2)

    elapsed = time.time() - start_time

    log_checkpoint(f"✅ Stage 1 Complete!")
    log_checkpoint(f"   Time: {elapsed:.1f} seconds")
    log_checkpoint(f"   Original refs: {len(grib_tree_store['refs'])}")
    log_checkpoint(f"   Deflated refs: {len(deflated_store['refs'])}")
    log_checkpoint(f"   Saved to: {stage1_output}")

    return deflated_store


# ==============================================================================
# STAGE 2: IDX + GCS TEMPLATES → MAPPED INDEX
# ==============================================================================

def test_stage2_with_templates():
    """Test Stage 2: Use GCS templates + fresh index files."""
    log_stage(2, "IDX + GCS TEMPLATES → MAPPED INDEX")

    start_time = time.time()

    try:
        from gefs_util import cs_create_mapped_index

        # Generate axes
        log_checkpoint("Generating axes...")
        axes = generate_axes(TEST_DATE)

        # Run Stage 2: mapped index creation
        log_checkpoint(f"Running cs_create_mapped_index()...")
        log_checkpoint(f"   Target date: {TEST_DATE}")
        log_checkpoint(f"   Reference date: {REFERENCE_DATE}")

        gefs_kind = cs_create_mapped_index(
            axes,
            GCS_BUCKET,
            TEST_DATE,
            TEST_MEMBER,
            gcp_service_account_json=GCP_SERVICE_ACCOUNT,
            reference_date_str=REFERENCE_DATE
        )

        # Save mapped index
        stage2_output = OUTPUT_DIR / f"stage2_{TEST_MEMBER}_mapped_index.parquet"
        gefs_kind.to_parquet(stage2_output)

        elapsed = time.time() - start_time

        log_checkpoint(f"✅ Stage 2 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Mapped entries: {len(gefs_kind)}")
        log_checkpoint(f"   Unique variables: {gefs_kind['varname'].nunique()}")
        log_checkpoint(f"   Saved to: {stage2_output}")

        # Show sample data
        print("\n📊 Sample mapped index:")
        print(gefs_kind[['varname', 'stepType', 'typeOfLevel', 'step']].head(10))

        return gefs_kind

    except Exception as e:
        log_checkpoint(f"❌ Stage 2 Failed: {e}")
        log_checkpoint(f"   This is expected if GCS templates don't exist")
        log_checkpoint(f"   You need to run Stage 0 preprocessing first!")
        return None


def test_stage2_without_templates():
    """Test Stage 2 alternative: Simple index parsing without GCS."""
    log_stage("2-ALT", "IDX PARSING (Without GCS Templates)")

    start_time = time.time()

    try:
        from kerchunk._grib_idx import parse_grib_idx

        log_checkpoint("Parsing GRIB index files directly...")

        all_idx_data = []
        for hour in TEST_HOURS:
            fname = (f"s3://noaa-gefs-pds/gefs.{TEST_DATE}/{TEST_RUN}/atmos/pgrb2sp25/"
                    f"{TEST_MEMBER}.t{TEST_RUN}z.pgrb2s.0p25.f{hour:03d}")

            log_checkpoint(f"   Parsing {fname}.idx...")
            idxdf = parse_grib_idx(basename=fname, storage_options={"anon": True})
            idxdf['forecast_hour'] = hour
            all_idx_data.append(idxdf)

        # Combine all hours
        combined_idx = pd.concat(all_idx_data, ignore_index=True)

        # Save
        stage2_alt_output = OUTPUT_DIR / f"stage2_alt_{TEST_MEMBER}_simple_index.parquet"
        combined_idx.to_parquet(stage2_alt_output)

        elapsed = time.time() - start_time

        log_checkpoint(f"✅ Stage 2-ALT Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Index entries: {len(combined_idx)}")
        log_checkpoint(f"   Saved to: {stage2_alt_output}")

        print("\n📊 Sample index data:")
        print(combined_idx[['attrs', 'forecast_hour']].head(10))

        return combined_idx

    except Exception as e:
        log_checkpoint(f"❌ Stage 2-ALT Failed: {e}")
        return None


# ==============================================================================
# STAGE 3: CREATE FINAL ZARR STORE
# ==============================================================================

def test_stage3(deflated_store, gefs_kind):
    """Test Stage 3: Merge stores and create final zarr parquet."""
    log_stage(3, "CREATE FINAL ZARR STORE")

    if gefs_kind is None:
        log_checkpoint("⚠️ Skipping Stage 3: No mapped index from Stage 2")
        return None

    start_time = time.time()

    try:
        # Generate time dimensions
        log_checkpoint("Calculating time dimensions...")
        axes = generate_axes(TEST_DATE)
        time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)

        log_checkpoint(f"   Time dimensions: {time_dims}")

        # Prepare zarr store
        log_checkpoint("Preparing zarr store...")
        zstore, chunk_index = prepare_zarr_store(deflated_store, gefs_kind)

        log_checkpoint(f"   Zarr store entries: {len(zstore)}")
        log_checkpoint(f"   Chunk index rows: {len(chunk_index)}")

        # Process unique groups
        log_checkpoint("Processing unique variable groups...")
        updated_zstore = process_unique_groups(
            zstore,
            chunk_index,
            time_dims,
            time_coords,
            times,
            valid_times,
            steps
        )

        log_checkpoint(f"   Updated zarr store entries: {len(updated_zstore)}")

        # Save as parquet
        stage3_output = OUTPUT_DIR / f"stage3_{TEST_MEMBER}_final.parquet"
        create_parquet_simple(updated_zstore, stage3_output)

        elapsed = time.time() - start_time

        log_checkpoint(f"✅ Stage 3 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Final store entries: {len(updated_zstore)}")
        log_checkpoint(f"   Saved to: {stage3_output}")

        return updated_zstore, stage3_output

    except Exception as e:
        log_checkpoint(f"❌ Stage 3 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# VALIDATION: TEST OPENING WITH XARRAY
# ==============================================================================

def validate_final_output(parquet_file):
    """Validate that the final parquet can be opened with xarray."""
    log_stage("VALIDATE", "TEST XARRAY OPENING")

    if parquet_file is None:
        log_checkpoint("⚠️ Skipping validation: No parquet file to test")
        return

    try:
        import fsspec

        log_checkpoint(f"Reading parquet: {parquet_file}")
        zstore = read_parquet_simple(parquet_file)

        log_checkpoint("Creating fsspec reference filesystem...")
        fs = fsspec.filesystem(
            "reference",
            fo={'refs': zstore, 'version': 1},
            remote_protocol='s3',
            remote_options={'anon': True}
        )
        mapper = fs.get_mapper("")

        log_checkpoint("Opening with xarray.open_datatree()...")
        dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

        log_checkpoint(f"✅ Successfully opened datatree!")

        # Show structure
        print("\n📊 DataTree Structure:")
        variables = list(dt.keys())
        print(f"   Variables: {len(variables)}")
        for var in variables[:10]:
            print(f"      - {var}")
        if len(variables) > 10:
            print(f"      ... and {len(variables) - 10} more")

        # Try accessing a variable
        if '/t2m' in str(variables) or 'TMP' in str(variables):
            log_checkpoint("\n🌡️ Testing temperature data access...")
            try:
                # Find temperature variable
                temp_path = [v for v in variables if 'TMP' in v or 't2m' in v][0]
                log_checkpoint(f"   Found temperature at: {temp_path}")

                # Access data (don't compute, just check structure)
                temp_data = dt[temp_path].ds
                log_checkpoint(f"   Temperature dataset: {temp_data}")

            except Exception as e:
                log_checkpoint(f"   ⚠️ Could not access temperature: {e}")

        log_checkpoint("\n✅ VALIDATION PASSED!")

    except Exception as e:
        log_checkpoint(f"❌ Validation Failed: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# MAIN TEST ROUTINE
# ==============================================================================

def main():
    """Run all three stages sequentially."""
    print("\n🚀 Starting GEFS Three-Stage Test Routine\n")

    overall_start = time.time()

    # Stage 0: Check GCS templates (optional)
    has_templates = check_stage0_templates()

    # Stage 1: Scan GRIB
    deflated_store = test_stage1()

    # Stage 2: IDX + GCS templates (or alternative)
    if has_templates:
        gefs_kind = test_stage2_with_templates()
    else:
        log_checkpoint("\n⚠️ GCS templates not available")
        log_checkpoint("   Trying alternative: simple index parsing")
        gefs_kind = test_stage2_without_templates()
        log_checkpoint("   NOTE: Stage 3 requires proper mapped index from GCS templates")

    # Stage 3: Create final zarr store
    result = test_stage3(deflated_store, gefs_kind)

    if result:
        updated_zstore, stage3_output = result

        # Validation: Test with xarray
        validate_final_output(stage3_output)

    # Summary
    total_time = time.time() - overall_start

    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("\n📁 Generated Files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")

    print("\n📝 Next Steps:")
    if not has_templates:
        print("   1. Run Stage 0 preprocessing to create GCS templates:")
        print(f"      python gefs_index_preprocessing_fixed.py \\")
        print(f"        --date {REFERENCE_DATE} --member {TEST_MEMBER} --bucket {GCS_BUCKET}")
        print("   2. Re-run this test to use GCS templates in Stage 2")
    else:
        print("   1. ✅ All stages working with GCS templates!")
        print("   2. Scale to all 30 members: run_day_gefs_ensemble_full.py")
        print("   3. Process different dates by changing TEST_DATE")

    print("="*80)


if __name__ == "__main__":
    main()
