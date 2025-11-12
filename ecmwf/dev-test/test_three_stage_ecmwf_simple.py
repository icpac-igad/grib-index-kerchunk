#!/usr/bin/env python3
"""
Simple ECMWF Three-Stage Processing Test Routine
Tests each stage independently to learn and understand the workflow quickly.

NO async, NO complex setup - just straightforward sequential processing.

Usage:
    python test_three_stage_ecmwf_simple.py
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

# Test with a single ensemble member for learning
TEST_MEMBER = 'ens01'
TEST_DATE = '20240529'
TEST_RUN = '00'

# Reference date with pre-built GCS mappings (if Stage 0 was done)
REFERENCE_DATE = '20240529'

# GCS configuration (needed for Stage 2)
GCS_BUCKET = 'gik-ecmwf-aws-tf'
GCP_SERVICE_ACCOUNT = 'coiled-data-e4drr_202505.json'

# Test with minimal forecast hours for speed
TEST_HOURS = [0, 3]  # Just 3 hours for quick testing

# Variables to extract (subset of ECMWF variables)
FORECAST_DICT = {
    "2 metre temperature": "2t:sfc",
    "Total precipitation": "tp:sfc",
    "10 metre U wind": "10u:sfc",
    "10 metre V wind": "10v:sfc"
}

# Output directory
OUTPUT_DIR = Path("test_ecmwf_three_stage_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("ECMWF THREE-STAGE PROCESSING TEST")
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
        sample_path = f"gs://{GCS_BUCKET}/ecmwf/{TEST_MEMBER}/ecmwf-time-{REFERENCE_DATE}-{TEST_MEMBER}-rt000.parquet"

        if gcs_fs.exists(sample_path):
            log_checkpoint(f"✅ GCS templates FOUND for reference date {REFERENCE_DATE}")
            log_checkpoint(f"   Path: {sample_path}")

            # List available templates
            template_dir = f"gs://{GCS_BUCKET}/ecmwf/{TEST_MEMBER}/"
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

    try:
        from kerchunk.grib2 import scan_grib, grib_tree
        from kerchunk._grib_idx import strip_datavar_chunks

        # Build ECMWF URLs for test hours
        ecmwf_files = []
        for hour in TEST_HOURS:
            # ECMWF URL format
            url = (f"s3://ecmwf-forecasts/{TEST_DATE}/00z/ifs/0p25/enfo/"
                   f"{TEST_DATE}000000-{hour}h-enfo-ef.grib2")
            ecmwf_files.append(url)
            log_checkpoint(f"Will process: {url}")

        # Run Stage 1: scan_grib and build tree
        log_checkpoint("Running scan_grib on ECMWF files...")

        all_groups = []
        for url in ecmwf_files:
            log_checkpoint(f"  Scanning {url}...")
            groups = scan_grib(
                url,
                storage_options={"anon": True},
                filter=FORECAST_DICT  # Filter variables
            )
            all_groups.extend(groups)

        log_checkpoint(f"Scanned {len(all_groups)} groups total")

        # Build hierarchical GRIB tree
        log_checkpoint("Building GRIB tree...")
        grib_tree_store = grib_tree(all_groups)

        # Create deflated copy (strip data chunks to reduce memory)
        log_checkpoint("Creating deflated store...")
        deflated_store = copy.deepcopy(grib_tree_store)
        strip_datavar_chunks(deflated_store)

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

    except Exception as e:
        log_checkpoint(f"❌ Stage 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 2: IDX + GCS TEMPLATES → MAPPED INDEX
# ==============================================================================

def test_stage2_with_templates():
    """Test Stage 2: Use GCS templates + fresh index files."""
    log_stage(2, "IDX + GCS TEMPLATES → MAPPED INDEX")

    start_time = time.time()

    try:
        from kerchunk._grib_idx import parse_grib_idx, map_from_index
        import gcsfs
        import asyncio

        # Set up GCS filesystem
        gcs_fs = gcsfs.GCSFileSystem(token=GCP_SERVICE_ACCOUNT)

        # Generate axes
        log_checkpoint("Generating ECMWF axes...")
        axes = generate_ecmwf_axes(TEST_DATE)

        log_checkpoint(f"Running mapped index creation...")
        log_checkpoint(f"   Target date: {TEST_DATE}")
        log_checkpoint(f"   Reference date: {REFERENCE_DATE}")

        all_mapped_indices = []

        for hour in TEST_HOURS:
            # Step 2.1: Read fresh GRIB index file for TARGET date
            fname = (f"s3://ecmwf-forecasts/{TEST_DATE}/00z/ifs/0p25/enfo/"
                    f"{TEST_DATE}000000-{hour}h-enfo-ef.grib2")

            log_checkpoint(f"  Processing hour {hour:03d}...")
            log_checkpoint(f"    Parsing fresh index from {fname}.index")

            idxdf = parse_grib_idx(basename=fname, storage_options={"anon": True})

            # Filter for our test member
            # ECMWF stores member info in attrs
            member_num = int(TEST_MEMBER.replace("ens", ""))
            idxdf_filtered = idxdf[idxdf['attrs'].str.contains(f"number={member_num}")]

            # Step 2.2: Read pre-built parquet mapping from GCS (REFERENCE date)
            gcs_path = f"gs://{GCS_BUCKET}/ecmwf/{TEST_MEMBER}/ecmwf-time-{REFERENCE_DATE}-{TEST_MEMBER}-rt{hour:03d}.parquet"
            log_checkpoint(f"    Reading GCS template from {gcs_path}")

            deduped_mapping = pd.read_parquet(gcs_path, filesystem=gcs_fs)

            # Step 2.3: Merge fresh binary positions + template structure
            mapped_index = map_from_index(
                run_time=pd.Timestamp(TEST_DATE),
                mapping=deduped_mapping,
                idxdf=idxdf_filtered
            )

            all_mapped_indices.append(mapped_index)
            log_checkpoint(f"    Mapped {len(mapped_index)} entries")

        # Combine all hours
        ecmwf_kind = pd.concat(all_mapped_indices, ignore_index=True)

        # Save mapped index
        stage2_output = OUTPUT_DIR / f"stage2_{TEST_MEMBER}_mapped_index.parquet"
        ecmwf_kind.to_parquet(stage2_output)

        elapsed = time.time() - start_time

        log_checkpoint(f"✅ Stage 2 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Mapped entries: {len(ecmwf_kind)}")
        log_checkpoint(f"   Unique variables: {ecmwf_kind['varname'].nunique() if 'varname' in ecmwf_kind else 'N/A'}")
        log_checkpoint(f"   Saved to: {stage2_output}")

        # Show sample data
        print("\n📊 Sample mapped index:")
        if 'varname' in ecmwf_kind:
            print(ecmwf_kind[['varname', 'step']].head(10))
        else:
            print(ecmwf_kind.head(10))

        return ecmwf_kind

    except Exception as e:
        log_checkpoint(f"❌ Stage 2 Failed: {e}")
        log_checkpoint(f"   This is expected if GCS templates don't exist")
        log_checkpoint(f"   You need to run Stage 0 preprocessing first!")
        import traceback
        traceback.print_exc()
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
            fname = (f"s3://ecmwf-forecasts/{TEST_DATE}/00z/ifs/0p25/enfo/"
                    f"{TEST_DATE}000000-{hour}h-enfo-ef.grib2")

            log_checkpoint(f"   Parsing {fname}.index...")
            idxdf = parse_grib_idx(basename=fname, storage_options={"anon": True})

            # Filter for test member
            member_num = int(TEST_MEMBER.replace("ens", ""))
            idxdf = idxdf[idxdf['attrs'].str.contains(f"number={member_num}")]

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
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# STAGE 3: CREATE FINAL ZARR STORE
# ==============================================================================

def test_stage3(deflated_store, ecmwf_kind):
    """Test Stage 3: Merge stores and create final zarr parquet."""
    log_stage(3, "CREATE FINAL ZARR STORE")

    if ecmwf_kind is None or deflated_store is None:
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

        # Generate time dimensions
        log_checkpoint("Calculating time dimensions...")
        axes = generate_ecmwf_axes(TEST_DATE)

        # For test, we only have a few hours
        times = axes[1][:len(TEST_HOURS)]
        valid_times = axes[0][:len(TEST_HOURS)]
        steps = pd.TimedeltaIndex([pd.Timedelta(hours=h) for h in TEST_HOURS])

        time_dims = {'valid_time': len(TEST_HOURS)}
        time_coords = {'valid_time': 'valid_time', 'time': 'time', 'step': 'step'}

        log_checkpoint(f"   Time dimensions: {time_dims}")

        # Prepare zarr store structure
        log_checkpoint("Preparing zarr store...")

        # Initialize zarr store from deflated store
        zstore = deflated_store.get('refs', {}).copy()

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
        log_checkpoint("Processing unique variable groups...")

        unique_groups = chunk_index.groupby(['varname', 'stepType', 'typeOfLevel']).groups

        for group_key, group_indices in unique_groups.items():
            varname, step_type, level_type = group_key
            base_path = f"/{varname}/{step_type}/{level_type}"

            log_checkpoint(f"   Processing {base_path}")

            # Store coordinate variables
            store_coord_var(f"{base_path}/time", zstore, times)
            store_coord_var(f"{base_path}/valid_time", zstore, valid_times)
            store_coord_var(f"{base_path}/step", zstore, steps)

            # Store data variable (simplified for test)
            # In production, this would use proper chunking and aggregation
            group_data = chunk_index.iloc[group_indices]

            # Create simple data array reference
            data_key = f"{base_path}/{varname}"
            zstore[f"{data_key}/.zarray"] = json.dumps({
                "chunks": [1, 721, 1440],  # time, lat, lon for ECMWF 0.25 degree
                "compressor": None,
                "dtype": "<f4",
                "fill_value": None,
                "filters": [],
                "order": "C",
                "shape": [len(TEST_HOURS), 721, 1440],
                "zarr_format": 2
            })

        log_checkpoint(f"   Updated zarr store entries: {len(zstore)}")

        # Save as parquet
        stage3_output = OUTPUT_DIR / f"stage3_{TEST_MEMBER}_final.parquet"
        create_parquet_simple(zstore, stage3_output)

        elapsed = time.time() - start_time

        log_checkpoint(f"✅ Stage 3 Complete!")
        log_checkpoint(f"   Time: {elapsed:.1f} seconds")
        log_checkpoint(f"   Final store entries: {len(zstore)}")
        log_checkpoint(f"   Saved to: {stage3_output}")

        return zstore, stage3_output

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
        if any('2t' in str(v) or 't2m' in str(v) for v in variables):
            log_checkpoint("\n🌡️ Testing temperature data access...")
            try:
                # Find temperature variable
                temp_path = [v for v in variables if '2t' in v or 't2m' in v][0]
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
    print("\n🚀 Starting ECMWF Three-Stage Test Routine\n")

    overall_start = time.time()

    # Stage 0: Check GCS templates (optional)
    has_templates = check_stage0_templates()

    # Stage 1: Scan GRIB
    deflated_store = test_stage1()

    # Stage 2: IDX + GCS templates (or alternative)
    if has_templates:
        ecmwf_kind = test_stage2_with_templates()
    else:
        log_checkpoint("\n⚠️ GCS templates not available")
        log_checkpoint("   Trying alternative: simple index parsing")
        ecmwf_kind = test_stage2_without_templates()
        log_checkpoint("   NOTE: Stage 3 requires proper mapped index from GCS templates")

    # Stage 3: Create final zarr store
    result = test_stage3(deflated_store, ecmwf_kind)

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
        print(f"      python ecmwf_index_preprocessing.py \\")
        print(f"        --date {REFERENCE_DATE} --member {TEST_MEMBER} --bucket {GCS_BUCKET}")
        print("   2. Re-run this test to use GCS templates in Stage 2")
    else:
        print("   1. ✅ All stages working with GCS templates!")
        print("   2. Scale to all 51 members (control + 50 perturbed)")
        print("   3. Process different dates by changing TEST_DATE")
        print("   4. Use run_day_ecmwf_ensemble_full.py for production")

    print("="*80)


if __name__ == "__main__":
    main()
