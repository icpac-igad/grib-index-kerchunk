#!/usr/bin/env python3
"""
GEFS Full Ensemble Processing - Parquet Creation Only
This script processes all 30 ensemble members and creates parquet files.
Plotting routines have been moved to run_gefs_24h_accumulation.py
"""

import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import warnings
from pathlib import Path
from datetime import datetime
import fsspec
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# Import from gefs_util
from gefs_util import generate_axes
from gefs_util import filter_build_grib_tree
from gefs_util import calculate_time_dimensions
from gefs_util import cs_create_mapped_index
from gefs_util import cs_create_mapped_index_local
from gefs_util import LocalTarGzMappingManager
from gefs_util import prepare_zarr_store
from gefs_util import process_unique_groups

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for NOAA data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

# Option 1: Specify forecast hour directly (hours from model run time)
FORECAST_HOUR = 12  # 12 hours from 00Z = 12:00 UTC = 15:00 Nairobi time

# Option 2: Specify local time in target timezone
TARGET_LOCAL_TIME = 14  # 14:00 local time
TARGET_TIMEZONE_OFFSET = 3  # UTC+3 for Nairobi
TARGET_UTC_TIME = TARGET_LOCAL_TIME - TARGET_TIMEZONE_OFFSET  # 11:00 UTC
FORECAST_HOUR_FROM_LOCAL = TARGET_UTC_TIME if TARGET_UTC_TIME >= 0 else TARGET_UTC_TIME + 24

# Calculate timestep index (GEFS uses 3-hour intervals)
TIMESTEP_INDEX = FORECAST_HOUR // 3  # For 12 hours: 12/3 = 4

# Configuration for the specific run
TARGET_DATE_STR = '20250918'
TARGET_RUN = '00'  # 00Z run
REFERENCE_DATE_STR = '20241112'  # Date with existing parquet mappings
ENSEMBLE_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # All 30 members

# Option to keep parquet files
KEEP_PARQUET_FILES = True  # Set to True to keep files, False to delete them

# Option to create parquet files after processing GRIB data
# NOTE: This has been fixed to work with Zarr v3 by removing xarray validation.
# The function now only creates parquet files without loading data into memory.
STREAM_AFTER_CREATION = True  # Creates parquet files (works with Zarr v3)

# ==============================================================================
# Mapping Source Configuration
# ==============================================================================
# Choose between GCS bucket (requires service account) or local tar.gz file
# Set USE_LOCAL_TARBALL = True to use local tar.gz file instead of GCS bucket
USE_LOCAL_TARBALL = False  # Set to True to use local tar.gz file

# Path to local tar.gz file containing parquet mappings
# Only used when USE_LOCAL_TARBALL = True
LOCAL_TARBALL_PATH = 'gik-fmrc-gefs-20241112.tar.gz'

# GCS configuration (only used when USE_LOCAL_TARBALL = False)
GCS_BUCKET_NAME = 'gik-fmrc'
GCP_SERVICE_ACCOUNT_JSON = 'coiled-data-e4drr_202505.json'

# East Africa bounding box
EA_LAT_MIN, EA_LAT_MAX = -12, 15
EA_LON_MIN, EA_LON_MAX = 25, 52

print(f"🚀 Processing GEFS full ensemble data for {TARGET_DATE_STR} {TARGET_RUN}Z run")
print(f"👥 Ensemble members: {len(ENSEMBLE_MEMBERS)} members (gep01-gep30)")
print(f"📊 Using reference mappings from: {REFERENCE_DATE_STR}")
print(f"🌍 East Africa region: {EA_LAT_MIN}°S-{EA_LAT_MAX}°N, {EA_LON_MIN}°E-{EA_LON_MAX}°E")
print(f"💾 Parquet files will be {'KEPT' if KEEP_PARQUET_FILES else 'DELETED'} after processing")
if USE_LOCAL_TARBALL:
    print(f"📦 Mapping source: LOCAL TAR.GZ FILE ({LOCAL_TARBALL_PATH})")
else:
    print(f"☁️  Mapping source: GCS BUCKET ({GCS_BUCKET_NAME})")


# Option 1: Specify forecast hour directly (hours from model run time)
FORECAST_HOUR = 12  # 12 hours from 00Z = 12:00 UTC = 15:00 Nairobi time

# Option 2: Specify local time in target timezone
TARGET_LOCAL_TIME = 14  # 14:00 local time
TARGET_TIMEZONE_OFFSET = 3  # UTC+3 for Nairobi
TARGET_UTC_TIME = TARGET_LOCAL_TIME - TARGET_TIMEZONE_OFFSET  # 11:00 UTC
FORECAST_HOUR_FROM_LOCAL = TARGET_UTC_TIME if TARGET_UTC_TIME >= 0 else TARGET_UTC_TIME + 24

# Calculate timestep index (GEFS uses 3-hour intervals)
TIMESTEP_INDEX = FORECAST_HOUR // 3  # For 12 hours: 12/3 = 4

# For 6-hour forecast: TIMESTEP_INDEX = 2 (6/3 = 2)
# For 12-hour forecast: TIMESTEP_INDEX = 4 (12/3 = 4)
# For 24-hour forecast: TIMESTEP_INDEX = 8 (24/3 = 8)

print(f"📍 Time Configuration:")
print(f"   - Model run time: {TARGET_DATE_STR} {TARGET_RUN}Z")
print(f"   - Forecast hour: +{FORECAST_HOUR}h")
print(f"   - UTC time: {(int(TARGET_RUN) + FORECAST_HOUR) % 24:02d}:00 UTC")
print(f"   - Nairobi time: {((int(TARGET_RUN) + FORECAST_HOUR + 3) % 24):02d}:00 EAT")
print(f"   - Timestep index: {TIMESTEP_INDEX}")




def create_output_directory(date_str: str, run: str) -> Path:
    """Create output directory structure for parquet files."""
    output_dir = Path(f"{date_str}_{run}")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir


def create_parquet_file_fixed(zstore: dict, output_parquet_file: str):
    """Fixed version that stores each zarr reference as an individual row."""
    data = []
    
    for key, value in zstore.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        elif isinstance(value, (int, float, np.integer, np.floating)):
            encoded_value = str(value).encode('utf-8')
        else:
            encoded_value = str(value).encode('utf-8')
        
        data.append((key, encoded_value))
    
    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_parquet_file)
    print(f"✅ Fixed parquet file saved: {output_parquet_file} ({len(df)} rows)")
    
    return df


def read_parquet_fixed(parquet_path):
    """Read parquet files with proper handling."""
    import ast
    
    df = pd.read_parquet(parquet_path)
    
    if 'refs' in df['key'].values and len(df) <= 2:
        refs_row = df[df['key'] == 'refs']
        refs_value = refs_row['value'].iloc[0]
        if isinstance(refs_value, bytes):
            refs_value = refs_value.decode('utf-8')
        zstore = ast.literal_eval(refs_value)
        print(f"✅ Extracted {len(zstore)} entries from old format")
    else:
        zstore = {}
        for _, row in df.iterrows():
            key = row['key']
            value = row['value']
            
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = json.loads(value)
                    except:
                        pass
            
            zstore[key] = value
        
        print(f"✅ Loaded {len(zstore)} entries from new format")
    
    if 'version' in zstore:
        del zstore['version']
    
    return zstore


def process_single_ensemble_member(member, target_date_str, target_run, reference_date_str,
                                 axes, forecast_dict, time_dims, time_coords, times, valid_times, steps,
                                 mapping_manager=None):
    """Process a single ensemble member and return its zarr store.

    Args:
        member: Ensemble member identifier (e.g., 'gep01')
        target_date_str: Target date string (YYYYMMDD)
        target_run: Run time (e.g., '00')
        reference_date_str: Reference date for parquet mappings
        axes: Time axes from generate_axes()
        forecast_dict: Dictionary of variables to process
        time_dims, time_coords, times, valid_times, steps: Time dimension info
        mapping_manager: Optional LocalTarGzMappingManager for local tar.gz usage
    """
    print(f"\n🎯 Processing ensemble member: {member}")

    # Define GEFS files for this member
    gefs_files = []
    for hour in [0, 3]:  # Only initial timesteps needed
        gefs_files.append(
            f"s3://noaa-gefs-pds/gefs.{target_date_str}/{target_run}/atmos/pgrb2sp25/"
            f"{member}.t{target_run}z.pgrb2s.0p25.f{hour:03d}"
        )

    try:
        # Build GRIB tree from files
        print(f"🔨 Building GRIB tree for {member}...")
        _, deflated_gefs_grib_tree_store = filter_build_grib_tree(gefs_files, forecast_dict)
        print(f"✅ GRIB tree built successfully for {member}")

        # Create zarr store using reference mappings
        print(f"🗃️ Creating zarr store for {member}...")

        try:
            # Choose between local tar.gz or GCS bucket based on configuration
            if USE_LOCAL_TARBALL and mapping_manager is not None:
                # Use local tar.gz file
                print(f"📦 Using local tar.gz mappings for {member}...")
                gefs_kind = cs_create_mapped_index_local(
                    axes, target_date_str, member,
                    tar_gz_path=LOCAL_TARBALL_PATH,
                    mapping_manager=mapping_manager
                )
            else:
                # Use GCS bucket (original method)
                print(f"☁️  Using GCS bucket mappings for {member}...")
                gefs_kind = cs_create_mapped_index(
                    axes, GCS_BUCKET_NAME, target_date_str, member,
                    gcp_service_account_json=GCP_SERVICE_ACCOUNT_JSON,
                    reference_date_str=reference_date_str
                )

            zstore, chunk_index = prepare_zarr_store(deflated_gefs_grib_tree_store, gefs_kind)
            updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords,
                                                 times, valid_times, steps)

            print(f"✅ Zarr store created for {member} using reference mappings")

            return member, updated_zstore, True

        except Exception as e:
            print(f"⚠️ Reference mapping failed for {member}: {e}")
            print(f"🔄 Using direct zarr store for {member}...")
            return member, deflated_gefs_grib_tree_store, False

    except Exception as e:
        print(f"❌ Error processing {member}: {e}")
        return member, None, False


def process_ensemble_members_batch(members_batch, target_date_str, target_run, reference_date_str,
                                  axes, forecast_dict, time_dims, time_coords, times, valid_times, steps,
                                  mapping_manager=None):
    """Process a batch of ensemble members.

    Args:
        members_batch: List of ensemble member identifiers
        target_date_str: Target date string (YYYYMMDD)
        target_run: Run time (e.g., '00')
        reference_date_str: Reference date for parquet mappings
        axes: Time axes from generate_axes()
        forecast_dict: Dictionary of variables to process
        time_dims, time_coords, times, valid_times, steps: Time dimension info
        mapping_manager: Optional LocalTarGzMappingManager for local tar.gz usage
    """
    results = {}

    for member in members_batch:
        member_name, member_store, success = process_single_ensemble_member(
            member, target_date_str, target_run, reference_date_str,
            axes, forecast_dict, time_dims, time_coords, times, valid_times, steps,
            mapping_manager=mapping_manager
        )

        if member_store is not None:
            results[member_name] = {
                'store': member_store,
                'success': success
            }

    return results


def stream_ensemble_precipitation(members_data, variable='tp', output_dir=None):
    """Create parquet files for all successful ensemble members.

    NOTE: This function only creates parquet files. It does NOT load data into memory.
    To process the parquet files, use run_gefs_24h_accumulation.py (which uses obstore method).
    """
    print(f"\n💾 Creating parquet files for {len(members_data)} ensemble members...")

    parquet_files_created = []

    for member, data in members_data.items():
        print(f"\n📊 Processing {member}...")

        try:
            # Create parquet file for this member
            if output_dir:
                # Save in organized directory structure
                parquet_file = output_dir / f"{member}.par"
                parquet_file_str = str(parquet_file)
            else:
                # Original behavior - save in current directory
                parquet_file_str = f'gefs_{member}_{TARGET_DATE_STR}_{TARGET_RUN}z_fixed.par'

            create_parquet_file_fixed(data['store'], parquet_file_str)

            # Simple validation: Check that parquet file was created
            if not os.path.exists(parquet_file_str):
                raise ValueError(f"Failed to create parquet file for {member}")

            print(f"✅ Parquet file created successfully for {member}")
            parquet_files_created.append(parquet_file_str)

            # NOTE: Skipping xarray validation to avoid Zarr v3 FSMap issues
            # Use run_gefs_24h_accumulation.py (with obstore method) to process these files

        except Exception as e:
            print(f"❌ Error creating parquet for {member}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ Successfully created parquet files for {len(parquet_files_created)} members")

    if KEEP_PARQUET_FILES and output_dir:
        # List saved files
        parquet_files = sorted(output_dir.glob("*.par"))
        print(f"\n💾 Saved {len(parquet_files)} parquet files in {output_dir}:")
        for pf in parquet_files[:5]:  # Show first 5
            print(f"   - {pf.name}")
        if len(parquet_files) > 5:
            print(f"   ... and {len(parquet_files) - 5} more")

        print(f"\n📝 Next step: Run the following command to process these files:")
        print(f"   python run_gefs_24h_accumulation.py")

    # Return None since we're not loading data into memory
    return None, None



def main():
    """Main processing function for full ensemble GEFS data."""
    print("="*80)
    print("GEFS Full Ensemble Processing (30 Members)")
    print("="*80)

    start_time = time.time()

    # Create output directory structure
    output_dir = None
    if KEEP_PARQUET_FILES:
        output_dir = create_output_directory(TARGET_DATE_STR, TARGET_RUN)

    # Initialize mapping manager if using local tar.gz
    mapping_manager = None
    if USE_LOCAL_TARBALL:
        print(f"\n📦 Initializing local tar.gz mapping manager...")
        if not os.path.exists(LOCAL_TARBALL_PATH):
            print(f"❌ Error: Local tarball not found: {LOCAL_TARBALL_PATH}")
            print(f"   Please download the tar.gz file or set USE_LOCAL_TARBALL = False")
            return False
        mapping_manager = LocalTarGzMappingManager(LOCAL_TARBALL_PATH)
        available_members = mapping_manager.list_ensemble_members()
        print(f"✅ Found mappings for {len(available_members)} ensemble members")
        print(f"   Available: {', '.join(available_members[:5])}{'...' if len(available_members) > 5 else ''}")

    # 1. Generate axes for the target date
    print(f"\n📅 Generating axes for {TARGET_DATE_STR}...")
    axes = generate_axes(TARGET_DATE_STR)

    # 2. Define forecast variables
    forecast_dict = {
        "Surface pressure": "PRES:surface",
        "2 metre temperature": "TMP:2 m above ground",
        "10 metre U wind component": "UGRD:10 m above ground",
        "10 metre V wind component": "VGRD:10 m above ground",
        "Precipitable water": "PWAT:entire atmosphere (considered as a single layer)",
        "Convective available potential energy": "CAPE:surface",
        "Mean sea level pressure": "MSLET:mean sea level",
        "Total Precipitation": "APCP:surface",
    }

    # 3. Calculate time dimensions (same for all members)
    print(f"\n⏰ Calculating time dimensions...")
    time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
    print(f"✅ Time dimensions: {len(times)} timesteps")

    # 4. Process ensemble members in batches
    print(f"\n🚀 Processing {len(ENSEMBLE_MEMBERS)} ensemble members...")

    # Process in batches of 5 to manage resources
    batch_size = 5
    all_results = {}

    try:
        for i in range(0, len(ENSEMBLE_MEMBERS), batch_size):
            batch = ENSEMBLE_MEMBERS[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(ENSEMBLE_MEMBERS) + batch_size - 1) // batch_size

            print(f"\n📦 Processing batch {batch_num}/{total_batches}: {', '.join(batch)}")

            batch_results = process_ensemble_members_batch(
                batch, TARGET_DATE_STR, TARGET_RUN, REFERENCE_DATE_STR,
                axes, forecast_dict, time_dims, time_coords, times, valid_times, steps,
                mapping_manager=mapping_manager
            )

            all_results.update(batch_results)

            print(f"✅ Batch {batch_num} completed: {len(batch_results)} successful")
    finally:
        # Clean up mapping manager if it was created
        if mapping_manager is not None:
            mapping_manager.cleanup()
    
    # Summary
    successful = [m for m, data in all_results.items() if data['success']]
    partial = [m for m, data in all_results.items() if not data['success'] and data['store'] is not None]
    failed = [m for m in ENSEMBLE_MEMBERS if m not in all_results]
    
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successful (with reference mappings): {len(successful)} members")
    if successful:
        print(f"      {', '.join(successful[:10])}{'...' if len(successful) > 10 else ''}")
    
    if partial:
        print(f"   ⚠️ Partial (fallback zarr store): {len(partial)} members")
        print(f"      {', '.join(partial[:10])}{'...' if len(partial) > 10 else ''}")
    
    if failed:
        print(f"   ❌ Failed: {len(failed)} members")
        print(f"      {', '.join(failed)}")
    
    if len(all_results) == 0:
        print("\n❌ No ensemble members processed successfully!")
        return False

    # 5. Optionally stream precipitation data (disabled by default with Zarr v3)
    ensemble_numpy = {}
    ensemble_xarray = {}

    if STREAM_AFTER_CREATION:
        print("\n💾 Creating parquet files (Zarr v3 compatible - no xarray validation)...")
        try:
            ensemble_numpy, ensemble_xarray = stream_ensemble_precipitation(all_results, 'tp', output_dir)
            print("\n✅ Parquet file creation successful!")
        except Exception as e:
            print(f"\n❌ Parquet creation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Parquet file creation is DISABLED!")
        print("   📝 Set STREAM_AFTER_CREATION=True to create parquet files")
        print("   📝 Then use run_gefs_24h_accumulation.py to process them")

    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ GEFS full ensemble processing completed!")
    print("="*80)
    
    print(f"\n📁 Results:")
    print(f"   - Total processing time: {total_time/60:.1f} minutes")

    if STREAM_AFTER_CREATION and KEEP_PARQUET_FILES and output_dir:
        # Count parquet files that were created
        parquet_files = list(output_dir.glob("*.par"))
        print(f"   - Parquet files created: {len(parquet_files)}")
        print(f"   - Output directory: {output_dir}")
    elif STREAM_AFTER_CREATION:
        print(f"   - Parquet files: Created but not kept")
    else:
        print(f"   - Parquet file creation: Disabled")

    print(f"\n📝 Next step: Run run_gefs_24h_accumulation.py to process the parquet files")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Full ensemble processing completed successfully!")
    else:
        print("\n❌ Full ensemble processing failed. Check error messages above.")
        exit(1)
