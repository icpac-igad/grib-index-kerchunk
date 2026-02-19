#!/usr/bin/env python3
"""
ECMWF Index-Based Processing for Steps 2-3
Implements index file parsing and mapped index creation for ECMWF ensemble processing.
Based on the GEFS cs_create_mapped_index pattern but adapted for ECMWF structure.
"""

import fsspec
import pandas as pd
import numpy as np
import json
import os
import re
import gcsfs
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import required kerchunk functions
try:
    from kerchunk._grib_idx import (
        build_idx_grib_mapping,
        strip_datavar_chunks,
        store_coord_var,
        store_data_var
    )
    from kerchunk.grib2 import grib_tree, scan_grib
except ImportError:
    print("Warning: kerchunk not available, some functions will be limited")


def download_index_file(s3_url: str, local_path: str = None) -> str:
    """
    Download ECMWF index file from S3 to local directory for inspection.

    Parameters:
    - s3_url: S3 URL to ECMWF GRIB file (index file will be derived)
    - local_path: Local path to save index file (optional)

    Returns:
    - Path to downloaded index file
    """
    # Construct index file URL
    index_url = f"{s3_url.rsplit('.', 1)[0]}.index"

    # Set up local filename
    if local_path is None:
        filename = index_url.split('/')[-1]
        local_path = f"./{filename}"

    print(f"Downloading index file from: {index_url}")
    print(f"Saving to: {local_path}")

    # Download using fsspec
    fs = fsspec.filesystem("s3", anon=True)

    try:
        # Check if file exists
        if not fs.exists(index_url):
            raise FileNotFoundError(f"Index file not found: {index_url}")

        # Download file
        fs.download(index_url, local_path)
        print(f"✅ Downloaded index file: {local_path}")

        # Show file size
        file_size = os.path.getsize(local_path)
        print(f"📊 File size: {file_size:,} bytes")

        return local_path

    except Exception as e:
        print(f"❌ Error downloading index file: {e}")
        raise


def parse_index_file_simple(index_file_path: str) -> pd.DataFrame:
    """
    Parse ECMWF index file and return DataFrame with basic analysis.

    Parameters:
    - index_file_path: Path to local index file

    Returns:
    - DataFrame with parsed index data
    """
    print(f"\n📋 Parsing index file: {index_file_path}")

    records = []
    line_count = 0
    parse_errors = 0

    with open(index_file_path, 'r') as f:
        for idx, line in enumerate(f):
            line_count += 1
            try:
                # Clean line
                clean_line = line.strip().rstrip(',')

                # Parse JSON
                data = json.loads(clean_line)

                # Extract key fields
                record = {
                    'idx': idx,
                    'offset': data.get('_offset', 0),
                    'length': data.get('_length', 0),
                    'number': data.get('number', None),  # Keep as None for NaN detection
                    'param': data.get('param', None),
                    'levtype': data.get('levtype', None),
                    'levelist': data.get('levelist', None),
                    'step': data.get('step', None),
                    'date': data.get('date', None),
                    'time': data.get('time', None),
                }

                records.append(record)

            except Exception as e:
                parse_errors += 1
                print(f"⚠️ Parse error at line {idx}: {e}")
                if parse_errors <= 5:  # Show first 5 errors
                    print(f"   Line content: {line[:100]}...")

    df = pd.DataFrame(records)

    print(f"📊 Parsing summary:")
    print(f"   Total lines: {line_count}")
    print(f"   Successfully parsed: {len(df)}")
    print(f"   Parse errors: {parse_errors}")

    return df


def analyze_ensemble_numbers(df: pd.DataFrame) -> Dict:
    """
    Analyze ensemble number distribution and NaN values.

    Parameters:
    - df: DataFrame from parse_index_file_simple

    Returns:
    - Analysis results dictionary
    """
    print(f"\n🔍 Analyzing ensemble numbers...")

    # Count total entries
    total_entries = len(df)

    # Check for NaN values in 'number' column
    nan_count = df['number'].isna().sum()
    non_nan_count = total_entries - nan_count

    # Get unique ensemble numbers (excluding NaN)
    unique_numbers = df['number'].dropna().unique()

    # Count occurrences of each ensemble number
    number_counts = df['number'].value_counts(dropna=False)

    print(f"📊 Ensemble number analysis:")
    print(f"   Total entries: {total_entries}")
    print(f"   NaN values: {nan_count} ({nan_count/total_entries*100:.1f}%)")
    print(f"   Non-NaN values: {non_nan_count} ({non_nan_count/total_entries*100:.1f}%)")
    print(f"   Unique ensemble numbers: {len(unique_numbers)}")

    if len(unique_numbers) > 0:
        print(f"   Range: {unique_numbers.min()} to {unique_numbers.max()}")
        print(f"   Ensemble numbers found: {sorted(unique_numbers)}")

    print(f"\n📈 Top 10 ensemble number frequencies:")
    for number, count in number_counts.head(10).items():
        if pd.isna(number):
            print(f"   NaN: {count}")
        else:
            print(f"   {int(number)}: {count}")

    return {
        'total_entries': total_entries,
        'nan_count': nan_count,
        'non_nan_count': non_nan_count,
        'unique_numbers': unique_numbers,
        'number_counts': number_counts
    }


def analyze_parameters(df: pd.DataFrame) -> Dict:
    """
    Analyze parameter distribution in the index.

    Parameters:
    - df: DataFrame from parse_index_file_simple

    Returns:
    - Parameter analysis results
    """
    print(f"\n🌡️ Analyzing parameters...")

    # Parameter counts
    param_counts = df['param'].value_counts()
    levtype_counts = df['levtype'].value_counts()

    print(f"📊 Parameter analysis:")
    print(f"   Unique parameters: {len(param_counts)}")
    print(f"   Unique level types: {len(levtype_counts)}")

    print(f"\n🔝 Top 10 parameters:")
    for param, count in param_counts.head(10).items():
        print(f"   {param}: {count}")

    print(f"\n📏 Level types:")
    for levtype, count in levtype_counts.items():
        print(f"   {levtype}: {count}")

    return {
        'param_counts': param_counts,
        'levtype_counts': levtype_counts
    }


def test_ensemble_member_mapping(df: pd.DataFrame) -> Dict[int, int]:
    """
    Test the ensemble member mapping logic with proper NaN handling.

    Parameters:
    - df: DataFrame from parse_index_file_simple

    Returns:
    - Mapping from index to ensemble number
    """
    print(f"\n🗺️ Testing ensemble member mapping...")

    idx_mapping = {}
    nan_count = 0
    member_counts = {}

    for idx, row in df.iterrows():
        ens_number = row['number']

        # Handle NaN values properly
        if pd.isna(ens_number):
            ens_number = -1  # Default to control member for NaN
            nan_count += 1

        try:
            member_num = int(float(ens_number))  # Convert via float first
            idx_mapping[idx] = member_num
            member_counts[member_num] = member_counts.get(member_num, 0) + 1
        except (ValueError, TypeError):
            print(f"⚠️ Invalid ensemble number at index {idx}: {ens_number}, defaulting to control (-1)")
            idx_mapping[idx] = -1
            member_counts[-1] = member_counts.get(-1, 0) + 1

    print(f"📊 Ensemble mapping summary:")
    print(f"   NaN values converted to control: {nan_count}")
    print(f"   Total mapped entries: {len(idx_mapping)}")
    print(f"   Ensemble members found: {sorted(member_counts.keys())}")

    print(f"\n👥 Ensemble member distribution:")
    for member, count in sorted(member_counts.items()):
        member_name = "control" if member == -1 else f"ens{member:02d}"
        print(f"   {member_name}: {count} messages")

    return idx_mapping


def validate_target_members(df: pd.DataFrame, target_members: list = None) -> Dict:
    """
    Validate that target ensemble members exist in the data.

    Parameters:
    - df: DataFrame from parse_index_file_simple
    - target_members: List of target ensemble members to check

    Returns:
    - Validation results
    """
    if target_members is None:
        target_members = [-1, 1, 2, 3, 4, 5]  # Default test members

    print(f"\n✅ Validating target ensemble members: {target_members}")

    # Create mapping
    idx_mapping = test_ensemble_member_mapping(df)

    # Check availability of target members
    available_members = set(idx_mapping.values())

    results = {}
    for member in target_members:
        count = len([idx for idx, mem in idx_mapping.items() if mem == member])
        member_name = "control" if member == -1 else f"ens{member:02d}"

        if member in available_members:
            results[member] = {
                'available': True,
                'count': count,
                'name': member_name
            }
            print(f"   ✅ {member_name}: {count} messages")
        else:
            results[member] = {
                'available': False,
                'count': 0,
                'name': member_name
            }
            print(f"   ❌ {member_name}: Not found")

    return results


# ================== Step 2: Time Dimension Functions ==================

def generate_axes(date_str: str) -> List[pd.Index]:
    """
    Generate temporal axes for ECMWF forecast (15-day period).

    Parameters:
    - date_str: Start date formatted as 'YYYYMMDD'

    Returns:
    - List containing valid_time and time indices
    """
    start_date = pd.Timestamp(date_str)

    # ECMWF forecast structure
    hours_3h = np.arange(0, 93, 3)      # 0-90h at 3h intervals
    hours_6h = np.arange(96, 150, 6)    # 96-144h at 6h intervals
    hours_12h = np.arange(156, 372, 12) # 156-360h at 12h intervals

    forecast_hours = np.concatenate([hours_3h, hours_6h, hours_12h])
    valid_times = start_date + pd.to_timedelta(forecast_hours, unit='h')

    valid_time_index = pd.Index(valid_times, name="valid_time")
    time_index = pd.Index([start_date], name="time")

    return [valid_time_index, time_index]


def create_ecmwf_time_dimensions(date_str: str, run: str = "00"):
    """
    Create time dimensions for ECMWF ensemble forecast.

    Returns:
    - time_dims: Dictionary of dimensions
    - time_coords: Dictionary of coordinate arrays
    - times, valid_times, steps: numpy arrays
    """
    reference_time = pd.Timestamp(f"{date_str} {run}:00:00")

    # ECMWF forecast hour structure
    hours_3h = np.arange(0, 93, 3)
    hours_6h = np.arange(96, 150, 6)
    hours_12h = np.arange(156, 372, 12)

    forecast_hours = np.concatenate([hours_3h, hours_6h, hours_12h])
    valid_times = reference_time + pd.to_timedelta(forecast_hours, unit='h')

    times = np.array([reference_time] * len(forecast_hours))
    steps = pd.to_timedelta(forecast_hours, unit='h')

    time_dims = {
        'time': len(times),
        'step': len(steps),
        'valid_time': len(valid_times)
    }

    time_coords = {
        'time': ('time',),
        'step': ('step',),
        'valid_time': ('valid_time',),
        'datavar': ('time', 'latitude', 'longitude')
    }

    return time_dims, time_coords, times, valid_times, steps


# ================== Step 3: Mapped Index Creation ==================

def parse_ecmwf_index_for_mapping(index_url: str) -> Dict[int, int]:
    """
    Parse ECMWF index file and create index-to-ensemble-member mapping.

    Parameters:
    - index_url: URL to ECMWF .index file

    Returns:
    - Dictionary mapping index to ensemble member number
    """
    fs = fsspec.filesystem("s3", anon=True)
    idx_to_member = {}

    with fs.open(index_url, 'r') as f:
        for idx, line in enumerate(f):
            try:
                clean_line = line.strip().rstrip(',')
                data = json.loads(clean_line)

                # Extract ensemble number with NaN handling
                ens_number = data.get('number', None)

                # Handle NaN: default to control member (-1)
                if ens_number is None or pd.isna(ens_number):
                    ens_number = -1
                else:
                    ens_number = int(float(ens_number))

                idx_to_member[idx] = ens_number

            except Exception as e:
                # Default to control for parse errors
                idx_to_member[idx] = -1

    return idx_to_member


def cs_create_mapped_index_ecmwf(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    member_number: int,
    gcp_service_account_json: str,
    reference_date_str: Optional[str] = None
) -> pd.DataFrame:
    """
    Create mapped index for ECMWF ensemble member using index files.
    Adapted from GEFS cs_create_mapped_index pattern.

    Parameters:
    - axes: Time axes from generate_axes()
    - gcs_bucket_name: GCS bucket containing scan_grib parquet files
    - date_str: Date string (YYYYMMDD)
    - member_number: Ensemble member (-1 for control, 1-50 for perturbed)
    - gcp_service_account_json: Path to GCS credentials
    - reference_date_str: Optional reference date for template mappings

    Returns:
    - DataFrame with mapped index for the ensemble member
    """
    print(f"Creating mapped index for ECMWF member {member_number} on {date_str}")

    # Setup GCS filesystem
    fs = gcsfs.GCSFileSystem(token=gcp_service_account_json)

    # Get list of forecast hour parquet files
    parquet_prefix = f"fmrc/scan_grib{date_str}/"
    parquet_files = fs.glob(f"{gcs_bucket_name}/{parquet_prefix}*.parquet")

    print(f"Found {len(parquet_files)} parquet files")

    mapped_indices = []

    for pfile in sorted(parquet_files):
        # Extract forecast hour from filename
        hour_match = re.search(r'_(\d+)h\.parquet', pfile)
        if not hour_match:
            continue
        forecast_hour = int(hour_match.group(1))

        print(f"Processing forecast hour {forecast_hour}h")

        # Load parquet file
        df = pd.read_parquet(f"gs://{pfile}", filesystem=fs)

        # Get GRIB URL from first row
        if 'uri' in df.columns:
            grib_url = df['uri'].iloc[0]
        else:
            # Construct GRIB URL from metadata
            grib_url = f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-{forecast_hour}h-enfo-ef.grib2"

        # Parse index file for ensemble mapping
        index_url = grib_url.replace('.grib2', '.index')
        idx_to_member = parse_ecmwf_index_for_mapping(index_url)

        # Filter groups for target ensemble member
        member_groups = []

        for _, row in df.iterrows():
            groups = json.loads(row['groups']) if isinstance(row['groups'], str) else row['groups']

            for group in groups:
                # Get index from group metadata
                group_idx = group.get('idx', 0)
                group_member = idx_to_member.get(group_idx, -1)

                # Keep only groups for target member
                if group_member == member_number:
                    # Add metadata
                    group['forecast_hour'] = forecast_hour
                    group['ensemble_member'] = member_number
                    member_groups.append(group)

        if member_groups:
            print(f"  Found {len(member_groups)} groups for member {member_number}")

            # Create mapping entry
            mapping_entry = {
                'forecast_hour': forecast_hour,
                'grib_url': grib_url,
                'index_url': index_url,
                'member_groups': member_groups,
                'group_count': len(member_groups)
            }
            mapped_indices.append(mapping_entry)

    # Convert to DataFrame
    if mapped_indices:
        mapped_df = pd.DataFrame(mapped_indices)
        print(f"Created mapped index with {len(mapped_df)} forecast hours")
        return mapped_df
    else:
        print(f"Warning: No data found for member {member_number}")
        return pd.DataFrame()


def process_ecmwf_member_to_zarr(
    date_str: str,
    member_number: int,
    gcs_bucket: str,
    credentials_path: str,
    output_path: str
) -> str:
    """
    Process ECMWF member to create zarr store parquet file.

    Parameters:
    - date_str: Date string (YYYYMMDD)
    - member_number: Ensemble member (-1 for control, 1-50)
    - gcs_bucket: GCS bucket name
    - credentials_path: GCS credentials JSON path
    - output_path: Output directory for parquet files

    Returns:
    - Path to created parquet file
    """
    print(f"\n{'='*60}")
    print(f"Processing ECMWF Member {member_number} for {date_str}")
    print(f"{'='*60}")

    start_time = time.time()

    # Step 2: Generate axes and time dimensions
    print("\n📅 Step 2: Creating time dimensions...")
    axes = generate_axes(date_str)
    time_dims, time_coords, times, valid_times, steps = create_ecmwf_time_dimensions(date_str)
    print(f"  ✅ Time dimensions created: {len(times)} timesteps")

    # Step 3: Create index-based mapped index
    print("\n🗺️ Step 3: Creating mapped index...")
    mapped_index = cs_create_mapped_index_ecmwf(
        axes,
        gcs_bucket,
        date_str,
        member_number,
        credentials_path
    )

    if mapped_index.empty:
        raise ValueError(f"No data found for member {member_number}")

    print(f"  ✅ Mapped index created with {len(mapped_index)} forecast hours")

    # Build zarr store structure
    print("\n🔨 Building zarr store...")
    zstore = {}

    # Add basic metadata
    zstore['.zattrs'] = json.dumps({
        'date': date_str,
        'ensemble_member': member_number,
        'forecast_hours': len(mapped_index),
        'created': pd.Timestamp.now().isoformat()
    })

    # Process each forecast hour
    for _, row in mapped_index.iterrows():
        forecast_hour = row['forecast_hour']
        groups = row['member_groups']

        # Add groups to store
        for group in groups:
            # Create zarr key from group metadata
            varname = group.get('varname', 'unknown')
            level = group.get('level', 'surface')

            key = f"{varname}/{level}/{forecast_hour}"
            zstore[key] = json.dumps(group)

    # Save as parquet file
    member_name = "control" if member_number == -1 else f"ens{member_number:02d}"
    output_file = f"{output_path}/ecmwf_{date_str}_{member_name}.par"

    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)

    # Save zarr store as parquet
    print("\n💾 Saving parquet file...")
    store_df = pd.DataFrame([
        {'key': k, 'value': v.encode('utf-8') if isinstance(v, str) else v}
        for k, v in zstore.items()
    ])
    store_df.to_parquet(output_file)

    elapsed = time.time() - start_time
    print(f"\n✅ Created: {output_file}")
    print(f"⏱️ Processing time: {elapsed:.1f} seconds")

    return output_file


def save_analysis_results(df: pd.DataFrame, output_dir: str = "./index_analysis"):
    """
    Save analysis results to files for inspection.

    Parameters:
    - df: DataFrame from parse_index_file_simple
    - output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n💾 Saving analysis results to: {output_path}")

    # Save full DataFrame
    df.to_csv(output_path / "index_full.csv", index=False)
    print(f"   📄 Full index data: index_full.csv")

    # Save summary statistics
    summary = df.describe(include='all')
    summary.to_csv(output_path / "index_summary.csv")
    print(f"   📊 Summary statistics: index_summary.csv")

    # Save ensemble number analysis
    ensemble_analysis = df['number'].value_counts(dropna=False)
    ensemble_analysis.to_csv(output_path / "ensemble_counts.csv", header=['count'])
    print(f"   👥 Ensemble counts: ensemble_counts.csv")

    # Save parameter analysis
    param_analysis = df['param'].value_counts()
    param_analysis.to_csv(output_path / "parameter_counts.csv", header=['count'])
    print(f"   🌡️ Parameter counts: parameter_counts.csv")

    print(f"✅ Analysis results saved to {output_path}")


def test_step2_step3_processing():
    """
    Test Step 2 and Step 3 processing for ECMWF ensemble.
    """
    print("="*80)
    print("ECMWF Step 2 & Step 3 Processing Test")
    print("="*80)

    # Configuration
    date_str = "20240529"
    gcs_bucket = "gik-ecmwf-aws-tf"
    credentials_path = "coiled-data.json"
    target_members = [-1, 1, 2]  # Test with control + 2 members
    output_dir = f"./ecmwf_{date_str}_test"

    print(f"📅 Date: {date_str}")
    print(f"🪣 GCS Bucket: {gcs_bucket}")
    print(f"👥 Target members: {target_members}")
    print(f"📁 Output directory: {output_dir}")

    successful = []
    failed = []

    # Process each member
    for member in target_members:
        try:
            print(f"\n{'='*60}")
            output_file = process_ecmwf_member_to_zarr(
                date_str,
                member,
                gcs_bucket,
                credentials_path,
                output_dir
            )
            successful.append(member)
            print(f"✅ Success: Member {member}")
        except Exception as e:
            failed.append(member)
            print(f"❌ Failed: Member {member}: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"✅ Successful: {len(successful)}/{len(target_members)}")
    if successful:
        print(f"   Members: {successful}")

    if failed:
        print(f"❌ Failed: {len(failed)}")
        print(f"   Members: {failed}")

    # List output files
    if os.path.exists(output_dir):
        files = sorted(Path(output_dir).glob("*.par"))
        print(f"\n📁 Output files:")
        for f in files:
            size = os.path.getsize(f) / 1024  # KB
            print(f"   {f.name}: {size:.1f} KB")

    return len(successful) == len(target_members)


def main():
    """
    Main routine with options for different test modes.
    """
    print("="*80)
    print("ECMWF Index-Based Processing Test Suite")
    print("="*80)

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--step2-3":
        # Test Step 2 and Step 3 processing
        success = test_step2_step3_processing()
        if success:
            print("\n🎉 Step 2-3 processing test completed successfully!")
            exit(0)
        else:
            print("\n❌ Step 2-3 processing test failed.")
            exit(1)

    # Default: Original index analysis
    print("\nRunning index file analysis...")
    print("(Use --step2-3 flag to test Step 2 and Step 3 processing)\n")

    # Test configuration
    ecmwf_url = "s3://ecmwf-forecasts/20250628/18z/ifs/0p25/enfo/20250628180000-0h-enfo-ef.grib2"
    target_members = [-1, 1, 2, 3, 4, 5]  # Control + first 5 ensemble members

    print(f"🎯 Testing ECMWF URL: {ecmwf_url}")
    print(f"👥 Target ensemble members: {target_members}")

    try:
        # Step 1: Download index file
        print(f"\n" + "="*50)
        print("Step 1: Download Index File")
        print("="*50)

        index_file = download_index_file(ecmwf_url)

        # Step 2: Parse index file
        print(f"\n" + "="*50)
        print("Step 2: Parse Index File")
        print("="*50)

        df = parse_index_file_simple(index_file)

        if df.empty:
            print("❌ No data parsed from index file!")
            return False

        # Step 3: Analyze ensemble numbers
        print(f"\n" + "="*50)
        print("Step 3: Analyze Ensemble Numbers")
        print("="*50)

        ensemble_analysis = analyze_ensemble_numbers(df)

        # Step 4: Analyze parameters
        print(f"\n" + "="*50)
        print("Step 4: Analyze Parameters")
        print("="*50)

        parameter_analysis = analyze_parameters(df)

        # Step 5: Test ensemble member mapping
        print(f"\n" + "="*50)
        print("Step 5: Test Ensemble Member Mapping")
        print("="*50)

        idx_mapping = test_ensemble_member_mapping(df)

        # Step 6: Validate target members
        print(f"\n" + "="*50)
        print("Step 6: Validate Target Members")
        print("="*50)

        validation_results = validate_target_members(df, target_members)

        # Step 7: Save analysis results
        print(f"\n" + "="*50)
        print("Step 7: Save Analysis Results")
        print("="*50)

        save_analysis_results(df)

        # Final summary
        print(f"\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        available_count = sum(1 for r in validation_results.values() if r['available'])
        total_target = len(target_members)

        print(f"📊 Index file analysis:")
        print(f"   Total messages: {len(df)}")
        print(f"   NaN ensemble numbers: {ensemble_analysis['nan_count']}")
        print(f"   Unique ensemble members: {len(ensemble_analysis['unique_numbers'])}")
        print(f"   Unique parameters: {len(parameter_analysis['param_counts'])}")

        print(f"\n🎯 Target member validation:")
        print(f"   Available: {available_count}/{total_target}")
        print(f"   Success rate: {available_count/total_target*100:.1f}%")

        if available_count == total_target:
            print(f"\n✅ All target ensemble members are available!")
            print(f"🚀 Ready to proceed with GRIB processing.")
        else:
            print(f"\n⚠️ Some target ensemble members are missing.")
            print(f"🔧 Consider adjusting target_members list or investigating data.")

        # Clean up
        os.remove(index_file)
        print(f"\n🗑️ Cleaned up temporary index file: {index_file}")

        return available_count == total_target

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🎉 Index file test completed successfully!")
        print(f"✅ Ready to run ecmwf_ensemble_par_creator.py")
    else:
        print(f"\n❌ Index file test failed.")
        print(f"🔧 Please check the analysis results and fix issues before proceeding.")
        exit(1)