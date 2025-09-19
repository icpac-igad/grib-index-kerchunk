#!/usr/bin/env python3
"""
Simple test routine to download and analyze ECMWF index files
before doing expensive GRIB processing. This helps debug ensemble
member extraction and NaN value issues.
"""

import fsspec
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Optional, Tuple
from pathlib import Path


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


def main():
    """
    Main test routine for ECMWF index file analysis.
    """
    print("="*80)
    print("ECMWF Index File Test & Analysis")
    print("="*80)

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