#!/usr/bin/env python3
"""
ECMWF Ensemble Parquet Creator
Following GEFS pattern to create individual parquet files for each ensemble member.
Main difference: ECMWF has all members in single GRIB file, requiring extraction.
"""

import fsspec
import pandas as pd
import numpy as np
import xarray as xr
import json
import copy
import os
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from kerchunk.grib2 import scan_grib, grib_tree
from kerchunk._grib_idx import strip_datavar_chunks
from kerchunk.combine import MultiZarrToZarr

warnings.filterwarnings('ignore')

# Set up anonymous S3 access for ECMWF data
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'


def log_message(message: str, level: str = "INFO"):
    """Simple logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


# Import the fixed ensemble grib tree function after log_message is defined
try:
    # Try importing from eutils.py first
    from eutils import fixed_ensemble_grib_tree
    log_message("Successfully imported fixed_ensemble_grib_tree from eutils.py")
except ImportError:
    try:
        # If not available, try from test file
        from test_run_ecmwf_step1_scangrib import fixed_ensemble_grib_tree
        log_message("Successfully imported fixed_ensemble_grib_tree from test file")
    except ImportError:
        # If neither works, we'll use a simple fallback
        log_message("Warning: Could not import fixed_ensemble_grib_tree, using grib_tree with remote_options", "WARNING")
        fixed_ensemble_grib_tree = None


def s3_parse_ecmwf_grib_idx(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    suffix: str = "index",
    tstamp: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Parse ECMWF GRIB index file to extract metadata."""
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"

    fs.invalidate_cache(fname)
    fs.invalidate_cache(basename)

    baseinfo = fs.info(basename)

    with fs.open(fname, "r") as f:
        splits = []
        for idx, line in enumerate(f):
            try:
                clean_line = line.strip().rstrip(',')
                data = json.loads(clean_line)

                lidx = idx
                offset = data.get("_offset", 0)
                length = data.get("_length", 0)
                date = data.get("date", "Unknown Date")
                ens_number = data.get("number", -1)  # -1 for control

                splits.append([
                    int(lidx),
                    int(offset),
                    int(length),
                    date,
                    data,
                    int(ens_number)
                ])
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse JSON from line: {line}") from e

    result = pd.DataFrame(
        splits,
        columns=["idx", "offset", "length", "date", "attr", "ens_number"]
    )

    result.loc[:, "idx_uri"] = fname
    result.loc[:, "grib_uri"] = basename

    if tstamp is None:
        tstamp = pd.Timestamp.now()
    result['indexed_at'] = tstamp

    if "s3" in fs.protocol:
        result.loc[:, "grib_etag"] = baseinfo.get("ETag")
        result.loc[:, "grib_updated_at"] = pd.to_datetime(
            baseinfo.get("LastModified")
        ).tz_localize(None)

        idxinfo = fs.info(fname)
        result.loc[:, "idx_etag"] = idxinfo.get("ETag")
        result.loc[:, "idx_updated_at"] = pd.to_datetime(
            idxinfo.get("LastModified")
        ).tz_localize(None)

    log_message(f"Parsed index with {len(result)} entries")
    return result.set_index("idx")


def create_ensemble_member_mapping(idx_df: pd.DataFrame) -> Dict[int, int]:
    """Create mapping from GRIB message index to ensemble member number."""
    # Expand the attr column to access ensemble number directly
    edf = pd.concat([
        idx_df.drop('attr', axis=1),
        idx_df['attr'].apply(pd.Series)
    ], axis=1)

    # Create mapping: index -> ensemble member number
    idx_mapping = {}
    for idx, row in edf.iterrows():
        ens_number = row.get('number', -1)  # -1 for control

        # Handle NaN values properly
        if pd.isna(ens_number):
            ens_number = -1  # Default to control member for NaN

        try:
            idx_mapping[idx] = int(float(ens_number))  # Convert via float first to handle string numbers
        except (ValueError, TypeError):
            log_message(f"Warning: Invalid ensemble number at index {idx}: {ens_number}, defaulting to control (-1)", "WARNING")
            idx_mapping[idx] = -1

    return idx_mapping


def extract_ensemble_member_groups(
    grib_url: str,
    target_member: int
) -> List[Dict]:
    """
    Extract GRIB groups for a specific ensemble member.

    Parameters:
    - grib_url: S3 URL to ECMWF GRIB file
    - target_member: Ensemble member number (-1 for control, 1-50 for perturbed)

    Returns:
    - List of scan_grib groups for the specified member
    """
    log_message(f"Extracting member {target_member} from {grib_url}")

    # Scan GRIB file
    storage_options = {"anon": True}
    all_groups = scan_grib(grib_url, storage_options=storage_options)

    # Get ensemble member mapping from index
    fs = fsspec.filesystem("s3", anon=True)
    idx_df = s3_parse_ecmwf_grib_idx(fs, grib_url)
    idx_mapping = create_ensemble_member_mapping(idx_df)

    # Filter groups for target member
    member_groups = []
    for i, group in enumerate(all_groups):
        if idx_mapping.get(i, -99) == target_member:
            # Add ensemble metadata to the group
            mod_group = copy.deepcopy(group)

            # Add ensemble information to zarr attributes
            if '.zattrs' in mod_group['refs']:
                try:
                    attrs = json.loads(mod_group['refs']['.zattrs'])
                    attrs['ensemble_member'] = target_member
                    mod_group['refs']['.zattrs'] = json.dumps(attrs)
                except json.JSONDecodeError:
                    pass

            member_groups.append(mod_group)

    log_message(f"Found {len(member_groups)} messages for member {target_member}")
    return member_groups


def build_member_grib_tree(member_groups: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Build GRIB tree for a single ensemble member, following GEFS pattern.
    Uses fixed_ensemble_grib_tree for proper anonymous S3 access.

    Returns:
    - Original and deflated GRIB tree stores
    """
    log_message(f"Building GRIB tree from {len(member_groups)} groups")

    # Set up remote options for anonymous S3 access
    remote_options = {"anon": True}

    if fixed_ensemble_grib_tree is not None:
        # Use the fixed ensemble grib tree function if available
        log_message("Using fixed_ensemble_grib_tree for proper anonymous S3 access")
        member_tree = fixed_ensemble_grib_tree(
            member_groups,
            remote_options=remote_options,
            debug_output=False
        )
    else:
        # Fallback to regular grib_tree with explicit remote options
        log_message("Using fallback grib_tree approach", "WARNING")

        # Try to set up fsspec for anonymous access
        import fsspec
        fsspec.config.conf['s3'] = {'anon': True}

        member_tree = grib_tree(member_groups)

    # Create deflated version
    deflated_tree = copy.deepcopy(member_tree)
    strip_datavar_chunks(deflated_tree)

    log_message(f"Original refs: {len(member_tree['refs'])}, Deflated refs: {len(deflated_tree['refs'])}")

    return member_tree, deflated_tree


def generate_ecmwf_axes(date_str: str, run: str, forecast_hours: int = 240) -> List[pd.Index]:
    """
    Generate temporal axes for ECMWF forecast.

    Parameters:
    - date_str: Date string 'YYYYMMDD'
    - run: Run hour ('00', '06', '12', '18')
    - forecast_hours: Total forecast hours (default 240 for 10-day)

    Returns:
    - List of time indices
    """
    start_date = pd.Timestamp(f"{date_str} {run}:00:00")
    end_date = start_date + pd.Timedelta(hours=forecast_hours)

    # ECMWF typically uses 3-hour intervals for ensemble forecasts
    valid_time_index = pd.date_range(start_date, end_date, freq="180min", name="valid_time")
    time_index = pd.Index([start_date], name="time")

    return [valid_time_index, time_index]


def calculate_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time dimensions following GEFS pattern.
    """
    axes_by_name = {pdi.name: pdi for pdi in axes}

    time_dims = {}
    time_coords = {}

    # Best available aggregation type (ECMWF style)
    time_dims["valid_time"] = len(axes_by_name["valid_time"])
    reference_time = axes_by_name["time"].to_numpy()[0]

    time_coords["step"] = ("valid_time",)
    time_coords["valid_time"] = ("valid_time",)
    time_coords["time"] = ("valid_time",)

    valid_times = axes_by_name["valid_time"].to_numpy()
    times = np.where(valid_times <= reference_time, valid_times, reference_time)
    steps = valid_times - times

    return time_dims, time_coords, times, valid_times, steps


def create_parquet_file(zstore: dict, output_parquet_file: str):
    """Save zarr store as parquet file following GEFS pattern."""
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
    log_message(f"Saved parquet file: {output_parquet_file} ({len(df)} rows)")


def process_single_ensemble_member(
    ecmwf_files: List[str],
    member_number: int,
    date_str: str,
    run: str,
    output_dir: Path
) -> bool:
    """
    Process a single ECMWF ensemble member and create parquet file.

    Parameters:
    - ecmwf_files: List of ECMWF GRIB file URLs
    - member_number: Ensemble member (-1 for control, 1-50 for perturbed)
    - date_str: Date string
    - run: Run hour
    - output_dir: Output directory path

    Returns:
    - Success status
    """
    member_name = "control" if member_number == -1 else f"ens{member_number:02d}"
    log_message(f"Processing ensemble member: {member_name}")

    try:
        # Extract groups for this member from all files
        all_member_groups = []
        for grib_file in ecmwf_files:
            member_groups = extract_ensemble_member_groups(grib_file, member_number)
            all_member_groups.extend(member_groups)

        if not all_member_groups:
            log_message(f"No groups found for member {member_name}", "WARNING")
            return False

        # Build GRIB tree
        member_tree, deflated_tree = build_member_grib_tree(all_member_groups)

        # Generate time axes
        axes = generate_ecmwf_axes(date_str, run)
        time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)

        # Process the zarr store (simplified version - full implementation would include
        # coordinate processing similar to GEFS)
        zstore = deflated_tree['refs']

        # Save as parquet
        parquet_file = output_dir / f"{member_name}.par"
        create_parquet_file(zstore, str(parquet_file))

        # Validate by trying to open
        try:
            fs = fsspec.filesystem("reference", fo={'refs': zstore, 'version': 1},
                                 remote_protocol='s3', remote_options={'anon': True})
            mapper = fs.get_mapper("")

            # Quick validation - just check if zarr can read it
            import zarr
            z = zarr.open(mapper, mode='r')
            log_message(f"Validation successful for {member_name}")
        except Exception as e:
            log_message(f"Validation failed for {member_name}: {e}", "WARNING")

        return True

    except Exception as e:
        log_message(f"Error processing member {member_name}: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def process_ecmwf_ensemble(
    date_str: str,
    run: str = '00',
    members_to_process: Optional[List[int]] = None,
    batch_size: int = 5
):
    """
    Main function to process ECMWF ensemble following GEFS pattern.

    Parameters:
    - date_str: Date string 'YYYYMMDD'
    - run: Run hour ('00', '06', '12', '18')
    - members_to_process: List of member numbers to process (default: all)
    - batch_size: Number of members to process in parallel
    """
    print("="*80)
    print("ECMWF Ensemble Parquet Creator (GEFS-style)")
    print("="*80)

    start_time = time.time()

    # Create output directory
    output_dir = Path(f"ecmwf_{date_str}_{run}")
    output_dir.mkdir(exist_ok=True)
    log_message(f"Output directory: {output_dir}")

    # Define ECMWF files (typically 3-hour intervals)
    ecmwf_files = [
        f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-0h-enfo-ef.grib2",
        f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-3h-enfo-ef.grib2",
        f"s3://ecmwf-forecasts/{date_str}/{run}z/ifs/0p25/enfo/{date_str}{run}0000-6h-enfo-ef.grib2",
    ]

    # Check if files exist
    fs = fsspec.filesystem("s3", anon=True)
    for f in ecmwf_files:
        if not fs.exists(f):
            log_message(f"File not found: {f}", "WARNING")
            ecmwf_files.remove(f)

    if not ecmwf_files:
        log_message("No valid ECMWF files found", "ERROR")
        return False

    log_message(f"Found {len(ecmwf_files)} ECMWF files")

    # Define ensemble members to process
    if members_to_process is None:
        # Process all members: control (-1) plus 50 perturbed (1-50)
        members_to_process = [-1] + list(range(1, 51))

    log_message(f"Processing {len(members_to_process)} ensemble members")

    # Process members in batches
    successful = []
    failed = []

    for i in range(0, len(members_to_process), batch_size):
        batch = members_to_process[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(members_to_process) + batch_size - 1) // batch_size

        log_message(f"Processing batch {batch_num}/{total_batches}: members {batch}")

        # Process each member in the batch (could be parallelized)
        for member in batch:
            success = process_single_ensemble_member(
                ecmwf_files, member, date_str, run, output_dir
            )

            member_name = "control" if member == -1 else f"ens{member:02d}"
            if success:
                successful.append(member_name)
            else:
                failed.append(member_name)

    # Summary
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("Processing Summary")
    print("="*80)

    print(f"\nResults:")
    print(f"  ✅ Successful: {len(successful)} members")
    if successful[:5]:
        print(f"     {', '.join(successful[:5])}{'...' if len(successful) > 5 else ''}")

    if failed:
        print(f"  ❌ Failed: {len(failed)} members")
        print(f"     {', '.join(failed)}")

    print(f"\nOutput:")
    print(f"  📁 Directory: {output_dir}")
    print(f"  📄 Parquet files: {len(list(output_dir.glob('*.par')))}")

    print(f"\nPerformance:")
    print(f"  ⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"  📊 Average per member: {total_time/len(members_to_process):.1f} seconds")

    # Create metadata file
    metadata = {
        "date_str": date_str,
        "run": run,
        "ecmwf_files": ecmwf_files,
        "members_processed": len(successful),
        "members_failed": len(failed),
        "successful_members": successful,
        "failed_members": failed,
        "processing_time_seconds": total_time,
        "output_directory": str(output_dir)
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    log_message(f"Metadata saved to {metadata_file}")

    return len(successful) > 0


def validate_parquet_files(output_dir: Path):
    """
    Validate that parquet files can be opened and contain expected structure.
    """
    log_message("Validating parquet files...")

    parquet_files = sorted(output_dir.glob("*.par"))

    for pf in parquet_files[:3]:  # Check first 3 files
        try:
            # Read parquet
            df = pd.read_parquet(pf)

            # Convert back to zarr store format
            zstore = {}
            for _, row in df.iterrows():
                key = row['key']
                value = row['value']
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                zstore[key] = value

            # Try to open with fsspec
            fs = fsspec.filesystem("reference", fo={'refs': zstore, 'version': 1},
                                 remote_protocol='s3', remote_options={'anon': True})
            mapper = fs.get_mapper("")

            # Open with xarray datatree
            dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

            log_message(f"✅ {pf.name}: Valid, contains {len(dt.keys())} variables")

        except Exception as e:
            log_message(f"❌ {pf.name}: Validation failed - {e}", "ERROR")


if __name__ == "__main__":
    # Example usage
    TARGET_DATE = '20250628'
    TARGET_RUN = '18'

    # Process subset for testing (control + first 5 members)
    TEST_MEMBERS = [-1, 1, 2, 3, 4, 5]

    success = process_ecmwf_ensemble(
        date_str=TARGET_DATE,
        run=TARGET_RUN,
        members_to_process=TEST_MEMBERS,
        batch_size=3
    )

    if success:
        # Validate the created files
        output_dir = Path(f"ecmwf_{TARGET_DATE}_{TARGET_RUN}")
        validate_parquet_files(output_dir)

        print("\n🎉 ECMWF ensemble processing completed successfully!")
    else:
        print("\n❌ ECMWF ensemble processing failed")
        exit(1)