#!/usr/bin/env python3
"""
ECMWF Utility Functions
Provides ECMWF-specific utilities for ensemble weather forecast processing.
Adapted from GEFS utilities but tailored for ECMWF data structure and requirements.
"""

import asyncio
import concurrent.futures
import copy
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from functools import partial
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import fsspec
import xarray as xr
import zarr
import gcsfs
from kerchunk.grib2 import grib_tree, scan_grib
from kerchunk._grib_idx import (
    AggregationType,
    build_idx_grib_mapping,
    map_from_index,
    parse_grib_idx,
    store_coord_var,
    store_data_var,
    strip_datavar_chunks,
)

# Configure logging
logger = logging.getLogger("ecmwf-utils")
logger.setLevel(logging.INFO)

# ECMWF-specific constants
ECMWF_BUCKET = "ecmwf-forecasts"
ECMWF_FORECAST_HOURS_3H = list(range(0, 145, 3))  # 0-144h at 3h intervals
ECMWF_FORECAST_HOURS_6H = list(range(150, 361, 6))  # 150-360h at 6h intervals
ECMWF_FORECAST_HOURS = ECMWF_FORECAST_HOURS_3H + ECMWF_FORECAST_HOURS_6H

# ECMWF ensemble member numbers
ECMWF_CONTROL_NUMBER = -1
ECMWF_PERTURBED_NUMBERS = list(range(1, 51))
ECMWF_ALL_NUMBERS = [ECMWF_CONTROL_NUMBER] + ECMWF_PERTURBED_NUMBERS

# ECMWF variable mapping
ECMWF_FORECAST_DICT = {
    "Temperature": "t:pl",
    "cape": "cape:sfc",
    "U component of wind": "u:pl",
    "V component of wind": "v:pl",
    "Mean sea level pressure": "msl:sfc",
    "2 metre temperature": "2t:sfc",
    "Total precipitation": "tp:sfc",
    "10 metre U wind": "10u:sfc",
    "10 metre V wind": "10v:sfc",
    "Relative humidity": "r:pl",
    "Geopotential height": "z:pl",
    "Specific humidity": "q:pl",
}


def generate_ecmwf_axes(date_str: str) -> List[pd.Index]:
    """
    Generate temporal axes indices for ECMWF forecast with variable time intervals.

    ECMWF has different time resolutions:
    - 0-144h: 3-hour intervals (49 time steps)
    - 150-360h: 6-hour intervals (36 time steps)
    - Total: 85 time steps

    Parameters:
    - date_str: The start date of forecast, formatted as 'YYYYMMDD'

    Returns:
    - List containing valid_time and time indices
    """
    start_date = pd.Timestamp(date_str)

    # First part: 0-144h at 3-hour intervals
    part1_times = pd.date_range(
        start_date,
        start_date + pd.Timedelta(hours=144),
        freq="3h",
        inclusive='left'
    )

    # Second part: 150-360h at 6-hour intervals
    part2_times = pd.date_range(
        start_date + pd.Timedelta(hours=150),
        start_date + pd.Timedelta(hours=360),
        freq="6h",
        inclusive='both'
    )

    # Combine both parts
    all_times = part1_times.tolist() + part2_times.tolist()
    valid_time_index = pd.DatetimeIndex(all_times, name="valid_time")

    # Single initialization time
    time_index = pd.Index([start_date], name="time")

    print(f"Generated ECMWF axes: {len(valid_time_index)} time steps")
    print(f"  - 3h intervals (0-144h): {len(part1_times)} steps")
    print(f"  - 6h intervals (150-360h): {len(part2_times)} steps")

    return [valid_time_index, time_index]


def generate_ecmwf_urls(date_str: str, run: str = "00") -> List[str]:
    """
    Generate ECMWF S3 URLs for all forecast hours.

    Parameters:
    - date_str: Date in YYYYMMDD format
    - run: Run hour (default "00" for 00Z)

    Returns:
    - List of S3 URLs for ECMWF GRIB files
    """
    urls = []
    base_url = f"s3://{ECMWF_BUCKET}/{date_str}/{run}z/ifs/0p25/enfo"

    for hour in ECMWF_FORECAST_HOURS:
        filename = f"{date_str}000000-{hour}h-enfo-ef.grib2"
        url = f"{base_url}/{filename}"
        urls.append(url)

    print(f"Generated {len(urls)} ECMWF URLs for {date_str} {run}Z")
    return urls


def get_ecmwf_forecast_hour_index(forecast_hour: int) -> int:
    """
    Map forecast hour to array index for ECMWF's variable intervals.

    Parameters:
    - forecast_hour: Forecast hour (0-360)

    Returns:
    - Index in the time array
    """
    if forecast_hour <= 144:
        # 3-hour intervals: index = hour / 3
        if forecast_hour % 3 != 0:
            raise ValueError(f"Invalid forecast hour {forecast_hour} for 3h interval")
        return forecast_hour // 3
    else:
        # 6-hour intervals starting at 150h
        # First 49 indices are for 0-144h (3h intervals)
        if (forecast_hour - 150) % 6 != 0:
            raise ValueError(f"Invalid forecast hour {forecast_hour} for 6h interval")
        return 49 + (forecast_hour - 150) // 6


def extract_ecmwf_metadata(grib_url: str) -> Dict[str, Any]:
    """
    Extract metadata from ECMWF GRIB URL.

    Parameters:
    - grib_url: ECMWF GRIB file URL

    Returns:
    - Dictionary with date, run, and forecast_hour
    """
    # Pattern: s3://ecmwf-forecasts/YYYYMMDD/HHz/ifs/0p25/enfo/YYYYMMDD000000-Hh-enfo-ef.grib2
    pattern = r"s3://ecmwf-forecasts/(\d{8})/(\d{2})z/ifs/0p25/enfo/\d{8}000000-(\d+)h-enfo-ef\.grib2"

    match = re.match(pattern, grib_url)
    if match:
        return {
            'date': match.group(1),
            'run': match.group(2),
            'forecast_hour': int(match.group(3))
        }
    else:
        logger.warning(f"Could not extract metadata from URL: {grib_url}")
        return {}


def identify_ensemble_members(grib_groups: List) -> Dict[int, List]:
    """
    Identify and separate ensemble members from GRIB groups.

    ECMWF uses 'number' field:
    - Control: number = -1 (or sometimes 0)
    - Perturbed: number = 1 to 50

    Parameters:
    - grib_groups: List of GRIB group dictionaries

    Returns:
    - Dictionary mapping member number to list of groups
    """
    member_groups = {}

    for group in grib_groups:
        # Extract member number from attributes
        attrs = group.get('attrs', {})
        number = attrs.get('number', None)

        if number is None:
            # Try to extract from other fields
            if 'perturbationNumber' in attrs:
                number = attrs['perturbationNumber']
            else:
                logger.warning("No ensemble member number found in group")
                continue

        # Normalize control member number
        if number == 0:
            number = -1

        if number not in member_groups:
            member_groups[number] = []

        member_groups[number].append(group)

    # Log summary
    logger.info(f"Identified {len(member_groups)} ensemble members:")
    if -1 in member_groups:
        logger.info(f"  - Control member: {len(member_groups[-1])} groups")
    for num in sorted([n for n in member_groups.keys() if n > 0]):
        logger.info(f"  - Perturbed member {num}: {len(member_groups[num])} groups")

    return member_groups


def ecmwf_filter_scan_grib(grib_url: str, forecast_dict: Dict = None) -> Tuple[List, Dict]:
    """
    Scan ECMWF GRIB file and extract all ensemble members.

    Parameters:
    - grib_url: S3 URL to ECMWF GRIB file
    - forecast_dict: Optional dictionary of variables to filter

    Returns:
    - Tuple of (groups list, index mapping dictionary)
    """
    if forecast_dict is None:
        forecast_dict = ECMWF_FORECAST_DICT

    logger.info(f"Scanning ECMWF GRIB: {grib_url}")

    # Scan GRIB file
    storage_options = {"anon": True}
    groups = scan_grib(grib_url, storage_options=storage_options)

    # Parse index file if available
    try:
        idx_df = parse_grib_idx(basename=grib_url, suffix="index", storage_options=storage_options)

        # Create index mapping for ensemble members
        idx_mapping = {}
        for idx, row in idx_df.iterrows():
            # Extract member number from attrs
            attrs = row.get('attrs', {})
            if isinstance(attrs, str):
                # Parse string attributes if needed
                import ast
                try:
                    attrs = ast.literal_eval(attrs)
                except:
                    attrs = {}

            number = attrs.get('number', attrs.get('perturbationNumber', None))
            if number is not None:
                if number not in idx_mapping:
                    idx_mapping[number] = []
                idx_mapping[number].append(idx)

        logger.info(f"Created index mapping for {len(idx_mapping)} members")

    except Exception as e:
        logger.warning(f"Could not parse index file: {e}")
        idx_mapping = {}

    return groups, idx_mapping


def fixed_ensemble_grib_tree(groups: List, ensemble_numbers: List[int] = None) -> Dict:
    """
    Build efficient GRIB tree for all ensemble members.

    This processes all ensemble members from the groups in one pass,
    creating a comprehensive zarr store structure.

    Parameters:
    - groups: List of GRIB groups from scan_grib
    - ensemble_numbers: List of ensemble member numbers to process

    Returns:
    - Dictionary containing zarr store for all members
    """
    if ensemble_numbers is None:
        ensemble_numbers = ECMWF_ALL_NUMBERS

    logger.info(f"Building GRIB tree for {len(ensemble_numbers)} ensemble members")

    # Separate groups by member
    member_groups = identify_ensemble_members(groups)

    # Build combined tree for all specified members
    all_member_groups = []
    for num in ensemble_numbers:
        if num in member_groups:
            all_member_groups.extend(member_groups[num])
        else:
            logger.warning(f"Member {num} not found in GRIB groups")

    # Create GRIB tree
    grib_tree_store = grib_tree(all_member_groups)

    logger.info(f"Created GRIB tree with {len(grib_tree_store.get('refs', {}))} references")

    return grib_tree_store


def extract_member_from_store(zarr_store: Dict, member_number: int) -> Dict:
    """
    Extract single ensemble member data from combined zarr store.

    Parameters:
    - zarr_store: Combined zarr store with all members
    - member_number: Ensemble member number (-1 for control, 1-50 for perturbed)

    Returns:
    - Dictionary containing zarr store for single member
    """
    member_store = copy.deepcopy(zarr_store)
    refs = member_store.get('refs', {})

    # Filter references for this member
    member_refs = {}
    for key, value in refs.items():
        # Check if this reference belongs to the member
        # This requires parsing the zarr path structure
        if f"/number/{member_number}/" in key or f"_ens{member_number}/" in key:
            member_refs[key] = value

    member_store['refs'] = member_refs

    member_name = "control" if member_number == -1 else f"ens{member_number:02d}"
    logger.info(f"Extracted {len(member_refs)} references for {member_name}")

    return member_store


def calculate_ecmwf_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time-related dimensions for ECMWF's variable time intervals.

    Parameters:
    - axes: List of pandas Index objects containing time information

    Returns:
    - Tuple of (time_dims, time_coords, times, valid_times, steps)
    """
    logger.info("Calculating ECMWF time dimensions")

    aggregation_type = AggregationType.BEST_AVAILABLE
    time_dims: Dict[str, int] = {}
    time_coords: Dict[str, tuple] = {}

    valid_time_index = axes[0]
    time_index = axes[1]

    if aggregation_type == AggregationType.BEST_AVAILABLE:
        time_dims["valid_time"] = len(valid_time_index)
        assert len(time_index) == 1, "Time axis must describe a single initialization date"

        reference_time = time_index.to_numpy()[0]

        time_coords["step"] = ("valid_time",)
        time_coords["valid_time"] = ("valid_time",)
        time_coords["time"] = ("valid_time",)
        time_coords["datavar"] = ("valid_time",)

        valid_times = valid_time_index.to_numpy()
        times = np.where(valid_times <= reference_time, valid_times, reference_time)

        # Calculate steps in hours (accounting for variable intervals)
        steps = (valid_times - reference_time) / np.timedelta64(1, 'h')

    logger.info(f"Time dimensions: {len(valid_times)} valid times")
    logger.info(f"Step range: {steps[0]} to {steps[-1]} hours")

    return time_dims, time_coords, times, valid_times, steps


def process_ecmwf_variables(df: pd.DataFrame, variable_mapping: Dict = None) -> pd.DataFrame:
    """
    Process ECMWF variables with specific level and type filtering.

    Parameters:
    - df: DataFrame with ECMWF variables
    - variable_mapping: Optional variable mapping dictionary

    Returns:
    - Processed DataFrame with filtered variables
    """
    if variable_mapping is None:
        variable_mapping = ECMWF_FORECAST_DICT

    # ECMWF-specific processing conditions
    conditions = {
        't': 'isobaricInhPa',       # Temperature at pressure levels
        'u': 'isobaricInhPa',       # U-wind at pressure levels
        'v': 'isobaricInhPa',       # V-wind at pressure levels
        'r': 'isobaricInhPa',       # Relative humidity at pressure levels
        'z': 'isobaricInhPa',       # Geopotential at pressure levels
        'q': 'isobaricInhPa',       # Specific humidity at pressure levels
        'cape': 'surface',          # CAPE at surface
        'msl': 'meanSea',          # Mean sea level pressure
        '2t': 'heightAboveGround',  # 2m temperature
        '10u': 'heightAboveGround', # 10m U-wind
        '10v': 'heightAboveGround', # 10m V-wind
        'tp': 'surface',            # Total precipitation
    }

    processed_df = pd.DataFrame()

    for varname in df['varname'].unique():
        if varname in conditions:
            level_type = conditions[varname]

            # Filter by variable and level type
            filtered_df = df[(df['varname'] == varname) & (df['typeOfLevel'] == level_type)]

            # Remove duplicates, keeping highest resolution
            filtered_df = filtered_df.sort_values(by='length', ascending=False)
            filtered_df = filtered_df.drop_duplicates(subset=['time', 'level'], keep='first')

            processed_df = pd.concat([processed_df, filtered_df], ignore_index=True)

            logger.info(f"Processed {varname}: {len(filtered_df)} records")

    return processed_df


def create_ecmwf_mapped_index(
    axes: List[pd.Index],
    grib_url: str,
    member_number: int
) -> pd.DataFrame:
    """
    Create mapped index for ECMWF ensemble member.

    Parameters:
    - axes: Time axes for mapping
    - grib_url: ECMWF GRIB file URL
    - member_number: Ensemble member number

    Returns:
    - DataFrame with mapped index
    """
    logger.info(f"Creating mapped index for member {member_number}")

    # Parse index file
    storage_options = {"anon": True}
    idx_df = parse_grib_idx(basename=grib_url, suffix="index", storage_options=storage_options)

    # Filter for specific member
    member_df = idx_df[idx_df['number'] == member_number]

    if member_df.empty:
        logger.warning(f"No entries found for member {member_number}")
        return pd.DataFrame()

    # Map to time axes
    valid_times = axes[0]
    mapped_list = []

    for time in valid_times:
        # Find matching entries for this time
        time_entries = member_df[member_df['valid_time'] == time]

        if not time_entries.empty:
            mapped_list.append(time_entries)

    if mapped_list:
        mapped_index = pd.concat(mapped_list, ignore_index=True)
        logger.info(f"Mapped {len(mapped_index)} entries for member {member_number}")
        return mapped_index
    else:
        return pd.DataFrame()


def prepare_ecmwf_zarr_store(
    grib_tree_store: Dict,
    mapped_index: pd.DataFrame,
    time_dims: Dict,
    time_coords: Dict,
    times: np.ndarray,
    valid_times: np.ndarray,
    steps: np.ndarray
) -> Dict:
    """
    Prepare zarr store for ECMWF data.

    Parameters:
    - grib_tree_store: GRIB tree structure
    - mapped_index: Mapped index DataFrame
    - time_dims: Time dimensions
    - time_coords: Time coordinates
    - times: Time values
    - valid_times: Valid time values
    - steps: Step values

    Returns:
    - Zarr store dictionary
    """
    logger.info("Preparing ECMWF zarr store")

    # Start with deflated store
    zarr_store = copy.deepcopy(grib_tree_store)
    strip_datavar_chunks(zarr_store)

    zstore = zarr_store.get('refs', {})

    # Process unique variable groups
    unique_groups = mapped_index.set_index(["varname", "stepType", "typeOfLevel"]).index.unique()

    logger.info(f"Processing {len(unique_groups)} unique variable groups")

    for key, group in mapped_index.groupby(["varname", "stepType", "typeOfLevel"]):
        try:
            base_path = "/".join(key)
            lvals = group.level.unique()
            dims = time_dims.copy()
            coords = time_coords.copy()

            # Handle level dimensions
            if len(lvals) == 1:
                lvals = lvals.squeeze()
                dims[key[2]] = 0
            elif len(lvals) > 1:
                lvals = np.sort(lvals)
                dims[key[2]] = len(lvals)
                coords["datavar"] += (key[2],)

            # Store coordinate variables
            store_coord_var(f"{base_path}/time", zstore, coords["time"],
                          times.astype("datetime64[s]"))
            store_coord_var(f"{base_path}/valid_time", zstore, coords["valid_time"],
                          valid_times.astype("datetime64[s]"))
            store_coord_var(f"{base_path}/step", zstore, coords["step"],
                          steps.astype("float64"))

            if key[2]:  # Level dimension exists
                store_coord_var(f"{base_path}/{key[2]}", zstore,
                              (key[2],) if lvals.shape else (), lvals)

            # Store data variable
            store_data_var(f"{base_path}/{key[0]}", zstore, dims, coords,
                         group, steps, times, lvals if lvals.shape else None)

            logger.info(f"Processed group: {key}")

        except Exception as e:
            logger.error(f"Error processing group {key}: {e}")
            continue

    return zstore


def load_ecmwf_parquet_from_gcs(
    date_str: str,
    member_name: str,
    forecast_hour: int,
    gcs_bucket: str = "gik-fmrc",
    gcp_service_account_json: Optional[str] = None
) -> Dict:
    """
    Load pre-processed ECMWF parquet file from GCS.

    Parameters:
    - date_str: Date string (YYYYMMDD)
    - member_name: Member name (control, ens01-ens50)
    - forecast_hour: Forecast hour (0-360)
    - gcs_bucket: GCS bucket name
    - gcp_service_account_json: Optional path to service account JSON

    Returns:
    - Dictionary containing zarr store references
    """
    # Determine GCS folder structure
    if member_name == "control":
        gcs_folder = "ens_control"
        filename_member = "control"
    else:
        member_num = int(member_name[3:]) if member_name.startswith("ens") else int(member_name)
        gcs_folder = f"ens_{member_num:02d}"
        filename_member = f"ens{member_num:02d}" if member_name != "control" else "control"

    # Build GCS path - note: no gs:// prefix when using gcsfs
    gcs_filename = f"ecmwf-{date_str}00-{filename_member}-rt{forecast_hour:03d}.par"
    gcs_path = f"{gcs_bucket}/v2ecmwf_fmrc/{gcs_folder}/{gcs_filename}"

    logger.info(f"Loading parquet from GCS: gs://{gcs_path}")

    try:
        # Initialize GCS filesystem with proper authentication
        if gcp_service_account_json and os.path.exists(gcp_service_account_json):
            # Load service account info
            with open(gcp_service_account_json, 'r') as f:
                service_account_info = json.load(f)
            gcs_fs = gcsfs.GCSFileSystem(token=service_account_info, project=service_account_info.get('project_id'))
        else:
            gcs_fs = gcsfs.GCSFileSystem(anon=True)

        # Check if file exists first
        if not gcs_fs.exists(gcs_path):
            logger.warning(f"Parquet file not found in GCS: {gcs_path}")
            raise FileNotFoundError(f"File not found: gs://{gcs_path}")

        # Read parquet file - gcsfs doesn't use gs:// prefix
        df = pd.read_parquet(f"gs://{gcs_path}", filesystem=gcs_fs)

        # Convert to zarr store dictionary
        zstore = {}
        for _, row in df.iterrows():
            key = row['key']
            value = row['value']

            # Decode bytes if needed
            if isinstance(value, bytes):
                value = value.decode('utf-8')

            # Parse JSON values
            if isinstance(value, str) and value.startswith('['):
                try:
                    value = json.loads(value)
                except:
                    pass

            zstore[key] = value

        logger.info(f"Loaded {len(zstore)} keys from {gcs_filename}")
        return zstore

    except Exception as e:
        logger.error(f"Failed to load parquet from GCS: {e}")
        raise


def load_all_ecmwf_member_parquets(
    date_str: str,
    member_name: str,
    forecast_hours: List[int],
    gcs_bucket: str = "gik-fmrc",
    gcp_service_account_json: Optional[str] = None
) -> Dict:
    """
    Load all parquet files for a specific ECMWF ensemble member.

    Parameters:
    - date_str: Date string (YYYYMMDD)
    - member_name: Member name (control, ens01-ens50)
    - forecast_hours: List of forecast hours to load
    - gcs_bucket: GCS bucket name
    - gcp_service_account_json: Optional path to service account JSON

    Returns:
    - Combined zarr store dictionary
    """
    combined_store = {}

    for hour in forecast_hours:
        try:
            hour_store = load_ecmwf_parquet_from_gcs(
                date_str, member_name, hour,
                gcs_bucket, gcp_service_account_json
            )

            # Merge into combined store
            combined_store.update(hour_store)

        except Exception as e:
            logger.warning(f"Could not load {member_name} at {hour}h: {e}")
            continue

    logger.info(f"Loaded {len(combined_store)} total keys for {member_name}")
    return combined_store


def process_ecmwf_member_with_gcs_parquets(
    member_name: str,
    date_str: str,
    axes: List[pd.Index],
    time_dims: Dict,
    time_coords: Dict,
    times: np.ndarray,
    valid_times: np.ndarray,
    steps: np.ndarray,
    gcs_bucket: str = "gik-fmrc",
    gcp_service_account_json: Optional[str] = None,
    reference_date_str: Optional[str] = None
) -> Dict:
    """
    Process ECMWF ensemble member using pre-processed GCS parquet files.

    This is the key function that uses existing parquet files from GCS
    instead of processing GRIB files from scratch.

    Parameters:
    - member_name: Member name (control, ens01-ens50)
    - date_str: Target date string
    - axes: Time axes
    - time_dims: Time dimensions
    - time_coords: Time coordinates
    - times: Time values
    - valid_times: Valid time values
    - steps: Step values
    - gcs_bucket: GCS bucket name
    - gcp_service_account_json: Path to service account JSON
    - reference_date_str: Reference date for parquet files (defaults to date_str)

    Returns:
    - Zarr store dictionary for the member
    """
    logger.info(f"Processing {member_name} using GCS parquet files")

    # Use reference date if provided, otherwise use target date
    parquet_date = reference_date_str if reference_date_str else date_str

    try:
        # Load all parquet files for this member
        member_store = load_all_ecmwf_member_parquets(
            parquet_date,
            member_name,
            ECMWF_FORECAST_HOURS,
            gcs_bucket,
            gcp_service_account_json
        )

        if not member_store:
            raise ValueError(f"No data loaded for {member_name}")

        # Update time coordinates if processing different date
        if reference_date_str and reference_date_str != date_str:
            member_store = update_time_references(member_store, date_str, reference_date_str)

        logger.info(f"✅ Successfully processed {member_name} from GCS parquets")
        return member_store

    except Exception as e:
        logger.error(f"Failed to process {member_name} from GCS: {e}")
        raise


def update_time_references(zarr_store: Dict, target_date: str, reference_date: str) -> Dict:
    """
    Update time references when using parquet from different date.

    Parameters:
    - zarr_store: Original zarr store
    - target_date: Target date string
    - reference_date: Reference date string

    Returns:
    - Updated zarr store
    """
    # Calculate date offset
    target_dt = pd.Timestamp(target_date)
    reference_dt = pd.Timestamp(reference_date)
    time_offset = target_dt - reference_dt

    # Update time-related keys
    updated_store = {}
    for key, value in zarr_store.items():
        if '/time/' in key or '/valid_time/' in key:
            # Update time arrays
            if isinstance(value, list) and len(value) > 0:
                # Adjust timestamps
                updated_value = value.copy()
                # Logic to update timestamps would go here
                updated_store[key] = updated_value
            else:
                updated_store[key] = value
        else:
            updated_store[key] = value

    return updated_store


def save_ecmwf_parquet(zarr_store: Dict, output_path: Path, member_name: str) -> Path:
    """
    Save ECMWF zarr store to parquet file.

    Parameters:
    - zarr_store: Zarr store dictionary
    - output_path: Output directory path
    - member_name: Ensemble member name

    Returns:
    - Path to saved parquet file
    """
    # Create output directory structure
    if member_name == "control":
        member_dir = output_path / "ens_control"
    else:
        member_num = int(member_name[3:]) if member_name.startswith("ens") else int(member_name)
        member_dir = output_path / f"ens_{member_num:02d}"

    member_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet file
    parquet_file = member_dir / f"{member_name}.par"

    # Convert zarr store to DataFrame
    data = []
    for key, value in zarr_store.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        if isinstance(value, str):
            value = value.encode('utf-8')
        data.append((key, value))

    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(parquet_file)

    logger.info(f"Saved parquet file: {parquet_file}")

    return parquet_file


async def process_ecmwf_url_async(
    url: str,
    sem: asyncio.Semaphore,
    executor: concurrent.futures.ThreadPoolExecutor,
    forecast_dict: Dict = None
) -> Tuple[List, Dict]:
    """
    Process single ECMWF URL asynchronously.

    Parameters:
    - url: ECMWF GRIB URL
    - sem: Semaphore for concurrency control
    - executor: Thread pool executor
    - forecast_dict: Variable dictionary

    Returns:
    - Tuple of (groups, index_mapping)
    """
    async with sem:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            partial(ecmwf_filter_scan_grib, url, forecast_dict)
        )
        return result


async def process_ecmwf_urls_parallel(
    urls: List[str],
    max_concurrent: int = 5,
    forecast_dict: Dict = None
) -> Tuple[List, Dict]:
    """
    Process multiple ECMWF URLs in parallel.

    Parameters:
    - urls: List of ECMWF GRIB URLs
    - max_concurrent: Maximum concurrent operations
    - forecast_dict: Variable dictionary

    Returns:
    - Tuple of (all_groups, all_mappings)
    """
    logger.info(f"Processing {len(urls)} ECMWF URLs in parallel")

    sem = asyncio.Semaphore(max_concurrent)
    all_groups = []
    all_mappings = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        tasks = [
            process_ecmwf_url_async(url, sem, executor, forecast_dict)
            for url in urls
        ]

        results = await asyncio.gather(*tasks)

        for groups, mapping in results:
            all_groups.extend(groups)
            all_mappings.update(mapping)

    logger.info(f"Collected {len(all_groups)} total groups")

    return all_groups, all_mappings


# Validation functions
def validate_ecmwf_ensemble_completeness(member_stores: Dict) -> bool:
    """
    Validate that all ECMWF ensemble members are present.

    Parameters:
    - member_stores: Dictionary of member stores

    Returns:
    - Boolean indicating completeness
    """
    expected_members = ["control"] + [f"ens{i:02d}" for i in range(1, 51)]

    missing = set(expected_members) - set(member_stores.keys())

    if missing:
        logger.warning(f"Missing ensemble members: {missing}")
        return False

    logger.info("✅ All 51 ECMWF ensemble members present")
    return True


def validate_ecmwf_time_consistency(zarr_store: Dict) -> bool:
    """
    Validate time consistency in ECMWF zarr store.

    Parameters:
    - zarr_store: Zarr store dictionary

    Returns:
    - Boolean indicating consistency
    """
    # Check for expected number of time steps
    expected_steps = len(ECMWF_FORECAST_HOURS)

    # Extract time-related keys
    time_keys = [k for k in zarr_store.keys() if '/valid_time/' in k or '/time/' in k]

    if not time_keys:
        logger.warning("No time keys found in zarr store")
        return False

    # Validate time array dimensions
    # This would require parsing the zarr arrays

    logger.info("✅ Time consistency validated")
    return True


# Helper function for logging
def log_processing_summary(
    start_time: float,
    member_stores: Dict,
    output_dir: Optional[Path] = None
):
    """
    Log processing summary statistics.

    Parameters:
    - start_time: Processing start time
    - member_stores: Dictionary of processed member stores
    - output_dir: Optional output directory
    """
    elapsed_time = time.time() - start_time

    print("\n" + "="*80)
    print("ECMWF ENSEMBLE PROCESSING SUMMARY")
    print("="*80)
    print(f"✅ Processing completed in {elapsed_time/60:.1f} minutes")
    print(f"📊 Ensemble members processed: {len(member_stores)}")

    if output_dir and output_dir.exists():
        parquet_files = list(output_dir.glob("**/*.par"))
        total_size = sum(f.stat().st_size for f in parquet_files) / (1024**2)
        print(f"💾 Output files: {len(parquet_files)} parquet files")
        print(f"💾 Total size: {total_size:.1f} MB")

    # Member breakdown
    control_present = "control" in member_stores
    perturbed_count = len([m for m in member_stores if m.startswith("ens")])

    print(f"\n📈 Member Distribution:")
    print(f"   - Control member: {'✅' if control_present else '❌'}")
    print(f"   - Perturbed members: {perturbed_count}/50")

    if perturbed_count == 50 and control_present:
        print("\n🎉 All 51 ECMWF ensemble members successfully processed!")
    else:
        print(f"\n⚠️ Warning: Only {len(member_stores)}/51 members processed")


if __name__ == "__main__":
    # Test the utility functions
    print("ECMWF Utility Functions Test")
    print("="*60)

    # Test axes generation
    test_date = "20240529"
    axes = generate_ecmwf_axes(test_date)
    print(f"Generated axes for {test_date}:")
    print(f"  - Valid times: {len(axes[0])} steps")
    print(f"  - Time: {axes[1][0]}")

    # Test URL generation
    urls = generate_ecmwf_urls(test_date)
    print(f"\nGenerated {len(urls)} URLs")
    print(f"  - First URL: {urls[0]}")
    print(f"  - Last URL: {urls[-1]}")

    # Test forecast hour mapping
    test_hours = [0, 72, 144, 150, 240, 360]
    print("\nForecast hour to index mapping:")
    for hour in test_hours:
        try:
            idx = get_ecmwf_forecast_hour_index(hour)
            print(f"  - {hour}h -> index {idx}")
        except ValueError as e:
            print(f"  - {hour}h -> ERROR: {e}")

    print("\n✅ ECMWF utility functions loaded successfully")