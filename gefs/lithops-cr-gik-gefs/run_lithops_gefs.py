#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lithops",
#     "google-cloud-storage",
#     "google-cloud-pubsub",
#     "google-api-python-client",
#     "google-auth",
#     "httplib2",
#     "pandas",
#     "numpy",
#     "pyarrow",
#     "fsspec",
#     "s3fs",
#     "gcsfs",
#     "requests",
#     "kerchunk",
#     "xarray",
#     "cfgrib",
#     "eccodes",
#     "h5py",
#     "dask",
#     "distributed",
#     "zarr",
# ]
# ///
"""
Lithops-native GEFS Parquet Generation (Cloud Run) — Self-Contained
====================================================================

Processes NOAA GEFS weather data through the GIK three-stage pipeline using
Lithops to directly execute Python functions on Cloud Run workers.

This is the **self-contained** version: all functions previously imported from
gefs_util.py are inlined below, so cloudpickle serialization works without
baking gefs_util.py into the Docker image.

Architecture:
    run_lithops_gefs.py (this script, runs locally via uv run)
         |
         v
    [Lithops FunctionExecutor(backend='gcp_cloudrun')]
      /    |    \\
     /     |     \\  (cloudpickle-serialized functions via GCS)
    v      v      v
  [CR 1]  [CR 2]  ... [CR N]  (Lithops proxy containers on Cloud Run)
    |      |           |
    v      v           v
  process_gefs_date() runs directly inside each container:
    Stage 1: Load deflated store from template parquet (~1s)
    Stage 2: Read .idx files from S3 + merge with mapping parquets (~3-5 min)
    Stage 3: Build zarr store + create final parquets (~1 min)
    Upload to GCS

Usage:
    # Build custom runtime (one-time setup)
    gcloud builds submit \\
      --config=cloudbuild.yaml \\
      --project=e4drr-crafd \\
      --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com

    # Single date
    uv run run_lithops_gefs.py --date 20250106

    # Last 7 days
    uv run run_lithops_gefs.py --days-back 7

    # Date range
    uv run run_lithops_gefs.py --start-date 20250101 --end-date 20250107

    # Custom parallelism
    uv run run_lithops_gefs.py --days-back 30 --max-workers 20

    # Dry run (show dates without executing)
    uv run run_lithops_gefs.py --days-back 7 --dry-run

    # Sequential (no Lithops, for local testing)
    uv run run_lithops_gefs.py --date 20250106 --sequential

    # Limit members (for testing)
    uv run run_lithops_gefs.py --date 20250106 --sequential --max-members 3

Configuration:
    Uses lithops_config.yaml in current directory, or set LITHOPS_CONFIG_FILE.

Author: ICPAC GIK Team
"""

import copy
import io
import os
import sys
import json
import time
import argparse
import shutil
import logging
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

from kerchunk._grib_idx import (
    AggregationType,
    map_from_index,
    parse_grib_idx,
    store_coord_var,
    store_data_var,
    strip_datavar_chunks,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION (serialized to Cloud Run workers via cloudpickle)
# ==============================================================================

GCS_BUCKET = os.environ.get('GCS_BUCKET', 'gik-gefs-aws-tf')
GCS_PARQUET_PREFIX = os.environ.get('GCS_PARQUET_PREFIX', 'run_par_gefs')

TEMPLATE_TAR_GZ_URL = (
    "https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main/"
    "gik-fmrc-gefs-20241112.tar.gz"
)
TEMPLATE_PARQUET_URL = (
    "https://huggingface.co/datasets/E4DRR/grib-index-kerchunk-templates/resolve/main/"
    "gefs-deflated-store-template-20241112.parquet"
)

# Use baked-in templates from Docker image if available, fallback to /tmp
TEMPLATE_TAR_GZ_PATH = os.environ.get(
    'GEFS_TEMPLATE_TAR_GZ',
    '/tmp/gik-fmrc-gefs-20241112.tar.gz'
)
TEMPLATE_PARQUET_PATH = os.environ.get(
    'GEFS_TEMPLATE_PARQUET',
    '/tmp/gefs-deflated-store-template-20241112.parquet'
)

# GEFS ensemble members
ALL_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]  # gep01-gep30


# ==============================================================================
# INLINED FUNCTIONS FROM gefs_util.py
# ==============================================================================
# The following classes and functions were previously imported from gefs_util.py.
# They are inlined here so that the script is fully self-contained and can be
# serialized by cloudpickle without requiring gefs_util.py in the Docker image.
# ==============================================================================


class LocalTarGzMappingManager:
    """
    Manages reading parquet mapping files from a local tar.gz archive.
    This provides an alternative to GCS bucket access for environments without service account credentials.
    """

    def __init__(self, tar_gz_path: str, extract_dir: Optional[str] = None):
        """
        Initialize the local tar.gz mapping manager.

        Args:
            tar_gz_path: Path to the tar.gz file containing parquet mappings
            extract_dir: Directory to extract files to (default: temp directory)
        """
        self.tar_gz_path = tar_gz_path
        self._extracted = False
        self._extract_dir = extract_dir
        self._temp_dir = None
        self._file_index = {}

        if not os.path.exists(tar_gz_path):
            raise FileNotFoundError(f"Tar.gz file not found: {tar_gz_path}")

        # Build index of files in the archive
        self._build_file_index()

    def _build_file_index(self):
        """Build an index of parquet files in the archive."""
        with tarfile.open(self.tar_gz_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.parquet'):
                    # Extract ensemble member and forecast hour from filename
                    # Pattern: gik-fmrc/gefs/{member}/gefs-time-{date}-{member}-rt{hour}.parquet
                    basename = os.path.basename(member.name)
                    parts = basename.replace('.parquet', '').split('-')
                    if len(parts) >= 4:
                        # Extract member (e.g., gep01) and hour (e.g., rt003 -> 003)
                        ensemble_member = parts[3]  # gep01, gep02, etc.
                        forecast_hour_str = parts[-1]  # rt000, rt003, etc.
                        if forecast_hour_str.startswith('rt'):
                            forecast_hour = int(forecast_hour_str[2:])
                            key = (ensemble_member, forecast_hour)
                            self._file_index[key] = member.name

        print(f"Indexed {len(self._file_index)} parquet files from {self.tar_gz_path}")

    def extract_all(self, extract_dir: Optional[str] = None) -> str:
        """
        Extract all files from the archive.

        Args:
            extract_dir: Directory to extract to (default: creates temp directory)

        Returns:
            Path to the extraction directory
        """
        if self._extracted and self._extract_dir:
            return self._extract_dir

        if extract_dir:
            self._extract_dir = extract_dir
        elif not self._extract_dir:
            self._temp_dir = tempfile.mkdtemp(prefix='gefs_mappings_')
            self._extract_dir = self._temp_dir

        os.makedirs(self._extract_dir, exist_ok=True)

        print(f"Extracting tar.gz to {self._extract_dir}...")
        with tarfile.open(self.tar_gz_path, 'r:gz') as tar:
            tar.extractall(self._extract_dir)

        self._extracted = True
        print(f"Extraction complete.")
        return self._extract_dir

    def get_mapping_path(self, ensemble_member: str, forecast_hour: int) -> Optional[str]:
        """
        Get the path to a specific mapping parquet file.

        Args:
            ensemble_member: Ensemble member identifier (e.g., 'gep01')
            forecast_hour: Forecast hour (e.g., 0, 3, 6, ...)

        Returns:
            Full path to the extracted parquet file, or None if not found
        """
        key = (ensemble_member, forecast_hour)
        if key not in self._file_index:
            return None

        # Ensure files are extracted
        if not self._extracted:
            self.extract_all()

        relative_path = self._file_index[key]
        full_path = os.path.join(self._extract_dir, relative_path)

        if os.path.exists(full_path):
            return full_path
        return None

    def read_mapping(self, ensemble_member: str, forecast_hour: int) -> Optional[pd.DataFrame]:
        """
        Read a mapping parquet file directly from the archive without extracting.

        Args:
            ensemble_member: Ensemble member identifier (e.g., 'gep01')
            forecast_hour: Forecast hour (e.g., 0, 3, 6, ...)

        Returns:
            DataFrame with the mapping data, or None if not found
        """
        key = (ensemble_member, forecast_hour)
        if key not in self._file_index:
            print(f"Warning: No mapping found for {ensemble_member} at forecast hour {forecast_hour}")
            return None

        member_path = self._file_index[key]

        with tarfile.open(self.tar_gz_path, 'r:gz') as tar:
            member = tar.getmember(member_path)
            f = tar.extractfile(member)
            if f is not None:
                return pd.read_parquet(io.BytesIO(f.read()))

        return None

    def list_ensemble_members(self) -> List[str]:
        """List all ensemble members available in the archive."""
        return sorted(list(set(key[0] for key in self._file_index.keys())))

    def list_forecast_hours(self, ensemble_member: str) -> List[int]:
        """List all forecast hours available for a specific ensemble member."""
        return sorted([key[1] for key in self._file_index.keys() if key[0] == ensemble_member])

    def cleanup(self):
        """Clean up temporary extraction directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
            self._extracted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def _load_deflated_parquet_from_tar(tar_path: str) -> pd.DataFrame:
    """Load the deflated store parquet from inside a tar.gz archive."""
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Look for the deflated store template file
        for member in tar.getnames():
            if 'deflated-store-template' in member and member.endswith('.parquet'):
                f = tar.extractfile(member)
                return pd.read_parquet(io.BytesIO(f.read()))
        raise FileNotFoundError(
            f"No deflated store template parquet found in {tar_path}. "
            "Expected a file matching '*deflated-store-template*.parquet'."
        )


def _filter_deflated_refs(refs: dict, filter_vars: Dict[str, str]) -> dict:
    """
    Filter deflated store refs to only include the requested variables.

    Keeps the root .zgroup and all entries under wanted variable prefixes.
    Removes everything for variables not in filter_vars.

    Parameters:
    - refs: Full deflated store refs dict.
    - filter_vars: Dict mapping display names to "VAR:level" strings,
      e.g. {"2 metre temperature": "TMP:2 m above ground"}.

    Returns:
    - dict: Filtered refs containing only the requested variable paths.
    """
    # GRIB .idx abbreviation -> cfgrib short name (zarr path prefix).
    # Built from the GEFS mapping parquets which contain both representations.
    GRIB_TO_CFGRIB = {
        'tmp': 't2m', 'dpt': 'd2m', 'rh': 'r2', 'ugrd': 'u10', 'vgrd': 'v10',
        'apcp': 'tp', 'pres': 'sp', 'hgt': 'gh', 'tsoil': 'st',
        'soilw': 'soilw', 'weasd': 'sdwe', 'snod': 'sde', 'icetk': 'sithick',
        'gust': 'gust', 'vis': 'vis', 'mslet': 'mslet', 'prmsl': 'prmsl',
        'cpofp': 'cpofp', 'cape': 'cape', 'cin': 'cin', 'pwat': 'pwat',
        'tcdc': 'tcc', 'hlcy': 'hlcy', 'tmax': 'tmax', 'tmin': 'tmin',
        'csnow': 'csnow', 'cicep': 'cicep', 'cfrzr': 'cfrzr', 'crain': 'crain',
        'lhtfl': 'avg_slhtf', 'shtfl': 'avg_ishf',
        'dswrf': 'sdswrf', 'dlwrf': 'sdlwrf', 'uswrf': 'suswrf', 'ulwrf': 'sulwrf',
    }

    # Resolve filter_vars to zarr path prefixes
    wanted_prefixes = set()
    for var_level in filter_vars.values():
        abbrev = var_level.split(':')[0].strip().lower()
        cfgrib_name = GRIB_TO_CFGRIB.get(abbrev, abbrev)
        wanted_prefixes.add(cfgrib_name)

    # Keep root .zgroup + all entries under wanted prefixes
    filtered = {}
    for key, value in refs.items():
        prefix = key.split('/')[0]
        if prefix.startswith('.'):
            filtered[key] = value
        elif prefix in wanted_prefixes:
            filtered[key] = value

    return filtered


def build_gefs_deflated_store_from_template(
    template_path: str,
    filter_vars: Optional[Dict[str, str]] = None
) -> dict:
    """
    Load the GEFS deflated store from a pre-built parquet template instead of
    running scan_grib. This replaces the ~30s-per-member scan_grib call with a
    <1s template load. The deflated store is identical across all 30 ensemble
    members, so it only needs to be loaded once.

    The template parquet is a key/value DataFrame with zarr metadata entries
    (.zarray, .zattrs, .zgroup) and base64-encoded coordinate data, generated
    by running scan_grib + grib_tree + strip_datavar_chunks on the reference
    date (20241112) and serializing the result.

    Parameters:
    - template_path: Path to the deflated store parquet template file, or path
      to a tar.gz archive containing 'gefs-deflated-store-template.parquet'.
    - filter_vars: Optional dict of variables to keep (same format as
      tofilter_gefs_var_dict used in filter_build_grib_tree). Keys are display
      names, values are "VAR:level" strings. If None, all variables are kept.

    Returns:
    - dict: Deflated GRIB tree store with 'version' and 'refs' keys,
      compatible with prepare_zarr_store().
    """
    print("Loading GEFS deflated store from template (skipping scan_grib)")
    start = time.time()

    # Load from parquet (may be inside a tar.gz)
    if template_path.endswith('.tar.gz'):
        template_df = _load_deflated_parquet_from_tar(template_path)
    else:
        template_df = pd.read_parquet(template_path)

    # Convert DataFrame to zarr store dict
    refs = {}
    for _, row in template_df.iterrows():
        key = row['key']
        value = row['value']
        if isinstance(value, bytes):
            try:
                decoded = value.decode('utf-8')
            except UnicodeDecodeError:
                refs[key] = value
                continue
            value = decoded
        # Keep values as strings -- kerchunk's store_coord_var/store_data_var
        # expect JSON strings in .zattrs/.zarray, not parsed dicts.
        refs[key] = value

    total_refs = len(refs)

    # Optionally filter to requested variables
    if filter_vars is not None:
        refs = _filter_deflated_refs(refs, filter_vars)

    elapsed = time.time() - start
    print(f"  Template loaded in {elapsed:.2f}s: {total_refs} total refs"
          + (f", filtered to {len(refs)}" if filter_vars else ""))

    return {'version': 1, 'refs': refs}


def generate_axes(date_str: str) -> List[pd.Index]:
    """
    Generate temporal axes indices for a given forecast start date for GEFS (10-day forecast period).

    Parameters:
    - date_str (str): The start date of the forecast, formatted as 'YYYYMMDD'.

    Returns:
    - List[pd.Index]: A list containing two pandas Index objects:
      1. 'valid_time' index with datetime stamps spaced 3 hours apart, covering a 10-day range from the start date.
      2. 'time' index representing the single start date of the forecast as a datetime object.
    """
    start_date = pd.Timestamp(date_str)
    end_date = start_date + pd.Timedelta(days=10)  # GEFS forecast period of 10 days

    valid_time_index = pd.date_range(start_date, end_date, freq="180min", name="valid_time")  # 3-hour intervals
    time_index = pd.Index([start_date], name="time")

    return [valid_time_index, time_index]


def calculate_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time-related dimensions and coordinates based on input axes for GEFS.

    Parameters:
    - axes (List[pd.Index]): List of pandas Index objects containing time information.

    Returns:
    - Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]: Time dimensions, coordinates, times, valid times, and steps.
    """
    print("Calculating GEFS Time Dimensions and Coordinates")
    axes_by_name: Dict[str, pd.Index] = {pdi.name: pdi for pdi in axes}
    aggregation_type = AggregationType.BEST_AVAILABLE
    time_dims: Dict[str, int] = {}
    time_coords: Dict[str, tuple[str, ...]] = {}

    if aggregation_type == AggregationType.BEST_AVAILABLE:
        time_dims["valid_time"] = len(axes_by_name["valid_time"])
        assert len(axes_by_name["time"]) == 1, "The time axes must describe a single 'as of' date for best available"
        reference_time = axes_by_name["time"].to_numpy()[0]

        time_coords["step"] = ("valid_time",)
        time_coords["valid_time"] = ("valid_time",)
        time_coords["time"] = ("valid_time",)
        time_coords["datavar"] = ("valid_time",)

        valid_times = axes_by_name["valid_time"].to_numpy()
        times = np.where(valid_times <= reference_time, valid_times, reference_time)
        steps = valid_times - times

    times = valid_times
    return time_dims, time_coords, times, valid_times, steps


def process_gefs_dataframe(df, varnames_to_process):
    """
    Filter and process the DataFrame by specific variable names for GEFS data.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to process.
    - varnames_to_process (list): List of variable names to filter and process in the DataFrame.

    Returns:
    - pd.DataFrame: Processed DataFrame with duplicates removed based on the 'time' column and sorted by 'length'.
    """
    conditions = {
        'pres': 'surface',
        'dswrf': 'surface',
        'cape': 'surface',
        'uswrf': 'surface',
        'apcp': 'surface',
        'gust': 'surface',
        'tmp': 'heightAboveGround',
        'rh': 'heightAboveGround',
        'ugrd': 'heightAboveGround',
        'vgrd': 'heightAboveGround',
        'pwat': ['entire atmosphere (considered as a single layer)', 'atmosphereSingleLayer'],
        'tcdc': 'atmosphere',
        'hgt': 'cloudCeiling'
    }
    processed_df = pd.DataFrame()

    for varname in varnames_to_process:
        if varname in conditions:
            level = conditions[varname]
            if isinstance(level, list):
                for l in level:
                    filtered_df = df[(df['varname'] == varname) & (df['typeOfLevel'] == l)]
                    if varname == 'pwat':
                        print(f"PWAT: Looking for typeOfLevel='{l}', found {len(filtered_df)} entries")
                    filtered_df = filtered_df.sort_values(by='length', ascending=False).drop_duplicates(subset=['time'], keep='first')
                    processed_df = pd.concat([processed_df, filtered_df], ignore_index=True)
            else:
                filtered_df = df[(df['varname'] == varname) & (df['typeOfLevel'] == level)]
                if varname == 'pwat':
                    print(f"PWAT: Looking for typeOfLevel='{level}', found {len(filtered_df)} entries")
                filtered_df = filtered_df.sort_values(by='length', ascending=False).drop_duplicates(subset=['time'], keep='first')
                processed_df = pd.concat([processed_df, filtered_df], ignore_index=True)
        else:
            if varname == 'pwat':
                print(f"WARNING: PWAT not found in conditions mapping!")

    return processed_df


def process_single_gefs_file_local(
    date_str: str,
    idx: int,
    datestr: pd.Timestamp,
    ensemble_member: str,
    mapping_manager: 'LocalTarGzMappingManager'
) -> Optional[pd.DataFrame]:
    """
    Process a single GEFS file using local tar.gz mappings.

    Parameters:
    - date_str: Target date for GRIB index reading (where fresh data comes from)
    - idx: Index for the forecast timestep
    - datestr: Timestamp for this forecast step
    - ensemble_member: Ensemble member identifier (e.g., 'gep01')
    - mapping_manager: LocalTarGzMappingManager instance for reading mappings

    Returns:
    - DataFrame with mapped index, or None if processing fails
    """
    try:
        # GEFS uses 3-hour intervals
        forecast_hour = idx * 3

        # Target date: where we get fresh GRIB data and index from
        fname = f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.t00z.pgrb2s.0p25.f{forecast_hour:03d}"

        # Read idx file (fresh binary positions from target date)
        storage_options = {"anon": True}
        idxdf = parse_grib_idx(basename=fname, storage_options=storage_options)

        # Read pre-built mapping from local tar.gz
        deduped_mapping = mapping_manager.read_mapping(ensemble_member, forecast_hour)

        if deduped_mapping is None:
            print(f"Warning: No mapping found for {ensemble_member} at forecast hour {forecast_hour}")
            return None

        # Process the mapping
        idxdf_filtered = idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
        mapped_index = map_from_index(datestr, deduped_mapping, idxdf_filtered)

        return mapped_index

    except Exception as e:
        print(f'Error in GEFS local processing: {str(e)}')
        return None


def process_gefs_files_local(
    axes: List[pd.Index],
    date_str: str,
    ensemble_member: str,
    mapping_manager: 'LocalTarGzMappingManager',
    max_timesteps: Optional[int] = None
) -> pd.DataFrame:
    """
    Process GEFS files using local tar.gz mappings (non-async version).

    Parameters:
    - axes: List of pandas Index objects containing time information
    - date_str: Target date for GRIB index reading
    - ensemble_member: Ensemble member identifier (e.g., 'gep01')
    - mapping_manager: LocalTarGzMappingManager instance for reading mappings
    - max_timesteps: Optional limit on number of timesteps to process

    Returns:
    - DataFrame with combined mapped indices
    """
    dtaxes = axes[0]

    # Limit timesteps if specified
    if max_timesteps:
        dtaxes = dtaxes[:max_timesteps]

    all_results = []

    for idx, datestr in enumerate(dtaxes):
        result = process_single_gefs_file_local(
            date_str,
            idx,
            datestr,
            ensemble_member,
            mapping_manager
        )
        if result is not None:
            all_results.append(result)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(dtaxes)} timesteps for {ensemble_member}")

    if not all_results:
        raise ValueError(f"No valid GEFS mapped indices created for date {date_str}, member {ensemble_member}")

    gefs_kind = pd.concat(all_results, ignore_index=True)

    # Process variables similar to GFS workflow
    gefs_kind_var = gefs_kind.drop_duplicates('varname')
    var_list = gefs_kind_var['varname'].tolist()
    var_to_remove = ['pres','dswrf','cape','uswrf','apcp','gust','tmp','rh','ugrd','vgrd','pwat','tcdc','hgt']
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))

    print(f"Variables found in gefs_kind: {var_list}")
    print(f"Variables to be specially processed: {[v for v in var_to_remove if v in var_list]}")
    print(f"Variables to be kept as-is: {var1_list}")

    gefs_kind1 = gefs_kind.loc[gefs_kind.varname.isin(var1_list)]
    to_process_df = gefs_kind[gefs_kind['varname'].isin(var_to_remove)]

    # Debug pwat specifically
    if 'pwat' in var_list:
        pwat_df = gefs_kind[gefs_kind['varname'] == 'pwat']
        print(f"PWAT entries before processing: {len(pwat_df)}")
        print(f"PWAT typeOfLevel values: {pwat_df['typeOfLevel'].unique()}")

    processed_df = process_gefs_dataframe(to_process_df, var_to_remove)

    # Debug pwat after processing
    if 'pwat' in var_to_remove:
        pwat_processed = processed_df[processed_df['varname'] == 'pwat']
        print(f"PWAT entries after processing: {len(pwat_processed)}")
        if not pwat_processed.empty:
            print(f"PWAT typeOfLevel after processing: {pwat_processed['typeOfLevel'].unique()}")
        else:
            print("WARNING: PWAT lost during processing!")

    final_df = pd.concat([gefs_kind1, processed_df], ignore_index=True)
    final_df = final_df.sort_values(by=['time', 'varname'])

    return final_df


def cs_create_mapped_index_local(
    axes: List[pd.Index],
    date_str: str,
    ensemble_member: str,
    tar_gz_path: str,
    mapping_manager: Optional['LocalTarGzMappingManager'] = None,
    max_timesteps: Optional[int] = None
) -> pd.DataFrame:
    """
    Create GEFS mapped index using local tar.gz file instead of GCS bucket.

    This is an alternative to cs_create_mapped_index() that doesn't require
    GCS service account credentials.

    Parameters:
    - axes: List of pandas Index objects containing time information
    - date_str: Target date for GRIB index reading
    - ensemble_member: Ensemble member identifier (e.g., 'gep01')
    - tar_gz_path: Path to the tar.gz file containing parquet mappings
    - mapping_manager: Optional pre-initialized LocalTarGzMappingManager (reuse for efficiency)
    - max_timesteps: Optional limit on number of timesteps to process

    Returns:
    - DataFrame with combined mapped indices
    """
    # Create or reuse mapping manager
    if mapping_manager is None:
        mapping_manager = LocalTarGzMappingManager(tar_gz_path)

    return process_gefs_files_local(
        axes,
        date_str,
        ensemble_member,
        mapping_manager,
        max_timesteps
    )


def prepare_zarr_store(deflated_gefs_grib_tree_store: dict, gefs_kind: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """
    Prepare Zarr store and related data for chunk processing based on GEFS kind DataFrame.

    Parameters:
    - deflated_gefs_grib_tree_store (dict): Deflated GRIB tree store containing reference data.
    - gefs_kind (pd.DataFrame): DataFrame containing GEFS data.

    Returns:
    - Tuple[dict, pd.DataFrame]: Zarr reference store and the DataFrame for chunk index.
    """
    print("Preparing GEFS Zarr Store")
    zarr_ref_store = deflated_gefs_grib_tree_store
    chunk_index = gefs_kind
    zstore = copy.deepcopy(zarr_ref_store["refs"])
    return zstore, chunk_index


def process_unique_groups(zstore: dict, chunk_index: pd.DataFrame, time_dims: Dict, time_coords: Dict,
                          times: np.ndarray, valid_times: np.ndarray, steps: np.ndarray) -> dict:
    """
    Process and update Zarr store by configuring data for unique variable groups for GEFS.

    Parameters:
    - zstore (dict): The initial Zarr store with references to original data.
    - chunk_index (pd.DataFrame): DataFrame containing metadata and paths for the chunks of data to be stored.
    - time_dims (Dict): Dictionary specifying dimensions for time-related data.
    - time_coords (Dict): Dictionary specifying coordinates for time-related data.
    - times (np.ndarray): Array of actual times from the data files.
    - valid_times (np.ndarray): Array of valid forecast times.
    - steps (np.ndarray): Time steps in seconds converted from time differences.

    Returns:
    - dict: Updated Zarr store with added datasets and metadata.
    """
    print("Processing GEFS Unique Groups and Updating Zarr Store")
    unique_groups = chunk_index.set_index(["varname", "stepType", "typeOfLevel"]).index.unique()

    # Debug: Print what we have in chunk_index
    print(f"Chunk index contains {len(chunk_index)} rows")
    print(f"Unique variables in chunk_index: {chunk_index['varname'].unique()}")
    print(f"Unique groups identified: {len(unique_groups)}")

    # Debug: Check specifically for pwat
    pwat_entries = chunk_index[chunk_index['varname'] == 'pwat']
    if not pwat_entries.empty:
        print(f"PWAT found in chunk_index with typeOfLevel: {pwat_entries['typeOfLevel'].unique()}")
    else:
        print("WARNING: PWAT not found in chunk_index!")

    # Count keys before deletion
    original_key_count = len(zstore.keys())
    keys_to_delete = []

    for key in list(zstore.keys()):
        lookup = tuple([val for val in os.path.dirname(key).split("/")[:3] if val != ""])
        if lookup not in unique_groups:
            keys_to_delete.append(key)

    # Debug: Check if pwat keys are being deleted
    pwat_keys_to_delete = [k for k in keys_to_delete if 'pwat' in k.lower()]
    if pwat_keys_to_delete:
        print(f"WARNING: About to delete {len(pwat_keys_to_delete)} pwat-related keys!")
        print(f"Sample pwat keys being deleted: {pwat_keys_to_delete[:3]}")

    # Delete the keys
    for key in keys_to_delete:
        del zstore[key]

    print(f"Deleted {len(keys_to_delete)} keys from zarr store (from {original_key_count} to {len(zstore.keys())})")

    for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
        try:
            base_path = "/".join(key)
            lvals = group.level.unique()
            dims = time_dims.copy()
            coords = time_coords.copy()

            # Check if this group has the required zarr structure
            required_keys = [
                f"{base_path}/time/.zattrs",
                f"{base_path}/valid_time/.zattrs",
                f"{base_path}/step/.zattrs",
                f"{base_path}/{key[2]}/.zattrs" if key[2] else None
            ]
            required_keys = [k for k in required_keys if k is not None]

            missing_keys = [k for k in required_keys if k not in zstore]
            if missing_keys:
                if key[0] == 'pwat':
                    print(f"PWAT missing zarr structure keys: {missing_keys}")
                    print(f"PWAT available keys matching pattern: {[k for k in zstore.keys() if 'pwat' in k.lower()][:5]}")
                    print(f"PWAT group will be skipped - not included in original GRIB tree structure")
                # Skip this group if it's missing required zarr structure
                continue

            if len(lvals) == 1:
                lvals = lvals.squeeze()
                dims[key[2]] = 0
            elif len(lvals) > 1:
                lvals = np.sort(lvals)
                dims[key[2]] = len(lvals)
                coords["datavar"] += (key[2],)
            else:
                raise ValueError("Invalid level values encountered")

            store_coord_var(key=f"{base_path}/time", zstore=zstore, coords=time_coords["time"], data=times.astype("datetime64[s]"))
            store_coord_var(key=f"{base_path}/valid_time", zstore=zstore, coords=time_coords["valid_time"], data=valid_times.astype("datetime64[s]"))
            store_coord_var(key=f"{base_path}/step", zstore=zstore, coords=time_coords["step"], data=steps.astype("timedelta64[s]").astype("float64") / 3600.0)
            store_coord_var(key=f"{base_path}/{key[2]}", zstore=zstore, coords=(key[2],) if lvals.shape else (), data=lvals)

            store_data_var(key=f"{base_path}/{key[0]}", zstore=zstore, dims=dims, coords=coords, data=group, steps=steps, times=times, lvals=lvals if lvals.shape else None)

            # Success message for PWAT
            if key[0] == 'pwat':
                print(f"PWAT successfully processed and stored in zarr!")

        except Exception as e:
            print(f"Skipping processing of GEFS group {key}: {str(e)}")
            if key[0] == 'pwat':  # Special attention to PWAT errors
                print(f"PWAT SPECIFIC ERROR: {type(e).__name__}: {str(e)}")
                print(f"PWAT group details: varname={key[0]}, stepType={key[1]}, typeOfLevel={key[2]}")
                print(f"PWAT group size: {len(group)}")
                print(f"PWAT level values: {group.level.unique()}")
                import traceback
                traceback.print_exc()

    return zstore


def zstore_dict_to_df(zstore: dict):
    """
    Helper function to convert dictionary to pandas DataFrame with columns 'key' and 'value'.

    Parameters:
    - zstore (dict): The dictionary representing the Zarr store.

    Returns:
    - pd.DataFrame: DataFrame with two columns: 'key' representing the dictionary keys, and 'value'
                    representing the dictionary values, which are encoded in UTF-8 if they are of type
                    dictionary, list, or numeric.
    """
    data = []
    for key, value in zstore.items():
        if isinstance(value, (dict, list, int, float, np.integer, np.floating)):
            value = str(value).encode('utf-8')
        data.append((key, value))
    return pd.DataFrame(data, columns=['key', 'value'])


def create_parquet_file(zstore: dict, output_parquet_file: str):
    """
    Converts a dictionary containing Zarr store data to a DataFrame and saves it as a Parquet file for GEFS.

    Parameters:
    - zstore (dict): The Zarr store dictionary containing all references and data needed for Zarr operations.
    - output_parquet_file (str): The path where the Parquet file will be saved.
    """
    gefs_store = dict(refs=zstore, version=1)
    zstore_df = zstore_dict_to_df(gefs_store)
    zstore_df.to_parquet(output_parquet_file)
    print(f"GEFS Parquet file saved to {output_parquet_file}")


# ==============================================================================
# TEMPLATE MANAGEMENT
# ==============================================================================

def _download_if_missing(url: str, local_path: str) -> str:
    """Download file from URL if not already present locally."""
    if os.path.exists(local_path):
        logger.info(f"Template cached: {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
        return local_path
    logger.info(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded: {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
    return local_path


def ensure_templates() -> Tuple[str, str]:
    """Ensure both GEFS templates are available (baked-in or downloaded)."""
    tar_gz = _download_if_missing(TEMPLATE_TAR_GZ_URL, TEMPLATE_TAR_GZ_PATH)
    parquet = _download_if_missing(TEMPLATE_PARQUET_URL, TEMPLATE_PARQUET_PATH)
    return tar_gz, parquet


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_idx_availability(date_str: str, run: str = "00") -> Tuple[bool, int]:
    """Check that .idx files are available on S3 for the target date."""
    import fsspec
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    # Check f003 (first non-zero timestep)
    idx_url = (
        f"s3://noaa-gefs-pds/gefs.{date_str}/{run}/atmos/pgrb2sp25/"
        f"gep01.t{run}z.pgrb2s.0p25.f003.idx"
    )
    try:
        fs = fsspec.filesystem("s3", anon=True)
        if not fs.exists(idx_url):
            logger.warning(f"IDX file not found: {idx_url}")
            return False, 0

        # Count lines to estimate message count
        with fs.open(idx_url, 'r') as f:
            lines = f.readlines()
        logger.info(f"IDX validation OK: {len(lines)} messages in f003")
        return True, len(lines)
    except Exception as e:
        logger.error(f"IDX validation failed: {e}")
        return False, 0


# ==============================================================================
# MAIN PROCESSING FUNCTION (runs on Cloud Run workers via Lithops)
# ==============================================================================

def process_gefs_date(
    date_str: str,
    run: str = "00",
    max_members: Optional[int] = None,
) -> Dict:
    """
    Process a single GEFS date through the three-stage GIK pipeline.

    This function is serialized by Lithops (cloudpickle) and executed on
    Cloud Run workers. All required functions are defined above in this
    file (no external gefs_util import needed).

    Args:
        date_str: Date in YYYYMMDD format
        run: Model run hour (default "00")
        max_members: Limit number of members (default: all 30)

    Returns:
        Dictionary with processing results
    """
    os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

    logger.info(f"Processing GEFS {date_str} {run}z")
    pipeline_start = time.time()

    result = {
        "date": date_str,
        "run": run,
        "success": False,
        "message": "",
        "gcs_path": None,
        "files_count": 0,
        "total_time_seconds": 0,
    }

    try:
        # All pipeline functions are defined above in this file (self-contained)

        # Validate .idx availability
        idx_valid, n_msgs = validate_idx_availability(date_str, run)
        if not idx_valid:
            result["message"] = f"IDX validation failed for {date_str}"
            return result

        # Ensure templates
        tar_gz_path, parquet_path = ensure_templates()

        # Select members
        members = ALL_MEMBERS[:max_members] if max_members else ALL_MEMBERS

        # -- Stage 1: Load deflated store from template --
        stage1_start = time.time()
        deflated_store = build_gefs_deflated_store_from_template(parquet_path)
        stage1_time = time.time() - stage1_start
        logger.info(f"Stage 1: {len(deflated_store['refs'])} refs in {stage1_time:.1f}s")

        # Initialize mapping manager (reused across members)
        mapping_manager = LocalTarGzMappingManager(tar_gz_path)

        # -- Stage 2 + 3: Process each member --
        output_dir = Path(f"/tmp/gefs_{date_str}_{run}z")
        output_dir.mkdir(parents=True, exist_ok=True)

        stage2_total = 0
        stage3_total = 0
        members_ok = 0

        for i, member in enumerate(members, 1):
            try:
                # Stage 2: Read .idx files, create mapped index
                s2_start = time.time()
                axes = generate_axes(date_str)
                gefs_kind = cs_create_mapped_index_local(
                    axes, date_str, member,
                    tar_gz_path=tar_gz_path,
                    mapping_manager=mapping_manager,
                )
                s2_time = time.time() - s2_start
                stage2_total += s2_time

                # Stage 3: Build zarr store, create parquet
                s3_start = time.time()
                time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
                zstore, chunk_index = prepare_zarr_store(deflated_store, gefs_kind)
                updated_zstore = process_unique_groups(
                    zstore, chunk_index, time_dims, time_coords,
                    times, valid_times, steps,
                )

                # Save parquet
                out_file = str(output_dir / f"{date_str}{run}z-{member}.parquet")
                create_parquet_file(updated_zstore, out_file)
                s3_time = time.time() - s3_start
                stage3_total += s3_time

                members_ok += 1
                if i % 5 == 0 or i == len(members):
                    logger.info(f"  [{i}/{len(members)}] {member} done "
                                f"(s2={s2_time:.1f}s s3={s3_time:.1f}s)")

            except Exception as e:
                logger.warning(f"  [{i}/{len(members)}] {member} FAILED: {e}")

        # Cleanup mapping manager
        mapping_manager.cleanup()

        if members_ok == 0:
            result["message"] = "No members processed successfully"
            return result

        # -- Upload to GCS --
        gcs_path = _upload_to_gcs(output_dir, date_str, run)

        # Cleanup /tmp
        shutil.rmtree(output_dir, ignore_errors=True)

        total_time = time.time() - pipeline_start

        result.update({
            "success": True,
            "message": (f"{members_ok}/{len(members)} members in {total_time:.1f}s "
                        f"(s1={stage1_time:.1f}s s2={stage2_total:.1f}s s3={stage3_total:.1f}s)"),
            "gcs_path": gcs_path,
            "files_count": members_ok,
            "total_time_seconds": round(total_time, 1),
            "members_processed": members_ok,
            "stage1_time": round(stage1_time, 1),
            "stage2_time": round(stage2_total, 1),
            "stage3_time": round(stage3_total, 1),
        })

        logger.info(f"Pipeline complete: {result['message']}")
        return result

    except Exception as e:
        total_time = time.time() - pipeline_start
        result["message"] = f"Pipeline error: {str(e)}"
        result["total_time_seconds"] = round(total_time, 1)
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return result


# ==============================================================================
# GCS UPLOAD
# ==============================================================================

def _upload_to_gcs(output_dir: Path, date_str: str, run: str) -> Optional[str]:
    """Upload final parquet files to GCS."""
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()

        year = date_str[:4]
        month = date_str[4:6]
        gcs_path = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{year}/{month}/{date_str}/{run}z"

        parquet_files = list(output_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No parquet files to upload")
            return None

        logger.info(f"Uploading {len(parquet_files)} files to gs://{gcs_path}")
        for pf in parquet_files:
            fs.put(str(pf), f"{gcs_path}/{pf.name}")

        logger.info(f"Uploaded {len(parquet_files)} files")
        return f"gs://{gcs_path}"

    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        return None


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def generate_date_list(args) -> List[str]:
    """Generate list of dates based on command-line arguments."""
    if args.date:
        return [args.date]
    elif args.days_back:
        end_date = datetime.now()
        return [(end_date - timedelta(days=i)).strftime("%Y%m%d")
                for i in range(args.days_back)]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y%m%d")
        end = datetime.strptime(args.end_date, "%Y%m%d")
        return [(start + timedelta(days=i)).strftime("%Y%m%d")
                for i in range((end - start).days + 1)]
    else:
        raise ValueError("Specify --date, --days-back, or --start-date/--end-date")


def run_sequential(dates: List[str], run: str = "00",
                   max_members: Optional[int] = None) -> List[Dict]:
    """Run processing sequentially (no Lithops, for local testing)."""
    print(f"\nRunning sequential processing for {len(dates)} dates")
    print("=" * 70)

    results = []
    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] Processing {date}...")
        result = process_gefs_date(date, run, max_members)

        status = "OK" if result.get('success') else "FAIL"
        print(f"{status} {date}: {result.get('message', 'Unknown')}")
        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")

        results.append(result)

    return results


def run_with_lithops(dates: List[str], run: str = "00",
                     max_workers: int = 10,
                     max_members: Optional[int] = None) -> List[Dict]:
    """Run processing in parallel using Lithops Cloud Run backend, with retry."""
    try:
        import lithops
    except ImportError:
        print("Error: lithops not installed. Install with: pip install lithops")
        sys.exit(1)

    MAX_RETRIES = 3
    WAIT_TIMEOUT = 600
    RETRY_DELAY = 30

    print(f"\nRunning Lithops parallel processing for {len(dates)} dates")
    print(f"Max workers: {max_workers}  |  Max retries: {MAX_RETRIES}")
    print("=" * 70)

    config_path = Path(__file__).parent / 'lithops_config.yaml'
    if config_path.exists():
        print(f"Config: {config_path}")
        fexec = lithops.FunctionExecutor(config_file=str(config_path))
    else:
        print("Config: using default lithops config")
        fexec = lithops.FunctionExecutor(backend='gcp_cloudrun')

    overall_start = time.time()
    all_results: Dict[str, Dict] = {}
    remaining = list(dates)

    for attempt in range(1, MAX_RETRIES + 1):
        if not remaining:
            break

        if attempt > 1:
            print(f"\n--- Retry {attempt}/{MAX_RETRIES}: {len(remaining)} dates "
                  f"— pausing {RETRY_DELAY}s for Cloud Run warm-up ---")
            time.sleep(RETRY_DELAY)

        print(f"\nAttempt {attempt}: dispatching {len(remaining)} worker(s)...")
        futures = fexec.map(
            process_gefs_date, remaining,
            extra_args=(run, max_members),
        )

        try:
            fexec.wait(futures, throw_except=False, timeout=WAIT_TIMEOUT)
        except TimeoutError:
            logger.warning(
                f"Attempt {attempt}: wait timeout ({WAIT_TIMEOUT}s) — "
                f"collecting completed results"
            )

        still_failing = []
        for i, f in enumerate(futures):
            date = remaining[i]
            if f.status == 'success':
                try:
                    result = f.result()
                    if isinstance(result, dict) and result.get('success'):
                        all_results[date] = result
                        logger.info(f"  OK   {date}: {result.get('message', '')}")
                    else:
                        msg = result.get('message', 'unknown') if isinstance(result, dict) else str(result)
                        all_results[date] = result if isinstance(result, dict) else {
                            'date': date, 'run': run, 'success': False, 'message': msg}
                        still_failing.append(date)
                        logger.warning(f"  FAIL {date}: {msg}, will retry")
                except Exception as exc:
                    all_results[date] = {'date': date, 'run': run, 'success': False,
                                         'message': str(exc)}
                    still_failing.append(date)
                    logger.warning(f"  FAIL {date}: result error ({exc}), will retry")
            else:
                msg = f"invoke failed (status={f.status})"
                all_results[date] = {'date': date, 'run': run, 'success': False, 'message': msg}
                still_failing.append(date)
                logger.warning(f"  FAIL {date}: {msg}, will retry")

        # GCS verification for dates that may have succeeded but status is stale
        if still_failing:
            try:
                import gcsfs
                _fs = gcsfs.GCSFileSystem()
                gcs_rescued = []
                for date in still_failing:
                    year, month = date[:4], date[4:6]
                    gcs_prefix = f"{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/{year}/{month}/{date}/{run}z"
                    try:
                        items = _fs.ls(gcs_prefix)
                        if items:
                            all_results[date] = {
                                'date': date, 'run': run, 'success': True,
                                'gcs_path': f"gs://{gcs_prefix}",
                                'message': 'GCS-verified (f.status stale)',
                            }
                            gcs_rescued.append(date)
                            logger.info(f"  GCS-OK {date}: found {len(items)} file(s)")
                    except Exception:
                        pass
                if gcs_rescued:
                    still_failing = [d for d in still_failing if d not in set(gcs_rescued)]
                    print(f"  GCS verification rescued {len(gcs_rescued)} date(s)")
            except Exception as gcs_exc:
                logger.warning(f"  GCS verification skipped: {gcs_exc}")

        succeeded = len(remaining) - len(still_failing)
        print(f"Attempt {attempt}: {succeeded}/{len(remaining)} succeeded, "
              f"{len(still_failing)} queued for retry")
        remaining = still_failing

    for date in remaining:
        all_results[date] = {'date': date, 'run': run, 'success': False,
                             'message': 'Max retries exceeded'}
        logger.error(f"  GIVE UP {date}: max retries ({MAX_RETRIES}) exceeded")

    total_time = time.time() - overall_start
    success_count = sum(1 for r in all_results.values() if r.get('success'))
    print(f"\nCompleted in {total_time:.1f}s ({total_time / 60:.1f} min) — "
          f"{success_count}/{len(dates)} successful")

    return [all_results.get(d, {'date': d, 'run': run, 'success': False,
                                'message': 'No result'}) for d in dates]


def print_summary(dates: List[str], results: List[Dict], total_time: float):
    """Print execution summary."""
    success_count = sum(1 for r in results if r.get('success'))
    failed_count = len(results) - success_count

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal dates: {len(dates)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    if success_count > 0:
        avg_time = sum(r.get('total_time_seconds', 0) for r in results
                       if r.get('success')) / success_count
        print(f"Average processing time: {avg_time:.1f}s ({avg_time / 60:.1f} min)")

    print("\nResults by date:")
    print("-" * 70)

    for date, result in zip(dates, results):
        status = "OK" if result.get('success') else "FAIL"
        message = result.get('message', 'No message')
        print(f"{status} {date}: {message}")

        if result.get('gcs_path'):
            print(f"   GCS: {result['gcs_path']}")
        if result.get('files_count'):
            print(f"   Files: {result['files_count']}")

    gcs_paths = [r.get('gcs_path') for r in results if r.get('gcs_path')]
    if gcs_paths:
        print("\n" + "=" * 70)
        print("GCS OUTPUT LOCATIONS")
        print("=" * 70)
        for path in gcs_paths:
            print(f"  {path}")
        print(f"\nList all parquets:")
        print(f"  gsutil ls gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")


def main():
    parser = argparse.ArgumentParser(
        description='Run GEFS parquet generation via Lithops Cloud Run workers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Date selection
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--date', type=str,
                            help='Single date to process (YYYYMMDD)')
    date_group.add_argument('--start-date', type=str,
                            help='Start date for range (YYYYMMDD)')
    date_group.add_argument('--days-back', type=int,
                            help='Process last N days')

    parser.add_argument('--end-date', type=str,
                        help='End date for range (YYYYMMDD), used with --start-date')
    parser.add_argument('--run', type=str, default='00',
                        choices=['00', '06', '12', '18'],
                        help='Model run hour (default: 00)')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Max parallel workers for Lithops (default: 10)')
    parser.add_argument('--max-members', type=int, default=None,
                        help='Limit ensemble members (default: all 30)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially without Lithops (local testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show dates without executing')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip interactive confirmation')

    args = parser.parse_args()

    if args.start_date and not args.end_date:
        parser.error("--end-date is required when using --start-date")
    if args.end_date and not args.start_date:
        parser.error("--start-date is required when using --end-date")

    try:
        dates = generate_date_list(args)
    except ValueError as e:
        parser.error(str(e))

    # Print configuration
    print("=" * 70)
    print("GEFS PARQUET GENERATION - LITHOPS CLOUD RUN")
    print("=" * 70)
    print(f"\nRun hour: {args.run}Z")
    print(f"Dates to process: {len(dates)}")
    print(f"  First: {dates[0]}")
    print(f"  Last: {dates[-1]}")
    print(f"Members: {args.max_members or 30}")
    mode = 'Sequential (local)' if args.sequential else f'Lithops Cloud Run (max {args.max_workers} workers)'
    print(f"Execution mode: {mode}")
    print(f"GCS output: gs://{GCS_BUCKET}/{GCS_PARQUET_PREFIX}/")

    if args.dry_run:
        print(f"\nDRY RUN - Dates that would be processed:")
        for date in dates:
            print(f"  {date}")
        return

    if len(dates) > 5 and not args.yes:
        response = input(f"\nProcess {len(dates)} dates? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    start_time = time.time()

    if args.sequential:
        results = run_sequential(dates, args.run, args.max_members)
    else:
        results = run_with_lithops(dates, args.run, args.max_workers, args.max_members)

    total_time = time.time() - start_time
    print_summary(dates, results, total_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
