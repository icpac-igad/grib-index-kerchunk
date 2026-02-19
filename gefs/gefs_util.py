import asyncio
import concurrent.futures
import copy
import datetime
import io
import json
import logging
import contextlib
import math
import os
import pathlib
import re
import sys
import tempfile
import tarfile
import time
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable, Union
import shutil
import os
import warnings

import dask
import fsspec
import gcsfs
import numpy as np
import pandas as pd
from distributed import get_worker
from google.auth import credentials
from google.cloud import storage
from google.oauth2 import service_account
from kerchunk.grib2 import grib_tree, scan_grib

import dask.dataframe as dd
from dask.distributed import Client, get_worker
import xarray as xr
import zarr
import gcsfs
import pathlib
import pandas as pd

from enum import Enum, auto
from kerchunk._grib_idx import (
    AggregationType,
    build_idx_grib_mapping,
    map_from_index,
    parse_grib_idx,
    store_coord_var,
    store_data_var,
    strip_datavar_chunks,
    _extract_single_group,
)

logger = logging.getLogger("gefs-utils-logs")


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


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": self.format_message(record.getMessage()),
            "function": record.funcName,
            "line": record.lineno,
        }

        try:
            return json.dumps(log_data)
        except (TypeError, ValueError):
            return f"{{\"timestamp\": \"{self.formatTime(record, self.datefmt)}\", \"level\": \"{record.levelname}\", \"function\": \"{record.funcName}\", \"line\": {record.lineno}, \"message\": \"{self.format_message(record.getMessage())}\"}}"

    def format_message(self, message):
        return message.replace('\n', ' ')


def setup_logging(log_level: int = logging.INFO, log_file: str = "gefs_processing.log"):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)


def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        func_name = func.__name__
        logger.info(json.dumps({"event": "function_start", "function": func_name}))

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = func(*args, **kwargs)

            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()

            if stdout:
                for line in stdout.splitlines():
                    logger.info(json.dumps({"event": "print_output", "function": func_name, "message": line}))

            if stderr:
                for line in stderr.splitlines():
                    logger.info(json.dumps({"event": "stderr_output", "function": func_name, "message": line}))

        logger.info(json.dumps({"event": "function_end", "function": func_name}))
        return result
    return wrapper


def build_gefs_grib_tree(gefs_files: List[str]) -> Tuple[dict, dict]:
    """
    Scan GEFS files, build a hierarchical tree structure for the data, and strip unnecessary data.

    Parameters:
    - gefs_files (List[str]): List of file paths to GEFS files.

    Returns:
    - Tuple[dict, dict]: Original and deflated GRIB tree stores.
    """
    print("Building GEFS Grib Tree")
    gefs_grib_tree_store = grib_tree([group for f in gefs_files for group in scan_grib(f)])
    deflated_gefs_grib_tree_store = copy.deepcopy(gefs_grib_tree_store)
    strip_datavar_chunks(deflated_gefs_grib_tree_store)
    print(f"Original references: {len(gefs_grib_tree_store['refs'])}")
    print(f"Stripped references: {len(deflated_gefs_grib_tree_store['refs'])}")
    return gefs_grib_tree_store, deflated_gefs_grib_tree_store


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


def create_gefs_mapped_index(axes: List[pd.Index], mapping_parquet_file_path: str, date_str: str, ensemble_member: str) -> pd.DataFrame:
    """
    Create a mapped index from GEFS files for a specific date and ensemble member.

    Parameters:
    - axes (List[pd.Index]): List of time axes to map.
    - mapping_parquet_file_path (str): File path to the mapping parquet file.
    - date_str (str): Date string for the data being processed.
    - ensemble_member (str): Ensemble member identifier (e.g., 'gep01').

    Returns:
    - pd.DataFrame: DataFrame containing the mapped index for the specified date and member.
    """
    print(f"Creating GEFS Mapped Index for date {date_str} and member {ensemble_member}")
    mapped_index_list = []
    dtaxes = axes[0]

    for idx, datestr in enumerate(dtaxes):
        try:
            # GEFS files use 3-hour intervals starting from 000
            forecast_hour = idx * 3
            fname = f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.t00z.pgrb2s.0p25.f{forecast_hour:03d}"
            
            idxdf = parse_grib_idx(basename=fname)
            deduped_mapping = pd.read_parquet(f"{mapping_parquet_file_path}gefs-mapping-{ensemble_member}-{forecast_hour:03d}.parquet")
            mapped_index = map_from_index(
                datestr,
                deduped_mapping,
                idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
            )
            mapped_index_list.append(mapped_index)
        except Exception as e:
            logger.error(f"Error processing file {fname}: {str(e)}")

    gefs_kind = pd.concat(mapped_index_list)
    gefs_kind_var = gefs_kind.drop_duplicates('varname')
    var_list = gefs_kind_var['varname'].tolist() 
    var_to_remove = ['pres','dswrf','cape','uswrf','apcp','gust','tmp','rh','ugrd','vgrd','pwat','tcdc','hgt']
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))
    gefs_kind1 = gefs_kind.loc[gefs_kind.varname.isin(var1_list)]
    
    to_process_df = gefs_kind[gefs_kind['varname'].isin(var_to_remove)]
    processed_df = process_gefs_dataframe(to_process_df, var_to_remove)
    
    final_df = pd.concat([gefs_kind1, processed_df], ignore_index=True)
    final_df = final_df.sort_values(by=['time', 'varname'])
    final_df_var = final_df.drop_duplicates('varname')
    final_var_list = final_df_var['varname'].tolist() 

    print(f"GEFS Mapped collected multiple variables index info: {len(final_var_list)} and {final_var_list}")
    return final_df


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
                print(f"✅ PWAT successfully processed and stored in zarr!")

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


def generate_gefs_dates(year: int, month: int) -> List[str]:
    """
    Generate a list of dates for a specific month and year, formatted as 'YYYYMMDD', for GEFS processing.

    Parameters:
    - year (int): The year for which the dates are to be generated.
    - month (int): The month for which the dates are to be generated.

    Returns:
    - List[str]: A list of dates in the format 'YYYYMMDD' for every day in the specified month and year.
    """
    _, last_day = monthrange(year, month)
    date_range = pd.date_range(start=f'{year}-{month:02d}-01', 
                               end=f'{year}-{month:02d}-{last_day}', 
                               freq='D')
    return date_range.strftime('%Y%m%d').tolist()


def get_gefs_details(url):
    """Extract date and time details from GEFS URL."""
    pattern = r"s3://noaa-gefs-pds/gefs\.(\d{8})/(\d{2})/atmos/pgrb2sp25/(gep\d{2})\.t(\d{2})z\.pgrb2s\.0p25\.f(\d{3})"
    match = re.match(pattern, url)
    if match:
        date = match.group(1)
        run = match.group(2)
        member = match.group(3)
        hour = match.group(5)
        return date, run, member, hour
    else:
        logger.warning(f"No match found for GEFS URL pattern: {url}")
        return None, None, None, None


def gefs_s3_url_maker(date_str, ensemble_member="gep01"):
    """Create S3 URLs for GEFS data."""
    fs_s3 = fsspec.filesystem("s3", anon=True)
    s3url_glob = fs_s3.glob(
        f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.*"
    )
    s3url_only_grib = [f for f in s3url_glob if f.split(".")[-1] != "idx"]
    fmt_s3og = sorted(["s3://" + f for f in s3url_only_grib])
    print(f"Generated {len(fmt_s3og)} GEFS URLs for date {date_str} and member {ensemble_member}")
    return fmt_s3og


def filter_gefs_scan_grib(gurl, tofilter_gefs_var_dict):
    """Filter and scan GEFS GRIB files based on variable dictionary."""
    suffix = "idx"
    storage_options = {"anon": True}
    gsc = scan_grib(gurl, storage_options=storage_options)
    idx_gefs = parse_grib_idx(basename=gurl, suffix=suffix, storage_options=storage_options)
    output_dict0, vl_gefs = map_forecast_to_indices(tofilter_gefs_var_dict, idx_gefs)

    # Debug PWAT filtering
    pwat_found = any('PWAT' in str(value) for value in tofilter_gefs_var_dict.values())
    if pwat_found:
        pwat_entries = idx_gefs[idx_gefs['attrs'].str.contains('PWAT', na=False)]
        print(f"GRIB filtering debug - PWAT entries in idx: {len(pwat_entries)}")
        if not pwat_entries.empty:
            print(f"PWAT attrs sample: {pwat_entries['attrs'].iloc[0]}")
        print(f"Indices selected for PWAT: {[i for i, v in output_dict0.items() if 'PWAT' in str(v)]}")

    return [gsc[i] for i in vl_gefs]


def filter_build_grib_tree(gefs_files: List[str], tofilter_gefs_var_dict: dict) -> Tuple[dict, dict]:
    """
    Scan GEFS files, build a hierarchical tree structure for the data, and strip unnecessary data.

    Parameters:
    - gefs_files (List[str]): List of file paths to GEFS files.
    - tofilter_gefs_var_dict (dict): Dictionary of variables to filter.

    Returns:
    - Tuple[dict, dict]: Original and deflated GRIB tree stores.
    """
    print("Building GEFS Grib Tree")
    sg_groups = [group for gurl in gefs_files for group in filter_gefs_scan_grib(gurl, tofilter_gefs_var_dict)]
    gefs_grib_tree_store = grib_tree(sg_groups)
    deflated_gefs_grib_tree_store = copy.deepcopy(gefs_grib_tree_store)
    strip_datavar_chunks(deflated_gefs_grib_tree_store)
    print(f"Original references: {len(gefs_grib_tree_store['refs'])}")
    print(f"Stripped references: {len(deflated_gefs_grib_tree_store['refs'])}")
    return gefs_grib_tree_store, deflated_gefs_grib_tree_store


def map_forecast_to_indices(forecast_dict: dict, df: pd.DataFrame) -> Tuple[dict, list]:
    """
    Map each forecast variable in forecast_dict to the index in df where its corresponding value in 'attrs' is found.
    
    Parameters:
    - forecast_dict (dict): Dictionary with forecast variables as keys and search strings as values.
    - df (pd.DataFrame): DataFrame containing a column 'attrs' where search is performed.
    
    Returns:
    - Tuple[dict, list]: Dictionary mapping each forecast variable to the found index, and list of all indices found.
    """
    output_dict = {}
    
    for key, value in forecast_dict.items():
        # Use regex=False to treat the search string as literal text, not regex
        matching_rows = df[df['attrs'].str.contains(value, na=False, regex=False)]

        if not matching_rows.empty:
            output_dict[key] = int(matching_rows.index[0] - 1)
            if 'PWAT' in value:  # Debug PWAT matching
                print(f"PWAT MATCH FOUND: {key} -> index {matching_rows.index[0] - 1}")
        else:
            output_dict[key] = 9999
            if 'PWAT' in value:  # Debug PWAT not matching
                print(f"PWAT NO MATCH: searching for '{value}' in {len(df)} entries")
                pwat_attrs = df[df['attrs'].str.contains('PWAT', na=False, regex=False)]['attrs'].tolist()
                print(f"Available PWAT attrs: {pwat_attrs[:3]}")
    
    values_list = [value for value in output_dict.values() if value != 9999]
    return output_dict, values_list


async def process_single_gefs_file(
    date_str: str,
    gcs_bucket_name: str,
    idx: int,
    datestr: pd.Timestamp,
    ensemble_member: str,
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    gcp_service_account_json: str,
    chunk_size: Optional[int] = None,
    reference_date_str: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Process a single GEFS file asynchronously using pre-built GCS mappings.
    
    Parameters:
    - date_str: Target date for GRIB index reading (where fresh data comes from)
    - reference_date_str: Reference date for parquet mapping templates (defaults to date_str)
    
    The function combines:
    1. Fresh GRIB index (.idx) from target date (binary positions)
    2. Existing parquet mappings from reference date (structure template)
    """
    async with sem:
        try:
            # GEFS uses 3-hour intervals
            forecast_hour = idx * 3
            
            # Target date: where we get fresh GRIB data and index from
            fname = f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.t00z.pgrb2s.0p25.f{forecast_hour:03d}"
            
            # Reference date: where we get parquet mapping templates from
            ref_date = reference_date_str if reference_date_str else date_str
            ref_year = ref_date[:4]
            gcs_mapping_path = f"gs://{gcs_bucket_name}/gefs/{ensemble_member}/gefs-time-{ref_date}-{ensemble_member}-rt{forecast_hour:03d}.parquet"
            
            gcs_fs = gcsfs.GCSFileSystem(token=gcp_service_account_json)
            loop = asyncio.get_event_loop()
            
            # Read idx file (fresh binary positions from target date)
            storage_options = {"anon": True}
            idxdf = await loop.run_in_executor(
                executor,
                partial(parse_grib_idx, basename=fname, storage_options=storage_options)
            )
            
            # Read pre-built mapping from GCS
            if chunk_size:
                deduped_mapping_chunks = []
                for chunk in pd.read_parquet(gcs_mapping_path, filesystem=gcs_fs, chunksize=chunk_size):
                    deduped_mapping_chunks.append(chunk)
                deduped_mapping = pd.concat(deduped_mapping_chunks, ignore_index=True)
            else:
                deduped_mapping = await loop.run_in_executor(
                    executor,
                    partial(pd.read_parquet, gcs_mapping_path, filesystem=gcs_fs)
                )
            
            # Process the mapping
            idxdf_filtered = idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
            mapped_index = await loop.run_in_executor(
                executor,
                partial(map_from_index, datestr, deduped_mapping, idxdf_filtered)
            )
            
            return mapped_index
            
        except Exception as e:
            print(f'Error in GEFS processing {str(e)} - ensure mappings exist in GCS')
            print(f'Expected mapping path: {gcs_mapping_path}')
            return None


async def process_gefs_files_in_batches(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    ensemble_member: str,
    max_concurrent: int = 3,
    batch_size: int = 5,
    chunk_size: Optional[int] = None,
    gcp_service_account_json: Optional[str] = None,
    reference_date_str: Optional[str] = None
) -> pd.DataFrame:
    """
    Process GEFS files in batches using pre-built GCS mappings.
    Reads from mappings created by gefs_index_preprocessing.py
    """
    dtaxes = axes[0]
    
    sem = asyncio.Semaphore(max_concurrent)
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        all_results = []
        
        for batch_start in range(0, len(dtaxes), batch_size):
            batch_end = min(batch_start + batch_size, len(dtaxes))
            batch_indices = range(batch_start, batch_end)
            
            tasks = [
                process_single_gefs_file(
                    date_str,
                    gcs_bucket_name,
                    idx,
                    dtaxes[idx],
                    ensemble_member,
                    sem,
                    executor,
                    gcp_service_account_json,
                    chunk_size,
                    reference_date_str
                )
                for idx in batch_indices
            ]
            
            batch_results = await asyncio.gather(*tasks)
            valid_results = [r for r in batch_results if r is not None]
            all_results.extend(valid_results)
            batch_results = None
            
            print(f"GEFS batch processed for {ensemble_member}")
    
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


def cs_create_mapped_index(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    ensemble_member: str,
    max_concurrent: int = 10,
    batch_size: int = 20,
    chunk_size: Optional[int] = None,
    gcp_service_account_json: Optional[str] = None,
    reference_date_str: Optional[str] = None
) -> pd.DataFrame:
    """
    Async wrapper for creating GEFS mapped index with memory management.

    Parameters:
    - date_str: Target date for GRIB index reading
    - reference_date_str: Reference date for parquet mapping templates
    """
    return asyncio.run(
        process_gefs_files_in_batches(
            axes,
            gcs_bucket_name,
            date_str,
            ensemble_member,
            max_concurrent,
            batch_size,
            chunk_size,
            gcp_service_account_json,
            reference_date_str
        )
    )


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


class KerchunkZarrDictStorageManager:
    """Manages storage and retrieval of Kerchunk Zarr dictionaries in Google Cloud Storage for GEFS."""
    
    def __init__(
        self, 
        bucket_name: str,
        service_account_file: Optional[str] = None,
        service_account_info: Optional[Dict] = None
    ):
        """
        Initialize the GEFS storage manager with GCP credentials.
        
        Args:
            bucket_name (str): Name of the GCS bucket
            service_account_file (str, optional): Path to service account JSON file
            service_account_info (dict, optional): Service account info as dictionary
        """
        self.bucket_name = bucket_name
        
        if service_account_file and service_account_info:
            raise ValueError("Provide either service_account_file OR service_account_info, not both")
        
        if service_account_file:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            with open(service_account_file, 'r') as f:
                self.service_account_info = json.load(f)
        elif service_account_info:
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.service_account_info = service_account_info
        else:
            credentials = None
            self.service_account_info = None
        
        self.storage_client = storage.Client(credentials=credentials)
        
        if self.service_account_info:
            self.gcs_fs = fsspec.filesystem(
                'gcs', 
                token=self.service_account_info
            )
        else:
            self.gcs_fs = fsspec.filesystem('gcs')


def process_gefs_data(date_str: str, ensemble_member: str, mapping_parquet_file_path: str, output_parquet_file: str, log_level: int = logging.INFO):
    """
    Orchestrates the end-to-end processing of Global Ensemble Forecast System (GEFS) data for a specific date and ensemble member.

    Parameters:
    - date_str (str): A date string in the format 'YYYYMMDD' representing the date for which GEFS data is to be processed.
    - ensemble_member (str): Ensemble member identifier (e.g., 'gep01').
    - mapping_parquet_file_path (str): Path to the parquet file that contains mapping information for the GEFS data.
    - output_parquet_file (str): Path where the output Parquet file will be saved after processing the data.
    - log_level (int): Logging level to use for reporting within this function. Default is logging.INFO.
    """
        
    try:
        print(f"Processing GEFS date: {date_str}, member: {ensemble_member}")
        axes = generate_axes(date_str)
        gefs_files = [
            f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.t00z.pgrb2s.0p25.f000",
            f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.t00z.pgrb2s.0p25.f003"
        ]
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
        _, deflated_gefs_grib_tree_store = filter_build_grib_tree(gefs_files, forecast_dict)
        time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
        gefs_kind = create_gefs_mapped_index(axes, mapping_parquet_file_path, date_str, ensemble_member)
        
        zstore, chunk_index = prepare_zarr_store(deflated_gefs_grib_tree_store, gefs_kind)
        updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, times, valid_times, steps)
        create_parquet_file(updated_zstore, output_parquet_file)
    except Exception as e:
        print(f"An error occurred during GEFS processing: {str(e)}")
        raise