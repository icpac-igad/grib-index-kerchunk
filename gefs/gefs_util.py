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
import time
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
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
        'pwat': 'atmosphere',
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
                    filtered_df = filtered_df.sort_values(by='length', ascending=False).drop_duplicates(subset=['time'], keep='first')
                    processed_df = pd.concat([processed_df, filtered_df], ignore_index=True)
            else:
                filtered_df = df[(df['varname'] == varname) & (df['typeOfLevel'] == level)]
                filtered_df = filtered_df.sort_values(by='length', ascending=False).drop_duplicates(subset=['time'], keep='first')
                processed_df = pd.concat([processed_df, filtered_df], ignore_index=True)

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

    for key in list(zstore.keys()):
        lookup = tuple([val for val in os.path.dirname(key).split("/")[:3] if val != ""])
        if lookup not in unique_groups:
            del zstore[key]

    for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
        try:
            base_path = "/".join(key)
            lvals = group.level.unique()
            dims = time_dims.copy()
            coords = time_coords.copy()

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
        except Exception as e:
            print(f"Skipping processing of GEFS group {key}: {str(e)}")

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
        matching_rows = df[df['attrs'].str.contains(value, na=False)]
        
        if not matching_rows.empty:
            output_dict[key] = int(matching_rows.index[0] - 1)
        else:
            output_dict[key] = 9999
    
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
    
    gefs_kind1 = gefs_kind.loc[gefs_kind.varname.isin(var1_list)]
    to_process_df = gefs_kind[gefs_kind['varname'].isin(var_to_remove)]
    processed_df = process_gefs_dataframe(to_process_df, var_to_remove)
    
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