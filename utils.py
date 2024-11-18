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
import datatree

from dynamic_zarr_store import (
    AggregationType,
    build_idx_grib_mapping,
    map_from_index,
    parse_grib_idx,
    store_coord_var,
    store_data_var,
    strip_datavar_chunks,
    _extract_single_group,
)


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
            # Handle cases where the logged message contains non-JSON-serializable data
            return f"{{\"timestamp\": \"{self.formatTime(record, self.datefmt)}\", \"level\": \"{record.levelname}\", \"function\": \"{record.funcName}\", \"line\": {record.lineno}, \"message\": \"{self.format_message(record.getMessage())}\"}}"

    def format_message(self, message):
        # Replace any newline characters with a space
        return message.replace('\n', ' ')

def setup_logging(log_level: int = logging.INFO, log_file: str = "gfs_processing.log"):
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

        # Capture print and stderr output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = func(*args, **kwargs)

            # Log the captured output
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





def build_grib_tree(gfs_files: List[str]) -> Tuple[dict, dict]:
    """
    Scan GFS files, build a hierarchical tree structure for the data, and strip unnecessary data.

    Parameters:
    - gfs_files (List[str]): List of file paths to GFS files.

    Returns:
    - Tuple[dict, dict]: Original and deflated GRIB tree stores.
    """
    print("Building Grib Tree")
    gfs_grib_tree_store = grib_tree([group for f in gfs_files for group in scan_grib(f)])
    deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
    strip_datavar_chunks(deflated_gfs_grib_tree_store)
    print(f"Original references: {len(gfs_grib_tree_store['refs'])}")
    print(f"Stripped references: {len(deflated_gfs_grib_tree_store['refs'])}")
    return gfs_grib_tree_store, deflated_gfs_grib_tree_store

def calculate_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time-related dimensions and coordinates based on input axes.

    Parameters:
    - axes (List[pd.Index]): List of pandas Index objects containing time information.

    Returns:
    - Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]: Time dimensions, coordinates, times, valid times, and steps.
    """
    print("Calculating Time Dimensions and Coordinates")
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


def process_dataframe(df, varnames_to_process):
    """
    Filter and process the DataFrame by specific variable names and their corresponding type of levels.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to process.
    - varnames_to_process (list): List of variable names to filter and process in the DataFrame.

    Returns:
    - pd.DataFrame: Processed DataFrame with duplicates removed based on the 'time' column and sorted by 'length'.
    """
    conditions = {
        'acpcp':'surface',
        'cape': 'surface',
        'cin': 'surface',
        'pres': 'heightAboveGround',
        'r': 'atmosphereSingleLayer',
        'soill': 'atmosphereSingleLayer',
        'soilw':'depthBelowLandLayer',  # Handling multiple levels for 'soill'
        'st': 'depthBelowLandLayer',
        't': 'surface',
        'tp': 'surface'
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


def create_mapped_index(axes: List[pd.Index], mapping_parquet_file_path: str, date_str: str) -> pd.DataFrame:
    """
    Create a mapped index from GFS files for a specific date, using the mapping from a parquet file.

    Parameters:
    - axes (List[pd.Index]): List of time axes to map.
    - mapping_parquet_file_path (str): File path to the mapping parquet file.
    - date_str (str): Date string for the data being processed.

    Returns:
    - pd.DataFrame: DataFrame containing the mapped index for the specified date.
    """
    print(f"Creating Mapped Index for date {date_str}")
    mapped_index_list = []
    dtaxes = axes[0]

    for idx, datestr in enumerate(dtaxes):
        try:
            fname = f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{idx:03}"
            
            idxdf = parse_grib_idx(
                fs=fsspec.filesystem("s3"),
                basename=fname
            )
            deduped_mapping = pd.read_parquet(f"{mapping_parquet_file_path}gfs-mapping-{idx:03}.parquet")
            mapped_index = map_from_index(
                datestr,
                deduped_mapping,
                idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
            )
            mapped_index_list.append(mapped_index)
        except Exception as e:
            logger.error(f"Error processing file {fname}: {str(e)}")

    gfs_kind = pd.concat(mapped_index_list)
    gfs_kind_var=gfs_kind.drop_duplicates('varname')
    var_list=gfs_kind_var['varname'].tolist() 
    var_to_remove=['acpcp','cape','cin','pres','r','soill','soilw','st','t','tp']
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))
    gfs_kind1=gfs_kind.loc[gfs_kind.varname.isin(var1_list)]
    #gfs_kind1 = gfs_kind.drop_duplicates('uri')
    # Process the data that needs to be filtered and modified
    to_process_df = gfs_kind[gfs_kind['varname'].isin(var_to_remove)]
    processed_df = process_dataframe(to_process_df, var_to_remove)
    # Concatenate the unprocessed and processed parts back together
    final_df = pd.concat([gfs_kind1, processed_df], ignore_index=True)
    # Optionally, you might want to sort or reorganize the DataFrame
    final_df = final_df.sort_values(by=['time', 'varname'])
    final_df_var=final_df.drop_duplicates('varname')
    final_var_list=final_df_var['varname'].tolist() 

    print(f"Mapped collected multiple variables index info: {len(final_var_list)} and {final_var_list}")
    return final_df

def cs_create_mapped_index(axes: List[pd.Index], gcs_bucket_name: str, date_str: str) -> pd.DataFrame:
    """
    Create a mapped index from GFS files for a specific date, using the mapping from a parquet file stored in GCS.
    
    Parameters:
    - axes (List[pd.Index]): List of time axes to map.
    - gcs_bucket_name (str): Name of the GCS bucket containing mapping files.
    - date_str (str): Date string for the data being processed (format: YYYYMMDD).
    
    Returns:
    - pd.DataFrame: DataFrame containing the mapped index for the specified date.
    """
    #logger = logging.getLogger()
    mapped_index_list = []
    dtaxes = axes[0]

    # Convert date_str to first day of the month
    first_day_of_month = pd.to_datetime(date_str).replace(day=1).strftime('%Y%m%d')
    
    # Initialize GCS filesystem
    gcs_fs = fsspec.filesystem('gcs')
    
    for idx, datestr in enumerate(dtaxes):
        try:
            # S3 path for the GFS data
            fname = f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{idx:03}"
            
            # Parse the idx file from S3
            idxdf = parse_grib_idx(
                fs=fsspec.filesystem("s3"),
                basename=fname
            )
            
            # Construct GCS path for mapping file
            gcs_mapping_path = f"gs://{gcs_bucket_name}/time_idx/20231201/agfs-time-20231201-rt{idx:03}.parquet"
            
            # Read parquet directly from GCS using fsspec
            deduped_mapping = pd.read_parquet(
                gcs_mapping_path,
                filesystem=gcs_fs
            )
            
            mapped_index = map_from_index(
                datestr,
                deduped_mapping,
                idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
            )
            mapped_index_list.append(mapped_index)
            
            logger.info(json.dumps({
                "event": "file_processed",
                "date": date_str,
                "file_index": idx,
                "mapping_file": gcs_mapping_path
            }))
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "error_processing_file",
                "date": date_str,
                "file_index": idx,
                "error": str(e)
            }))
            continue

    if not mapped_index_list:
        raise ValueError(f"No valid mapped indices created for date {date_str}")

    # Combine all mapped indices
    gfs_kind = pd.concat(mapped_index_list)
    
    # Get unique variables
    gfs_kind_var = gfs_kind.drop_duplicates('varname')
    var_list = gfs_kind_var['varname'].tolist()
    
    # Define variables to process separately
    var_to_remove = ['acpcp', 'cape', 'cin', 'pres', 'r', 'soill', 'soilw', 'st', 't', 'tp']
    
    # Filter variables
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))
    gfs_kind1 = gfs_kind.loc[gfs_kind.varname.isin(var1_list)]
    
    # Process special variables
    to_process_df = gfs_kind[gfs_kind['varname'].isin(var_to_remove)]
    processed_df = process_dataframe(to_process_df, var_to_remove)
    
    # Combine processed and unprocessed data
    final_df = pd.concat([gfs_kind1, processed_df], ignore_index=True)
    final_df = final_df.sort_values(by=['time', 'varname'])
    
    # Get final variable list for logging
    final_df_var = final_df.drop_duplicates('varname')
    final_var_list = final_df_var['varname'].tolist()
        
    return final_df

def prepare_zarr_store(deflated_gfs_grib_tree_store: dict, gfs_kind: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """
    Prepare Zarr store and related data for chunk processing based on GFS kind DataFrame.

    Parameters:
    - deflated_gfs_grib_tree_store (dict): Deflated GRIB tree store containing reference data.
    - gfs_kind (pd.DataFrame): DataFrame containing GFS data.

    Returns:
    - Tuple[dict, pd.DataFrame]: Zarr reference store and the DataFrame for chunk index.
    """
    print("Preparing Zarr Store")
    zarr_ref_store = deflated_gfs_grib_tree_store
    #chunk_index = gfs_kind.loc[gfs_kind.varname.isin(["t2m"])]
    chunk_index = gfs_kind
    zstore = copy.deepcopy(zarr_ref_store["refs"])
    return zstore, chunk_index


def process_unique_groups(zstore: dict, chunk_index: pd.DataFrame, time_dims: Dict, time_coords: Dict,
                          times: np.ndarray, valid_times: np.ndarray, steps: np.ndarray) -> dict:
    """
    Process and update Zarr store by configuring data for unique variable groups. This involves setting time dimensions,
    coordinates, and updating Zarr store paths with processed data arrays.

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
    
    This function processes each unique combination of 'varname', 'stepType', and 'typeOfLevel' found in the chunk_index.
    For each group, it determines appropriate dimensions and coordinates based on the unique levels present and updates
    the Zarr store with the processed data. It removes any data references that do not match the existing unique groups.
    """
    print("Processing Unique Groups and Updating Zarr Store")
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

            # Store coordinates and data variables in the Zarr store
            store_coord_var(key=f"{base_path}/time", zstore=zstore, coords=time_coords["time"], data=times.astype("datetime64[s]"))
            store_coord_var(key=f"{base_path}/valid_time", zstore=zstore, coords=time_coords["valid_time"], data=valid_times.astype("datetime64[s]"))
            store_coord_var(key=f"{base_path}/step", zstore=zstore, coords=time_coords["step"], data=steps.astype("timedelta64[s]").astype("float64") / 3600.0)
            store_coord_var(key=f"{base_path}/{key[2]}", zstore=zstore, coords=(key[2],) if lvals.shape else (), data=lvals)

            store_data_var(key=f"{base_path}/{key[0]}", zstore=zstore, dims=dims, coords=coords, data=group, steps=steps, times=times, lvals=lvals if lvals.shape else None)
        except Exception as e:
            print(f"Error processing group {key}: {str(e)}")

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
        # Convert dictionaries, lists, or numeric types to UTF-8 encoded strings
        if isinstance(value, (dict, list, int, float, np.integer, np.floating)):
            value = str(value).encode('utf-8')
        data.append((key, value))
    return pd.DataFrame(data, columns=['key', 'value'])


def create_parquet_file(zstore: dict, output_parquet_file: str):
    """
    Converts a dictionary containing Zarr store data to a DataFrame and saves it as a Parquet file.

    This function encapsulates the Zarr store data within a dictionary, converts this dictionary to a pandas DataFrame,
    and then writes the DataFrame to a Parquet file. This is useful for persisting Zarr metadata and references
    in a compressed and efficient format that can be easily reloaded.

    Parameters:
    - zstore (dict): The Zarr store dictionary containing all references and data needed for Zarr operations.
    - output_parquet_file (str): The path where the Parquet file will be saved.

    This function first creates an internal dictionary that includes versioning information, then iterates over
    the items in the Zarr store. For each item, it checks if the value is a dictionary, list, or a numeric type,
    and encodes it as a UTF-8 string if necessary. This encoded data is then used to create a DataFrame, which
    is subsequently written to a Parquet file. The function logs both the beginning of the operation and its
    successful completion, noting the location of the saved Parquet file.
    """
    gfs_store = dict(refs=zstore, version=1)  # Include versioning for the store structure  
    zstore_df = zstore_dict_to_df(gfs_store)
    zstore_df.to_parquet(output_parquet_file)
    print(f"Parquet file saved to {output_parquet_file}")


def create_parquet_df(zstore: dict, date_str: str, run_str: str, source: str = "aws_s3") -> pd.DataFrame:
    """
    Converts a dictionary containing Zarr store data to a DataFrame with additional columns.

    This function encapsulates the Zarr store data within a dictionary,
    converts this dictionary to a pandas DataFrame, and returns the DataFrame.
    Additional columns are added for metadata such as date, run, and source.

    Parameters:
    - zstore (dict): The Zarr store dictionary containing all references
      and data needed for Zarr operations.
    - date_str (str): A string representing the date to be added to the DataFrame.
    - run_str (str): A string representing the run to be added to the DataFrame.
    - source (str): A string representing the data source to be added to the DataFrame.
      Defaults to "aws_s3".

    Returns:
    - pd.DataFrame: The resulting DataFrame representing the Zarr store data
      with additional metadata columns.
    """
    gfs_store = dict(refs=zstore, version=1)  # Include versioning for the store structure
    zstore_df = zstore_dict_to_df(gfs_store)

    # Add additional metadata columns
    zstore_df["date"] = date_str
    zstore_df["run"] = run_str
    zstore_df["source"] = source

    return zstore_df



def generate_axes(date_str: str) -> List[pd.Index]:
    """
    Generate temporal axes indices for a given forecast start date over a predefined forecast period.
    
    This function creates two pandas Index objects: one for 'valid_time' and another for 'time'. 
    The 'valid_time' index represents a sequence of datetime stamps for each hour over a 5-day forecast period, 
    starting from the given start date. The 'time' index captures the single forecast initiation date.

    Parameters:
    - date_str (str): The start date of the forecast, formatted as 'YYYYMMDD'.

    Returns:
    - List[pd.Index]: A list containing two pandas Index objects:
      1. 'valid_time' index with datetime stamps spaced one hour apart, covering a 5-day range from the start date.
      2. 'time' index representing the single start date of the forecast as a datetime object.

    Example:
    For a given start date '20230101', this function will return two indices:
    - The first index will span from '2023-01-01 00:00' to '2023-01-06 00:00' with hourly increments.
    - The second index will contain just the single datetime '2023-01-01 00:00'.

    These indices are typically used to set up time coordinates in weather or climate models and datasets,
    facilitating data alignment and retrieval based on forecast times.
    """
    start_date = pd.Timestamp(date_str)
    end_date = start_date + pd.Timedelta(days=5)  # Forecast period of 5 days

    valid_time_index = pd.date_range(start_date, end_date, freq="60min", name="valid_time")
    time_index = pd.Index([start_date], name="time")

    return [valid_time_index, time_index]




def generate_gfs_dates(year: int, month: int) -> List[str]:
    """
    Generate a list of dates for a specific month and year, formatted as 'YYYYMMDD', to cover the full range of days in the specified month.

    This function computes the total number of days in the given month of the specified year and generates a complete list of dates.
    This is particularly useful for scheduling tasks or simulations that require a complete temporal scope of a month for processes like
    weather forecasting or data collection where daily granularity is needed.

    Parameters:
    - year (int): The year for which the dates are to be generated.
    - month (int): The month for which the dates are to be generated, where 1 represents January and 12 represents December.

    Returns:
    - List[str]: A list of dates in the format 'YYYYMMDD' for every day in the specified month and year.

    Example:
    For inputs year=2023 and month=1, the output will be:
    ['20230101', '20230102', '20230103', ..., '20230130', '20230131']
    
    This example demonstrates generating dates for January 2023, resulting in a list that includes every day from January 1st to 31st.
    """
    # Get the last day of the month using the monthrange function from the calendar module
    _, last_day = monthrange(year, month)
    
    # Generate a date range for the entire month from the first to the last day
    date_range = pd.date_range(start=f'{year}-{month:02d}-01', 
                               end=f'{year}-{month:02d}-{last_day}', 
                               freq='D')
    
    # Convert the date range to a list of strings formatted as 'YYYYMMDD'
    return date_range.strftime('%Y%m%d').tolist()





def process_gfs_data(date_str: str, mapping_parquet_file_path: str, output_parquet_file: str, log_level: int = logging.INFO):
    """
    Orchestrates the end-to-end processing of Global Forecast System (GFS) data for a specific date. This function
    integrates several steps including reading GFS files, calculating time dimensions, mapping data, preparing Zarr
    stores, processing unique data groups, and finally saving the processed data to a Parquet file.

    This function is designed to automate the workflow for daily GFS data processing, ensuring that each step is
    logged and any issues are reported for troubleshooting.

    Parameters:
    - date_str (str): A date string in the format 'YYYYMMDD' representing the date for which GFS data is to be processed.
    - mapping_parquet_file_path (str): Path to the parquet file that contains mapping information for the GFS data.
    - output_parquet_file (str): Path where the output Parquet file will be saved after processing the data.
    - log_level (int): Logging level to use for reporting within this function. Default is logging.INFO.

    Workflow:
    1. Set up logging based on the specified log level.
    2. Generate time axes for the date specified by `date_str`.
    3. List GFS file paths for the initial and subsequent model outputs.
    4. Build a GFS grib tree for storing and managing data hierarchically.
    5. Calculate the time dimensions and coordinates necessary for data processing.
    6. Create a mapped index based on the generated axes and specified parquet mapping file.
    7. Prepare the Zarr store for efficient data manipulation and storage.
    8. Process unique data groups, organizing and formatting the data as required.
    9. Save the processed data to a Parquet file for durable storage.

    Exceptions:
    - Raises an exception if an error occurs during the processing steps, with the error logged for diagnostic purposes.

    Example:
    To process GFS data for January 1st, 2021, call the function as follows:
    process_gfs_data('20210101', '/path/to/mapping.parquet/', '/output/path/20210101_processed.parquet')
    
    This will execute the entire data processing pipeline for the specified date and save the results in the designated output file.
    """
        
    try:
        print(f"Processing date: {date_str}")
        axes = generate_axes(date_str)
        gfs_files = [
            f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f000",
            f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f001"
        ]
        _, deflated_gfs_grib_tree_store = build_grib_tree(gfs_files)
        time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
        gfs_kind = create_mapped_index(axes, mapping_parquet_file_path, date_str)
        
        zstore, chunk_index = prepare_zarr_store(deflated_gfs_grib_tree_store, gfs_kind)
        updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, times, valid_times, steps)
        create_parquet_file(updated_zstore, output_parquet_file)
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        raise

def get_details(url):
    """Extract date and time details from GFS URL."""
    pattern = r"s3://noaa-gfs-bdp-pds/gfs\.(\d{8})/(\d{2})/atmos/gfs\.t(\d{2})z\.pgrb2\.0p25\.f(\d{3})"
    match = re.match(pattern, url)
    if match:
        date = match.group(1)  # Captures '20241010'
        run = match.group(2)   # Captures '00'
        hour = match.group(4)  # Captures '003'
        return date, run, hour
    else:
        logger.warning(f"No match found for URL pattern: {url}")
        return None, None, None

def gfs_s3_url_maker(date_str):
    """Create S3 URLs for GFS data."""
    fs_s3 = fsspec.filesystem("s3", anon=True)
    s3url_glob = fs_s3.glob(
        f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f*"
    )
    s3url_only_grib = [f for f in s3url_glob if f.split(".")[-1] != "idx"]
    fmt_s3og = sorted(["s3://" + f for f in s3url_only_grib])
    print(f"Generated {len(fmt_s3og)} URLs for date {date_str}")
    return fmt_s3og


def get_filename_from_path(file_path):
    """Extract filename from full path"""
    return os.path.basename(file_path)

def nonclusterworker_upload_to_gcs(bucket_name, source_file_name, destination_blob_name, dask_worker_credentials_path):
    """Uploads a file to the GCS bucket using provided service account credentials."""
    try:
        # Get just the filename from the credentials path
        #creds_filename = get_filename_from_path(credentials_path)
        # Construct the worker-local path
        #worker_creds_path = os.path.join(os.getcwd(), creds_filename)
        #credentials_path = "/app/coiled-data-key.json"
        #credentials_path = os.path.join(os.getcwd(), creds_filename)
        #credentials_path = os.path.join(tempfile.gettempdir(), creds_filename)
        print(f"Using credentials file at: {dask_worker_credentials_path}")
        
        if not os.path.exists(dask_worker_credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {dask_worker_credentials_path}")
            
        storage_client = storage.Client.from_service_account_json(dask_worker_credentials_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload file to GCS: {str(e)}")
        raise

def get_worker_creds_path(dask_worker):
    return str(pathlib.Path(dask_worker.local_directory) / 'coiled-data-key.json')


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket using provided service account credentials."""
    try:
        # Get the worker's local directory path for credentials
                    
        # Get current worker's credentials path
        worker = get_worker()
        worker_creds_path = get_worker_creds_path(worker)
        
        print(f"Using credentials file at: {worker_creds_path}")
        
        storage_client = storage.Client.from_service_account_json(worker_creds_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload file to GCS: {str(e)}")
        raise


@dask.delayed
def process_gfs_time_idx_data(s3url, bucket_name):
    """Process GFS data and upload to GCS."""
    # Ensure logger is set up in the worker process
    worker_logger = setup_logger()
    
    try:
        worker_logger.info(f"Processing: {s3url}")
        date_str, runz, runtime = get_details(s3url)
        #worker_creds_path = os.path.join(dask_worker.local_directory, credentials_path)
        #worker_logger.info(f"Using credentials from: {worker_creds_path}")
        
        if not all([date_str, runz, runtime]):
            worker_logger.error(f"Invalid URL format: {s3url}")
            return False

        # Build mapping for the specified runtime
        mapping = build_idx_grib_mapping(
            fs=fsspec.filesystem("s3"),
            basename=s3url 
        )
        deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
        deduped_mapping.set_index('attrs', inplace=True)
        
        # Save deduped mapping as Parquet
        output_dir = f"gfs_mapping_{date_str}"
        os.makedirs(output_dir, exist_ok=True)
        parquet_path = os.path.join(output_dir, f"gfs-time-{date_str}-rt{int(runtime):03}.parquet")
        deduped_mapping.to_parquet(parquet_path, index=True)
        
        # Upload to GCS
        destination_blob_name = f"time_idx/2023/{date_str}/{os.path.basename(parquet_path)}"
        upload_to_gcs(bucket_name, parquet_path, destination_blob_name)
        
        # Cleanup
        os.remove(parquet_path)
        worker_logger.info(f"Data for {date_str} runtime {runtime} has been processed and uploaded successfully.")
        return True
    except Exception as e:
        worker_logger.error(f"Failed to process data for URL {s3url}: {str(e)}")
        worker_logger.error(traceback.format_exc())
        raise


#@dask.delayed
def logged_process_gfs_time_idx_data(s3url, bucket_name):
    """Process GFS data and upload to GCS, logging results to individual GCS logs."""
    
    # Parse date_str and other details from the URL
    date_str, runz, runtime = get_details(s3url)
    year=date_str[0:3]
    
    # Create a temporary log file for this specific URL with date_str in the name
    with tempfile.NamedTemporaryFile(mode="w+", suffix=f"_{date_str}_{runtime}.log", delete=False) as log_file:
        log_filename = log_file.name
        worker_logger = setup_logger(log_file.name)  # Log messages will go to this file

        try:
            worker_logger.info(f"Processing: {s3url}")

            if not all([date_str, runz, runtime]):
                worker_logger.error(f"Invalid URL format: {s3url}")
                return False

            # Build mapping for the specified runtime
            mapping = build_idx_grib_mapping(
                fs=fsspec.filesystem("s3"),
                basename=s3url 
            )
            deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
            deduped_mapping.set_index('attrs', inplace=True)

            # Save deduped mapping as Parquet
            output_dir = f"gfs_mapping_{date_str}"
            os.makedirs(output_dir, exist_ok=True)
            parquet_path = os.path.join(output_dir, f"gfs-time-{date_str}-rt{int(runtime):03}.parquet")
            deduped_mapping.to_parquet(parquet_path, index=True)

            # Upload to GCS
            destination_blob_name = f"time_idx/year/{date_str}/{os.path.basename(parquet_path)}"
            upload_to_gcs(bucket_name, parquet_path, destination_blob_name)

            # Cleanup
            os.remove(parquet_path)
            worker_logger.info(f"Data for {date_str} runtime {runtime} has been processed and uploaded successfully.")
            process_success = True
        except Exception as e:
            worker_logger.error(f"Failed to process data for URL {s3url}: {str(e)}")
            worker_logger.error(traceback.format_exc())
            process_success = False
        finally:
            # Upload log file to GCS for later inspection
            gcs_log_path = f"time_idx/2023/logs/{date_str}/{os.path.basename(log_filename)}"
            upload_to_gcs(bucket_name, log_filename, gcs_log_path)
            os.remove(log_filename)  # Remove the temporary log file after uploading

        return process_success


def s3_ecmwf_build_idx_grib_mapping(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    date_str: str,
    idx:int,
    suffix: str = "index",
    mapper: Optional[Callable] = None,
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Mapping method combines the idx and grib metadata to make a mapping from one to the other for a particular
    model horizon file. This should be generally applicable to all forecasts for the given horizon.
    :param fs: the file system to read metatdata from
    :param basename: the full path for the grib2 file
    :param suffix: the suffix for the index file
    :param mapper: the mapper if any to apply (used for hrrr subhf)
    :param tstamp: the timestamp to use for when the data was indexed
    :param validate: assert mapping is correct or fail before returning
    :return: the merged dataframe with the results of the two operations joined on the grib message group number
    """
    #grib_file_index = _map_grib_file_by_group(fname=basename, mapper=mapper)
    grib_file_index = pd.read_parquet(f'{date_str}/ecmwf_scangrib_metadata_table_{date_str}_{idx}.parquet')
    idx_file_index = s3_parse_ecmwf_grib_idx(
        fs=fs, basename=basename, suffix=suffix, tstamp=tstamp
    )
    result = idx_file_index.merge(
        # Left merge because the idx file should be authoritative - one record per grib message
        grib_file_index,
        on="idx",
        how="left",
        suffixes=("_idx", "_grib"),
    )

    if validate:
        # If any of these conditions fail - run the method in colab for the same file and inspect the result manually.
        all_match_offset = (
            (result.loc[:, "offset_idx"] == result.loc[:, "offset_grib"])
            | pd.isna(result.loc[:, "offset_grib"])
            | ~pd.isna(result.loc[:, "inline_value"])
        )
        all_match_length = (
            (result.loc[:, "length_idx"] == result.loc[:, "length_grib"])
            | pd.isna(result.loc[:, "length_grib"])
            | ~pd.isna(result.loc[:, "inline_value"])
        )

        if not all_match_offset.all():
            vcs = all_match_offset.value_counts()
            raise ValueError(
                f"Failed to match message offset mapping for grib file {basename}: {vcs[True]} matched, {vcs[False]} didn't"
            )

        if not all_match_length.all():
            vcs = all_match_length.value_counts()
            raise ValueError(
                f"Failed to match message length mapping for grib file {basename}: {vcs[True]} matched, {vcs[False]} didn't"
            )

        if not result["attrs"].is_unique:
            dups = result.loc[result["attrs"].duplicated(keep=False), :]
            logger.warning(
                "The idx attribute mapping for %s is not unique for %d variables: %s",
                basename,
                len(dups),
                dups.varname.tolist(),
            )

        r_index = result.set_index(
            ["varname", "typeOfLevel", "stepType", "level", "valid_time"]
        )
        if not r_index.index.is_unique:
            dups = r_index.loc[r_index.index.duplicated(keep=False), :]
            logger.warning(
                "The grib hierarchy in %s is not unique for %d variables: %s",
                basename,
                len(dups),
                dups.index.get_level_values("varname").tolist(),
            )

    return result


def s3_parse_ecmwf_grib_idx(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    suffix: str = "index",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Standalone method used to extract metadata from a grib2 index file

    :param fs: the file system to read from
    :param basename: the base name is the full path to the grib file
    :param suffix: the suffix is the ending for the index file
    :param tstamp: the timestamp to record for this index process
    :return: the data frame containing the results
    """
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"

    fs.invalidate_cache(fname)
    fs.invalidate_cache(basename)

    baseinfo = fs.info(basename)

    with fs.open(fname, "r") as f:
        splits = []
        for idx, line in enumerate(f):
            try:
                # Removing the trailing characters if there's any at the end of the line
                clean_line = line.strip().rstrip(',')
                # Convert the JSON-like string to a dictionary
                data = json.loads(clean_line)
                # Extracting required fields using .get() method to handle missing keys
                lidx = idx
                offset = data.get("_offset", 0)  # Default to 0 if missing
                length = data.get("_length", 0)
                date = data.get("date", "Unknown Date")  # Default to 'Unknown Date' if missing
                ens_number = data.get("number", -1)  # Default to -1 if missing
                # Append to the list as integers or the original data type
                splits.append([int(lidx), int(offset),int(length), date, data, int(ens_number)])
            except json.JSONDecodeError as e:
                # Handle cases where JSON conversion fails
                raise ValueError(f"Could not parse JSON from line: {line}") from e

    result = pd.DataFrame(splits, columns=["idx", "offset", "length", "date", "attr", "ens_number"])

    # Subtract the next offset to get the length using the filesize for the last value 

    result.loc[:, "idx_uri"] = fname
    result.loc[:, "grib_uri"] = basename

    if tstamp is None:
        tstamp = pd.Timestamp.now()
    #result.loc[:, "indexed_at"] = tstamp
    result['indexed_at'] = result.apply(lambda x: tstamp, axis=1)

    # Check for S3 or GCS filesystem instances to handle metadata
    if "s3" in fs.protocol:
        # Use ETag as the S3 equivalent to crc32c
        result.loc[:, "grib_etag"] = baseinfo.get("ETag")
        result.loc[:, "grib_updated_at"] = pd.to_datetime(
            baseinfo.get("LastModified")
        ).tz_localize(None)

        idxinfo = fs.info(fname)
        result.loc[:, "idx_etag"] = idxinfo.get("ETag")
        result.loc[:, "idx_updated_at"] = pd.to_datetime(
            idxinfo.get("LastModified")
        ).tz_localize(None)
    else:
        # TODO: Fix metadata for other filesystems
        result.loc[:, "grib_crc32"] = None
        result.loc[:, "grib_updated_at"] = None
        result.loc[:, "idx_crc32"] = None
        result.loc[:, "idx_updated_at"] = None

    if validate and not result["attrs"].is_unique:
        raise ValueError(f"Attribute mapping for grib file {basename} is not unique)")
    print(f'Completed index files and found {len(result.index)} entries in it')
    return result.set_index("idx")


class KerchunkZarrDictStorageManager:
    """Manages storage and retrieval of Kerchunk Zarr dictionaries in Google Cloud Storage."""
    
    def __init__(
        self, 
        bucket_name: str,
        service_account_file: Optional[str] = None,
        service_account_info: Optional[Dict] = None
    ):
        """
        Initialize the storage manager with GCP credentials.
        
        Args:
            bucket_name (str): Name of the GCS bucket
            service_account_file (str, optional): Path to service account JSON file
            service_account_info (dict, optional): Service account info as dictionary
            
        Note:
            Provide either service_account_file OR service_account_info, not both.
            If neither is provided, defaults to application default credentials.
        """
        self.bucket_name = bucket_name
        
        # Handle credentials
        if service_account_file and service_account_info:
            raise ValueError("Provide either service_account_file OR service_account_info, not both")
        
        if service_account_file:
            # Load credentials from file
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            # Load the JSON content for fsspec
            with open(service_account_file, 'r') as f:
                self.service_account_info = json.load(f)
        elif service_account_info:
            # Use provided service account info
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.service_account_info = service_account_info
        else:
            credentials = None
            self.service_account_info = None
        
        # Initialize storage client
        self.storage_client = storage.Client(credentials=credentials)
        
        # Initialize fsspec filesystem with the same credentials
        if self.service_account_info:
            self.gcs_fs = fsspec.filesystem(
                'gcs', 
                token=self.service_account_info
            )
        else:
            self.gcs_fs = fsspec.filesystem('gcs')    

    def _process_zarr_value(self, value: Any) -> Any:
        """Process Zarr store values for storage."""
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(value).decode('utf-8')
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    def _restore_zarr_value(self, value: str, key: str) -> Any:
        """Restore original Zarr store value from stored format."""
        try:
            # Try to parse as JSON first (for lists and dicts)
            return json.loads(value)
        except json.JSONDecodeError:
            # Check if it's base64 encoded
            try:
                return base64.b64decode(value.encode('utf-8'))
            except:
                # Return as is if not base64
                return value

    def save_zarr_store(self, zarr_store: Dict[str, Any], filepath: str, 
                       chunk_size: int = 10000) -> None:
        """
        Save Kerchunk Zarr store dictionary to GCS.
        
        Args:
            zarr_store: Zarr store dictionary from grib_tree
            filepath: GCS path (e.g., 'gs://bucket/path/zarr_store.parquet')
            chunk_size: Number of keys per chunk for large stores
        """
        # Convert zarr store to rows
        rows = []
        for key, value in zarr_store.items():
            processed_value = self._process_zarr_value(value)
            rows.append({
                'key': key,
                'value': processed_value,
                'value_type': type(value).__name__
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Save in chunks if necessary
        if len(df) > chunk_size:
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_df = df.iloc[i*chunk_size:(i+1)*chunk_size]
                chunk_table = pa.Table.from_pandas(chunk_df)
                chunk_path = f"{filepath}.chunk{i}"
                
                pq.write_table(
                    chunk_table,
                    chunk_path,
                    filesystem=self.gcs_fs,
                    compression='ZSTD',
                    use_dictionary=True
                )
            
            # Save metadata
            metadata = {
                'num_chunks': num_chunks,
                'total_keys': len(df),
                'schema_version': '1.0'
            }
            self.gcs_fs.write_text(
                f"{filepath}.metadata",
                json.dumps(metadata)
            )
        else:
            # Save as single file
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                filepath,
                filesystem=self.gcs_fs,
                compression='ZSTD',
                use_dictionary=True
            )

    def load_zarr_store(self, filepath: str) -> Dict[str, Any]:
        """
        Load Kerchunk Zarr store dictionary from GCS.
        
        Args:
            filepath: GCS path
        
        Returns:
            Dict representing the Zarr store
        """
        # Check if it's a chunked store
        try:
            metadata = json.loads(self.gcs_fs.read_text(f"{filepath}.metadata"))
            is_chunked = True
        except:
            is_chunked = False
        
        if is_chunked:
            # Load all chunks
            dfs = []
            for i in range(metadata['num_chunks']):
                chunk_path = f"{filepath}.chunk{i}"
                chunk_table = pq.read_table(chunk_path, filesystem=self.gcs_fs)
                dfs.append(chunk_table.to_pandas())
            df = pd.concat(dfs, ignore_index=True)
        else:
            # Load single file
            table = pq.read_table(filepath, filesystem=self.gcs_fs)
            df = table.to_pandas()
        
        # Reconstruct zarr store
        zarr_store = {}
        for _, row in df.iterrows():
            zarr_store[row['key']] = self._restore_zarr_value(row['value'], row['key'])
        
        return zarr_store

    def verify_zarr_store(self, zarr_store: Dict[str, Any], filepath: str) -> bool:
        """
        Verify the integrity of a saved Zarr store.
        
        Args:
            zarr_store: Original Zarr store dictionary
            filepath: GCS path where it was saved
            
        Returns:
            bool: True if verification passes
        """
        loaded_store = self.load_zarr_store(filepath)
        
        # Compare keys
        if set(zarr_store.keys()) != set(loaded_store.keys()):
            return False
        
        # Compare values
        for key in zarr_store:
            if isinstance(zarr_store[key], (list, dict)):
                if json.dumps(zarr_store[key]) != json.dumps(loaded_store[key]):
                    return False
            elif isinstance(zarr_store[key], (bytes, bytearray)):
                if zarr_store[key] != loaded_store[key]:
                    return False
            elif str(zarr_store[key]) != str(loaded_store[key]):
                return False
        
        return True



async def old_process_single_file(
    date_str: str,
    first_day_of_month: str,
    gcs_bucket_name: str,
    idx: int,
    datestr: pd.Timestamp,
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    chunk_size: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Process a single file asynchronously with memory management.
    
    Parameters:
    - date_str: The date string for the GFS data
    - first_day_of_month: First day of the month for mapping files
    - gcs_bucket_name: GCS bucket name
    - idx: File index
    - datestr: Timestamp for the data
    - sem: Semaphore for controlling concurrent operations
    - executor: ThreadPoolExecutor for CPU-bound operations
    - chunk_size: Optional size for chunked reading
    """
    async with sem:  # Control concurrent operations
        try:
            # S3 path for GFS data
            fname = f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{idx:03}"
            #gcs_mapping_path = f"gs://{gcs_bucket_name}/gfs_mapping/{first_day_of_month}/gfs-mapping-{idx:03}.parquet"
            gcs_mapping_path = f"gs://{gcs_bucket_name}/time_idx/20231201/agfs-time-20231201-rt{idx:03}.parquet"

            
            # Use ThreadPoolExecutor for I/O operations
            loop = asyncio.get_event_loop()
            
            # Read idx file
            idxdf = await loop.run_in_executor(
                executor,
                partial(parse_grib_idx, fs=fsspec.filesystem("s3"), basename=fname)
            )
            
            # Initialize GCS filesystem
            gcs_fs = fsspec.filesystem('gcs')
            
            # Read parquet in chunks if chunk_size is specified
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
            print(f'Error in {str(e)}')
            return None

async def process_single_file(
    date_str: str,
    first_day_of_month: str,
    gcs_bucket_name: str,
    idx: int,
    datestr: pd.Timestamp,
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    gcp_service_account_json: str,
    chunk_size: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Process a single file asynchronously with memory management.
    
    Additional Parameter:
    - gcp_service_account_json: Path to the GCP service account JSON file.
    """
    async with sem:  # Control concurrent operations
        try:
            fname = f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{idx:03}"
            gcs_mapping_path = f"gs://{gcs_bucket_name}/time_idx/20231201/agfs-time-20231201-rt{idx:03}.parquet"
            
            # Initialize GCS filesystem with credentials
            gcs_fs = gcsfs.GCSFileSystem(token=gcp_service_account_json)
            
            loop = asyncio.get_event_loop()
            
            # Read idx file
            idxdf = await loop.run_in_executor(
                executor,
                partial(parse_grib_idx, fs=fsspec.filesystem("s3"), basename=fname)
            )
            
            # Read parquet in chunks if chunk_size is specified
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
            print(f'Error in {str(e)}')
            return None



async def process_files_in_batches(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    max_concurrent: int = 3,
    batch_size: int = 5,
    chunk_size: Optional[int] = None,
    gcp_service_account_json: Optional[str] = None  # Add the parameter here
) -> pd.DataFrame:
    """
    Process files in batches with controlled concurrency and memory usage.
    
    Additional Parameter:
    - gcp_service_account_json: Path to the GCP service account JSON file.
    """
    dtaxes = axes[0]
    first_day_of_month = pd.to_datetime(date_str).replace(day=1).strftime('%Y%m%d')
    
    # Create semaphore for controlling concurrent operations
    sem = asyncio.Semaphore(max_concurrent)
    
    # Initialize thread pool executor
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        all_results = []
        
        # Process in batches
        for batch_start in range(0, len(dtaxes), batch_size):
            batch_end = min(batch_start + batch_size, len(dtaxes))
            batch_indices = range(batch_start, batch_end)
            
            # Create tasks for current batch
            tasks = [
                process_single_file(
                    date_str,
                    first_day_of_month,
                    gcs_bucket_name,
                    idx,
                    dtaxes[idx],
                    sem,
                    executor,
                    gcp_service_account_json,  # Pass the parameter here
                    chunk_size
                )
                for idx in batch_indices
            ]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks)
            
            # Filter out None results and extend all_results
            valid_results = [r for r in batch_results if r is not None]
            all_results.extend(valid_results)
            
            # Optional: Clear memory after each batch
            batch_results = None
            
            print("event batch_processed")
    
    # Combine results and process
    if not all_results:
        raise ValueError(f"No valid mapped indices created for date {date_str}")
    
    gfs_kind = pd.concat(all_results, ignore_index=True)
    
    # Process variables as before
    gfs_kind_var = gfs_kind.drop_duplicates('varname')
    var_list = gfs_kind_var['varname'].tolist()
    var_to_remove = ['acpcp', 'cape', 'cin', 'pres', 'r', 'soill', 'soilw', 'st', 't', 'tp']
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))
    
    # Split processing into smaller chunks if needed
    gfs_kind1 = gfs_kind.loc[gfs_kind.varname.isin(var1_list)]
    to_process_df = gfs_kind[gfs_kind['varname'].isin(var_to_remove)]
    processed_df = process_dataframe(to_process_df, var_to_remove)
    
    final_df = pd.concat([gfs_kind1, processed_df], ignore_index=True)
    final_df = final_df.sort_values(by=['time', 'varname'])
    
    return final_df

def cs_create_mapped_index(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    max_concurrent: int = 10,
    batch_size: int = 20,
    chunk_size: Optional[int] = None,
    gcp_service_account_json: Optional[str] = None  # Add the parameter here
) -> pd.DataFrame:
    """
    Async wrapper for creating mapped index with memory management.
    
    Additional Parameter:
    - gcp_service_account_json: Path to the GCP service account JSON file.
    """
    return asyncio.run(
        process_files_in_batches(
            axes,
            gcs_bucket_name,
            date_str,
            max_concurrent,
            batch_size,
            chunk_size,
            gcp_service_account_json  # Pass the parameter here
        )
    )


def aws_parse_grib_idx(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    suffix: str = "idx",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Standalone method used to extract metadata from a grib2 idx file from NODD
    :param fs: the file system to read from
    :param basename: the base name is the full path to the grib file
    :param suffix: the suffix is the ending for the idx file
    :param tstamp: the timestamp to record for this index process
    :return: the data frame containing the results
    """

    fname = f"{basename}.{suffix}"
    fs.invalidate_cache(fname)
    fs.invalidate_cache(basename)

    baseinfo = fs.info(basename)

    with fs.open(fname, "r") as f:
        splits = []
        for line in f.readlines():
            try:
                idx, offset, date, attrs = line.split(":", maxsplit=3)
                splits.append([int(idx), int(offset), date, attrs])
            except ValueError:
                # Wrap the ValueError in a new one that includes the bad line
                # If building the mapping, pick a different forecast run where the idx file is not broken
                # If indexing a forecast using the mapping, fall back to reading the grib file
                raise ValueError(f"Could not parse line: {line}")

    result = pd.DataFrame(
        data=splits,
        columns=["idx", "offset", "date", "attrs"],
    )

    # Subtract the next offset to get the length using the filesize for the last value
    result.loc[:, "length"] = (
        result.offset.shift(periods=-1, fill_value=baseinfo["size"]) - result.offset
    )

    result.loc[:, "idx_uri"] = fname
    result.loc[:, "grib_uri"] = basename
    if tstamp is None:
        tstamp = pd.Timestamp.now()
    result.loc[:, "indexed_at"] = tstamp

    if "s3" in fs.protocol:
        # Use ETag as the S3 equivalent to crc32c
        result.loc[:, "grib_etag"] = baseinfo.get("ETag")
        result.loc[:, "grib_updated_at"] = pd.to_datetime(
            baseinfo.get("LastModified")
        ).tz_localize(None)

        idxinfo = fs.info(fname)
        result.loc[:, "idx_etag"] = idxinfo.get("ETag")
        result.loc[:, "idx_updated_at"] = pd.to_datetime(
            idxinfo.get("LastModified")
        ).tz_localize(None)
    if "gcs" in fs.protocol:
        result.loc[:, "grib_crc32"] = baseinfo["crc32c"]
        result.loc[:, "grib_updated_at"] = pd.to_datetime(
            baseinfo["updated"]
        ).tz_localize(None)

        idxinfo = fs.info(fname)
        result.loc[:, "idx_crc32"] = idxinfo["crc32c"]
        result.loc[:, "idx_updated_at"] = pd.to_datetime(
            idxinfo["updated"]
        ).tz_localize(None)
    else:
        # TODO: Fix metadata for other filesystems
        result.loc[:, "grib_crc32"] = None
        result.loc[:, "grib_updated_at"] = None
        result.loc[:, "idx_crc32"] = None
        result.loc[:, "idx_updated_at"] = None

    if validate and not result["attrs"].is_unique:
        raise ValueError(f"Attribute mapping for grib file {basename} is not unique)")

    return result.set_index("idx")

def subset_list_by_indices(lst, id_val):
    """
    Subset a list by a list of index values.
    
    Parameters:
    lst (list): The original list to be subsetted.
    id_val (list): The list of index values to use for subsetting.
    
    Returns:
    list: The subsetted list containing only the elements at the specified indices.
    """
    return [lst[i] for i in id_val]

def map_forecast_to_indices(forecast_dict: dict, df: pd.DataFrame) -> Tuple[dict, list]:
    """
    Map each forecast variable in forecast_dict to the index in df where its corresponding value in 'attrs' is found.
    
    Parameters:
    - forecast_dict (dict): Dictionary with forecast variables as keys and search strings as values.
    - df (pd.DataFrame): DataFrame containing a column 'attrs' where search is performed.
    
    Returns:
    - Tuple[dict, list]: 
        - Dictionary mapping each forecast variable to the found index in df, adjusted by -1, or default_index if not found.
        - List of all indices found that are not equal to 9999.
    """
    output_dict = {}
    
    # Iterate over each key-value pair in forecast_dict
    for key, value in forecast_dict.items():
        # Check if `value` is present in any row of the 'attrs' column
        matching_rows = df[df['attrs'].str.contains(value, na=False)]
        
        # If there are matching rows, get the row index (adjusted by -1)
        if not matching_rows.empty:
            output_dict[key] = int(matching_rows.index[0] - 1)
        else:
            output_dict[key] = 9999
    
    # Generate the list of non-default indices
    values_list = [value for value in output_dict.values() if value != 9999]

    return output_dict, values_list

@log_function_call
def _map_grib_file_by_group(
    fname: str,
    total_groups:int,
    mapper: Optional[Callable] = None,
):
    """
    Helper method used to read the cfgrib metadata associated with each message (group) in the grib file
    This method does not add metadata
    :param fname: the file name to read with scan_grib
    :param mapper: the mapper if any to apply (used for hrrr subhf)
    :return: the pandas dataframe
    """
    mapper = (lambda x: x) if mapper is None else mapper
    #total_groups = 4233  # Your specified value
    
    start_time = time.time()
    print(f" Starting to process {total_groups} groups from file: {fname}")
    
    processed_groups = 0
    successful_groups = 0
    failed_groups = 0
    last_update_time = start_time

    def process_groups():
        nonlocal processed_groups, successful_groups, failed_groups, last_update_time
        
        for i, group in enumerate(scan_grib(fname), start=1):
            try:
                result = _extract_single_group(mapper(group), i)
                processed_groups += 1
                
                if result is not None:
                    successful_groups += 1
                else:
                    failed_groups += 1
                
                # Log progress every 10% or when processing single digits
                if total_groups <= 10 or processed_groups % max(1, total_groups // 10) == 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    time_since_last_update = current_time - last_update_time
                    last_update_time = current_time
                    
                    progress_percentage = (processed_groups / total_groups) * 100
                    groups_per_second = processed_groups / elapsed_time if elapsed_time > 0 else 0
                    
                    # Estimate remaining time
                    remaining_groups = total_groups - processed_groups
                    estimated_remaining_time = remaining_groups / groups_per_second if groups_per_second > 0 else 0
                    
                    print(
                        f"Progress: {processed_groups}/{total_groups} groups processed "
                        f"({progress_percentage:.1f}%) - "
                        f"Successful: {successful_groups}, Failed: {failed_groups}, "
                        f"Remaining: {total_groups - processed_groups} | "
                        f"Rate: {groups_per_second:.1f} groups/sec | "
                        f"Elapsed: {elapsed_time:.1f}s | "
                        f"Est. Remaining: {estimated_remaining_time:.1f}s",
                        end=""
                    )
                
                yield result
                
            except Exception as e:
                failed_groups += 1
                processed_groups += 1
                print(f"Error processing group {i}: {str(e)}")
                continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_df = pd.concat(
            list(
                filter(
                    lambda item: item is not None,
                    process_groups()
                )
            )
        ).set_index("idx")

    # Calculate final statistics
    end_time = time.time()
    total_time = end_time - start_time
    average_rate = processed_groups / total_time if total_time > 0 else 0

    # Print final summary with timestamp
    print(f"Completed processing {fname}:")
    print(f"Total groups: {total_groups}")
    print(f"Processed groups: {processed_groups}")
    print(f"Successful groups: {successful_groups}")
    print(f"Failed groups: {failed_groups}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing rate: {average_rate:.1f} groups/second")
    print(f"Success rate: {(successful_groups/processed_groups*100):.1f}%")

    return result_df

def zip_folder(folder_path, output_zip_path):
    """
    Compresses the contents of a folder into a zip file.

    Parameters:
    - folder_path (str): The path to the folder you want to compress.
    - output_zip_path (str): The path for the output zip file (without .zip extension).
    """
    # Ensure the output path does not include the .zip extension (it will be added automatically)
    output_zip_path = os.path.splitext(output_zip_path)[0]
    
    # Create the zip file
    shutil.make_archive(output_zip_path, 'zip', folder_path)
    print(f"Folder '{folder_path}' has been compressed to '{output_zip_path}.zip'")

def recreate_folder(folder_path):
    """
    Creates a new folder. If the folder already exists, it is removed and recreated.
    
    Parameters:
    - folder_path (str): The path to the folder to create.
    """
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the existing folder and its contents
        shutil.rmtree(folder_path)
    
    # Create the new folder
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' has been recreated.")


def filter_gfs_scan_grib(gurl,tofilter_cgan_var_dict):
    fs=fsspec.filesystem("s3")
    suffix= "idx"
    gsc = scan_grib(gurl)
    idx_gfs = aws_parse_grib_idx(fs=fs, basename=gurl, suffix=suffix)
    output_dict0, vl_gfs = map_forecast_to_indices(tofilter_cgan_var_dict, idx_gfs)
    return [gsc[i] for i in vl_gfs]


def filter_build_grib_tree(gfs_files: List[str],tofilter_cgan_var_dict: dict) -> Tuple[dict, dict]:
    """
    Scan GFS files, build a hierarchical tree structure for the data, and strip unnecessary data.

    Parameters:
    - gfs_files (List[str]): List of file paths to GFS files.

    Returns:
    - Tuple[dict, dict]: Original and deflated GRIB tree stores.
    """
    print("Building Grib Tree")
    sg_groups = [group for gurl in gfs_files for group in filter_gfs_scan_grib(gurl,tofilter_cgan_var_dict)]
    gfs_grib_tree_store = grib_tree(sg_groups)
    deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
    strip_datavar_chunks(deflated_gfs_grib_tree_store)
    print(f"Original references: {len(gfs_grib_tree_store['refs'])}")
    print(f"Stripped references: {len(deflated_gfs_grib_tree_store['refs'])}")
    return gfs_grib_tree_store, deflated_gfs_grib_tree_store




def download_parquet_from_gcs(
    gcs_bucket_name: str,
    year: str,
    date_str: str,
    run_str: str,
    service_account_json: str,
    local_save_path: str = "./"):
    """
    Downloads a Parquet file from GCS and saves it locally.

    Parameters:
    - gcs_bucket_name (str): The name of the GCS bucket.
    - year (str): The year of the data (e.g., "2024").
    - date_str (str): The date string in YYYYMMDD format.
    - run_str (str): The run string (e.g., "00", "06", "12", "18").
    - service_account_json (str): Path to the GCP service account JSON key file.
    - local_save_path (str): The local directory where the file should be saved. Defaults to "./".

    Returns:
    - str: The local path to the saved Parquet file.
    """
    # Create credentials object
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
    )

    # Construct the GCS file URL
    gcs_file_url = f"gs://{gcs_bucket_name}/gik_day_parqs/{year}/gfs_{date_str}_{run_str}.par"

    # Read Parquet file from GCS
    ddf = dd.read_parquet(
        gcs_file_url,
        storage_options={"token": credentials},
        engine="pyarrow",
    )
    pdf = ddf.compute()

    # Define the local file path
    local_file_path = os.path.join(local_save_path, f"gfs_{date_str}_{run_str}.par")

    # Save to local Parquet file
    pdf.to_parquet(local_file_path)

    print(f"Parquet file saved locally at: {local_file_path}")
    return local_file_path



def worker_upload_required_files(client, parquet_path: str, credentials_path: str):
    """Upload required files to all workers."""
    client.upload_file(credentials_path)
    client.upload_file(parquet_path)
    
def get_worker_paths(dask_worker, date_str: str):
    """Get paths for required files on worker."""
    local_dir = pathlib.Path(dask_worker.local_directory)
    return {
        'credentials': str(local_dir / 'coiled-data-key.json'),
        'parquet': str(local_dir / f'gfs_{date_str}.par')
    }

def load_datatree_on_worker(parquet_path: str):
    """Load datatree from parquet file on worker."""
    zstore_df = pd.read_parquet(parquet_path)
    zstore_df['value'] = zstore_df['value'].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
    )
    zstore = {row['key']: eval(row['value']) for _, row in zstore_df.iterrows()}
    return datatree.open_datatree(
        fsspec.filesystem("reference", fo=zstore).get_mapper(""),
        engine="zarr",
        consolidated=False
    )

def process_and_save(path: str, gcs_bucket: str, gcs_path: str, date_str: str):
    """Process a single path and save to GCS, appending if zarr store exists."""
    try:
        # Get worker-specific paths
        worker = get_worker()
        paths = get_worker_paths(worker, date_str)
        
        # Initialize GCS filesystem with proper credentials
        gcs = gcsfs.GCSFileSystem(
            token=paths['credentials'],
            project='your-project-id'  # Add your GCP project ID here
        )
        
        # Load datatree on worker
        gfs_dt = load_datatree_on_worker(paths['parquet'])
        
        # Process dataset
        ds = gfs_dt[path].to_dataset()
        if 'step' in ds.coords:
            ds = ds.drop_vars(['step'])
        if 'isobaricInhPa' in ds.coords:
            ds = ds.sel(isobaricInhPa=200)
            
        # Construct store path
        var_name = path.strip('/').replace('/', '_')
        store_path = f"{gcs_bucket}/{gcs_path}/{var_name}.zarr"
        
        # Check if zarr store already exists
        store_exists = gcs.exists(store_path)
        
        if store_exists:
            # If store exists, try to append
            try:
                existing_ds = xr.open_zarr(gcs.get_mapper(store_path))
                
                # Check if we have new times to append
                new_times = ds.valid_times.values
                existing_times = existing_ds.valid_times.values
                times_to_append = new_times[~np.isin(new_times, existing_times)]
                
                if len(times_to_append) > 0:
                    # Filter dataset to only new times
                    ds_to_append = ds.sel(valid_times=times_to_append)
                    
                    # Append to existing store
                    ds_to_append.to_zarr(
                        store=gcs.get_mapper(store_path),
                        mode='a',  # Append mode
                        append_dim='valid_times',  # Dimension to append along
                        consolidated=True
                    )
                    return f"Appended {len(times_to_append)} new times to gs://{store_path}"
                else:
                    return f"No new times to append for gs://{store_path}"
                    
            except Exception as e:
                # If append fails, log error and create new store
                print(f"Error appending to store: {e}")
                print("Creating new store instead")
                store_exists = False
        
        if not store_exists:
            # Create new store if doesn't exist or append failed
            ds.to_zarr(
                store=gcs.get_mapper(store_path),
                mode='w',  # Write mode (creates new store)
                consolidated=True
            )
            return f"Created new store at gs://{store_path}"
            
    except Exception as e:
        return f"Error processing {path}: {str(e)}"

def process_and_upload_datatree(
    parquet_path: str,
    gcs_bucket: str, 
    gcs_path: str,
    client: Client,
    credentials_path: str,
    date_str: str,
    project_id: str  # Add project_id parameter
) -> List[str]:
    """Process datatree and upload to GCS as Zarr stores."""
    # Upload required files to workers
    worker_upload_required_files(client, parquet_path, credentials_path)
    
    # Set GCS credentials for client
    client.run(lambda: gcsfs.GCSFileSystem(token='coiled-data-key.json', project=project_id))
    
    # Load datatree locally just to get paths
    temp_dt = load_datatree_on_worker(parquet_path)
    paths = [node.path for node in temp_dt.subtree if node.has_data]
    
    # Submit jobs in batches
    futures = []
    for path in paths:
        future = client.submit(
            process_and_save,
            path, 
            gcs_bucket, 
            gcs_path,
            date_str,
            pure=False
        )
        futures.append(future)
    
    return client.gather(futures)
