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

from enum import Enum, auto
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


import fsspec
import pandas as pd
import numpy as np
import copy
import json
import zarr
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable

#import datatree
from kerchunk.grib2 import scan_grib, grib_tree
from kerchunk.combine import MultiZarrToZarr

import fsspec
import pandas as pd
import ast
import json
import numpy as np
from typing import Optional, Dict, Any, List


import ast
import json
import pandas as pd
from typing import Optional

def generate_axes(date_str: str, fhr_min: str) -> List[pd.Index]:
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

    valid_time_index = pd.date_range(start_date, end_date, freq=fhr_min, name="valid_time")
    time_index = pd.Index([start_date], name="time")

    forecast_hours = ((valid_time_index - start_date).total_seconds() / 3600).astype(int).tolist()


    return [valid_time_index, forecast_hours, time_index] 



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
            print(f"Skipping of processing group {key}: {str(e)}")

    return zstore

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


def store_coord_var(key: str, zstore: dict, coords: tuple[str, ...], data: np.array):
    if np.isnan(data).any():
        if f"{key}/.zarray" not in zstore:
            logger.debug("Skipping nan coordinate with no variable %s", key)
            return
        else:
            logger.info("Trying to add coordinate var %s with nan value!", key)

    zattrs = ujson.loads(zstore[f"{key}/.zattrs"])
    zarray = ujson.loads(zstore[f"{key}/.zarray"])
    # Use list not tuple
    zarray["chunks"] = [*data.shape]
    zarray["shape"] = [*data.shape]
    zattrs["_ARRAY_DIMENSIONS"] = [
        COORD_DIM_MAPPING[v] if v in COORD_DIM_MAPPING else v for v in coords
    ]

    zstore[f"{key}/.zarray"] = ujson.dumps(zarray)
    zstore[f"{key}/.zattrs"] = ujson.dumps(zattrs)

    vkey = ".".join(["0" for _ in coords])
    data_bytes = data.tobytes()
    try:
        enocded_val = data_bytes.decode("ascii")
    except UnicodeDecodeError:
        enocded_val = (b"base64:" + base64.b64encode(data_bytes)).decode("ascii")
    zstore[f"{key}/{vkey}"] = enocded_val


def store_data_var(
    key: str,
    zstore: dict,
    dims: dict[str, int],
    coords: dict[str, tuple[str, ...]],
    data: pd.DataFrame,
    steps: np.array,
    times: np.array,
    lvals: Optional[np.array],
):
    zattrs = ujson.loads(zstore[f"{key}/.zattrs"])
    zarray = ujson.loads(zstore[f"{key}/.zarray"])

    dcoords = coords["datavar"]

    # The lat/lon y/x coordinates are always the last two
    lat_lon_dims = {
        k: v for k, v in zip(zattrs["_ARRAY_DIMENSIONS"][-2:], zarray["shape"][-2:])
    }
    full_coords = dcoords + tuple(lat_lon_dims.keys())
    full_dims = dict(**dims, **lat_lon_dims)

    # all chunk dimensions are 1 except for lat/lon or x/y
    zarray["chunks"] = [
        1 if c not in lat_lon_dims else lat_lon_dims[c] for c in full_coords
    ]
    zarray["shape"] = [full_dims[k] for k in full_coords]
    if zarray["fill_value"] is None:
        # Check dtype first?
        zarray["fill_value"] = np.nan

    zattrs["_ARRAY_DIMENSIONS"] = [
        COORD_DIM_MAPPING[v] if v in COORD_DIM_MAPPING else v for v in full_coords
    ]

    zstore[f"{key}/.zarray"] = ujson.dumps(zarray)
    zstore[f"{key}/.zattrs"] = ujson.dumps(zattrs)

    idata = data.set_index(["time", "step", "level"]).sort_index()

    for idx in itertools.product(*[range(dims[k]) for k in dcoords]):
        # Build an iterator over each of the single dimension chunks
        # TODO Replace this with a reindex operation and iterate the result if the .loc call is slow inside the loop
        dim_idx = {k: v for k, v in zip(dcoords, idx)}

        iloc: tuple[Any, ...] = (
            times[tuple([dim_idx[k] for k in coords["time"]])],
            steps[tuple([dim_idx[k] for k in coords["step"]])],
        )
        if lvals is not None:
            iloc = iloc + (lvals[idx[-1]],)  # type:ignore[assignment]

        try:
            # Squeeze if needed to get a series. Noop if already a series Df has multiple rows
            dval = idata.loc[iloc].squeeze()
        except KeyError:
            logger.info(f"Error getting vals {iloc} for in path {key}")
            continue

        assert isinstance(
            dval, pd.Series
        ), f"Got multiple values for iloc {iloc} in key {key}: {dval}"

        if pd.isna(dval.inline_value):
            # List of [URI(Str), offset(Int), length(Int)] using python (not numpy) types.
            record = [dval.uri, dval.offset.item(), dval.length.item()]
        else:
            record = dval.inline_value
        # lat/lon y/x have only the zero chunk
        vkey = ".".join([str(v) for v in (idx + (0, 0))])
        zstore[f"{key}/{vkey}"] = record





def ecmwf_enfo_local_parse_index(
    basename: str,
    suffix: str = "index",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Standalone method used to extract ALL metadata from a GRIB2 index file from a local disk using ast.literal_eval.

    :param basename: The full path to the GRIB file (excluding the suffix).
    :param suffix: The suffix for the index file (default: "index").
    :param tstamp: The timestamp to record for this index process.
    :param validate: Whether to validate uniqueness of attributes.
    :return: A pandas DataFrame containing the results with all JSON attributes.
    """
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"  # Construct the index filename

    # Lists to store parsed data
    records = []

    try:
        with open(fname, "r", encoding="utf-8") as f:  # Open the local file
            for idx, line in enumerate(f):
                try:
                    # Removing the trailing characters if there's any at the end of the line
                    clean_line = line.strip().rstrip(',')

                    # Attempt parsing using ast.literal_eval first
                    try:
                        prepared_str = clean_line.replace('"', "'")  # Ensure single quotes for eval
                        data = ast.literal_eval(prepared_str)
                    except (SyntaxError, ValueError):
                        data = json.loads(clean_line)  # Fallback to json.loads

                    # Store parsed data
                    record = {
                        "idx": idx,
                        "offset": data.get("_offset", 0),
                        "length": data.get("_length", 0),
                        "number": data.get("number", -1),
                        "idx_uri": fname,
                        "grib_uri": basename
                    }

                    # Add remaining fields
                    for key, value in data.items():
                        if key not in record:
                            record[key] = value

                    records.append(record)

                except Exception as e:
                    print(f"Warning: Could not parse line {idx} in {fname}: {e}")
                    continue  # Skip problematic lines instead of raising an error

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {fname} not found.")

    # Convert to DataFrame
    result = pd.DataFrame(records)

    # If empty, return early
    if result.empty:
        print(f"Warning: No valid entries found in {fname}. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Add timestamp column
    result["indexed_at"] = tstamp or pd.Timestamp.now()

    # Validate uniqueness if required
    if validate and result.duplicated().any():
        raise ValueError(f"Attribute mapping for GRIB file {basename} is not unique")

    print(f"Completed indexing {len(result)} entries from {fname}")
    return result.set_index("idx")



def ecmwf_enfo_s3_parse_index_with_ast(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    suffix: str = "index",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Standalone method used to extract ALL metadata from a grib2 index file using ast.literal_eval
    
    :param fs: the file system to read from
    :param basename: the base name is the full path to the grib file
    :param suffix: the suffix is the ending for the index file
    :param tstamp: the timestamp to record for this index process
    :param validate: whether to validate uniqueness of attributes
    :return: the data frame containing the results with all json attributes
    """
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"

    fs.invalidate_cache(fname)
    fs.invalidate_cache(basename)

    baseinfo = fs.info(basename)

    # Lists to store all parsed data
    records = []
    
    with fs.open(fname, "r") as f:
        for idx, line in enumerate(f):
            try:
                # Removing the trailing characters if there's any at the end of the line
                clean_line = line.strip().rstrip(',')
                
                # Using ast.literal_eval instead of json.loads
                try:
                    # Convert double quotes to single quotes
                    prepared_str = clean_line.replace('"', "'")
                    data = ast.literal_eval(prepared_str)
                except (SyntaxError, ValueError) as e:
                    # Fallback to json.loads if ast.literal_eval fails
                    data = json.loads(clean_line)
                
                # Store data along with index and file info
                record = {
                    'idx': idx,
                    'offset': data.get('_offset', 0),
                    'length': data.get('_length', 0),
                    'number': data.get('number', -1),
                    'idx_uri': fname,
                    'grib_uri': basename
                }
                
                # Add all fields from the JSON to the record
                for key, value in data.items():
                    if key not in record:  # Avoid overwriting index fields
                        record[key] = value
                
                records.append(record)
                
            except Exception as e:
                # Handle cases where both conversion methods fail
                raise ValueError(f"Could not parse data from line: {line}") from e

    # Create the DataFrame from all records
    result = pd.DataFrame(records)
    
    # Add timestamps
    if tstamp is None:
        tstamp = pd.Timestamp.now()
    result['indexed_at'] = tstamp
    
    # Add filesystem metadata
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
        # For other filesystems
        result.loc[:, "grib_crc32"] = None
        result.loc[:, "grib_updated_at"] = None
        result.loc[:, "idx_crc32"] = None
        result.loc[:, "idx_updated_at"] = None

    if validate and len(result) != len(result.drop_duplicates()):
        raise ValueError(f"Attribute mapping for grib file {basename} is not unique)")
    
    print(f'Completed index files and found {len(result.index)} entries in it')
    return result.set_index("idx")

def ecmwf_enfo_duplicate_dict_ens_mem(var_dict):
    """
    Create duplicates of parameter entries for each ensemble member.
    
    Parameters:
    -----------
    var_dict : dict
        Dictionary with unique parameter combinations
    
    Returns:
    --------
    updated_data_dict : dict
        Dictionary with parameter entries for all ensemble members
    """
    # Generate sequence for ensemble members 1-50, with control (-1) at the start
    ens_numbers = list(range(1, 51))
    ens_numbers.insert(0, -1)
    
    updated_data_dict = {}
    for ens_number in ens_numbers:
        for key, subdict in var_dict.items():
            updated_subdict = subdict.copy()
            updated_subdict['number'] = int(ens_number)  # Ensure number is integer
            new_key = f"{key}_ens{ens_number}"
            updated_data_dict[new_key] = updated_subdict
    
    print(f"Total parameter-ensemble combinations: {len(updated_data_dict)}")
    return updated_data_dict


def ecmwf_enfo_index_with_target_variables(edf, target_variables):
    """
    Filter ECMWF index DataFrame according to specified target variables.
    
    Parameters:
    -----------
    edf : pd.DataFrame
        The DataFrame containing the ECMWF index data
    target_variables : dict
        Dictionary specifying which variables to extract at which levels
        Format: {'param_name': {'levels': [list_of_levels], 'levtype': 'level_type'}}
    
    Returns:
    --------
    combined_dict : dict
        Dictionary with unique parameter-level combinations
    """
    print(f"Total rows in original index: {len(edf)}")
    
    # Convert numeric columns that might be mixed types
    if 'levelist' in edf.columns:
        edf['levelist'] = edf['levelist'].astype(str).fillna('null')
    
    # Make sure number column is consistently typed
    if 'number' in edf.columns:
        # Try to convert to int where possible, handle errors
        try:
            edf['number'] = edf['number'].astype(int)
        except (ValueError, TypeError):
            # If conversion fails, convert everything to string
            edf['number'] = edf['number'].astype(str)
    
    selected_rows = []
    
    # Process each target variable
    for param_name, param_info in target_variables.items():
        levtype = param_info['levtype']
        levels = param_info['levels']
        
        if levtype == 'pl' and levels:
            # For pressure levels with specific levels
            for level in levels:
                level_str = str(level)
                mask = (edf['param'] == param_name) & (edf['levtype'] == levtype) & (edf['levelist'].astype(str) == level_str)
                matching = edf[mask]
                print(f"Variable: {param_name}, levtype: {levtype}, level: {level} - Found {len(matching)} rows")
                selected_rows.append(matching)
        elif levtype == 'sfc':
            # For surface variables (no level needed)
            mask = (edf['param'] == param_name) & (edf['levtype'] == levtype)
            matching = edf[mask]
            print(f"Variable: {param_name}, levtype: {levtype} - Found {len(matching)} rows")
            selected_rows.append(matching)
    
    # Combine all selected rows
    if not selected_rows:
        return {}
    
    filtered_df = pd.concat(selected_rows)
    print(f"Total selected rows: {len(filtered_df)}")
    
    # Group by parameter, level type, and level list
    # For each group, take the first ensemble member number (we'll create all ensemble members later)
    # Use a safer approach with explicit type conversion
    def safe_first_number(x):
        try:
            # Try to convert all values to integers for sorting
            nums = [int(val) for val in x.unique()]
            return sorted(nums)[0]
        except (ValueError, TypeError):
            # If conversion fails, return the first value
            return x.iloc[0]
    
    grouped = filtered_df.groupby(['param', 'levtype', 'levelist']).agg({
        'number': safe_first_number
    }).reset_index()
    
    # Create the dictionary
    combined_dict = {}
    for _, row in grouped.iterrows():
        key = f"{row['param']}_{row['levtype']}_{row['levelist']}"
        combined_dict[key] = {
            'param': row['param'],
            'levtype': row['levtype'],
            'number': row['number'],
            'levelist': row['levelist']
        }
    
    print(f"Unique parameter combinations: {len(combined_dict)}")
    return combined_dict


def convert_forecast_dict_to_target_variables(forecast_dict):
    """
    Convert a forecast dictionary with shorthand notation to target variables format
    
    Parameters:
    -----------
    forecast_dict : dict
        Dictionary with parameter names and shorthand notation like "t:pl"
    
    Returns:
    --------
    target_variables : dict
        Dictionary in the format required by ecmwf_enfo_index_with_target_variables
    """
    target_variables = {}
    
    for param_name, param_code in forecast_dict.items():
        # Split the code to get parameter and level type
        parts = param_code.split(':')
        
        if len(parts) != 2:
            print(f"Warning: Invalid code format for {param_name}: {param_code}")
            continue
            
        ecmwf_param, levtype = parts
        
        # Set up the levels list
        levels = []
        if levtype == 'pl':
            # For pressure level variables, we want level 50
            levels = [50]
        
        # Add to target variables
        target_variables[ecmwf_param] = {
            'levels': levels,
            'levtype': levtype
        }
    
    return target_variables


def ecmwf_enfo_index_df_create_with_keys(ecmwf_s3url, forecast_dict):
    # Parse the ECMWF index file
    fs = fsspec.filesystem("s3")
    suffix = 'index'
    #edf = ecmwf_enfo_s3_parse_index_with_ast(fs=fs, basename=ecmwf_s3url, suffix=suffix)
    edf = ecmwf_enfo_local_parse_index(basename=ecmwf_s3url)
    
    # Convert number to consistent type if it exists
    if 'number' in edf.columns:
        try:
            edf['number'] = edf['number'].astype(int)
        except (ValueError, TypeError):
            # If conversion fails for any reason, convert to string
            edf['number'] = edf['number'].astype(str)
    
    # Convert forecast dictionary to target variables format
    target_variables = convert_forecast_dict_to_target_variables(forecast_dict)
    
    # Filter the dataframe according to target variables
    combined_dict = ecmwf_enfo_index_with_target_variables(edf, target_variables)
    
    # Create ensemble member variations
    all_em = ecmwf_enfo_duplicate_dict_ens_mem(combined_dict)
    
    # Map and filter index entries
    idx_mapping = {}
    subset_dfs = []

    for ens_key, conditions in all_em.items():
        # Initialize mask with all True values
        mask = pd.Series(True, index=edf.index)
        
        # Apply conditions one by one
        for col, value in conditions.items():
            if col == 'number':
                # Try to handle both int and string comparisons
                try:
                    value_int = int(value)
                    col_int = edf[col].astype(int)
                    mask &= (col_int == value_int)
                except (ValueError, TypeError):
                    # Fall back to string comparison
                    mask &= (edf[col].astype(str) == str(value))
            elif value == 'null':
                # Handle null values
                mask &= edf[col].isna()
            else:
                # Try both string and original comparison
                col_vals = edf[col]
                str_mask = (col_vals.astype(str) == str(value))
                orig_mask = (col_vals == value)
                mask &= (str_mask | orig_mask)
        
        matching_rows = edf[mask]
        if not matching_rows.empty:
            subset_dfs.append(matching_rows)
            for idx in matching_rows.index:
                idx_mapping[idx] = ens_key

    # Combine all subset DataFrames into a single DataFrame
    subset_edf = pd.concat(subset_dfs) if subset_dfs else pd.DataFrame(columns=edf.columns)
    print(f"Final filtered dataset: {len(subset_edf)} rows")
    
    return edf, subset_edf, idx_mapping, combined_dict



def ecmwf_filter_scan_grib(ecmwf_s3url):
    """
    Scan an ECMWF GRIB file, add ensemble information to the Zarr references,
    and return a list of modified groups along with an index mapping.
    """
    esc_groups = scan_grib(ecmwf_s3url)
    print(f"Completed scan_grib for {ecmwf_s3url}, found {len(esc_groups)} messages")
    forecast_dict = {
        "Temperature": "t:pl",
        "Cape": "cape:sfc",
        "U component of wind": "u:pl",
        "V component of wind": "v:pl",
        "Mean sea level pressure": "msl:sfc",
        "2 metre temperature": "2t:sfc",
        "Total precipitation": "tp:sfc",
    }
    _, _, idx_mapping, _ = ecmwf_enfo_index_df_create_with_keys(ecmwf_s3url,forecast_dict)
    print(f"Found {len(idx_mapping)} matching indices")
    modified_groups = []
    for i, group in enumerate(esc_groups):
        if i in idx_mapping:
            ens_key = idx_mapping[i]
            ens_number = int(ens_key.split('ens')[-1]) if 'ens' in ens_key else -1
            mod_group = copy.deepcopy(group)
            refs = mod_group['refs']
            data_vars = []
            for key in refs:
                if key.endswith('/.zattrs'):
                    var_name = key.split('/')[0]
                    if not var_name.startswith('.'):
                        try:
                            attrs = json.loads(refs[key])
                            if '_ARRAY_DIMENSIONS' in attrs and len(attrs['_ARRAY_DIMENSIONS']) > 0:
                                if var_name not in ['latitude', 'longitude', 'number', 'time', 'step', 'valid_time']:
                                    data_vars.append(var_name)
                        except json.JSONDecodeError:
                            print(f"Error decoding {key}")
            if '.zattrs' in refs:
                try:
                    root_attrs = json.loads(refs['.zattrs'])
                    root_attrs['ensemble_member'] = ens_number
                    root_attrs['ensemble_key'] = ens_key
                    if 'coordinates' in root_attrs:
                        coords = root_attrs['coordinates'].split()
                        if 'number' not in coords:
                            coords.append('number')
                            root_attrs['coordinates'] = ' '.join(coords)
                    refs['.zattrs'] = json.dumps(root_attrs)
                except json.JSONDecodeError:
                    print(f"Error updating root attributes for group {i}")
            for var_name in data_vars:
                attr_key = f"{var_name}/.zattrs"
                if attr_key in refs:
                    try:
                        var_attrs = json.loads(refs[attr_key])
                        var_attrs['ensemble_member'] = ens_number
                        var_attrs['ensemble_key'] = ens_key
                        refs[attr_key] = json.dumps(var_attrs)
                    except json.JSONDecodeError:
                        print(f"Error updating attributes for {var_name}")
            has_number = any(key == 'number/.zattrs' or key.endswith('/number/.zattrs') for key in refs)
            if not has_number:
                print(f"Adding number coordinate for group {i}, ensemble {ens_number}")
                refs['number/.zarray'] = json.dumps({
                    "chunks": [],
                    "compressor": None,
                    "dtype": "<i8",
                    "fill_value": None,
                    "filters": None,
                    "order": "C",
                    "shape": [],
                    "zarr_format": 2
                })
                refs['number/.zattrs'] = json.dumps({
                    "_ARRAY_DIMENSIONS": [],
                    "long_name": "ensemble member numerical id",
                    "standard_name": "realization",
                    "units": "1"
                })
                ens_num_array = np.array(ens_number, dtype=np.int64)
                refs['number/0'] = ens_num_array.tobytes().decode('latin1')
            modified_groups.append(mod_group)
    return modified_groups, idx_mapping



def fixed_ensemble_grib_tree(
    message_groups: Iterable[Dict],
    remote_options=None,
    debug_output=False
) -> Dict:
    """
    Build a hierarchical data model from a set of scanned grib messages with proper ensemble support
    and correct zarr path structure.
    
    This function handles ensemble dimensions correctly while maintaining the proper zarr structure
    needed by datatree.
    
    Parameters
    ----------
    message_groups: iterable[dict]
        a collection of zarr store like dictionaries as produced by scan_grib
    remote_options: dict
        remote options to pass to MultiZarrToZarr
    debug_output: bool
        If True, prints detailed debugging information

    Returns
    -------
    dict: A zarr store like dictionary with proper ensemble support
    """
    # Hard code the filters in the correct order for the group hierarchy
    filters = ["stepType", "typeOfLevel"]

    # Use a regular dictionary for storage
    zarr_store = {'.zgroup': json.dumps({'zarr_format': 2})}
    zroot = zarr.group()
    
    # Track information by path
    aggregations = defaultdict(list)
    ensemble_dimensions = defaultdict(set)
    level_dimensions = defaultdict(set)
    path_counts = defaultdict(int)

    # Process each message group and determine paths
    for msg_ind, group in enumerate(message_groups):
        if "version" not in group or group["version"] != 1:
            if debug_output:
                print(f"Skipping message {msg_ind}: Invalid version")
            continue

        # Extract ensemble member information
        ensemble_member = None
        try:
            # Check various potential locations for ensemble info
            if ".zattrs" in group["refs"]:
                root_attrs = json.loads(group["refs"][".zattrs"])
                if "ensemble_member" in root_attrs:
                    ensemble_member = root_attrs["ensemble_member"]
            
            # Look for number variable which typically holds ensemble number
            if ensemble_member is None:
                for key in group["refs"]:
                    if key == "number/0" or key.endswith("/number/0"):
                        val = group["refs"][key]
                        if isinstance(val, str):
                            try:
                                arr = np.frombuffer(val.encode('latin1'), dtype=np.int64)
                                if len(arr) == 1:
                                    ensemble_member = int(arr[0])
                                    break
                            except:
                                pass
        except Exception as e:
            if debug_output:
                print(f"Warning: Error extracting ensemble information for msg {msg_ind}: {e}")
        
        # Try to extract coordinates from the root attributes
        try:
            gattrs = json.loads(group["refs"][".zattrs"])
            coordinates = gattrs["coordinates"].split(" ")
        except Exception as e:
            if debug_output:
                print(f"Warning: Issue with attributes for message {msg_ind}: {e}")
            continue

        # Find the data variable
        vname = None
        for key in group["refs"]:
            name = key.split("/")[0]
            if name not in [".zattrs", ".zgroup"] and name not in coordinates:
                vname = name
                break

        if vname is None or vname == "unknown":
            if debug_output:
                print(f"Warning: No valid data variable found for message {msg_ind}")
            continue

        # Extract attributes for this variable
        try:
            dattrs = json.loads(group["refs"][f"{vname}/.zattrs"])
        except Exception as e:
            if debug_output:
                print(f"Warning: Issue with variable attributes for {vname} in message {msg_ind}: {e}")
            continue

        # Build path based on filter attributes
        gfilters = {}
        for key in filters:
            attr_val = dattrs.get(f"GRIB_{key}")
            if attr_val and attr_val != "unknown":
                gfilters[key] = attr_val

        # Start with variable name
        path_parts = [vname]
        
        # Add filter values to path
        for key, value in gfilters.items():
            if value:
                path_parts.append(value)
        
        # The base path excludes ensemble information
        base_path = "/".join(path_parts)
        
        # Add group to aggregations
        group_copy = copy.deepcopy(group)
        if ensemble_member is not None:
            group_copy["ensemble_member"] = ensemble_member
        
        aggregations[base_path].append(group_copy)
        path_counts[base_path] += 1
        
        # Track ensemble dimension
        if ensemble_member is not None:
            ensemble_dimensions[base_path].add(ensemble_member)
        
        # Track level information
        for key, entry in group["refs"].items():
            name = key.split("/")[0]
            if name == gfilters.get("typeOfLevel") and key.endswith("0"):
                if isinstance(entry, list):
                    entry = tuple(entry)
                level_dimensions[base_path].add(entry)
    
    # Print diagnostics for paths if debug is enabled
    if debug_output:
        print(f"Found {len(aggregations)} unique paths from {len(message_groups)} messages")
        for path, groups in sorted(aggregations.items(), key=lambda x: len(x[1]), reverse=True):
            ensemble_count = len(ensemble_dimensions.get(path, set()))
            level_count = len(level_dimensions.get(path, set()))
            print(f"  {path}: {len(groups)} groups, {ensemble_count} ensemble members, {level_count} levels")
    
    # Process each path with MultiZarrToZarr and ensure proper hierarchical structure
    for path, groups in aggregations.items():
        # Build groups for each level in the hierarchy
        path_parts = path.split("/")
        current_path = ""
        for i, part in enumerate(path_parts):
            prev_path = current_path
            
            if current_path:
                current_path = f"{current_path}/{part}"
            else:
                current_path = part
            
            # Add .zgroup for this level if not already present
            if f"{current_path}/.zgroup" not in zarr_store:
                zarr_store[f"{current_path}/.zgroup"] = json.dumps({'zarr_format': 2})
            
            # Add .zattrs for this level
            if f"{current_path}/.zattrs" not in zarr_store:
                # Add appropriate attributes based on the level
                attrs = {}
                
                # Add filter-specific attributes
                if i == 1 and len(path_parts) > 1:  # stepType level
                    attrs["stepType"] = path_parts[i]
                if i == 2 and len(path_parts) > 2:  # typeOfLevel level
                    attrs["typeOfLevel"] = path_parts[i]
                
                zarr_store[f"{current_path}/.zattrs"] = json.dumps(attrs)
        
        # Get dimensions for this path
        catdims = ["time", "step"]  # Always concatenate time and step
        idims = ["longitude", "latitude"]  # Latitude and longitude are always identical
        
        # Handle level dimensions
        level_count = len(level_dimensions.get(path, set()))
        level_name = path_parts[-1] if len(path_parts) > 0 else None
        
        if level_count == 1:
            # Single level - treat as identical dimension
            if level_name and level_name not in idims:
                idims.append(level_name)
        elif level_count > 1:
            # Multiple levels - treat as concat dimension
            if level_name and level_name not in catdims:
                catdims.append(level_name)
        
        # Handle ensemble dimension
        ensemble_count = len(ensemble_dimensions.get(path, set()))
        if ensemble_count > 1 and "number" not in catdims:
            catdims.append("number")
            # Sort groups by ensemble number for consistent processing
            groups.sort(key=lambda g: g.get("ensemble_member", 0))
        
        if debug_output:
            print(f"Processing {path} with concat_dims={catdims}, identical_dims={idims}")
        
        try:
            # Create aggregation
            mzz = MultiZarrToZarr(
                groups,
                remote_options=remote_options,
                concat_dims=catdims,
                identical_dims=idims,
            )
            
            # Get result and store references
            group_result = mzz.translate()
            
            # Add each reference with proper path prefix
            for key, value in group_result["refs"].items():
                if key == ".zattrs" or key == ".zgroup":
                    # Don't overwrite existing group metadata
                    if f"{path}/{key}" not in zarr_store:
                        zarr_store[f"{path}/{key}"] = value
                else:
                    # Data or other references
                    zarr_store[f"{path}/{key}"] = value
            
        except Exception as e:
            if debug_output:
                print(f"Error processing path {path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Convert all byte values to strings for compatibility
    zarr_store = {
        key: (val.decode('utf-8') if isinstance(val, bytes) else val)
        for key, val in zarr_store.items()
    }
    
    return {
        "refs": zarr_store,
        "version": 1
    }


def analyze_grib_tree_output(original_tree, ensembe_tree):
    """
    Analyze and compare outputs from different grib_tree functions
    """
    # Count references by path prefix
    def count_by_prefix(refs_dict):
        prefix_counts = {}
        for key in refs_dict:
            # Extract first part of the path
            parts = key.split('/')
            if len(parts) > 0:
                prefix = parts[0]
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = 0
                prefix_counts[prefix] += 1
        
        return prefix_counts
    
    # Count references by group level
    def count_by_level(refs_dict):
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for key in refs_dict:
            # Count levels in the path
            level = key.count('/')
            if level in level_counts:
                level_counts[level] += 1
            else:
                level_counts[5] += 1  # Group anything deeper than level 4
        
        return level_counts
    
    # Look for ensemble-related entries
    def find_ensemble_refs(refs_dict):
        ensemble_refs = []
        for key in refs_dict:
            if 'number' in key or 'ensemble' in key:
                ensemble_refs.append(key)
        
        return ensemble_refs

    # Analyze original tree
    orig_prefix_counts = count_by_prefix(original_tree['refs'])
    orig_level_counts = count_by_level(original_tree['refs'])
    orig_ensemble_refs = find_ensemble_refs(original_tree['refs'])
    
    # Analyze ensemble tree
    ens_prefix_counts = count_by_prefix(ensembe_tree['refs'])
    ens_level_counts = count_by_level(ensembe_tree['refs'])
    ens_ensemble_refs = find_ensemble_refs(ensembe_tree['refs'])
    
    # Print analysis
    print("=== ORIGINAL TREE ANALYSIS ===")
    print(f"Total references: {len(original_tree['refs'])}")
    print("\nReferences by variable prefix:")
    for prefix, count in sorted(orig_prefix_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prefix}: {count}")
    
    print("\nReferences by path depth:")
    for level, count in orig_level_counts.items():
        print(f"  Level {level}: {count}")
    
    print(f"\nEnsemble-related references: {len(orig_ensemble_refs)}")
    if orig_ensemble_refs:
        print("  Examples:")
        for ref in orig_ensemble_refs[:5]:  # Show up to 5 examples
            print(f"    {ref}")
    
    print("\n=== ENSEMBLE TREE ANALYSIS ===")
    print(f"Total references: {len(ensembe_tree['refs'])}")
    print("\nReferences by variable prefix:")
    for prefix, count in sorted(ens_prefix_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prefix}: {count}")
    
    print("\nReferences by path depth:")
    for level, count in ens_level_counts.items():
        print(f"  Level {level}: {count}")
    
    print(f"\nEnsemble-related references: {len(ens_ensemble_refs)}")
    if ens_ensemble_refs:
        print("  Examples:")
        for ref in ens_ensemble_refs[:5]:  # Show up to 5 examples
            print(f"    {ref}")
    
    # Compare structure of a sample variable
    print("\n=== SAMPLE VARIABLE COMPARISON ===")
    # Find a common variable prefix
    common_prefixes = set(orig_prefix_counts.keys()) & set(ens_prefix_counts.keys())
    if common_prefixes:
        sample_var = next(iter(common_prefixes))
        print(f"Sample variable: {sample_var}")
        
        # Get all paths for this variable
        orig_var_paths = [p for p in original_tree['refs'] if p.startswith(f"{sample_var}/")]
        ens_var_paths = [p for p in ensembe_tree['refs'] if p.startswith(f"{sample_var}/")]
        
        print(f"Original tree paths: {len(orig_var_paths)}")
        for path in sorted(orig_var_paths)[:5]:  # Show up to 5 examples
            print(f"  {path}")
            
        print(f"Ensemble tree paths: {len(ens_var_paths)}")
        for path in sorted(ens_var_paths)[:5]:  # Show up to 5 examples
            print(f"  {path}")
    
    return {
        "original": {
            "total": len(original_tree['refs']),
            "by_prefix": orig_prefix_counts,
            "by_level": orig_level_counts,
            "ensemble_refs": len(orig_ensemble_refs)
        },
        "ensemble": {
            "total": len(ensembe_tree['refs']),
            "by_prefix": ens_prefix_counts,
            "by_level": ens_level_counts,
            "ensemble_refs": len(ens_ensemble_refs)
        }
    }



