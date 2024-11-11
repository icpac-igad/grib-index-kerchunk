import os
import logging
import traceback 
import copy
import tempfile
from typing import List, Dict, Tuple, final
from calendar import monthrange
import re
import pathlib

import pandas as pd
import numpy as np
import fsspec
import dask 
import json
from distributed import get_worker

from google.cloud import storage 
from dynamic_zarr_store import (
    AggregationType, grib_tree, scan_grib, strip_datavar_chunks,
    parse_grib_idx, map_from_index, store_coord_var, store_data_var,
    build_idx_grib_mapping
)


logger = logging.getLogger(__name__)

def setup_logging(log_level: int = logging.INFO):
    """
    Configure the logging level and format for the application.

    Parameters:
    - log_level (int): Logging level to use.
    """
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def build_grib_tree(gfs_files: List[str]) -> Tuple[dict, dict]:
    """
    Scan GFS files, build a hierarchical tree structure for the data, and strip unnecessary data.

    Parameters:
    - gfs_files (List[str]): List of file paths to GFS files.

    Returns:
    - Tuple[dict, dict]: Original and deflated GRIB tree stores.
    """
    logger.info("Building Grib Tree")
    gfs_grib_tree_store = grib_tree([group for f in gfs_files for group in scan_grib(f)])
    deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
    strip_datavar_chunks(deflated_gfs_grib_tree_store)
    logger.info(f"Original references: {len(gfs_grib_tree_store['refs'])}")
    logger.info(f"Stripped references: {len(deflated_gfs_grib_tree_store['refs'])}")
    return gfs_grib_tree_store, deflated_gfs_grib_tree_store

def calculate_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time-related dimensions and coordinates based on input axes.

    Parameters:
    - axes (List[pd.Index]): List of pandas Index objects containing time information.

    Returns:
    - Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]: Time dimensions, coordinates, times, valid times, and steps.
    """
    logger.info("Calculating Time Dimensions and Coordinates")
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
    logger.info(f"Creating Mapped Index for date {date_str}")
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

    logger.info(f"Mapped collected multiple variables index info: {len(final_var_list)} and {final_var_list}")
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
    logger = logging.getLogger()
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
    
    logger.info(json.dumps({
        "event": "mapping_completed",
        "date": date_str,
        "variables_count": len(final_var_list),
        "variables": final_var_list
    }))
    
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
    logger.info("Preparing Zarr Store")
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
    logger.info("Processing Unique Groups and Updating Zarr Store")
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
            logger.error(f"Error processing group {key}: {str(e)}")

    return zstore


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
    logger.info("Creating and Saving Parquet File")
    gfs_store = dict(refs=zstore, version=1)  # Include versioning for the store structure

    def dict_to_df(zstore: dict):
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

    zstore_df = dict_to_df(gfs_store)
    zstore_df.to_parquet(output_parquet_file)
    logger.info(f"Parquet file saved to {output_parquet_file}")



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
    setup_logging(log_level)
    
    try:
        logger.info(f"Processing date: {date_str}")
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
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

# Set up basic logging configuration
def setup_logger():
    """Configure logging for both main process and workers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )
    return logging.getLogger(__name__)

# Create logger instance
logger = setup_logger()

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
    logger.debug(f"Generated {len(fmt_s3og)} URLs for date {date_str}")
    return fmt_s3og


def get_filename_from_path(file_path):
    """Extract filename from full path"""
    return os.path.basename(file_path)

def old_upload_to_gcs(bucket_name, source_file_name, destination_blob_name, dask_worker_credentials_path):
    """Uploads a file to the GCS bucket using provided service account credentials."""
    try:
        # Get just the filename from the credentials path
        #creds_filename = get_filename_from_path(credentials_path)
        # Construct the worker-local path
        #worker_creds_path = os.path.join(os.getcwd(), creds_filename)
        #credentials_path = "/app/coiled-data-key.json"
        #credentials_path = os.path.join(os.getcwd(), creds_filename)
        #credentials_path = os.path.join(tempfile.gettempdir(), creds_filename)
        logger.info(f"Using credentials file at: {dask_worker_credentials_path}")
        
        if not os.path.exists(dask_worker_credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {dask_worker_credentials_path}")
            
        storage_client = storage.Client.from_service_account_json(dask_worker_credentials_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logger.info(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        logger.error(f"Failed to upload file to GCS: {str(e)}")
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
        
        logger.info(f"Using credentials file at: {worker_creds_path}")
        
        storage_client = storage.Client.from_service_account_json(worker_creds_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logger.info(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        logger.error(f"Failed to upload file to GCS: {str(e)}")
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

