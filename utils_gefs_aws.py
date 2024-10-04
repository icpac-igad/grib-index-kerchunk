import os
import logging
import pandas as pd
import numpy as np
import fsspec
import copy
from typing import List, Dict, Tuple, final
from dynamic_zarr_store import (
    AggregationType, grib_tree, scan_grib, strip_datavar_chunks,
    parse_grib_idx, map_from_index, store_coord_var, store_data_var,
    build_idx_grib_mapping
)
from calendar import monthrange


logger = logging.getLogger(__name__)

def setup_logging(log_level: int = logging.INFO):
    """
    Configure the logging level and format for the application.

    Parameters:
    - log_level (int): Logging level to use.
    """
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


def process_gep_members():
    for member in range(1, 31):  # Loop over members 01 to 30
        for runtime in range(0, 123, 3):
            fname = f"s3://noaa-gefs-pds/gefs.20240923/00/atmos/pgrb2sp25/gep{member:02}.t00z.pgrb2s.0p25.f{runtime:03}"
            print(fname)
            mapping = build_idx_grib_mapping(
                fs=fsspec.filesystem("s3"),
                basename=fname
            )
            deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
            deduped_mapping.set_index('attrs', inplace=True)
            deduped_mapping.to_parquet(
                f"aws_gefs_mapping/aws-gefs-mapping-gep{member:02}-{runtime:03}.parquet",
                index=True
            )
            mapping = []

def build_grib_tree(gfs_files: List[str]) -> Tuple[dict, dict]:
    """
    Scan GFS files, build a hierarchical tree structure for the data, and strip unnecessary data.

    Parameters:
    - gfs_files (List[str]): List of file paths to GFS files.

    Returns:
    - Tuple[dict, dict]: Original and deflated GRIB tree stores.
    """
    logger.info("Building Grib Tree")
    gefs_grib_tree_store = grib_tree([group for f in gfs_files for group in scan_grib(f)])
    deflated_gefs_grib_tree_store = copy.deepcopy(gefs_grib_tree_store)
    strip_datavar_chunks(deflated_gefs_grib_tree_store)
    logger.info(f"Original references: {len(gefs_grib_tree_store['refs'])}")
    logger.info(f"Stripped references: {len(deflated_gefs_grib_tree_store['refs'])}")
    return gefs_grib_tree_store, deflated_gefs_grib_tree_store

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
    gefs_idx_range=np.arange(3,243,3)
    for idx, datestr in enumerate(dtaxes[1:]):
        try:
            #fname = f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{idx:03}"
            gefs_idx=gefs_idx_range[idx]
            fname = f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f{gefs_idx:03}"
            
            idxdf = parse_grib_idx(
                fs=fsspec.filesystem("s3"),
                basename=fname
            )
            deduped_mapping = pd.read_parquet(f"{mapping_parquet_file_path}aws-gefs-mapping-gep01-{gefs_idx:03}.parquet")
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

def gefs_s3_utl_maker(date, run):
    fs_s3 = fsspec.filesystem("s3", anon=True)
    members = [str(i).zfill(2) for i in range(1, 31)]
    s3url_ll = []
    for ensemble_member in members:
        s3url_glob = fs_s3.glob(
            f"s3://noaa-gefs-pds/gefs.{date}/{run}/atmos/pgrb2sp25/gep{ensemble_member}.*"
        )
        s3url_only_grib = [f for f in s3url_glob if f.split(".")[-1] != "idx"]
        fmt_s3og = sorted(["s3://" + f for f in s3url_only_grib])
        s3url_ll.append(fmt_s3og[1:])
    gefs_url = [item for sublist in s3url_ll for item in sublist]
    return gefs_url

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

    valid_time_index = pd.date_range(start_date, end_date, freq="180min", name="valid_time")
    time_index = pd.Index([start_date], name="time")

    return [valid_time_index, time_index]



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
            f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f000",
            f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003"
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


