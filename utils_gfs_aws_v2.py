import os
import logging
import json
import pandas as pd
import numpy as np
import fsspec
import copy
from typing import List, Dict, Tuple, Any
from dynamic_zarr_store import (
    AggregationType, grib_tree, scan_grib, strip_datavar_chunks,
    parse_grib_idx, map_from_index, store_coord_var, store_data_var
)
from calendar import monthrange

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "function": record.funcName,
            "line": record.lineno,
        }
        return json.dumps(log_data)

def setup_logging(log_level: int = logging.INFO, log_file: str = "gfs_processing.log"):
    """
    Configure the logging level and format for the application.

    Parameters:
    - log_level (int): Logging level to use.
    - log_file (str): File to save logs to.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        func_name = func.__name__
        logger.info(json.dumps({"event": "function_start", "function": func_name}))
        result = func(*args, **kwargs)
        logger.info(json.dumps({"event": "function_end", "function": func_name}))
        return result
    return wrapper

@log_function_call
def build_grib_tree(gfs_files: List[str]) -> Tuple[dict, dict]:
    """
    Scan GFS files, build a hierarchical tree structure for the data, and strip unnecessary data.

    Parameters:
    - gfs_files (List[str]): List of file paths to GFS files.

    Returns:
    - Tuple[dict, dict]: Original and deflated GRIB tree stores.
    """
    logger = logging.getLogger()
    gfs_grib_tree_store = grib_tree([group for f in gfs_files for group in scan_grib(f)])
    deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
    strip_datavar_chunks(deflated_gfs_grib_tree_store)
    logger.info(json.dumps({
        "event": "grib_tree_built",
        "original_refs": len(gfs_grib_tree_store['refs']),
        "stripped_refs": len(deflated_gfs_grib_tree_store['refs'])
    }))
    return gfs_grib_tree_store, deflated_gfs_grib_tree_store

@log_function_call
def calculate_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time-related dimensions and coordinates based on input axes.

    Parameters:
    - axes (List[pd.Index]): List of pandas Index objects containing time information.

    Returns:
    - Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]: Time dimensions, coordinates, times, valid times, and steps.
    """
    logger = logging.getLogger()
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
    logger.info(json.dumps({
        "event": "time_dimensions_calculated",
        "time_dims": time_dims,
        "time_coords": time_coords
    }))
    return time_dims, time_coords, times, valid_times, steps

@log_function_call
def process_dataframe(df: pd.DataFrame, varnames_to_process: List[str]) -> pd.DataFrame:
    """
    Filter and process the DataFrame by specific variable names and their corresponding type of levels.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to process.
    - varnames_to_process (list): List of variable names to filter and process in the DataFrame.

    Returns:
    - pd.DataFrame: Processed DataFrame with duplicates removed based on the 'time' column and sorted by 'length'.
    """
    logger = logging.getLogger()
    conditions = {
        'acpcp':'surface',
        'cape': 'surface',
        'cin': 'surface',
        'pres': 'heightAboveGround',
        'r': 'atmosphereSingleLayer',
        'soill': 'atmosphereSingleLayer',
        'soilw':'depthBelowLandLayer',
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

    logger.info(json.dumps({
        "event": "dataframe_processed",
        "processed_rows": len(processed_df),
        "processed_variables": list(processed_df['varname'].unique())
    }))
    return processed_df

@log_function_call
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
    logger = logging.getLogger()
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
            logger.error(json.dumps({
                "event": "error_processing_file",
                "file": fname,
                "error": str(e)
            }))

    gfs_kind = pd.concat(mapped_index_list)
    gfs_kind_var = gfs_kind.drop_duplicates('varname')
    var_list = gfs_kind_var['varname'].tolist() 
    var_to_remove = ['acpcp','cape','cin','pres','r','soill','soilw','st','t','tp']
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))
    gfs_kind1 = gfs_kind.loc[gfs_kind.varname.isin(var1_list)]
    to_process_df = gfs_kind[gfs_kind['varname'].isin(var_to_remove)]
    processed_df = process_dataframe(to_process_df, var_to_remove)
    final_df = pd.concat([gfs_kind1, processed_df], ignore_index=True)
    final_df = final_df.sort_values(by=['time', 'varname'])
    final_df_var = final_df.drop_duplicates('varname')
    final_var_list = final_df_var['varname'].tolist() 

    logger.info(json.dumps({
        "event": "mapped_index_created",
        "variables_count": len(final_var_list),
        "variables": final_var_list
    }))
    return final_df

@log_function_call
def prepare_zarr_store(deflated_gfs_grib_tree_store: dict, gfs_kind: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """
    Prepare Zarr store and related data for chunk processing based on GFS kind DataFrame.

    Parameters:
    - deflated_gfs_grib_tree_store (dict): Deflated GRIB tree store containing reference data.
    - gfs_kind (pd.DataFrame): DataFrame containing GFS data.

    Returns:
    - Tuple[dict, pd.DataFrame]: Zarr reference store and the DataFrame for chunk index.
    """
    logger = logging.getLogger()
    zarr_ref_store = deflated_gfs_grib_tree_store
    chunk_index = gfs_kind
    zstore = copy.deepcopy(zarr_ref_store["refs"])
    logger.info(json.dumps({
        "event": "zarr_store_prepared",
        "chunk_index_rows": len(chunk_index)
    }))
    return zstore, chunk_index



@log_function_call
def process_unique_groups(zstore: dict, chunk_index: pd.DataFrame, time_dims: Dict, time_coords: Dict,
                          times: np.ndarray, valid_times: np.ndarray, steps: np.ndarray) -> dict:
    """
    Process and update Zarr store by configuring data for unique variable groups.

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
    logger = logging.getLogger()
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
            
            logger.info(json.dumps({
                "event": "group_processed",
                "varname": key[0],
                "stepType": key[1],
                "typeOfLevel": key[2]
            }))
        except Exception as e:
            logger.error(json.dumps({
                "event": "error_processing_group",
                "group": key,
                "error": str(e)
            }))

    logger.info(json.dumps({
        "event": "unique_groups_processed",
        "total_groups": len(unique_groups)
    }))
    return zstore

@log_function_call
def create_parquet_file(zstore: dict, output_parquet_file: str):
    """
    Converts a dictionary containing Zarr store data to a DataFrame and saves it as a Parquet file.

    Parameters:
    - zstore (dict): The Zarr store dictionary containing all references and data needed for Zarr operations.
    - output_parquet_file (str): The path where the Parquet file will be saved.
    """
    logger = logging.getLogger()
    gfs_store = dict(refs=zstore, version=1)  # Include versioning for the store structure

    def dict_to_df(zstore: dict):
        data = []
        for key, value in zstore.items():
            if isinstance(value, (dict, list, int, float, np.integer, np.floating)):
                value = str(value).encode('utf-8')
            data.append((key, value))
        return pd.DataFrame(data, columns=['key', 'value'])

    zstore_df = dict_to_df(gfs_store)
    zstore_df.to_parquet(output_parquet_file)
    logger.info(json.dumps({
        "event": "parquet_file_created",
        "file_path": output_parquet_file,
        "rows_count": len(zstore_df)
    }))

@log_function_call
def generate_axes(date_str: str) -> List[pd.Index]:
    """
    Generate temporal axes indices for a given forecast start date over a predefined forecast period.
    
    Parameters:
    - date_str (str): The start date of the forecast, formatted as 'YYYYMMDD'.

    Returns:
    - List[pd.Index]: A list containing two pandas Index objects for 'valid_time' and 'time'.
    """
    logger = logging.getLogger()
    start_date = pd.Timestamp(date_str)
    end_date = start_date + pd.Timedelta(days=5)  # Forecast period of 5 days

    valid_time_index = pd.date_range(start_date, end_date, freq="60min", name="valid_time")
    time_index = pd.Index([start_date], name="time")

    logger.info(json.dumps({
        "event": "axes_generated",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "valid_time_count": len(valid_time_index)
    }))

    return [valid_time_index, time_index]

@log_function_call
def generate_gfs_dates(year: int, month: int) -> List[str]:
    """
    Generate a list of dates for a specific month and year, formatted as 'YYYYMMDD'.

    Parameters:
    - year (int): The year for which the dates are to be generated.
    - month (int): The month for which the dates are to be generated.

    Returns:
    - List[str]: A list of dates in the format 'YYYYMMDD' for every day in the specified month and year.
    """
    logger = logging.getLogger()
    _, last_day = monthrange(year, month)
    
    date_range = pd.date_range(start=f'{year}-{month:02d}-01', 
                               end=f'{year}-{month:02d}-{last_day}', 
                               freq='D')
    
    date_list = date_range.strftime('%Y%m%d').tolist()

    logger.info(json.dumps({
        "event": "gfs_dates_generated",
        "year": year,
        "month": month,
        "dates_count": len(date_list)
    }))

    return date_list


@log_function_call
def process_gfs_data(date_str: str, mapping_parquet_file_path: str, output_parquet_file: str, log_level: int = logging.INFO):
    """
    Orchestrates the end-to-end processing of Global Forecast System (GFS) data for a specific date.

    Parameters:
    - date_str (str): A date string in the format 'YYYYMMDD' representing the date for which GFS data is to be processed.
    - mapping_parquet_file_path (str): Path to the parquet file that contains mapping information for the GFS data.
    - output_parquet_file (str): Path where the output Parquet file will be saved after processing the data.
    - log_level (int): Logging level to use.
    """
    setup_logging(log_level, f"gfs_processing_{date_str}.log")
    logger = logging.getLogger()
    
    try:
        logger.info(json.dumps({
            "event": "processing_started",
            "date": date_str
        }))

        # Step 1: Generate axes
        logger.info(json.dumps({"event": "step_started", "step": "generate_axes"}))
        axes = generate_axes(date_str)
        logger.info(json.dumps({"event": "step_completed", "step": "generate_axes"}))

        # Step 2: Define GFS files
        logger.info(json.dumps({"event": "step_started", "step": "define_gfs_files"}))
        gfs_files = [
            f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f000",
            f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f001"
        ]
        logger.info(json.dumps({"event": "step_completed", "step": "define_gfs_files", "files": gfs_files}))

        # Step 3: Build GRIB tree
        logger.info(json.dumps({"event": "step_started", "step": "build_grib_tree"}))
        _, deflated_gfs_grib_tree_store = build_grib_tree(gfs_files)
        logger.info(json.dumps({"event": "step_completed", "step": "build_grib_tree"}))

        # Step 4: Calculate time dimensions
        logger.info(json.dumps({"event": "step_started", "step": "calculate_time_dimensions"}))
        time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
        logger.info(json.dumps({"event": "step_completed", "step": "calculate_time_dimensions"}))

        # Step 5: Create mapped index
        logger.info(json.dumps({"event": "step_started", "step": "create_mapped_index"}))
        gfs_kind = create_mapped_index(axes, mapping_parquet_file_path, date_str)
        logger.info(json.dumps({"event": "step_completed", "step": "create_mapped_index"}))

        # Step 6: Prepare Zarr store
        logger.info(json.dumps({"event": "step_started", "step": "prepare_zarr_store"}))
        zstore, chunk_index = prepare_zarr_store(deflated_gfs_grib_tree_store, gfs_kind)
        logger.info(json.dumps({"event": "step_completed", "step": "prepare_zarr_store"}))

        # Step 7: Process unique groups
        logger.info(json.dumps({"event": "step_started", "step": "process_unique_groups"}))
        updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, times, valid_times, steps)
        logger.info(json.dumps({"event": "step_completed", "step": "process_unique_groups"}))

        # Step 8: Create Parquet file
        logger.info(json.dumps({"event": "step_started", "step": "create_parquet_file"}))
        create_parquet_file(updated_zstore, output_parquet_file)
        logger.info(json.dumps({"event": "step_completed", "step": "create_parquet_file"}))

        logger.info(json.dumps({
            "event": "processing_completed",
            "date": date_str,
            "output_file": output_parquet_file
        }))
    except Exception as e:
        logger.error(json.dumps({
            "event": "processing_error",
            "date": date_str,
            "step": logger.info['event'],  # This will capture the last step that was logged
            "error": str(e)
        }))
        raise
