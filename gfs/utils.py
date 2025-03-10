import asyncio
import copy
import json
import logging
import os
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Callable

import fsspec
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd
from dask.distributed import get_worker
from google.cloud import storage
from google.oauth2 import service_account
from kerchunk.grib2 import grib_tree, scan_grib
from kerchunk._grib_idx import (
    AggregationType,
    store_coord_var,
    store_data_var,
    strip_datavar_chunks,
    parse_grib_idx,
    map_from_index,
)

# --- Functions used by run_day_gfs_gik.py ---

def generate_axes(date_str: str) -> List[pd.Index]:
    """
    Generate two temporal axes:
      - 'valid_time': hourly timestamps over a 5‑day forecast period
      - 'time': a single timestamp representing the forecast start.
    """
    start_date = pd.Timestamp(date_str)
    end_date = start_date + pd.Timedelta(days=5)
    valid_time_index = pd.date_range(start_date, end_date, freq="60min", name="valid_time")
    time_index = pd.Index([start_date], name="time")
    return [valid_time_index, time_index]


def aws_parse_grib_idx(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    suffix: str = "idx",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Extract metadata from a GRIB2 index file stored on S3.
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
                raise ValueError(f"Could not parse line: {line}")
    result = pd.DataFrame(splits, columns=["idx", "offset", "date", "attrs"])
    result.loc[:, "length"] = result.offset.shift(periods=-1, fill_value=baseinfo["size"]) - result.offset
    result.loc[:, "idx_uri"] = fname
    result.loc[:, "grib_uri"] = basename
    if tstamp is None:
        tstamp = pd.Timestamp.now()
    result.loc[:, "indexed_at"] = tstamp
    if "s3" in fs.protocol:
        result.loc[:, "grib_etag"] = baseinfo.get("ETag")
        result.loc[:, "grib_updated_at"] = pd.to_datetime(baseinfo.get("LastModified")).tz_localize(None)
        idxinfo = fs.info(fname)
        result.loc[:, "idx_etag"] = idxinfo.get("ETag")
        result.loc[:, "idx_updated_at"] = pd.to_datetime(idxinfo.get("LastModified")).tz_localize(None)
    else:
        result.loc[:, "grib_crc32"] = None
        result.loc[:, "grib_updated_at"] = None
        result.loc[:, "idx_crc32"] = None
        result.loc[:, "idx_updated_at"] = None
    if validate and not result["attrs"].is_unique:
        raise ValueError(f"Attribute mapping for grib file {basename} is not unique)")
    return result.set_index("idx")


def map_forecast_to_indices(forecast_dict: dict, df: pd.DataFrame) -> Tuple[dict, list]:
    """
    For each forecast variable, search for its matching string in the DataFrame's 'attrs' column.
    Returns:
      - A dictionary mapping variable names to a corresponding index (adjusted by –1)
      - A list of found indices (excluding defaults).
    """
    output_dict = {}
    for key, value in forecast_dict.items():
        matching_rows = df[df['attrs'].str.contains(value, na=False)]
        if not matching_rows.empty:
            output_dict[key] = int(matching_rows.index[0] - 1)
        else:
            output_dict[key] = 9999
    values_list = [v for v in output_dict.values() if v != 9999]
    return output_dict, values_list


def filter_gfs_scan_grib(gurl, tofilter_cgan_var_dict):
    """
    Scan a GFS file and filter GRIB groups using the provided forecast variable mapping.
    """
    fs = fsspec.filesystem("s3")
    suffix = "idx"
    gsc = scan_grib(gurl)
    idx_gfs = aws_parse_grib_idx(fs, basename=gurl, suffix=suffix)
    _, vl_gfs = map_forecast_to_indices(tofilter_cgan_var_dict, idx_gfs)
    return [gsc[i] for i in vl_gfs]


def filter_build_grib_tree(gfs_files: List[str], tofilter_cgan_var_dict: dict) -> Tuple[dict, dict]:
    """
    Build a GRIB tree from a list of GFS files after filtering GRIB groups.
    """
    print("Building Grib Tree")
    sg_groups = [group for gurl in gfs_files for group in filter_gfs_scan_grib(gurl, tofilter_cgan_var_dict)]
    gfs_grib_tree_store = grib_tree(sg_groups)
    deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
    strip_datavar_chunks(deflated_gfs_grib_tree_store)
    print(f"Original references: {len(gfs_grib_tree_store['refs'])}")
    print(f"Stripped references: {len(deflated_gfs_grib_tree_store['refs'])}")
    return gfs_grib_tree_store, deflated_gfs_grib_tree_store


def calculate_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time dimensions and coordinates using 'BEST_AVAILABLE' aggregation.
    """
    print("Calculating Time Dimensions and Coordinates")
    axes_by_name: Dict[str, pd.Index] = {pdi.name: pdi for pdi in axes}
    aggregation_type = AggregationType.BEST_AVAILABLE
    time_dims: Dict[str, int] = {}
    time_coords: Dict[str, Tuple] = {}
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
    return time_dims, time_coords, times, valid_times, steps


class KerchunkZarrDictStorageManager:
    """
    Manages storage and retrieval of Kerchunk Zarr dictionaries in Google Cloud Storage.
    """
    def __init__(
        self, 
        bucket_name: str,
        service_account_file: Optional[str] = None,
        service_account_info: Optional[Dict] = None
    ):
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
            self.gcs_fs = fsspec.filesystem('gcs', token=self.service_account_info)
        else:
            self.gcs_fs = fsspec.filesystem('gcs')


def process_dataframe(df, varnames_to_process):
    """
    Filter and process a DataFrame for the specified variable names.
    """
    conditions = {
        'acpcp': 'surface',
        'cape': 'surface',
        'cin': 'surface',
        'pres': 'heightAboveGround',
        'r': 'atmosphereSingleLayer',
        'soill': 'atmosphereSingleLayer',
        'soilw': 'depthBelowLandLayer',
        'st': 'depthBelowLandLayer',
        't': 'surface',
        'tp': 'surface'
    }
    processed_df = pd.DataFrame()
    for varname in varnames_to_process:
        if varname in conditions:
            level = conditions[varname]
            filtered_df = df[(df['varname'] == varname) & (df['typeOfLevel'] == level)]
            filtered_df = filtered_df.sort_values(by='length', ascending=False).drop_duplicates(subset=['time'], keep='first')
            processed_df = pd.concat([processed_df, filtered_df], ignore_index=True)
    return processed_df


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
    Process a single GFS file asynchronously:
      - Parse the GRIB index from the S3 URL,
      - Read a parquet mapping from GCS,
      - Map the index.
    """
    async with sem:
        try:
            fname = f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f{idx:03}"
            gcs_mapping_path = f"gs://{gcs_bucket_name}/time_idx/20231201/agfs-time-20231201-rt{idx:03}.parquet"
            gcs_fs = gcsfs.GCSFileSystem(token=gcp_service_account_json)
            loop = asyncio.get_event_loop()
            idxdf = await loop.run_in_executor(
                executor,
                partial(parse_grib_idx, basename=fname)
            )
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
            idxdf_filtered = idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
            mapped_index = await loop.run_in_executor(
                executor,
                partial(map_from_index, datestr, deduped_mapping, idxdf_filtered)
            )
            return mapped_index
        except Exception as e:
            print(f'Error in {str(e)}')
            return None


def process_files_in_batches(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    max_concurrent: int = 3,
    batch_size: int = 5,
    chunk_size: Optional[int] = None,
    gcp_service_account_json: Optional[str] = None
) -> pd.DataFrame:
    """
    Process files in batches with controlled concurrency.
    Combines the mapped indices and applies variable-specific processing.
    """
    dtaxes = axes[0]
    first_day_of_month = pd.to_datetime(date_str).replace(day=1).strftime('%Y%m%d')
    sem = asyncio.Semaphore(max_concurrent)
    results = []
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        for batch_start in range(0, len(dtaxes), batch_size):
            batch_end = min(batch_start + batch_size, len(dtaxes))
            batch_indices = range(batch_start, batch_end)
            tasks = [
                process_single_file(
                    date_str,
                    first_day_of_month,
                    gcs_bucket_name,
                    idx,
                    dtaxes[idx],
                    sem,
                    executor,
                    gcp_service_account_json,
                    chunk_size
                )
                for idx in batch_indices
            ]
            batch_results = loop.run_until_complete(asyncio.gather(*tasks))
            valid_results = [r for r in batch_results if r is not None]
            results.extend(valid_results)
            print("event batch_processed")
    if not results:
        raise ValueError(f"No valid mapped indices created for date {date_str}")
    gfs_kind = pd.concat(results, ignore_index=True)
    gfs_kind_var = gfs_kind.drop_duplicates('varname')
    var_list = gfs_kind_var['varname'].tolist()
    var_to_remove = ['acpcp', 'cape', 'cin', 'pres', 'r', 'soill', 'soilw', 'st', 't', 'tp']
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))
    gfs_kind1 = gfs_kind.loc[gfs_kind.varname.isin(var1_list)]
    to_process_df = gfs_kind.loc[gfs_kind['varname'].isin(var_to_remove)]
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
    gcp_service_account_json: Optional[str] = None
) -> pd.DataFrame:
    """
    Async wrapper for creating the mapped index.
    """
    return asyncio.run(
        process_files_in_batches(
            axes,
            gcs_bucket_name,
            date_str,
            max_concurrent,
            batch_size,
            chunk_size,
            gcp_service_account_json
        )
    )


def prepare_zarr_store(deflated_gfs_grib_tree_store: dict, gfs_kind: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """
    Prepare a Zarr store and associated chunk index from the deflated GRIB tree and mapped index.
    """
    print("Preparing Zarr Store")
    zarr_ref_store = deflated_gfs_grib_tree_store
    chunk_index = gfs_kind
    zstore = copy.deepcopy(zarr_ref_store["refs"])
    return zstore, chunk_index


def process_unique_groups(zstore: dict, chunk_index: pd.DataFrame, time_dims: Dict, time_coords: Dict,
                          times: np.ndarray, valid_times: np.ndarray, steps: np.ndarray) -> dict:
    """
    Process each unique group (by varname, stepType, typeOfLevel) and update the Zarr store.
    """
    print("Processing Unique Groups and Updating Zarr Store")
    unique_groups = chunk_index.set_index(["varname", "stepType", "typeOfLevel"]).index.unique()
    for key in list(zstore.keys()):
        lookup = tuple([val for val in os.path.dirname(key).split("/")[:3] if val])
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
            store_coord_var(key=f"{base_path}/step", zstore=zstore, coords=time_coords["step"],
                            data=steps.astype("timedelta64[s]").astype("float64") / 3600.0)
            store_coord_var(key=f"{base_path}/{key[2]}", zstore=zstore,
                            coords=(key[2],) if hasattr(lvals, 'shape') and lvals.shape else (),
                            data=lvals)
            store_data_var(key=f"{base_path}/{key[0]}", zstore=zstore, dims=dims, coords=coords,
                           data=group, steps=steps, times=times,
                           lvals=lvals if hasattr(lvals, 'shape') and lvals.shape else None)
        except Exception as e:
            print(f"Skipping processing group {key}: {str(e)}")
    return zstore


def zstore_dict_to_df(zstore: dict):
    """
    Convert a Zarr store dictionary to a pandas DataFrame.
    """
    data = []
    for key, value in zstore.items():
        if isinstance(value, (dict, list, int, float, np.integer, np.floating)):
            value = str(value).encode('utf-8')
        data.append((key, value))
    return pd.DataFrame(data, columns=['key', 'value'])


def create_parquet_df(zstore: dict, date_str: str, run_str: str, source: str = "aws_s3") -> pd.DataFrame:
    """
    Convert a Zarr store dictionary to a DataFrame with additional metadata columns.
    """
    gfs_store = dict(refs=zstore, version=1)
    zstore_df = zstore_dict_to_df(gfs_store)
    zstore_df["date"] = date_str
    zstore_df["run"] = run_str
    zstore_df["source"] = source
    return zstore_df


def nonclusterworker_upload_to_gcs(bucket_name, source_file_name, destination_blob_name, dask_worker_credentials_path):
    """
    Upload a file to a GCS bucket using the provided service account credentials.
    """
    try:
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

# --- Functions used by run_day_stream_gfs_gik_to_zarr.py ---

from enum import Enum, auto

class StorageType(Enum):
    LOCAL = auto()
    GCS = auto()


def download_parquet_from_gcs(
    gcs_bucket_name: str,
    year: str,
    date_str: str,
    run_str: str,
    service_account_json: str,
    local_save_path: str = "./"
):
    """
    Download a Parquet file from GCS and save it locally.
    """
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
    )
    gcs_file_url = f"gs://{gcs_bucket_name}/gik_day_parqs/{year}/gfs_{date_str}_{run_str}.par"
    ddf = dd.read_parquet(
        gcs_file_url,
        storage_options={"token": credentials},
        engine="pyarrow",
    )
    pdf = ddf.compute()
    local_file_path = os.path.join(local_save_path, f"gfs_{date_str}_{run_str}.par")
    pdf.to_parquet(local_file_path)
    print(f"Parquet file saved locally at: {local_file_path}")
    return local_file_path


def worker_upload_required_files(client, parquet_path: str, credentials_path: str):
    """
    Upload required files to all workers.
    """
    client.upload_file(credentials_path)
    client.upload_file(parquet_path)
    client.upload_file('./utils.py')


def load_datatree_on_worker(parquet_path: str):
    """
    Load a datatree from a local Parquet file.
    """
    zstore_df = pd.read_parquet(parquet_path)
    zstore_df['value'] = zstore_df['value'].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
    )
    zstore = {row['key']: eval(row['value']) for _, row in zstore_df.iterrows()}
    return xr.open_datatree(
        fsspec.filesystem("reference", fo=zstore).get_mapper(""),
        engine="zarr",
        consolidated=False
    )


def get_worker_paths(dask_worker, date_str: str, run_str: str):
    """
    Get file paths for required files on the worker.
    """
    local_dir = pathlib.Path(dask_worker.local_directory)
    return {
        'credentials': str(local_dir / 'coiled-data-key.json'),
        'parquet': str(local_dir / f'gfs_{date_str}_{run_str}.par')
    }


def process_dataset_on_worker(
    var_path: str,
    date_str: str,
    run_str: str, 
    latlon_bounds: Dict[str, float],
) -> Tuple[str, xr.Dataset]:
    """
    Process a dataset on the worker and return a safe name along with the processed xarray Dataset.
    """
    worker = get_worker()
    location_paths_worker = get_worker_paths(worker, date_str, run_str)
    gfs_dt = load_datatree_on_worker(location_paths_worker['parquet'])
    ds = gfs_dt[var_path].to_dataset()
    if "step" in ds.coords:
        ds = ds.drop_vars(["step"])
    if "isobaricInhPa" in ds.coords:
        ds = ds.sel(isobaricInhPa=200)
    safe_name = var_path.strip("/").replace("/", "_")
    ds1 = ds.compute()
    print(f'message from worker {safe_name}') 
    return safe_name, ds1


def save_dataset_locally(
    safe_name: str,
    dataset: xr.Dataset,
    local_save_path: str
) -> str:
    """
    Save the processed dataset locally as a Zarr store.
    """
    try:
        os.makedirs(local_save_path, exist_ok=True)
        store_path = os.path.join(local_save_path, f"{safe_name}.zarr")
        dataset.to_zarr(store_path, mode="w", consolidated=True)
        return f"Created new store at {store_path}"
    except Exception as e:
        return f"Error saving dataset {safe_name}: {str(e)}"


def process_and_save(
    var_path: str,
    gcs_bucket: str,
    gcs_path: str,
    date_str: str,
    run_str: str,
    project_id: str,
    latlon_bounds: Dict[str, float] = {"lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0}
) -> str:
    """
    Process a single dataset and save (or append) it to a Zarr store on GCS.
    """
    try:
        worker = get_worker()
        location_paths_worker = get_worker_paths(worker, date_str, run_str)
        gcs = gcsfs.GCSFileSystem(
            token=location_paths_worker['credentials'],
            project=project_id
        )
        gfs_dt = load_datatree_on_worker(location_paths_worker['parquet'])
        ds = gfs_dt[var_path].to_dataset()
        if 'step' in ds.coords:
            ds = ds.drop_vars(['step'])
        if 'isobaricInhPa' in ds.coords:
            ds = ds.sel(isobaricInhPa=200)
        var_name = var_path.strip('/').replace('/', '_')
        store_path = f"{gcs_bucket}/{gcs_path}/{var_name}.zarr"
        store_exists = gcs.exists(store_path)
        if store_exists:
            try:
                existing_ds = xr.open_zarr(gcs.get_mapper(store_path))
                new_times = ds.valid_times.values
                existing_times = existing_ds.valid_times.values
                times_to_append = new_times[~np.isin(new_times, existing_times)]
                if len(times_to_append) > 0:
                    ds_to_append = ds.sel(valid_times=times_to_append)
                    if not (np.array_equal(existing_ds.lat.values, ds_to_append.lat.values) and
                            np.array_equal(existing_ds.lon.values, ds_to_append.lon.values)):
                        raise ValueError("Lat/lon bounds don't match existing dataset")
                    ds_to_append.to_zarr(
                        store=gcs.get_mapper(store_path),
                        mode='a',
                        append_dim='valid_times',
                        consolidated=True
                    )
                    return f"Appended {len(times_to_append)} new times to gs://{store_path}"
                else:
                    return f"No new times to append for gs://{store_path}"
            except Exception as e:
                print(f"Error appending to store: {e}")
                print("Creating new store instead")
                store_exists = False
        if not store_exists:
            ds.to_zarr(
                store=gcs.get_mapper(store_path),
                mode='w',
                consolidated=True
            )
            return f"Created new store at gs://{store_path}"
    except Exception as e:
        return f"Error processing {var_path}: {str(e)}"


def process_and_upload_datatree(
    parquet_path: str,
    gcs_bucket: str, 
    gcs_path: str,
    client,
    credentials_path: str,
    date_str: str,
    run_str: str, 
    project_id: str,
    storage_type: StorageType = StorageType.LOCAL,
    local_save_path: Optional[str] = None,
    latlon_bounds: Dict[str, float] = {"lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0}
) -> List[str]:
    """
    Process a datatree from a Parquet file and upload the resulting Zarr stores.
    Depending on storage_type, the stores are either saved locally or uploaded to GCS.
    """
    if local_save_path is None:
        local_save_path = os.path.join("zarr_stores", f"{date_str}_{run_str}")
    if storage_type == StorageType.LOCAL:
        os.makedirs(local_save_path, exist_ok=True)
    worker_upload_required_files(client, parquet_path, credentials_path)
    temp_dt = load_datatree_on_worker(parquet_path)
    variable_paths = [node.path for node in temp_dt.subtree if node.has_data]
    results = []
    for var_path in variable_paths:
        try:
            if storage_type == StorageType.LOCAL:
                print(f'original path {var_path}')
                future = client.submit(
                    process_dataset_on_worker,
                    var_path=var_path,
                    date_str=date_str,
                    run_str=run_str,
                    latlon_bounds=latlon_bounds
                )
                safe_name, processed_ds = future.result()
                print(safe_name) 
                result = save_dataset_locally(safe_name, processed_ds, local_save_path)
                results.append(result)
            else:
                future = client.submit(
                    process_and_save,
                    var_path=var_path,
                    gcs_bucket=gcs_bucket,
                    gcs_path=gcs_path,
                    date_str=date_str,
                    run_str=run_str,
                    project_id=project_id,
                    latlon_bounds=latlon_bounds
                )
                results.append(future.result())
        except Exception as e:
            results.append(f"Error processing {var_path}: {str(e)}")
    storage_type_str = "locally" if storage_type == StorageType.LOCAL else "to GCS"
    print(f"Processed {len(results)} datasets {storage_type_str}")
    return results

