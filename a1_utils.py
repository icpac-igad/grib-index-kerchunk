
import os
import importlib


from datetime import datetime, timedelta
import copy
import xarray as xr
import numpy as np
import pandas as pd
import fsspec
import kerchunk
from kerchunk.grib2 import scan_grib, grib_tree
import gcsfs
import datatree
import pickle
# This could be generalized to any gridded FMRC dataset but right now it works with NOAA's Grib2 files
import dynamic_zarr_store
from dotenv import load_dotenv


def get_gefs_gcs_gribtree(ens_mem):
    gfs_files = [
    f"gs://gfs-ensemble-forecast-system/gefs.20201103/00/atmos/pgrb2sp25/gep{ens_mem}.t00z.pgrb2s.0p25.f000",
    f"gs://gfs-ensemble-forecast-system/gefs.20201103/00/atmos/pgrb2sp25/gep{ens_mem}.t00z.pgrb2s.0p25.f003"]
    # This operation reads two of the large Grib2 files from GCS
    # scan_grib extracts the zarr kerchunk metadata for each individual grib message
    # grib_tree builds a zarr/xarray compatible hierarchical view of the dataset
    gfs_grib_tree_store = grib_tree([group for f in gfs_files for group in scan_grib(f)])
    deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
    dynamic_zarr_store.strip_datavar_chunks(deflated_gfs_grib_tree_store)
    with open(f'gefs_deflated_grib_{ens_mem}_tree_store.pkl', 'wb') as pickle_file:
        pickle.dump(deflated_gfs_grib_tree_store, pickle_file)
    print(ens_mem)



def get_gefs_gcs_mapping(ens_mem):
    mapping = dynamic_zarr_store.build_idx_grib_mapping(
    fs=fsspec.filesystem("gcs"),basename=f"gs://gfs-ensemble-forecast-system/gefs.20201103/00/atmos/pgrb2sp25/gep{ens_mem}.t00z.pgrb2s.0p25.f003")
    mapping.to_parquet(f'gefs_ens_{ens_mem}_mapping_table.parquet', engine='pyarrow')



def needed_time_axes(date_str):
    """
    getting 1+2 days of 3 hour data
    """
    original_date = datetime.strptime(date_str, '%Y%m%d')
    formatted_date_one = original_date.strftime('%Y-%m-%dT03:00')
    # Add two days for the second date and format it
    date_two = original_date + timedelta(days=2)
    formatted_date_two = date_two.strftime('%Y-%m-%dT00:00')
    axes = [
    pd.Index(
        [
            pd.timedelta_range(start="0 hours", end="3 hours", freq="3h", closed="right", name="3 hour"),
        ],
        name="step"
    ),
    pd.date_range(formatted_date_one, formatted_date_two, freq="3H", name="valid_time")]
    return axes



def ens_read(ens_mem,db_par):
    # Opening the pickle file in binary read mode
    with open(f'gefs_tree/gefs_deflated_grib_{ens_mem}_tree_store.pkl', 'rb') as pickle_file:
        # Loading the data from the pickle file
        #deflated_gfs_grib_tree_store = pickle.load(pickle_file)
        pkl_ld = pickle.load(pickle_file)
    mapping= pd.read_parquet(f'table/gefs_ens_{ens_mem}_mapping_table.parquet')
    #dmap = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
    e_db=db_par[db_par['ens_mem']==ens_mem]
    return pkl_ld,mapping,e_db


def make_store(date_str,ens_mem):
    #date_str=
    need_axes=needed_time_axes(date_str)
    db_par=pd.read_parquet(f'gefs-{date_str}18.paraquet')
    #ens_mem='01'
    pkl_ld,dmap,e_db=ens_read(ens_mem,db_par)
    gfs_store = dynamic_zarr_store.reinflate_grib_store(
        axes=need_axes,
        aggregation_type=dynamic_zarr_store.AggregationType.HORIZON,
        chunk_index=e_db.loc[~e_db.varname.isin(["cape","cin","cpofp","gh","hlcy","mslet","soilw","st","vis"])],
        #chunk_index=e_db,
        zarr_ref_store=pkl_ld)
    gfs_dt = datatree.open_datatree(fsspec.filesystem("reference", fo=gfs_store).get_mapper(""), engine="zarr", consolidated=False)
    #gfs_dt.data = gfs_dt.data.assign(member=int(ens_mem))
    return gfs_dt
