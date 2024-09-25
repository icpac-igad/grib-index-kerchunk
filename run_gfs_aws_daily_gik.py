import os
import logging
import pandas as pd
import numpy as np
import fsspec
import copy
from typing import List, Dict, Tuple
from dynamic_zarr_store import (
    AggregationType, grib_tree, scan_grib, strip_datavar_chunks,
    parse_grib_idx, map_from_index, store_coord_var, store_data_var
)

logger = logging.getLogger(__name__)

def setup_logging(log_level: int = logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

def build_grib_tree(gfs_files: List[str]) -> Tuple[dict, dict]:
    """Build and deflate the Grib tree."""
    logger.info("Building Grib Tree")
    gfs_grib_tree_store = grib_tree([group for f in gfs_files for group in scan_grib(f)])
    deflated_gfs_grib_tree_store = copy.deepcopy(gfs_grib_tree_store)
    strip_datavar_chunks(deflated_gfs_grib_tree_store)
    logger.info(f"Original references: {len(gfs_grib_tree_store['refs'])}")
    logger.info(f"Stripped references: {len(deflated_gfs_grib_tree_store['refs'])}")
    return gfs_grib_tree_store, deflated_gfs_grib_tree_store

def calculate_time_dimensions(axes: List[pd.Index]) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate time dimensions and coordinates."""
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

def create_mapped_index(axes: List[pd.Index], mapping_parquet_file_path: str) -> pd.DataFrame:
    """Create mapped index from GFS files."""
    logger.info("Creating Mapped Index")
    mapped_index_list = []
    dtaxes = axes[0]

    for idx, datestr in enumerate(dtaxes):
        try:
            fname = f"s3://noaa-gfs-bdp-pds/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f{idx:03}"
            
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
    logger.info(f"Mapped collected multiple variables index info: {gfs_kind1.info()}")
    return gfs_kind1

def prepare_zarr_store(deflated_gfs_grib_tree_store: dict, gfs_kind: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """Prepare Zarr store and chunk index."""
    logger.info("Preparing Zarr Store")
    zarr_ref_store = deflated_gfs_grib_tree_store
    #chunk_index = gfs_kind.loc[gfs_kind.varname.isin(["t2m"])]
    chunk_index = gfs_kind
    zstore = copy.deepcopy(zarr_ref_store["refs"])
    return zstore, chunk_index

def process_unique_groups(zstore: dict, chunk_index: pd.DataFrame, time_dims: Dict, time_coords: Dict,
                          times: np.ndarray, valid_times: np.ndarray, steps: np.ndarray) -> dict:
    """Process unique groups and update Zarr store."""
    logger.info("Processing Unique Groups and Updating Zarr Store")
    unique_groups = chunk_index.set_index(
        ["varname", "stepType", "typeOfLevel"]
    ).index.unique()

    for key in list(zstore.keys()):
        lookup = tuple(
            [val for val in os.path.dirname(key).split("/")[:3] if val != ""]
        )
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
                raise ValueError("Invalid lvals")

            store_coord_var(
                key=f"{base_path}/time",
                zstore=zstore,
                coords=time_coords["time"],
                data=times.astype("datetime64[s]"),
            )

            store_coord_var(
                key=f"{base_path}/valid_time",
                zstore=zstore,
                coords=time_coords["valid_time"],
                data=valid_times.astype("datetime64[s]"),
            )

            store_coord_var(
                key=f"{base_path}/step",
                zstore=zstore,
                coords=time_coords["step"],
                data=steps.astype("timedelta64[s]").astype("float64") / 3600.0,
            )

            store_coord_var(
                key=f"{base_path}/{key[2]}",
                zstore=zstore,
                coords=(key[2],) if lvals.shape else (),
                data=lvals,
            )

            store_data_var(
                key=f"{base_path}/{key[0]}",
                zstore=zstore,
                dims=dims,
                coords=coords,
                data=group,
                steps=steps,
                times=times,
                lvals=lvals if lvals.shape else None,
            )
        except Exception as e:
            logger.error(f"Error processing group {key}: {str(e)}")

    return zstore

def create_parquet_file(zstore: dict, output_parquet_file: str):
    """Create and save Parquet file."""
    logger.info("Creating and Saving Parquet File")
    gfs_store = dict(refs=zstore, version=1)

    def dict_to_df(zstore: dict):
        data = []
        for key, value in zstore.items():
            if isinstance(value, (dict, list)):
                value = str(value).encode('utf-8')
            elif isinstance(value, (int, float, np.integer, np.floating)):
                value = str(value).encode('utf-8')
            data.append((key, value))
        return pd.DataFrame(data, columns=['key', 'value'])

    zstore_df = dict_to_df(gfs_store)
    zstore_df.to_parquet(output_parquet_file)
    logger.info(f"Parquet file saved to {output_parquet_file}")

def process_gfs_data(gfs_files: List[str], axes: List[pd.Index], mapping_parquet_file_path: str, output_parquet_file: str, log_level: int = logging.INFO):
    """Main function to process GFS data."""
    setup_logging(log_level)
    
    try:
        _, deflated_gfs_grib_tree_store = build_grib_tree(gfs_files)
        time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
        gfs_kind = create_mapped_index(axes, mapping_parquet_file_path)
        zstore, chunk_index = prepare_zarr_store(deflated_gfs_grib_tree_store, gfs_kind)
        updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, times, valid_times, steps)
        create_parquet_file(updated_zstore, output_parquet_file)
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    gfs_files = [
        "s3://noaa-gfs-bdp-pds/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f000",
        "s3://noaa-gfs-bdp-pds/gfs.20230928/00/atmos/gfs.t00z.pgrb2.0p25.f001"
    ]
    axes = [
        pd.Index(pd.date_range("2023-09-01T00:00", "2023-09-03T12:00", freq="60min", name="valid_time")),
        pd.Index([pd.Timestamp("2023-09-01T00:00")], name="time")
    ]
    mapping_parquet_file_path = 'gfs_mapping3_120hr/'
    output_parquet_file = 'aws-best-avlbl-61hrs-t2m.parquet'

    process_gfs_data(gfs_files, axes, mapping_parquet_file_path, output_parquet_file)
