import logging
import fsspec
import json 

from utils import setup_logging
from utils import s3_parse_ecmwf_grib_idx
from utils import _map_grib_file_by_group
from utils import s3_parse_ecmwf_grib_idx
from utils import zip_folder
from utils import recreate_folder

date_str='20240529'
recreate_folder(f'date_str')
log_level=logging.INFO
log_file = f"{date_str}/ecmwf_buildidx_table_{date_str}.log"
setup_logging(log_level, log_file)
logger = logging.getLogger()

try:
    basename=f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-3h-enfo-ef.grib2"
    # with open('ecmwf_deflated_grib_tree_store.pkl', 'rb') as pickle_file:
    #     # Loading the data from the pickle file
    #     deflated_gfs_grib_tree_store = pickle.load(pickle_file)
    fs=fsspec.filesystem("s3")
    suffix= "index"
    idx_file_index = s3_parse_ecmwf_grib_idx(
        fs=fs, basename=basename, suffix=suffix
    )
    idx_file_index.to_parquet(f'{date_str}/ecmwf_index_{date_str}.parquet', engine='pyarrow')
    total_groups=len(idx_file_index.index)
    dd=_map_grib_file_by_group(basename,total_groups)
    dd.to_parquet(f'{date_str}/ecmwf_scangrib_metadata_table_{date_str}.parquet', engine='pyarrow')
    mapping = s3_ecwmf_build_idx_grib_mapping(
        fs=fsspec.filesystem("s3"),
        basename=f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-3h-enfo-ef.grib2",
        date_str=date_str
    )
    output_parquet_file=f'{date_str}/ecmwf_buildidx_table_{date_str}.parquet'
    mapping.to_parquet(output_parquet_file, engine='pyarrow')
    logger.info(json.dumps({
                "event": "processing_completed",
                "date": date_str,
                "output_file": output_parquet_file
            }))
    zip_folder(f'{date_str}', f'{date_str}.zip')
except Exception as e:
    logger.error(f"Error occurred: {str(e)}")
finally:
    # Close the log file
    logging.shutdown()
    print("Log file closed gracefully.")

