import pandas as pd

from utils import generate_axes
from utils import filter_build_grib_tree 
from utils import KerchunkZarrDictStorageManager

from utils import calculate_time_dimensions
from utils import cs_create_mapped_index

from utils import prepare_zarr_store
from utils import process_unique_groups
from utils import create_parquet_df 
from utils import nonclusterworker_upload_to_gcs

date_str = '20241113'
run_str='00'

axes = generate_axes(date_str)

gfs_files = [
    f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f000",
    f"s3://noaa-gfs-bdp-pds/gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.0p25.f001"
]

forecast_dict = {
    "Convective available potential energy": "CAPE:surface",
    "Convective precipitation (water)": "ACPCP:surface",
    "Medium cloud cover": "MCDC:middle cloud layer",
    "Surface pressure": "PRES:surface",
    "Surface upward short-wave radiation flux": "USWRF:surface",
    "Surface downward short-wave radiation flux": "DSWRF:surface",
    "2 metre temperature": "TMP:2 m above ground",
    "Cloud water": "CWAT",
    "Precipitable water": "PWAT",
    "Ice water mixing ratio": "ICMR:200 mb",
    "Cloud mixing ratio": "CLMR:200 mb",
    "Rain mixing ratio": "RWMR:200 mb",
    "Total Precipitation": "APCP:surface",
    "U component of wind": "UGRD:200 mb",
    "V component of wind": "VGRD:200 mb"
}

_, deflated_gfs_grib_tree_store = filter_build_grib_tree(gfs_files,forecast_dict)



time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
gcs_bucket_name = $env
gcp_service_account_json = $env 

gfs_kind = cs_create_mapped_index(
    axes, gcs_bucket_name, date_str, gcp_service_account_json=gcp_service_account_json
)
#gfs_kind.to_parquet(f'{date_str}idx.parquet')
time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
#gfs_kind=pd.read_parquet(f'{date_str}idx.parquet')
zstore, chunk_index = prepare_zarr_store(deflated_gfs_grib_tree_store, gfs_kind)
updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, times, valid_times, steps)
zdf=create_parquet_df(updated_zstore,date_str,run_str)

output_parquet_file=f'gfs_{date_str}_{run_str}.par'
zdf.to_parquet(output_parquet_file)


