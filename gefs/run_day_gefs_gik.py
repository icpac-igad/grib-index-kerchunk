import pandas as pd

from gefs_util import generate_axes
from gefs_util import filter_build_grib_tree 
from gefs_util import KerchunkZarrDictStorageManager

from gefs_util import calculate_time_dimensions
from gefs_util import cs_create_mapped_index

from gefs_util import prepare_zarr_store
from gefs_util import process_unique_groups
from gefs_util import create_parquet_file

date_str = '20241112'
ensemble_member = 'gep01'
axes = generate_axes(date_str)

gefs_files = [
    f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.t00z.pgrb2s.0p25.f000",
    f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.t00z.pgrb2s.0p25.f003"
]

forecast_dict = {
    "Surface pressure": "PRES:surface",
    "Downward short-wave radiation flux": "DSWRF:surface",
    "Convective available potential energy": "CAPE:surface",
    "Upward short-wave radiation flux": "USWRF:surface",
    "Total Precipitation": "APCP:surface",
    "Wind speed (gust)": "GUST:surface",
    "2 metre temperature": "TMP:2 m above ground",
    "2 metre relative humidity": "RH:2 m above ground",
    "10 metre U wind component": "UGRD:10 m above ground",
    "10 metre V wind component": "VGRD:10 m above ground",
    "Precipitable water": "PWAT:atmosphere",
    "Total Cloud Cover": "TCDC:atmosphere",
    "Geopotential height": "HGT:cloud ceiling"
}

_, deflated_gefs_grib_tree_store = filter_build_grib_tree(gefs_files, forecast_dict)

time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
gcs_bucket_name = 'gik-gefs-aws-tf'
gcp_service_account_json = 'coiled-data-key.json'

gefs_kind = cs_create_mapped_index(
    axes, gcs_bucket_name, date_str, ensemble_member, gcp_service_account_json=gcp_service_account_json
)

zstore, chunk_index = prepare_zarr_store(deflated_gefs_grib_tree_store, gefs_kind)
updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords, times, valid_times, steps)
output_parquet_file = f'gefs_{ensemble_member}_{date_str}.par'

create_parquet_file(updated_zstore, output_parquet_file)