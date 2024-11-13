from utils import generate_axes
from utils import calculate_time_dimensions
from utils import cs_create_mapped_index

date_str = '20241104'
axes = generate_axes(date_str)
time_dims, time_coords, times, valid_times, steps = calculate_time_dimensions(axes)
gcs_bucket_name = 'gik-gfs-aws-tf'
gcp_service_account_json = 'coiled-data-key.json'

gfs_kind = cs_create_mapped_index(
    axes, gcs_bucket_name, date_str, gcp_service_account_json=gcp_service_account_json
)
gfs_kind.to_parquet(f'{date_str}idx.parquet')
