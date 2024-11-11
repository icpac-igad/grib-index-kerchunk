from utils_gfs_aws_v2 import process_gfs_data
from utils_gfs_aws_v2 import upload_file_to_gcs


mapping_parquet_file_path='aws_gfs_built_map/'

list_ds=['20231001']

#list_ds=['20240102']

for date_str in list_ds:
    output_parquet_file = f'gfs-z00-{date_str}-allvar.parquet'
    output_log_file= f'gfs_processing_{date_str}.log'
    try:
        process_gfs_data(date_str, mapping_parquet_file_path, output_parquet_file)
        bucket_name= 'gik-gfs-aws-tf'
        year = date_str[:4]
        month = date_str[4:6]
        log_destination=  f'{year}/{month}/{output_log_file}'
        upload_file_to_gcs(bucket_name, output_log_file, log_destination)
        parquet_destination= f'{year}/{month}/{output_parquet_file}'
        upload_file_to_gcs(bucket_name, output_parquet_file, parquet_destination)
    except Exception as e:  # Catch generic exceptions or specify the type of exception
        print(f"Failed to process data for {date_str}: {e}")
    else:
        print(f"Successfully processed data for {date_str}")
    finally:
        print(f"Completed processing attempt for {date_str}")





