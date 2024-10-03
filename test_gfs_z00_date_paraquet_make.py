from utils_gfs_aws import process_gfs_data



mapping_parquet_file_path='gfs_mapping3_120hr/'

list_ds=['20210101','20210125','20210501','20210803','20211108','20220104','20220501','20220815',
         '20221201','20230108','20240405','20240928','20241001']

#list_ds=['20240102']

for date_str in list_ds:
    output_parquet_file = f'gfs-z00-{date_str}-allvar.parquet'
    try:
        process_gfs_data(date_str, mapping_parquet_file_path, output_parquet_file)
    except Exception as e:  # Catch generic exceptions or specify the type of exception
        print(f"Failed to process data for {date_str}: {e}")
    else:
        print(f"Successfully processed data for {date_str}")
    finally:
        print(f"Completed processing attempt for {date_str}")





