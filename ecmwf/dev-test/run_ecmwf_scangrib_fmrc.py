import logging
import fsspec
import json
import os
import pandas as pd
import numpy as np
import re
import coiled 
from dask.distributed import Client, get_worker

from fmrc_utils import setup_logging
from fmrc_utils import s3_ecmwf_scan_grib_storing
from fmrc_utils import nonclusterworker_upload_to_gcs
from fmrc_utils import ecmwf_s3_url_maker



date_str='20240529'
#urls = generate_urls(date_str)
urls=ecmwf_s3_url_maker(date_str)
#create_folder(date_str)

#coiled notebook start --name gik-day --vm-type n2-standard-2 --software gik-coiled-v6
@coiled.function(
    # memory="2 GiB",
    vm_type="n2-standard-2",
    #software="latex-improver-docker-coiled-env-v1-test",
    software="gik-coiled-v6",
    #container="367165744789.dkr.ecr.us-east-1.amazonaws.com/latex-coiled:v1-test",
    name=f"func-ecmwf3",
    region="us-east1",  # Specific region
    arm=False,  # Change architecture
    idle_timeout="30 minutes",
)
def run_esg(url):
    gcs_bucket_name='gik-ecmwf-aws-tf'
    #gcp_service_account_json='coiled-data.json'
    # Resolve the path to the uploaded credentials file
    # Check current working directory and files
    worker = get_worker()  # Call the function, don't just reference it
    local_dir = worker.local_directory  # Access the attribute directly
    
    # Construct path to the credentials file
    gcp_service_account_json = os.path.join(local_dir, "coiled-data.json")

    if not os.path.exists(gcp_service_account_json):
        raise FileNotFoundError(f"Credentials file not found at {gcp_service_account_json}")

    fs = fsspec.filesystem("s3")
    suffix = "index"
    #hr = url.split('-')[-1].split('h')[0]  # Extract the hour value from the URL
    match = re.search(r'\b\d+h\b', url)
    if match:
        ecmwf_hr = match.group()
    log_level = logging.INFO
    log_file = f"e_sg_mdt_{date_str}_{ecmwf_hr}.log"
    
    # Set up logging for each URL
    setup_logging(log_level, log_file)
    logger = logging.getLogger()
    #ecmwf_hr=ehr_str
    try:        
        # Build mapping and save to Parquet
        s3_ecmwf_scan_grib_storing(fs, url, date_str,suffix,ecmwf_hr,gcs_bucket_name,gcp_service_account_json)
        #output_parquet_file = f'{date_str}/ecmwf_buildidx_table_{date_str}_{idx}.parquet'
        #mapping.to_parquet(output_parquet_file, engine='pyarrow')
    except Exception as e:
        # Log the error specific to the current hour's URL
        logger.error(f"Error occurred while processing hour {ecmwf_hr}: {str(e)}")
    
    finally:
        # Close the log for the current hour and reset logging for next loop
        logging.shutdown()
        log_file_destination_blob_name=f'fmrc/scan_grib{date_str}/{log_file}'
        nonclusterworker_upload_to_gcs(
            bucket_name=gcs_bucket_name,
            source_file_name=log_file,
            destination_blob_name=log_file_destination_blob_name,
            dask_worker_credentials_path=gcp_service_account_json
        )
        print(f"Log for hour {ecmwf_hr} closed gracefully.")
        return f"run {ecmwf_hr}"
    # Zip the output folder after processing all URLs
    #zip_folder(f'{date_str}', f'{date_str}.zip')w

#for url in urls[0:1]:
run_esg.cluster.adapt(minimum=1, maximum=10)
func_env = run_esg.cluster.get_client()
func_env.upload_file("utils.py")
func_env.upload_file("dynamic_zarr_store.py")
func_env.upload_file("coiled-data.json")
#future = run_esg.submit(url)
future = run_esg.map(urls)
future.result()

