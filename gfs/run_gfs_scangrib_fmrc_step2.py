# main.py
import os
import sys
import dask
import coiled
import logging
from dotenv import load_dotenv

import dask
import coiled
import pathlib
from datetime import datetime, timedelta

load_dotenv()

# Add the GFS_PROCESSOR_PATH to sys.path
gfs_processor_path = os.getenv("sgutils_path")
gcs_sa_path = os.getenv("gcs_sa_path")
if gfs_processor_path and os.path.isdir(gfs_processor_path):
    sys.path.append(gfs_processor_path)
else:
    raise ValueError("GFS_PROCESSOR_PATH is not defined or is not a valid directory.")

from utils import gfs_s3_url_maker, process_gfs_time_idx_data, setup_logger, get_filename_from_path

#from utils_gfs_aws import gfs_s3_url_maker, process_gfs_time_idx_data, setup_logger, get_filename_from_path


# Step 1: Generate the list of dates (1st day of each month in 2023)
date_list = [datetime(2023, month, 1).strftime('%Y%m%d') for month in range(1, 13)]

# Placeholder for all URLs, with each month's URLs stored in a sublist
cont_date = []

# Step 2: Generate URLs for each date and append to cont_date
for date_str in date_list:
    urls = gfs_s3_url_maker(date_str)  # Assuming this function returns a list of URLs
    cont_date.append(urls)

# Step 3: Flatten cont_date into a single list
flat = [url for sublist in cont_date for url in sublist]

# Print results
print("Date List:", date_list)
print("Flattened URL List:", flat)

# Set up basic logging configuration
def setup_logger():
    """Configure logging for both main process and workers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )
    return logging.getLogger(__name__)


# Set up logging for main process
logger = setup_logger()
    
logger.info("Initializing Dask cluster...")

credentials_full_path = gcs_sa_path
# Configuration
#credentials_path = gcs_sa_path
csv_bucket_name = os.environ.get('CSV_BUCKET_NAME', 'gik-gfs-aws-tf')

if not credentials_full_path:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

# Get just the filename
credentials_filename = get_filename_from_path(credentials_full_path)
logger.info(f"Using credentials file: {credentials_filename}")

logger.info("Initializing Dask cluster...")

# Initialize cluster
cluster = coiled.Cluster(
    n_workers=50,
    name=f"gik-test2",
    software="gik-coiled-v6",
    scheduler_vm_types=["n2-standard-2"],
    region="us-east1",
    arm=False,
    compute_purchase_option="spot",
    tags={"workload": "gefs-arm-test0"},
    worker_vm_types="n2-standard-2",)

client = cluster.get_client()
#cluster =coiled.Cluster(name='gik-test2')
#client = cluster.get_client()
    
# Upload all necessary files to workers
logger.info("Uploading files to workers...")
client.upload_file(f"{gfs_processor_path}dynamic_zarr_store.py")
client.upload_file(f"{gfs_processor_path}utils_gfs_aws.py")
# Upload the credentials file from its full path to workers
logger.info(f"Uploading credentials file from {credentials_full_path}")
client.upload_file(credentials_full_path)


def get_worker_credentials_path(dask_worker):
    creds_path = pathlib.Path(dask_worker.local_directory) / 'coiled-data-key.json'
    return str(creds_path)
    
#worker_cred_path = client.run(get_worker_credentials_path)

try:
    # Process data
    logger.info("Starting data processing...")
    #date_str="20240101"
    #urls = gfs_s3_url_maker(date_str)  # Example date
    results = []
    logger.info(f"% of files to process on {date_str} run on 00: {len(urls)}")
    for url in urls:
        result = process_gfs_time_idx_data(
            url,
            csv_bucket_name,
        )
        results.append(result)
    
    logger.info("Computing results...")
    final_results = dask.compute(results)
    logger.info("Processing completed successfully")
    
except Exception as e:
    logger.error(f"Error during processing: {str(e)}")
    raise
finally:
    # Cleanup
    logger.info("Cleaning up resources...")
    client.close()
    cluster.close()
    cluster.shutdown()
