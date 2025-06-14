"""
GEFS Index Pre-processing Module

This module creates and stores GEFS index mappings in GCS bucket to enable fast processing.
This is the equivalent of the GFS pre-processing routine that creates mapping files
stored in GCS to circumvent index creation during main processing.

Usage:
    1. Run this pre-processing once per date/ensemble member to create mappings
    2. Main GEFS processing routine will use these pre-built mappings for fast execution
"""

import asyncio
import concurrent.futures
import os
import pathlib
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional
import logging

import dask
import fsspec
import gcsfs
import pandas as pd
from distributed import get_worker
from google.cloud import storage
from google.oauth2 import service_account

from kerchunk._grib_idx import (
    build_idx_grib_mapping,
    parse_grib_idx,
)

logger = logging.getLogger("gefs-preprocessing-logs")


def setup_gefs_logging(log_level: int = logging.INFO, log_file: str = "gefs_preprocessing.log"):
    """Set up logging for GEFS preprocessing."""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)


def get_gefs_details(url):
    """Extract date, run, member, and forecast hour from GEFS URL."""
    import re
    pattern = r"s3://noaa-gefs-pds/gefs\.(\d{8})/(\d{2})/atmos/pgrb2sp25/(gep\d{2})\.t(\d{2})z\.pgrb2s\.0p25\.f(\d{3})"
    match = re.match(pattern, url)
    if match:
        date = match.group(1)    # '20241112'
        run = match.group(2)     # '00'
        member = match.group(3)  # 'gep01'
        hour = match.group(5)    # '003'
        return date, run, member, hour
    else:
        logger.warning(f"No match found for GEFS URL pattern: {url}")
        return None, None, None, None


def gefs_s3_url_maker(date_str, ensemble_member="gep01", max_forecast_hour=240):
    """
    Create S3 URLs for GEFS data for a specific ensemble member.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format
    - ensemble_member (str): Ensemble member (e.g., 'gep01', 'gep02', etc.)
    - max_forecast_hour (int): Maximum forecast hour (default 240 for 10 days)
    
    Returns:
    - List[str]: List of S3 URLs for GEFS files
    """
    fs_s3 = fsspec.filesystem("s3", anon=True)
    s3url_glob = fs_s3.glob(
        f"s3://noaa-gefs-pds/gefs.{date_str}/00/atmos/pgrb2sp25/{ensemble_member}.*"
    )
    s3url_only_grib = [f for f in s3url_glob if f.split(".")[-1] != "idx"]
    
    # Filter by forecast hour if needed
    fmt_s3og = []
    for url in s3url_only_grib:
        full_url = "s3://" + url
        _, _, _, hour_str = get_gefs_details(full_url)
        if hour_str and int(hour_str) <= max_forecast_hour:
            fmt_s3og.append(full_url)
    
    fmt_s3og = sorted(fmt_s3og)
    print(f"Generated {len(fmt_s3og)} GEFS URLs for date {date_str}, member {ensemble_member}")
    return fmt_s3og


def get_worker_creds_path_gefs(dask_worker):
    """Get credentials path for GEFS processing on worker."""
    return str(pathlib.Path(dask_worker.local_directory) / 'coiled-data-key.json')


def upload_gefs_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload GEFS mapping file to GCS bucket using worker credentials."""
    try:
        worker = get_worker()
        worker_creds_path = get_worker_creds_path_gefs(worker)
        
        print(f"Using GEFS credentials file at: {worker_creds_path}")
        
        storage_client = storage.Client.from_service_account_json(worker_creds_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"GEFS file {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload GEFS file to GCS: {str(e)}")
        raise


def noncluster_upload_gefs_to_gcs(bucket_name, source_file_name, destination_blob_name, credentials_path):
    """Upload GEFS file to GCS without Dask worker (for local testing)."""
    try:
        print(f"Using GEFS credentials file at: {credentials_path}")
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
            
        storage_client = storage.Client.from_service_account_json(credentials_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"GEFS file {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload GEFS file to GCS: {str(e)}")
        raise


@dask.delayed
def process_gefs_time_idx_data(s3url, bucket_name):
    """
    Process GEFS data and create index mapping, then upload to GCS.
    This is the GEFS equivalent of process_gfs_time_idx_data.
    
    Parameters:
    - s3url (str): S3 URL to GEFS file
    - bucket_name (str): GCS bucket name for storage
    
    Returns:
    - bool: Success status
    """
    try:
        date_str, runz, member, forecast_hour = get_gefs_details(s3url)
        
        if not all([date_str, runz, member, forecast_hour]):
            logger.error(f"Invalid GEFS URL format: {s3url}")
            return False

        print(f"Processing GEFS: {date_str}, {member}, forecast hour {forecast_hour}")

        # Build mapping for the specified GEFS file
        mapping = build_idx_grib_mapping(
            fs=fsspec.filesystem("s3"),
            basename=s3url 
        )
        deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
        deduped_mapping.set_index('attrs', inplace=True)
        
        # Save deduped mapping as Parquet
        output_dir = f"gefs_mapping_{date_str}_{member}"
        os.makedirs(output_dir, exist_ok=True)
        parquet_path = os.path.join(output_dir, f"gefs-time-{date_str}-{member}-rt{int(forecast_hour):03}.parquet")
        deduped_mapping.to_parquet(parquet_path, index=True)
        
        # Upload to GCS in organized structure
        year = date_str[:4]
        destination_blob_name = f"time_idx/gefs/{year}/{date_str}/{member}/{os.path.basename(parquet_path)}"
        upload_gefs_to_gcs(bucket_name, parquet_path, destination_blob_name)
        
        # Cleanup local file
        os.remove(parquet_path)
        print(f"GEFS data for {date_str} {member} forecast hour {forecast_hour} processed and uploaded successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process GEFS data for URL {s3url}: {str(e)}")
        raise


def logged_process_gefs_time_idx_data(s3url, bucket_name):
    """
    Process GEFS data with enhanced logging, equivalent to logged_process_gfs_time_idx_data.
    """
    date_str, runz, member, forecast_hour = get_gefs_details(s3url)
    
    # Create temporary log file with GEFS-specific naming
    with tempfile.NamedTemporaryFile(mode="w+", suffix=f"_gefs_{date_str}_{member}_{forecast_hour}.log", delete=False) as log_file:
        log_filename = log_file.name
        gefs_logger = logging.getLogger(f"gefs_{date_str}_{member}")
        
        try:
            gefs_logger.info(f"Processing GEFS: {s3url}")

            if not all([date_str, runz, member, forecast_hour]):
                gefs_logger.error(f"Invalid GEFS URL format: {s3url}")
                return False

            # Build mapping for the specified GEFS file
            mapping = build_idx_grib_mapping(
                fs=fsspec.filesystem("s3"),
                basename=s3url 
            )
            deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
            deduped_mapping.set_index('attrs', inplace=True)

            # Save deduped mapping as Parquet  
            output_dir = f"gefs_mapping_{date_str}_{member}"
            os.makedirs(output_dir, exist_ok=True)
            parquet_path = os.path.join(output_dir, f"gefs-time-{date_str}-{member}-rt{int(forecast_hour):03}.parquet")
            deduped_mapping.to_parquet(parquet_path, index=True)

            # Upload to GCS
            year = date_str[:4]
            destination_blob_name = f"time_idx/gefs/{year}/{date_str}/{member}/{os.path.basename(parquet_path)}"
            upload_gefs_to_gcs(bucket_name, parquet_path, destination_blob_name)

            # Cleanup
            os.remove(parquet_path)
            gefs_logger.info(f"GEFS data for {date_str} {member} forecast hour {forecast_hour} processed successfully.")
            process_success = True
            
        except Exception as e:
            gefs_logger.error(f"Failed to process GEFS data for URL {s3url}: {str(e)}")
            process_success = False
            
        finally:
            # Upload log file to GCS
            year = date_str[:4] if date_str else "unknown"
            gcs_log_path = f"time_idx/gefs/{year}/logs/{date_str}/{member}/{os.path.basename(log_filename)}"
            upload_gefs_to_gcs(bucket_name, log_filename, gcs_log_path)
            os.remove(log_filename)

        return process_success


def create_gefs_ensemble_member_mappings(date_str: str, ensemble_member: str, bucket_name: str, 
                                       max_forecast_hour: int = 240, use_dask: bool = True):
    """
    Create index mappings for all forecast hours of a single GEFS ensemble member.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format
    - ensemble_member (str): Ensemble member (e.g., 'gep01')
    - bucket_name (str): GCS bucket name
    - max_forecast_hour (int): Maximum forecast hour to process
    - use_dask (bool): Whether to use Dask for parallel processing
    
    Returns:
    - List: Results from processing
    """
    print(f"Creating GEFS mappings for {date_str}, member {ensemble_member}")
    
    # Generate all GEFS URLs for this member
    gefs_urls = gefs_s3_url_maker(date_str, ensemble_member, max_forecast_hour)
    
    if use_dask:
        # Use Dask for parallel processing
        results = []
        for s3url in gefs_urls:
            result = process_gefs_time_idx_data(s3url, bucket_name)
            results.append(result)
        
        # Compute all results
        final_results = dask.compute(*results)
        return final_results
    else:
        # Process sequentially (for testing)
        results = []
        for s3url in gefs_urls:
            try:
                result = logged_process_gefs_time_idx_data(s3url, bucket_name)
                results.append(result)
            except Exception as e:
                print(f"Error processing {s3url}: {str(e)}")
                results.append(False)
        return results


def create_gefs_full_ensemble_mappings(date_str: str, bucket_name: str, 
                                     ensemble_members: List[str] = None,
                                     max_forecast_hour: int = 240):
    """
    Create index mappings for all ensemble members for a given date.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format  
    - bucket_name (str): GCS bucket name
    - ensemble_members (List[str]): List of ensemble members (default: gep01-gep30)
    - max_forecast_hour (int): Maximum forecast hour to process
    
    Returns:
    - Dict: Results organized by ensemble member
    """
    if ensemble_members is None:
        # Default GEFS ensemble members (gep01 through gep30)
        ensemble_members = [f"gep{i:02d}" for i in range(1, 31)]
    
    print(f"Creating GEFS mappings for {date_str}, {len(ensemble_members)} ensemble members")
    
    all_results = {}
    
    for member in ensemble_members:
        print(f"Processing ensemble member: {member}")
        try:
            member_results = create_gefs_ensemble_member_mappings(
                date_str, member, bucket_name, max_forecast_hour, use_dask=True
            )
            all_results[member] = member_results
            print(f"Completed processing for {member}")
        except Exception as e:
            print(f"Error processing member {member}: {str(e)}")
            all_results[member] = None
    
    return all_results


def verify_gefs_mappings_in_gcs(bucket_name: str, date_str: str, ensemble_member: str, 
                               credentials_path: str):
    """
    Verify that GEFS mapping files were successfully uploaded to GCS.
    
    Parameters:
    - bucket_name (str): GCS bucket name
    - date_str (str): Date in YYYYMMDD format
    - ensemble_member (str): Ensemble member to verify
    - credentials_path (str): Path to GCS credentials file
    
    Returns:
    - List[str]: List of found mapping files
    """
    gcs_fs = gcsfs.GCSFileSystem(token=credentials_path)
    year = date_str[:4]
    
    # Check for mapping files
    mapping_pattern = f"gs://{bucket_name}/time_idx/gefs/{year}/{date_str}/{ensemble_member}/gefs-time-{date_str}-{ensemble_member}-rt*.parquet"
    
    try:
        found_files = gcs_fs.glob(mapping_pattern)
        print(f"Found {len(found_files)} GEFS mapping files for {ensemble_member} on {date_str}")
        return found_files
    except Exception as e:
        print(f"Error checking GEFS mappings in GCS: {str(e)}")
        return []


# Example usage functions
def main_create_daily_gefs_mappings(date_str: str, bucket_name: str = "gik-gefs-aws-tf"):
    """
    Main function to create GEFS mappings for a full day (all ensemble members).
    Run this once per date to set up mappings for fast GEFS processing.
    """
    setup_gefs_logging()
    
    print(f"Starting GEFS mapping creation for {date_str}")
    start_time = time.time()
    
    # Create mappings for all ensemble members
    results = create_gefs_full_ensemble_mappings(date_str, bucket_name)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"GEFS mapping creation completed in {processing_time:.2f} seconds")
    print(f"Results summary: {len([r for r in results.values() if r is not None])} successful members")
    
    return results


def main_create_single_member_mappings(date_str: str, ensemble_member: str, 
                                     bucket_name: str = "gik-gefs-aws-tf"):
    """
    Create mappings for a single GEFS ensemble member.
    Useful for testing or processing specific members.
    """
    setup_gefs_logging()
    
    print(f"Creating GEFS mappings for {date_str}, member {ensemble_member}")
    
    results = create_gefs_ensemble_member_mappings(date_str, ensemble_member, bucket_name)
    
    print(f"Completed GEFS mapping creation for {ensemble_member}")
    return results


if __name__ == "__main__":
    # Example usage - create mappings for a specific date
    date_str = "20241112"
    bucket_name = "gik-gefs-aws-tf"
    
    # Option 1: Create mappings for all ensemble members
    # results = main_create_daily_gefs_mappings(date_str, bucket_name)
    
    # Option 2: Create mappings for a single member (for testing)
    results = main_create_single_member_mappings(date_str, "gep01", bucket_name)
    
    print("GEFS preprocessing completed!")