"""
CFS Index Processing Module

This module creates and stores CFS index mappings in GCS bucket to enable fast processing.
Adapted from GEFS preprocessing for Climate Forecast System flux files.

CFS Data Structure:
- Location: s3://noaa-cfs-pds/cfs.{date}/{run}/6hrly_grib_01/
- Files: flxf{YYYYMMDDHH}.01.{INIT_DATE}.grb2[.idx]
- Temporal: 6-hourly intervals (vs 3-hourly for GEFS)
- Forecast: Up to 215 days (vs 10 days for GEFS)

Usage:
    python cfs_index_processing.py --date 20250801 --run 00 --bucket gik-fmrc
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

logger = logging.getLogger("cfs-preprocessing-logs")


def setup_cfs_logging(log_level: int = logging.INFO, log_file: str = "cfs_preprocessing.log"):
    """Set up logging for CFS preprocessing."""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)


def get_cfs_details(url):
    """Extract date, run, and forecast hour from CFS URL."""
    import re
    # Pattern: s3://noaa-cfs-pds/cfs.20250801/00/6hrly_grib_01/flxf2025080100.01.2025080100.grb2
    pattern = r"s3://noaa-cfs-pds/cfs\.(\d{8})/(\d{2})/6hrly_grib_01/flxf(\d{10})\.01\.(\d{10})\.grb2"
    match = re.match(pattern, url)
    if match:
        date = match.group(1)      # '20250801'
        run = match.group(2)       # '00'
        forecast_time = match.group(3)  # '2025080100'
        init_time = match.group(4)      # '2025080100'
        
        # Calculate forecast hour from init and forecast times
        from datetime import datetime
        init_dt = datetime.strptime(init_time, "%Y%m%d%H")
        forecast_dt = datetime.strptime(forecast_time, "%Y%m%d%H")
        forecast_hour = int((forecast_dt - init_dt).total_seconds() / 3600)
        
        return date, run, forecast_hour, init_time
    else:
        logger.warning(f"No match found for CFS URL pattern: {url}")
        return None, None, None, None


def cfs_s3_url_generator(date, run, file_pattern="flxf*.grb2"):
    """
    Generate CFS S3 URLs for flux files.
    
    Parameters:
    - date (str): Date in YYYYMMDD format
    - run (str): Run time (00, 06, 12, 18)
    - file_pattern (str): File pattern (default: 'flxf*.grb2')
    
    Returns:
    - str: Formatted S3 URL pattern
    """
    url = f"s3://noaa-cfs-pds/cfs.{date}/{run}/6hrly_grib_01/{file_pattern}"
    return url


def cfs_s3_url_maker(date_str, run="00", max_forecast_hours=1440):
    """
    Create S3 URLs for CFS flux data.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format
    - run (str): Run time (00, 06, 12, 18) - default '00'
    - max_forecast_hours (int): Maximum forecast hours (default 1440 = 60 days)
    
    Returns:
    - List[str]: List of S3 URLs for CFS flux files
    """
    fs_s3 = fsspec.filesystem("s3", anon=True)
    # Get all flux files for this run
    base_pattern = cfs_s3_url_generator(date_str, run, "flxf*.grb2")
    s3url_glob = fs_s3.glob(base_pattern.replace("s3://", ""))
    s3url_only_grib = [f for f in s3url_glob if f.split(".")[-1] != "idx"]
    
    # Filter by forecast hour if needed
    fmt_s3og = []
    for url in s3url_only_grib:
        full_url = "s3://" + url
        _, _, forecast_hour, _ = get_cfs_details(full_url)
        if forecast_hour is not None and forecast_hour <= max_forecast_hours:
            fmt_s3og.append(full_url)
    
    fmt_s3og = sorted(fmt_s3og)
    print(f"Generated {len(fmt_s3og)} CFS URLs for date {date_str}, run {run} (max {max_forecast_hours}h)")
    print(f"Forecast hours range: {min([get_cfs_details(url)[2] for url in fmt_s3og if get_cfs_details(url)[2] is not None])}-{max([get_cfs_details(url)[2] for url in fmt_s3og if get_cfs_details(url)[2] is not None])}")
    return fmt_s3og


def get_worker_creds_path_cfs(dask_worker):
    """Get credentials path for CFS processing on worker."""
    return str(pathlib.Path(dask_worker.local_directory) / 'coiled-data-e4drr_202505.json')


def upload_cfs_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload CFS mapping file to GCS bucket using worker credentials."""
    try:
        worker = get_worker()
        worker_creds_path = get_worker_creds_path_cfs(worker)
        
        print(f"Using CFS credentials file at: {worker_creds_path}")
        
        storage_client = storage.Client.from_service_account_json(worker_creds_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"CFS file {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload CFS file to GCS: {str(e)}")
        raise


def noncluster_upload_cfs_to_gcs(bucket_name, source_file_name, destination_blob_name, credentials_path):
    """Upload CFS file to GCS without Dask worker (for local testing)."""
    try:
        print(f"Using CFS credentials file at: {credentials_path}")
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
            
        storage_client = storage.Client.from_service_account_json(credentials_path)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"CFS file {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Failed to upload CFS file to GCS: {str(e)}")
        raise


@dask.delayed
def process_cfs_time_idx_data(s3url, bucket_name):
    """
    Process CFS data and create index mapping, then upload to GCS.
    
    Parameters:
    - s3url (str): S3 URL to CFS flux file
    - bucket_name (str): GCS bucket name for storage
    
    Returns:
    - bool: Success status
    """
    try:
        date_str, runz, forecast_hour, init_time = get_cfs_details(s3url)
        
        if not all([date_str, runz, forecast_hour is not None, init_time]):
            logger.error(f"Invalid CFS URL format: {s3url}")
            return False

        print(f"Processing CFS: {date_str}, run {runz}, forecast hour {forecast_hour}")

        # Build mapping for the specified CFS file with anonymous S3 access
        storage_options = {"anon": True, "asynchronous": False}
        mapping = build_idx_grib_mapping(s3url, storage_options=storage_options)
        deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
        deduped_mapping.set_index('attrs', inplace=True)
        
        # Save deduped mapping as Parquet
        output_dir = f"cfs_mapping_{date_str}_{runz}"
        os.makedirs(output_dir, exist_ok=True)
        parquet_path = os.path.join(output_dir, f"cfs-time-{date_str}-{runz}-rt{forecast_hour:03}.parquet")
        deduped_mapping.to_parquet(parquet_path, index=True)
        
        # Upload to GCS in organized structure for CFS
        year = date_str[:4]
        destination_blob_name = f"cfs/time_idx/{year}/{date_str}/{runz}/{os.path.basename(parquet_path)}"
        
        # Try to use Dask worker upload, fallback to non-cluster upload
        try:
            upload_cfs_to_gcs(bucket_name, parquet_path, destination_blob_name)
        except ValueError:  # No worker found
            noncluster_upload_cfs_to_gcs(bucket_name, parquet_path, destination_blob_name, "coiled-data-e4drr_202505.json")
        
        # Cleanup local file
        os.remove(parquet_path)
        print(f"CFS data for {date_str} run {runz} forecast hour {forecast_hour} processed and uploaded successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process CFS data for URL {s3url}: {str(e)}")
        raise


def logged_process_cfs_time_idx_data(s3url, bucket_name):
    """
    Process CFS data with enhanced logging.
    """
    date_str, runz, forecast_hour, init_time = get_cfs_details(s3url)
    
    # Create temporary log file with CFS-specific naming
    with tempfile.NamedTemporaryFile(mode="w+", suffix=f"_cfs_{date_str}_{runz}_{forecast_hour}.log", delete=False) as log_file:
        log_filename = log_file.name
        cfs_logger = logging.getLogger(f"cfs_{date_str}_{runz}")
        
        try:
            cfs_logger.info(f"Processing CFS: {s3url}")

            if not all([date_str, runz, forecast_hour is not None, init_time]):
                cfs_logger.error(f"Invalid CFS URL format: {s3url}")
                return False

            # Build mapping for the specified CFS file with anonymous S3 access
            storage_options = {"anon": True, "asynchronous": False}
            mapping = build_idx_grib_mapping(s3url, storage_options=storage_options)
            deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
            deduped_mapping.set_index('attrs', inplace=True)

            # Save deduped mapping as Parquet  
            output_dir = f"cfs_mapping_{date_str}_{runz}"
            os.makedirs(output_dir, exist_ok=True)
            parquet_path = os.path.join(output_dir, f"cfs-time-{date_str}-{runz}-rt{forecast_hour:03}.parquet")
            deduped_mapping.to_parquet(parquet_path, index=True)

            # Upload to GCS
            year = date_str[:4]
            destination_blob_name = f"cfs/time_idx/{year}/{date_str}/{runz}/{os.path.basename(parquet_path)}"
            
            # Try to use Dask worker upload, fallback to non-cluster upload
            try:
                upload_cfs_to_gcs(bucket_name, parquet_path, destination_blob_name)
            except ValueError:  # No worker found
                noncluster_upload_cfs_to_gcs(bucket_name, parquet_path, destination_blob_name, "coiled-data-e4drr_202505.json")

            # Cleanup
            os.remove(parquet_path)
            cfs_logger.info(f"CFS data for {date_str} run {runz} forecast hour {forecast_hour} processed successfully.")
            process_success = True
            
        except Exception as e:
            cfs_logger.error(f"Failed to process CFS data for URL {s3url}: {str(e)}")
            process_success = False
            
        finally:
            # Upload log file to GCS
            year = date_str[:4] if date_str else "unknown"
            gcs_log_path = f"cfs/time_idx/{year}/logs/{date_str}/{runz}/{os.path.basename(log_filename)}"
            try:
                upload_cfs_to_gcs(bucket_name, log_filename, gcs_log_path)
            except:
                try:
                    noncluster_upload_cfs_to_gcs(bucket_name, log_filename, gcs_log_path, "coiled-data-e4drr_202505.json")
                except Exception as log_upload_error:
                    print(f"Failed to upload log file: {log_upload_error}")
            os.remove(log_filename)

        return process_success


def create_cfs_run_mappings(date_str: str, run: str, bucket_name: str, 
                           max_forecast_hours: int = 1440, use_dask: bool = True):
    """
    Create index mappings for all forecast hours of a CFS run.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format
    - run (str): Run time (00, 06, 12, 18)
    - bucket_name (str): GCS bucket name
    - max_forecast_hours (int): Maximum forecast hours to process
    - use_dask (bool): Whether to use Dask for parallel processing
    
    Returns:
    - List: Results from processing
    """
    print(f"Creating CFS mappings for {date_str}, run {run}")
    
    # Generate all CFS URLs for this run
    cfs_urls = cfs_s3_url_maker(date_str, run, max_forecast_hours)
    
    if use_dask:
        # Use Dask for parallel processing
        results = []
        for s3url in cfs_urls:
            result = process_cfs_time_idx_data(s3url, bucket_name)
            results.append(result)
        
        # Compute all results
        final_results = dask.compute(*results)
        return final_results
    else:
        # Process sequentially (for testing)
        results = []
        for s3url in cfs_urls:
            try:
                result = logged_process_cfs_time_idx_data(s3url, bucket_name)
                results.append(result)
            except Exception as e:
                print(f"Error processing {s3url}: {str(e)}")
                results.append(False)
        return results


def test_single_cfs_timestep(date_str: str, run: str = "00", forecast_hour: int = 0):
    """
    Test processing a single CFS timestep without GCS upload.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format
    - run (str): Run time (00, 06, 12, 18)
    - forecast_hour (int): Specific forecast hour to test
    
    Returns:
    - bool: Success status
    """
    print(f"Testing single CFS timestep: {date_str}, run {run}, forecast hour {forecast_hour}")
    
    try:
        # Generate URL for specific forecast hour
        from datetime import datetime, timedelta
        
        # Calculate the forecast time string
        init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
        forecast_dt = init_dt + timedelta(hours=forecast_hour)
        forecast_time_str = forecast_dt.strftime("%Y%m%d%H")
        init_time_str = init_dt.strftime("%Y%m%d%H")
        
        # Build the S3 URL
        s3url = f"s3://noaa-cfs-pds/cfs.{date_str}/{run}/6hrly_grib_01/flxf{forecast_time_str}.01.{init_time_str}.grb2"
        
        print(f"Processing CFS URL: {s3url}")
        
        # Check if file exists first
        import fsspec
        fs_s3 = fsspec.filesystem("s3", anon=True)
        s3_path = s3url.replace("s3://", "")
        
        if not fs_s3.exists(s3_path):
            print(f"File does not exist: {s3url}")
            return False
        
        print("File exists, proceeding with index parsing...")
        
        # Try simpler approach - just parse the index file directly
        try:
            print("Attempting to parse GRIB index directly...")
            idxdf = parse_grib_idx(basename=s3url)
            print(f"Index DataFrame created with {len(idxdf)} entries")
            print(f"Index columns: {list(idxdf.columns)}")
            
            # Show sample data
            if len(idxdf) > 0:
                print("Sample index entries:")
                print(idxdf.head(3))
                
            # Save index as Parquet
            output_dir = f"cfs_test_{date_str}_{run}"
            os.makedirs(output_dir, exist_ok=True)
            parquet_path = os.path.join(output_dir, f"cfs-index-{date_str}-{run}-f{forecast_hour:03}.parquet")
            idxdf.to_parquet(parquet_path, index=False)
            
            print(f"✅ CFS index parsing successful!")
            print(f"   Index parquet file: {parquet_path}")
            print(f"   File size: {os.path.getsize(parquet_path)} bytes")
            print(f"   Simplified GCS path would be: gik-fmrc/cfs/{date_str}/")
            
            return True
            
        except Exception as idx_error:
            print(f"Index parsing failed: {str(idx_error)}")
            print("Falling back to full kerchunk mapping (may fail due to async issues)...")
            
            # Fallback to original approach
            storage_options = {"anon": True, "asynchronous": False}
            mapping = build_idx_grib_mapping(s3url, storage_options=storage_options)
            
            print(f"Initial mapping created with {len(mapping)} entries")
            
            # Remove duplicates
            deduped_mapping = mapping.loc[~mapping["attrs"].duplicated(keep="first"), :]
            deduped_mapping.set_index('attrs', inplace=True)
            
            print(f"Deduped mapping has {len(deduped_mapping)} entries")
            
            # Save deduped mapping as Parquet locally (no GCS upload)
            output_dir = f"cfs_test_{date_str}_{run}"
            os.makedirs(output_dir, exist_ok=True)
            parquet_path = os.path.join(output_dir, f"cfs-test-{date_str}-{run}-f{forecast_hour:03}.parquet")
            deduped_mapping.to_parquet(parquet_path, index=True)
            
            print(f"✅ Single CFS timestep processed successfully!")
            print(f"   Parquet file saved: {parquet_path}")
            print(f"   File size: {os.path.getsize(parquet_path)} bytes")
            print(f"   Simplified GCS path would be: gik-fmrc/cfs/{date_str}/")
            
            return True
        
    except Exception as e:
        print(f"❌ Failed to process single CFS timestep: {str(e)}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return False


def test_simple_cfs_index_only(date_str: str, run: str = "00", forecast_hour: int = 0):
    """
    Simple test that only parses the CFS index file without complex mapping.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format
    - run (str): Run time (00, 06, 12, 18)  
    - forecast_hour (int): Specific forecast hour to test
    
    Returns:
    - bool: Success status
    """
    print(f"Simple CFS index test: {date_str}, run {run}, forecast hour {forecast_hour}")
    
    try:
        from datetime import datetime, timedelta
        
        # Calculate the forecast time string
        init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
        forecast_dt = init_dt + timedelta(hours=forecast_hour)
        forecast_time_str = forecast_dt.strftime("%Y%m%d%H")
        init_time_str = init_dt.strftime("%Y%m%d%H")
        
        # Build the S3 URL
        s3url = f"s3://noaa-cfs-pds/cfs.{date_str}/{run}/6hrly_grib_01/flxf{forecast_time_str}.01.{init_time_str}.grb2"
        
        print(f"Processing CFS URL: {s3url}")
        
        # Configure storage options for anonymous S3 access
        import fsspec
        storage_options = {"anon": True}
        
        # Check if the index file exists first
        idx_url = s3url + ".idx"
        print(f"Checking for index file: {idx_url}")
        
        fs_s3 = fsspec.filesystem("s3", anon=True)
        idx_path = idx_url.replace("s3://", "")
        
        if not fs_s3.exists(idx_path):
            print("❌ Index file not found!")
            return False
            
        print("Index file exists, proceeding with manual parsing...")
        
        # Try manual parsing to bypass kerchunk issues
        try:
            print("Reading index file content...")
            with fs_s3.open(idx_path, 'r') as f:
                lines = f.readlines()
            
            print(f"Index file has {len(lines)} entries")
            
            # Parse the index file manually
            index_data = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                # GRIB index format: record:offset:date:variable:level:forecast_time:
                parts = line.split(':')
                if len(parts) >= 6:
                    try:
                        record = int(parts[0])
                        offset = int(parts[1])
                        date_info = parts[2]
                        variable = parts[3]
                        level = parts[4]
                        forecast_info = parts[5] if len(parts) > 5 else ''
                        
                        index_data.append({
                            'record': record,
                            'offset': offset,
                            'date': date_info,
                            'varname': variable,
                            'level': level,
                            'forecast': forecast_info,
                            'line_num': line_num
                        })
                    except ValueError as pe:
                        print(f"Skipping line {line_num} due to parsing error: {pe}")
                        print(f"  Line content: {line}")
                        continue
            
            print(f"✅ Manual parsing successful: {len(index_data)} valid entries")
            
            # Convert to DataFrame
            import pandas as pd
            idxdf = pd.DataFrame(index_data)
            
            if len(idxdf) > 0:
                print(f"Variables found: {sorted(idxdf['varname'].unique()[:15])}")  # Show first 15 variables
                print(f"Date info: {idxdf['date'].unique()}")
                print(f"Level types: {sorted(idxdf['level'].unique()[:10])}")  # Show first 10 levels
                
                # Save locally
                output_dir = f"cfs_simple_{date_str}_{run}"
                os.makedirs(output_dir, exist_ok=True)
                parquet_path = os.path.join(output_dir, f"cfs-simple-{date_str}-{run}-f{forecast_hour:03}.parquet")
                idxdf.to_parquet(parquet_path, index=False)
                
                print(f"✅ CFS simple index test successful!")
                print(f"   Manual index file: {parquet_path}")
                print(f"   File size: {os.path.getsize(parquet_path)} bytes")
                print(f"   Simplified GCS path would be: gik-fmrc/cfs/{date_str}/")
                
                return True
            else:
                print("❌ No valid entries found in index file")
                return False
                
        except Exception as manual_error:
            print(f"Manual parsing failed: {str(manual_error)}")
            print("Attempting kerchunk parsing as fallback...")
            
            # Fallback to kerchunk (will likely fail but let's try)
            try:
                idxdf = parse_grib_idx(basename=s3url, storage_options=storage_options)
                print(f"✅ Kerchunk parsing worked: {len(idxdf)} entries")
                
                # Save locally
                output_dir = f"cfs_simple_{date_str}_{run}"
                os.makedirs(output_dir, exist_ok=True)
                parquet_path = os.path.join(output_dir, f"cfs-simple-{date_str}-{run}-f{forecast_hour:03}.parquet")
                idxdf.to_parquet(parquet_path, index=False)
                
                print(f"Kerchunk index file saved: {parquet_path}")
                return True
                
            except Exception as kerchunk_error:
                print(f"Kerchunk parsing also failed: {str(kerchunk_error)}")
                return False
        
    except Exception as e:
        print(f"❌ Simple index test failed: {str(e)}")
        return False


def download_and_store_cfs_files(date_str: str, run: str = "00", forecast_hour: int = 0):
    """
    Download IDX and GRIB files from AWS S3 and create parquet files for research purposes.
    
    Parameters:
    - date_str (str): Date in YYYYMMDD format
    - run (str): Run time (00, 06, 12, 18)
    - forecast_hour (int): Specific forecast hour to process
    
    Returns:
    - bool: Success status
    """
    print(f"Downloading CFS files for research: {date_str}, run {run}, forecast hour {forecast_hour}")
    
    try:
        from datetime import datetime, timedelta
        import shutil
        
        # Calculate the forecast time string
        init_dt = datetime.strptime(f"{date_str}{run}", "%Y%m%d%H")
        forecast_dt = init_dt + timedelta(hours=forecast_hour)
        forecast_time_str = forecast_dt.strftime("%Y%m%d%H")
        init_time_str = init_dt.strftime("%Y%m%d%H")
        
        # Build the S3 URLs
        s3url = f"s3://noaa-cfs-pds/cfs.{date_str}/{run}/6hrly_grib_01/flxf{forecast_time_str}.01.{init_time_str}.grb2"
        idx_url = s3url + ".idx"
        
        print(f"GRIB file URL: {s3url}")
        print(f"Index file URL: {idx_url}")
        
        # Create research directory
        research_dir = f"cfs_research_{date_str}_{run}_f{forecast_hour:03}"
        os.makedirs(research_dir, exist_ok=True)
        print(f"Created research directory: {research_dir}")
        
        # Configure S3 filesystem
        import fsspec
        fs_s3 = fsspec.filesystem("s3", anon=True)
        
        # Check if files exist
        grib_path = s3url.replace("s3://", "")
        idx_path = idx_url.replace("s3://", "")
        
        if not fs_s3.exists(grib_path):
            print(f"❌ GRIB file not found: {s3url}")
            return False
            
        if not fs_s3.exists(idx_path):
            print(f"❌ Index file not found: {idx_url}")
            return False
            
        print("✅ Both GRIB and IDX files exist on S3")
        
        # Download IDX file
        print("Downloading IDX file...")
        local_idx_path = os.path.join(research_dir, f"flxf{forecast_time_str}.01.{init_time_str}.grb2.idx")
        
        with fs_s3.open(idx_path, 'rb') as remote_file:
            with open(local_idx_path, 'wb') as local_file:
                shutil.copyfileobj(remote_file, local_file)
        
        idx_size = os.path.getsize(local_idx_path)
        print(f"✅ IDX file downloaded: {local_idx_path} ({idx_size:,} bytes)")
        
        # Download GRIB file (this might be large)
        print("Downloading GRIB file (this may take some time)...")
        local_grib_path = os.path.join(research_dir, f"flxf{forecast_time_str}.01.{init_time_str}.grb2")
        
        with fs_s3.open(grib_path, 'rb') as remote_file:
            with open(local_grib_path, 'wb') as local_file:
                # Copy in chunks to show progress for large files
                chunk_size = 8192 * 1024  # 8MB chunks
                total_bytes = 0
                while True:
                    chunk = remote_file.read(chunk_size)
                    if not chunk:
                        break
                    local_file.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes % (50 * 1024 * 1024) == 0:  # Show progress every 50MB
                        print(f"  Downloaded {total_bytes / (1024*1024):.1f} MB...")
        
        grib_size = os.path.getsize(local_grib_path)
        print(f"✅ GRIB file downloaded: {local_grib_path} ({grib_size:,} bytes = {grib_size/(1024*1024):.1f} MB)")
        
        # Parse the downloaded IDX file to create parquet
        print("Parsing downloaded IDX file to create parquet...")
        
        with open(local_idx_path, 'r') as f:
            lines = f.readlines()
        
        print(f"Index file has {len(lines)} entries")
        
        # Parse the index file manually (same logic as test_simple_cfs_index_only)
        index_data = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # GRIB index format: record:offset:date:variable:level:forecast_time:
            parts = line.split(':')
            if len(parts) >= 6:
                try:
                    record = int(parts[0])
                    offset = int(parts[1])
                    date_info = parts[2]
                    variable = parts[3]
                    level = parts[4]
                    forecast_info = parts[5] if len(parts) > 5 else ''
                    
                    index_data.append({
                        'record': record,
                        'offset': offset,
                        'date': date_info,
                        'varname': variable,
                        'level': level,
                        'forecast': forecast_info,
                        'line_num': line_num
                    })
                except ValueError as pe:
                    print(f"Skipping line {line_num} due to parsing error: {pe}")
                    continue
        
        print(f"✅ Manual parsing successful: {len(index_data)} valid entries")
        
        # Convert to DataFrame and save as parquet
        import pandas as pd
        idxdf = pd.DataFrame(index_data)
        
        if len(idxdf) > 0:
            parquet_path = os.path.join(research_dir, f"cfs-research-{date_str}-{run}-f{forecast_hour:03}.parquet")
            idxdf.to_parquet(parquet_path, index=False)
            parquet_size = os.path.getsize(parquet_path)
            
            print(f"✅ Parquet file created: {parquet_path} ({parquet_size:,} bytes)")
            print(f"Variables found: {sorted(idxdf['varname'].unique()[:15])}")
            print(f"Level types: {sorted(idxdf['level'].unique()[:10])}")
        else:
            print("❌ No valid entries found in index file")
            return False
        
        # Summary
        print(f"\n🎉 Research files successfully downloaded and processed!")
        print(f"Research directory: {research_dir}")
        print(f"Files created:")
        print(f"  - IDX file:     {local_idx_path} ({idx_size:,} bytes)")
        print(f"  - GRIB file:    {local_grib_path} ({grib_size:,} bytes = {grib_size/(1024*1024):.1f} MB)")
        print(f"  - Parquet file: {parquet_path} ({parquet_size:,} bytes)")
        print(f"Total storage:    {(idx_size + grib_size + parquet_size)/(1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download and store CFS files: {str(e)}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return False


def verify_cfs_mappings_in_gcs(bucket_name: str, date_str: str, run: str, 
                              credentials_path: str):
    """
    Verify that CFS mapping files were successfully uploaded to GCS.
    
    Parameters:
    - bucket_name (str): GCS bucket name
    - date_str (str): Date in YYYYMMDD format
    - run (str): Run time to verify
    - credentials_path (str): Path to GCS credentials file
    
    Returns:
    - List[str]: List of found mapping files
    """
    gcs_fs = gcsfs.GCSFileSystem(token=credentials_path)
    year = date_str[:4]
    
    # Check for mapping files
    mapping_pattern = f"gs://{bucket_name}/cfs/time_idx/{year}/{date_str}/{run}/cfs-time-{date_str}-{run}-rt*.parquet"
    
    try:
        found_files = gcs_fs.glob(mapping_pattern)
        print(f"Found {len(found_files)} CFS mapping files for run {run} on {date_str}")
        return found_files
    except Exception as e:
        print(f"Error checking CFS mappings in GCS: {str(e)}")
        return []


# Example usage functions
def main_create_cfs_mappings(date_str: str, run: str = "00", bucket_name: str = "gik-fmrc", max_forecast_hours: int = 1440):
    """
    Main function to create CFS mappings for a specific date and run.
    """
    setup_cfs_logging()
    
    print(f"Starting CFS mapping creation for {date_str}, run {run}")
    start_time = time.time()
    
    # Create mappings for the CFS run
    results = create_cfs_run_mappings(date_str, run, bucket_name, max_forecast_hours)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"CFS mapping creation completed in {processing_time:.2f} seconds")
    successful_results = sum(1 for r in results if r)
    print(f"Results summary: {successful_results}/{len(results)} successful files")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process CFS data and create index mappings")
    parser.add_argument("--date", required=True, help="Date in YYYYMMDD format")
    parser.add_argument("--run", default="00", help="Run time (00, 06, 12, 18) - default: 00")
    parser.add_argument("--bucket", default="gik-fmrc", help="GCS bucket name")
    parser.add_argument("--max-hours", type=int, default=1440, help="Maximum forecast hours to process (default: 1440)")
    parser.add_argument("--test-single", action="store_true", help="Test single timestep processing without GCS upload")
    parser.add_argument("--test-simple", action="store_true", help="Simple test - only parse CFS index without complex mapping")
    parser.add_argument("--research-download", action="store_true", help="Download IDX and GRIB files from S3 for research (saves files locally)")
    parser.add_argument("--forecast-hour", type=int, default=0, help="Forecast hour for single timestep test (default: 0)")
    
    args = parser.parse_args()
    
    # Validate run parameter
    if args.run not in ["00", "06", "12", "18"]:
        print("Run parameter must be one of: 00, 06, 12, 18")
        exit(1)
    
    if args.test_simple:
        # Simple CFS index test
        print("=== Simple CFS Index Test ===")
        print(f"Date: {args.date}, Run: {args.run}, Forecast Hour: {args.forecast_hour}")
        success = test_simple_cfs_index_only(args.date, args.run, args.forecast_hour)
        if success:
            print("✅ Simple index test completed successfully!")
        else:
            print("❌ Simple index test failed!")
            exit(1)
    elif args.research_download:
        # Research download - save IDX, GRIB, and parquet files locally
        print("=== CFS Research File Download ===")
        print(f"Date: {args.date}, Run: {args.run}, Forecast Hour: {args.forecast_hour}")
        success = download_and_store_cfs_files(args.date, args.run, args.forecast_hour)
        if success:
            print("✅ Research download completed successfully!")
        else:
            print("❌ Research download failed!")
            exit(1)
    elif args.test_single:
        # Test single timestep processing
        print("=== Single CFS Timestep Test ===")
        print(f"Date: {args.date}, Run: {args.run}, Forecast Hour: {args.forecast_hour}")
        success = test_single_cfs_timestep(args.date, args.run, args.forecast_hour)
        if success:
            print("✅ Single timestep test completed successfully!")
        else:
            print("❌ Single timestep test failed!")
            exit(1)
    else:
        print(f"CFS preprocessing starting for date: {args.date}, run: {args.run}")
        print(f"Target: s3://noaa-cfs-pds/cfs.{args.date}/{args.run}/6hrly_grib_01/")
        print(f"Max forecast hours: {args.max_hours}")
        
        # Process CFS run
        results = main_create_cfs_mappings(args.date, args.run, args.bucket, args.max_hours)
        
        print("CFS preprocessing completed!")
        successful_count = sum(1 for r in results if r)
        print(f"Results: {successful_count}/{len(results)} files processed successfully")