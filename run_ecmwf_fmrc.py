import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from utils import (
    generate_axes,
    cs_create_mapped_index,
    create_parquet_df,
    nonclusterworker_upload_to_gcs,
)


async def process_single_file_ecmwf(
    date_str: str,
    first_day_of_month: str,
    gcs_bucket_name: str,
    idx: int,
    datestr: pd.Timestamp,
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    gcp_service_account_json: str,
    chunk_size: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Process a single ECMWF file asynchronously with memory management.
    
    Additional Parameter:
    - gcp_service_account_json: Path to the GCP service account JSON file.
    """
    async with sem:  # Control concurrent operations
        try:
            fname = f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/ifs-0p25-enfo-ef_{date_str}_{idx:03}h.grib2"
            gcs_mapping_path = f"gs://{gcs_bucket_name}/fmrc/{date_str}/{idx:03}/ecmwf_buildidx_table_{date_str}_{idx:03}.parquet"
            
            # Initialize GCS filesystem with credentials
            gcs_fs = gcsfs.GCSFileSystem(
                token=gcp_service_account_json,
                project='your-project-id'  # Add your GCP project ID here
            )
            
            loop = asyncio.get_event_loop()
            
            # Read idx file
            idxdf = await loop.run_in_executor(
                executor,
                partial(parse_grib_idx, fs=fsspec.filesystem("s3"), basename=fname)
            )
            
            # Read parquet in chunks if chunk_size is specified
            if chunk_size:
                deduped_mapping_chunks = []
                for chunk in pd.read_parquet(gcs_mapping_path, filesystem=gcs_fs, chunksize=chunk_size):
                    deduped_mapping_chunks.append(chunk)
                deduped_mapping = pd.concat(deduped_mapping_chunks, ignore_index=True)
            else:
                deduped_mapping = await loop.run_in_executor(
                    executor,
                    partial(pd.read_parquet, gcs_mapping_path, filesystem=gcs_fs)
                )
            
            # Process the mapping
            idxdf_filtered = idxdf.loc[~idxdf["attrs"].duplicated(keep="first"), :]
            mapped_index = await loop.run_in_executor(
                executor,
                partial(map_from_index, datestr, deduped_mapping, idxdf_filtered)
            )
            
            return mapped_index
            
        except Exception as e:
            print(f'Error in {str(e)}')
            return None

async def process_files_in_batches_ecmwf(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    max_concurrent: int = 3,
    batch_size: int = 5,
    chunk_size: Optional[int] = None,
    gcp_service_account_json: Optional[str] = None
) -> pd.DataFrame:
    """
    Process ECMWF files in batches with controlled concurrency and memory usage.
    
    Additional Parameter:
    - gcp_service_account_json: Path to the GCP service account JSON file.
    """
    dtaxes = axes[0]
    first_day_of_month = pd.to_datetime(date_str).replace(day=1).strftime('%Y%m%d')
    
    # Create semaphore for controlling concurrent operations
    sem = asyncio.Semaphore(max_concurrent)
    
    # Initialize thread pool executor
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        all_results = []
        
        # Process in batches
        for batch_start in range(0, len(dtaxes), batch_size):
            batch_end = min(batch_start + batch_size, len(dtaxes))
            batch_indices = range(batch_start, batch_end)
            
            # Create tasks for current batch
            tasks = [
                process_single_file_ecmwf(
                    date_str,
                    first_day_of_month,
                    gcs_bucket_name,
                    idx,
                    dtaxes[idx],
                    sem,
                    executor,
                    gcp_service_account_json,
                    chunk_size
                )
                for idx in batch_indices
            ]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks)
            
            # Filter out None results and extend all_results
            valid_results = [r for r in batch_results if r is not None]
            all_results.extend(valid_results)
            
            # Optional: Clear memory after each batch
            batch_results = None
            
            print("event batch_processed")
    
    # Combine results and process
    if not all_results:
        raise ValueError(f"No valid mapped indices created for date {date_str}")
    
    gfs_kind = pd.concat(all_results, ignore_index=True)
    
    # Process variables as before
    gfs_kind_var = gfs_kind.drop_duplicates('varname')
    var_list = gfs_kind_var['varname'].tolist()
    var_to_remove = ['acpcp', 'cape', 'cin', 'pres', 'r', 'soill', 'soilw', 'st', 't', 'tp']
    
    # Filter variables
    var1_list = list(filter(lambda x: x not in var_to_remove, var_list))
    gfs_kind1 = gfs_kind.loc[gfs_kind.varname.isin(var1_list)]
    
    # Process special variables
    to_process_df = gfs_kind[gfs_kind['varname'].isin(var_to_remove)]
    processed_df = process_dataframe(to_process_df, var_to_remove)
    
    # Combine processed and unprocessed data
    final_df = pd.concat([gfs_kind1, processed_df], ignore_index=True)
    final_df = final_df.sort_values(by=['time', 'varname'])
    
    return final_df

def cs_create_mapped_index_ecmwf(
    axes: List[pd.Index],
    gcs_bucket_name: str,
    date_str: str,
    max_concurrent: int = 10,
    batch_size: int = 20,
    chunk_size: Optional[int] = None,
    gcp_service_account_json: Optional[str] = None
) -> pd.DataFrame:
    """
    Async wrapper for creating mapped index with memory management for ECMWF data.
    
    Additional Parameter:
    - gcp_service_account_json: Path to the GCP service account JSON file.
    """
    return asyncio.run(
        process_files_in_batches_ecmwf(
            axes,
            gcs_bucket_name,
            date_str,
            max_concurrent,
            batch_size,
            chunk_size,
            gcp_service_account_json
        )
    )




# Load environment variables
load_dotenv(dotenv_path='./env_gik')
gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
gcp_service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

if not gcs_bucket_name or not gcp_service_account_json:
    raise ValueError("GCS_BUCKET_NAME or GCP_SERVICE_ACCOUNT_JSON not set in the environment.")


date_str='20250215'    
# Generate axes for processing
axes = generate_axes(date_str)

# Create mapped index
e_kind = cs_create_mapped_index(
    axes, gcs_bucket_name, date_str, gcp_service_account_json=gcp_service_account_json
)


