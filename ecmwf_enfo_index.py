import asyncio
import concurrent.futures
from functools import partial
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import re

from eutils import ecmwf_enfo_index_df_create_with_keys
import asyncio
import concurrent.futures
from functools import partial
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any

forecast_dict = {
    "Temperature": "t:pl",
    "cape": "cape:sfc",
    "U component of wind": "u:pl",
    "V component of wind": "v:pl",
    "Mean sea level pressure": "msl:sfc",
    "2 metre temperature": "2t:sfc",
    "Total precipitation": "tp:sfc",
}


async def process_ecmwf_url(
    url: str,
    sem: asyncio.Semaphore,
    executor: concurrent.futures.ThreadPoolExecutor,
    forecast_dict: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Process a single ECMWF URL asynchronously."""
    async with sem:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                partial(ecmwf_enfo_index_df_create_with_keys, url, forecast_dict)
            )
            
            # Unpack the tuple
            full_df, subset_df, idx_mapping, params_dict = result
            
            return full_df, subset_df, idx_mapping, params_dict
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return None


async def process_urls_in_batches(
    urls: List[str],
    forecast_dict: Dict,
    max_concurrent: int = 10,
    batch_size: int = 20
) -> List[Tuple]:
    """Process ECMWF URLs in batches with controlled concurrency."""
    sem = asyncio.Semaphore(max_concurrent)
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        for batch_start in range(0, len(urls), batch_size):
            batch_end = min(batch_start + batch_size, len(urls))
            batch_urls = urls[batch_start:batch_end]
            
            tasks = [
                process_ecmwf_url(url, sem, executor, forecast_dict)
                for url in batch_urls
            ]
            
            batch_results = await asyncio.gather(*tasks)
            valid_results = [r for r in batch_results if r is not None]
            all_results.extend(valid_results)
            
            # Clear memory
            batch_results = None
            
            print(f"Batch processed: {batch_start} to {batch_end}")
    
    return all_results


def parallel_process_ecmwf_urls(
    urls: List[str],
    forecast_dict: Dict,
    max_concurrent: int = 10,
    batch_size: int = 20
) -> Tuple[List[pd.DataFrame], Dict, Dict]:
    """
    Process ECMWF URLs in parallel and combine results.
    
    Parameters:
    - urls: List of ECMWF URLs to process
    - forecast_dict: Dictionary mapping forecast variables to ECMWF parameters
    - max_concurrent: Maximum number of concurrent tasks
    - batch_size: Size of each processing batch
    
    Returns:
    - Tuple containing:
      - Combined full DataFrame
      - Combined subset DataFrame
      - Combined index mapping dictionary
      - Combined parameter dictionary
    """
    # Run the async processing
    result_tuples = asyncio.run(
        process_urls_in_batches(urls, forecast_dict, max_concurrent, batch_size)
    )
    
    # Check if we have any valid results
    if not result_tuples:
        raise ValueError("No valid results were created")
    
    # Separate the tuple components
    full_dfs = []
    subset_dfs = []
    combined_idx_mapping = {}
    all_combined_dicts = {}
    
    for full_df, subset_df, idx_mapping, params_dict in result_tuples:
        if full_df is not None:
            full_dfs.append(full_df.reset_index(drop=True))
        if subset_df is not None:
            subset_dfs.append(subset_df.reset_index(drop=True))
        if idx_mapping:
            combined_idx_mapping.update(idx_mapping)
        if params_dict:
            all_combined_dicts.update(params_dict)
    
    # Combine DataFrames if any exist
    combined_full_df = pd.concat(full_dfs, ignore_index=True) if full_dfs else pd.DataFrame()
    combined_subset_df = pd.concat(subset_dfs, ignore_index=True) if subset_dfs else pd.DataFrame()
    
    print(f"Processed {len(combined_subset_df)} filtered records from {len(urls)} URLs")
    print(f"Collected mappings for {len(combined_idx_mapping)} indices")
    print(f"Collected {len(all_combined_dicts)} parameter dictionaries")
    
    return combined_full_df, combined_subset_df, combined_idx_mapping, all_combined_dicts

def generate_urls(date_str):
    """
    Generate a list of URLs based on the specified date and hour increments.
    
    Parameters:
    - date_str (str): The date in 'YYYYMMDD' format.
    
    Returns:
    - List[str]: A list of URLs following the specified pattern.
    """
    urls = []
    
    # Generate URLs for hours from 0 to 144 in increments of 3
    for hr in range(0, 145, 3):
        url = f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-{hr}h-enfo-ef.grib2"
        urls.append(url)
    
    # Generate URLs for hours from 144 to 360 in increments of 6
    for hr in range(144, 361, 6):
        url = f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-{hr}h-enfo-ef.grib2"
        urls.append(url)
    
    return urls

def extract_metadata_from_idx(idx_df):
    """
    Extract structured metadata from index DataFrame.
    
    Parameters:
    ----------
    idx_df : pd.DataFrame
        DataFrame from parse_ecmwf_grib_idx
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with expanded metadata
    """
    # Extract metadata from attr column
    expanded_df = idx_df
    
    # Standardize column names for downstream processing
    metadata_mapping = {
        'param': 'varname',           # Parameter code to variable name
        'levtype': 'typeOfLevel',     # Level type
        'levelist': 'level',          # Level value
        'step': 'step'                # Forecast step
    }
    
    # Rename columns that exist
    existing_columns = set(expanded_df.columns) & set(metadata_mapping.keys())
    rename_dict = {col: metadata_mapping[col] for col in existing_columns}
    expanded_df = expanded_df.rename(columns=rename_dict)
    
    # Add missing columns with default values
    for source, target in metadata_mapping.items():
        if target not in expanded_df.columns:
            expanded_df[target] = None
    
    # Add additional required columns
    expanded_df['stepType'] = 'instant'  # Default for ECMWF
    expanded_df['name'] = expanded_df['varname']  # Name mirrors variable name
    
    # Convert step to pandas timedelta if not already
    if 'step' in expanded_df.columns and not pd.api.types.is_timedelta64_dtype(expanded_df['step']):
        try:
            # Try to convert to integer first, then to timedelta
            expanded_df['step'] = pd.to_timedelta(expanded_df['step'].astype(int), unit='h')
        except:
            # If that fails, set to 0
            expanded_df['step'] = pd.Timedelta(hours=0)
    
    # Make sure level is numeric
    expanded_df['level'] = pd.to_numeric(expanded_df['level'], errors='coerce').fillna(0)
    
    # Additional required columns for downstream processing
    expanded_df['uri'] = expanded_df['grib_uri']
    # Extract date and hour from URL
    # Extract date and forecast hour from idx_uri
    expanded_df['forecast_hour'] = expanded_df['idx_uri'].str.extract(r'-(\d+)h-').astype(int)
    expanded_df['date_str'] = expanded_df['idx_uri'].str.extract(r'/(\d{8})/').iloc[0, 0]
    
    # Set time and valid_time
    base_time = pd.Timestamp(expanded_df['date_str'].iloc[0] + '000000')
    expanded_df['time'] = base_time
    
    # Calculate valid_time for each row based on its forecast hour
    expanded_df['valid_time'] = expanded_df.apply(
        lambda row: base_time + pd.Timedelta(hours=row['forecast_hour']), 
        axis=1
    )
    
    # Set step as timedelta
    expanded_df['step'] = expanded_df.apply(
        lambda row: pd.Timedelta(hours=row['forecast_hour']),
        axis=1
    )
    
    return expanded_df

def create_ecmwf_mapped_index_parallel(
    date_str: str,
    forecast_dict: Dict[str, str],
    max_concurrent: int = 5,
    batch_size: int = 5,
    url_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a mapped index for ECMWF files in parallel.
    
    Parameters:
    ----------
    date_str : str
        Date string in YYYYMMDD format
    forecast_dict : Dict[str, str]
        Dictionary mapping variable names to param:level strings
    max_concurrent : int
        Maximum number of concurrent operations
    batch_size : int
        Size of batches to process at once
    url_limit : Optional[int]
        Maximum number of URLs to process (for testing)
        
    Returns:
    -------
    pd.DataFrame
        DataFrame containing the mapped index with standardized column names
    """
    print(f"Creating parallel mapped index for date {date_str}")
    
    # Generate URLs
    urls = generate_urls(date_str)
    
    if url_limit is not None:
        urls = urls[:url_limit]
    
    print(f"Processing {len(urls)} URLs...")
    
    # Process URLs in parallel
    _, subset_df0, _, _ = parallel_process_ecmwf_urls(
        urls,
        forecast_dict,
        max_concurrent=max_concurrent,
        batch_size=batch_size
    )
    subset_df=extract_metadata_from_idx(subset_df0)
    
    if subset_df.empty:
        raise ValueError(f"No valid mapped indices created for date {date_str}")

    
    
    # Ensure all required columns are present
    required_columns = [
        "varname", "typeOfLevel", "stepType", "name", "step", "level",
        "time", "valid_time", "uri", "offset", "length"
    ]
    
    for col in required_columns:
        if col not in subset_df.columns:
            if col == "inline_value":
                subset_df[col] = None
            else:
                raise ValueError(f"Required column '{col}' is missing from mapped index")
    
    # Add any missing columns needed by downstream processing
    if "inline_value" not in subset_df.columns:
        subset_df["inline_value"] = None
    
    # Sort by time and variable name
    subset_df = subset_df.sort_values(by=["time", "varname"])
    
    print(f"Created mapped index with {len(subset_df)} rows")
    return subset_df


# Example usage:
if __name__ == "__main__":
    date_str = '20240529'
    
    # Define forecast dictionary
    forecast_dict = {
        "Temperature": "t:pl",
        "Cape": "cape:sfc",
        "U component of wind": "u:pl",
        "V component of wind": "v:pl",
        "Mean sea level pressure": "msl:sfc",
        "2 metre temperature": "2t:sfc",
        "Total precipitation": "tp:sfc",
    }
    
    # Create mapped index
    mapped_index = create_ecmwf_mapped_index_parallel(
        date_str,
        forecast_dict,
        max_concurrent=5,
        batch_size=5,
        url_limit=10  # Limit for testing
    )
    
    print(f"Mapped index shape: {mapped_index.shape}")
    print(f"Columns: {mapped_index.columns.tolist()}")
    
    # Save to CSV for inspection
    mapped_index.to_csv(f"ecmwf_mapped_index_{date_str}.csv")
    print(f"Saved mapped index to ecmwf_mapped_index_{date_str}.csv")