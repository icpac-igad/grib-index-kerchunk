import asyncio
import concurrent.futures
from functools import partial
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import re
import ast
import json
import fsspec

forecast_dict = {
    "Temperature": "t:pl",
    "cape": "cape:sfc",
    "U component of wind": "u:pl",
    "V component of wind": "v:pl",
    "Mean sea level pressure": "msl:sfc",
    "2 metre temperature": "2t:sfc",
    "Total precipitation": "tp:sfc",
}


def ecmwf_enfo_local_parse_index(
    basename: str,
    suffix: str = "index",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Standalone method used to extract ALL metadata from a GRIB2 index file from a local disk using ast.literal_eval.

    :param basename: The full path to the GRIB file (excluding the suffix).
    :param suffix: The suffix for the index file (default: "index").
    :param tstamp: The timestamp to record for this index process.
    :param validate: Whether to validate uniqueness of attributes.
    :return: A pandas DataFrame containing the results with all JSON attributes.
    """
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"  # Construct the index filename

    # Lists to store parsed data
    records = []

    try:
        with open(fname, "r", encoding="utf-8") as f:  # Open the local file
            for idx, line in enumerate(f):
                try:
                    # Removing the trailing characters if there's any at the end of the line
                    clean_line = line.strip().rstrip(',')

                    # Attempt parsing using ast.literal_eval first
                    try:
                        prepared_str = clean_line.replace('"', "'")  # Ensure single quotes for eval
                        data = ast.literal_eval(prepared_str)
                    except (SyntaxError, ValueError):
                        data = json.loads(clean_line)  # Fallback to json.loads

                    # Store parsed data
                    record = {
                        "idx": idx,
                        "offset": data.get("_offset", 0),
                        "length": data.get("_length", 0),
                        "number": data.get("number", -1),
                        "idx_uri": fname,
                        "grib_uri": basename
                    }

                    # Add remaining fields
                    for key, value in data.items():
                        if key not in record:
                            record[key] = value

                    records.append(record)

                except Exception as e:
                    print(f"Warning: Could not parse line {idx} in {fname}: {e}")
                    continue  # Skip problematic lines instead of raising an error

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {fname} not found.")

    # Convert to DataFrame
    result = pd.DataFrame(records)

    # If empty, return early
    if result.empty:
        print(f"Warning: No valid entries found in {fname}. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Add timestamp column
    result["indexed_at"] = tstamp or pd.Timestamp.now()

    # Validate uniqueness if required
    if validate and result.duplicated().any():
        raise ValueError(f"Attribute mapping for GRIB file {basename} is not unique")

    print(f"Completed indexing {len(result)} entries from {fname}")
    return result.set_index("idx")


def ecmwf_enfo_s3_parse_index_with_ast(
    fs: fsspec.AbstractFileSystem,
    basename: str,
    suffix: str = "index",
    tstamp: Optional[pd.Timestamp] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """
    Standalone method used to extract ALL metadata from a grib2 index file using ast.literal_eval

    :param fs: the file system to read from
    :param basename: the base name is the full path to the grib file
    :param suffix: the suffix is the ending for the index file
    :param tstamp: the timestamp to record for this index process
    :param validate: whether to validate uniqueness of attributes
    :return: the data frame containing the results with all json attributes
    """
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"

    fs.invalidate_cache(fname)
    fs.invalidate_cache(basename)

    baseinfo = fs.info(basename)

    # Lists to store all parsed data
    records = []

    with fs.open(fname, "r") as f:
        for idx, line in enumerate(f):
            try:
                # Removing the trailing characters if there's any at the end of the line
                clean_line = line.strip().rstrip(',')

                # Using ast.literal_eval instead of json.loads
                try:
                    # Convert double quotes to single quotes
                    prepared_str = clean_line.replace('"', "'")
                    data = ast.literal_eval(prepared_str)
                except (SyntaxError, ValueError) as e:
                    # Fallback to json.loads if ast.literal_eval fails
                    data = json.loads(clean_line)

                # Store data along with index and file info
                record = {
                    'idx': idx,
                    'offset': data.get('_offset', 0),
                    'length': data.get('_length', 0),
                    'number': data.get('number', -1),
                    'idx_uri': fname,
                    'grib_uri': basename
                }

                # Add all fields from the JSON to the record
                for key, value in data.items():
                    if key not in record:  # Avoid overwriting index fields
                        record[key] = value

                records.append(record)

            except Exception as e:
                # Handle cases where both conversion methods fail
                raise ValueError(f"Could not parse data from line: {line}") from e

    # Create the DataFrame from all records
    result = pd.DataFrame(records)

    # Add timestamps
    if tstamp is None:
        tstamp = pd.Timestamp.now()
    result['indexed_at'] = tstamp

    # Add filesystem metadata
    if "s3" in fs.protocol:
        # Use ETag as the S3 equivalent to crc32c
        result.loc[:, "grib_etag"] = baseinfo.get("ETag")
        result.loc[:, "grib_updated_at"] = pd.to_datetime(
            baseinfo.get("LastModified")
        ).tz_localize(None)

        idxinfo = fs.info(fname)
        result.loc[:, "idx_etag"] = idxinfo.get("ETag")
        result.loc[:, "idx_updated_at"] = pd.to_datetime(
            idxinfo.get("LastModified")
        ).tz_localize(None)
    else:
        # For other filesystems
        result.loc[:, "grib_crc32"] = None
        result.loc[:, "grib_updated_at"] = None
        result.loc[:, "idx_crc32"] = None
        result.loc[:, "idx_updated_at"] = None

    if validate and len(result) != len(result.drop_duplicates()):
        raise ValueError(f"Attribute mapping for grib file {basename} is not unique)")

    print(f'Completed index files and found {len(result.index)} entries in it')
    return result.set_index("idx")


def convert_forecast_dict_to_target_variables(forecast_dict):
    """
    Convert a forecast dictionary with shorthand notation to target variables format

    Parameters:
    -----------
    forecast_dict : dict
        Dictionary with parameter names and shorthand notation like "t:pl"

    Returns:
    --------
    target_variables : dict
        Dictionary in the format required by ecmwf_enfo_index_with_target_variables
    """
    target_variables = {}

    for param_name, param_code in forecast_dict.items():
        # Split the code to get parameter and level type
        parts = param_code.split(':')

        if len(parts) != 2:
            print(f"Warning: Invalid code format for {param_name}: {param_code}")
            continue

        ecmwf_param, levtype = parts

        # Set up the levels list
        levels = []
        if levtype == 'pl':
            # For pressure level variables, we want level 50
            levels = [50]

        # Add to target variables
        target_variables[ecmwf_param] = {
            'levels': levels,
            'levtype': levtype
        }

    return target_variables


def ecmwf_enfo_index_with_target_variables(edf, target_variables):
    """
    Filter ECMWF index DataFrame according to specified target variables.

    Parameters:
    -----------
    edf : pd.DataFrame
        The DataFrame containing the ECMWF index data
    target_variables : dict
        Dictionary specifying which variables to extract at which levels
        Format: {'param_name': {'levels': [list_of_levels], 'levtype': 'level_type'}}

    Returns:
    --------
    combined_dict : dict
        Dictionary with unique parameter-level combinations
    """
    print(f"Total rows in original index: {len(edf)}")

    # Convert numeric columns that might be mixed types
    if 'levelist' in edf.columns:
        edf['levelist'] = edf['levelist'].astype(str).fillna('null')

    # Make sure number column is consistently typed
    if 'number' in edf.columns:
        # Try to convert to int where possible, handle errors
        try:
            edf['number'] = edf['number'].astype(int)
        except (ValueError, TypeError):
            # If conversion fails, convert everything to string
            edf['number'] = edf['number'].astype(str)

    selected_rows = []

    # Process each target variable
    for param_name, param_info in target_variables.items():
        levtype = param_info['levtype']
        levels = param_info['levels']

        if levtype == 'pl' and levels:
            # For pressure levels with specific levels
            for level in levels:
                level_str = str(level)
                mask = (edf['param'] == param_name) & (edf['levtype'] == levtype) & (edf['levelist'].astype(str) == level_str)
                matching = edf[mask]
                print(f"Variable: {param_name}, levtype: {levtype}, level: {level} - Found {len(matching)} rows")
                selected_rows.append(matching)
        elif levtype == 'sfc':
            # For surface variables (no level needed)
            mask = (edf['param'] == param_name) & (edf['levtype'] == levtype)
            matching = edf[mask]
            print(f"Variable: {param_name}, levtype: {levtype} - Found {len(matching)} rows")
            selected_rows.append(matching)

    # Combine all selected rows
    if not selected_rows:
        return {}

    filtered_df = pd.concat(selected_rows)
    print(f"Total selected rows: {len(filtered_df)}")

    # Group by parameter, level type, and level list
    # For each group, take the first ensemble member number (we'll create all ensemble members later)
    # Use a safer approach with explicit type conversion
    def safe_first_number(x):
        try:
            # Try to convert all values to integers for sorting
            nums = [int(val) for val in x.unique()]
            return sorted(nums)[0]
        except (ValueError, TypeError):
            # If conversion fails, return the first value
            return x.iloc[0]

    grouped = filtered_df.groupby(['param', 'levtype', 'levelist']).agg({
        'number': safe_first_number
    }).reset_index()

    # Create the dictionary
    combined_dict = {}
    for _, row in grouped.iterrows():
        key = f"{row['param']}_{row['levtype']}_{row['levelist']}"
        combined_dict[key] = {
            'param': row['param'],
            'levtype': row['levtype'],
            'number': row['number'],
            'levelist': row['levelist']
        }

    print(f"Unique parameter combinations: {len(combined_dict)}")
    return combined_dict


def ecmwf_enfo_duplicate_dict_ens_mem(var_dict):
    """
    Create duplicates of parameter entries for each ensemble member.

    Parameters:
    -----------
    var_dict : dict
        Dictionary with unique parameter combinations

    Returns:
    --------
    updated_data_dict : dict
        Dictionary with parameter entries for all ensemble members
    """
    # Generate sequence for ensemble members 1-50, with control (-1) at the start
    ens_numbers = list(range(1, 51))
    ens_numbers.insert(0, -1)

    updated_data_dict = {}
    for ens_number in ens_numbers:
        for key, subdict in var_dict.items():
            updated_subdict = subdict.copy()
            updated_subdict['number'] = int(ens_number)  # Ensure number is integer
            new_key = f"{key}_ens{ens_number}"
            updated_data_dict[new_key] = updated_subdict

    print(f"Total parameter-ensemble combinations: {len(updated_data_dict)}")
    return updated_data_dict


def ecmwf_enfo_index_df_create_with_keys(ecmwf_s3url, forecast_dict):
    # Parse the ECMWF index file - detect if it's S3 URL or local file
    if ecmwf_s3url.startswith('s3://'):
        # Handle S3 URLs - use anonymous access for public ECMWF bucket
        fs = fsspec.filesystem("s3", anon=True)
        suffix = 'index'
        edf = ecmwf_enfo_s3_parse_index_with_ast(fs=fs, basename=ecmwf_s3url, suffix=suffix)
    else:
        # Handle local files
        edf = ecmwf_enfo_local_parse_index(basename=ecmwf_s3url)

    # Convert number to consistent type if it exists
    if 'number' in edf.columns:
        try:
            edf['number'] = edf['number'].astype(int)
        except (ValueError, TypeError):
            # If conversion fails for any reason, convert to string
            edf['number'] = edf['number'].astype(str)

    # Convert forecast dictionary to target variables format
    target_variables = convert_forecast_dict_to_target_variables(forecast_dict)

    # Filter the dataframe according to target variables
    combined_dict = ecmwf_enfo_index_with_target_variables(edf, target_variables)

    # Create ensemble member variations
    all_em = ecmwf_enfo_duplicate_dict_ens_mem(combined_dict)

    # Map and filter index entries
    idx_mapping = {}
    subset_dfs = []

    for ens_key, conditions in all_em.items():
        # Initialize mask with all True values
        mask = pd.Series(True, index=edf.index)

        # Apply conditions one by one
        for col, value in conditions.items():
            if col == 'number':
                # Try to handle both int and string comparisons
                try:
                    value_int = int(value)
                    col_int = edf[col].astype(int)
                    mask &= (col_int == value_int)
                except (ValueError, TypeError):
                    # Fall back to string comparison
                    mask &= (edf[col].astype(str) == str(value))
            elif value == 'null':
                # Handle null values
                mask &= edf[col].isna()
            else:
                # Try both string and original comparison
                col_vals = edf[col]
                str_mask = (col_vals.astype(str) == str(value))
                orig_mask = (col_vals == value)
                mask &= (str_mask | orig_mask)

        matching_rows = edf[mask]
        if not matching_rows.empty:
            subset_dfs.append(matching_rows)
            for idx in matching_rows.index:
                idx_mapping[idx] = ens_key

    # Combine all subset DataFrames into a single DataFrame
    subset_edf = pd.concat(subset_dfs) if subset_dfs else pd.DataFrame(columns=edf.columns)
    print(f"Final filtered dataset: {len(subset_edf)} rows")

    return edf, subset_edf, idx_mapping, combined_dict


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
    
     # Convert step to int32 (directly from JSON's "step" field)
    if 'step' in expanded_df.columns:
        # Handle non-numeric values and convert to integer
        expanded_df['step'] = (
            pd.to_numeric(expanded_df['step'], errors='coerce')
            .fillna(0)
            .astype('int32')
        )
    else:
        expanded_df['step'] = 0  # Default for missing steps    # Make sure level is numeric
        
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
