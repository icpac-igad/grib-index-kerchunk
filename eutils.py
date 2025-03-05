import fsspec
import pandas as pd
import ast
import json
import numpy as np
from typing import Optional, Dict, Any, List

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


def ecmwf_enfo_index_df_create_with_keys(ecmwf_s3url, forecast_dict):
    # Parse the ECMWF index file
    fs = fsspec.filesystem("s3")
    suffix = 'index'
    edf = ecmwf_enfo_s3_parse_index_with_ast(fs=fs, basename=ecmwf_s3url, suffix=suffix)
    
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