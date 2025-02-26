import fsspec
import pandas as pd
import numpy as np
import copy

from kerchunk.grib2 import scan_grib, grib_tree
from utils import s3_parse_ecmwf_grib_idx
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
from dynamic_zarr_store import strip_datavar_chunks


fs=fsspec.filesystem("s3")
#basename=f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-3h-enfo-ef.grib2"
suffix= "index"

def ecmwf_idx_unique_dict(edf):
    # Fill empty rows or missing values in 'levelist' with 'null'
    edf['levelist'] = edf['levelist'].fillna('null')
    # Filter for both pl (level 50) and sfc parameters
    combined_params = edf[
       ((edf['levtype'] == 'pl') & (edf['levelist'] == '50')) |
       (edf['levtype'] == 'sfc')
    ].groupby(['param', 'levtype', 'levelist']).agg({
        'ens_number': lambda x: -1 if (-1 in x.values) else x.iloc[0]
    }).reset_index()
    
    combined_dict = {}
    for _, row in combined_params.iterrows():
        key = f"{row['param']}_{row['levtype']}"
        combined_dict[key] = {
            'param': row['param'],
            'levtype': row['levtype'],
            'ens_number': row['ens_number'],
            'levelist': 'null' if row['levtype'] == 'sfc' else '50'
        }
    return combined_dict

def ecmwf_duplicate_dict_ens_mem(var_dict):
   # Generate sequence for ensemble members 1-50
   ens_numbers = np.arange(1, 51)
   ens_numbers = np.insert(ens_numbers, 0, -1)
   # Initialize dictionary with original entries (control forecast)
   updated_data_dict = var_dict.copy()
   
   # Add ensemble member entries
   for ens_number in ens_numbers:
       for key, subdict in var_dict.items():
           updated_subdict = subdict.copy()
           updated_subdict['ens_number'] = int(ens_number)
           new_key = f"{key}_ens{ens_number}"
           updated_data_dict[new_key] = updated_subdict
           
   return updated_data_dict
    

def ecmwf_get_matching_indices(filter_dict, df):
    """
    Get the indices of rows in a DataFrame that match the criteria in a dictionary.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to filter.
        criteria_dict (dict): A dictionary where keys are labels, and values are sub-dictionaries 
                              containing column-value pairs for filtering.

    Returns:
        list: A sorted list of unique indices where the rows match the criteria.
    """
    # Create filter conditions for each entry in the dictionary
    filtered_dfs = []
    for key, conditions in filter_dict.items():
       mask = True
       for col, value in conditions.items():
           if value == 'null':
               mask = mask & (df[col] == 'null')
           else:
               mask = mask & (df[col] == value)
       filtered_df = df[mask]
       if not filtered_df.empty:
           filtered_dfs.append(filtered_df)
    
    # Combine all filtered results
    final_df = pd.concat(filtered_dfs)
    
    # Optional: Sort by param and levtype to match dictionary order
    final_df = final_df.sort_values(['param', 'levtype'])
    idx_index=final_df.index.tolist()
    return idx_index


def ecmwf_idx_df_create_with_keys(ecmwf_s3url):
    fs = fsspec.filesystem("s3")
    suffix = 'index'
    idx_file_index = s3_parse_ecmwf_grib_idx(fs=fs, basename=ecmwf_s3url, suffix=suffix)
    edf = pd.concat([
        idx_file_index.drop('attr', axis=1),
        idx_file_index['attr'].apply(pd.Series)
    ], axis=1)
    
    # Get the unique dict and duplicate it for ensembles.
    combined_dict = ecmwf_idx_unique_dict(edf)
    all_em = ecmwf_duplicate_dict_ens_mem(combined_dict)
    
    # Build a mapping of index -> ensemble key
    idx_mapping = {}
    for ens_key, conditions in all_em.items():
        # Build the mask for these conditions:
        mask = True
        for col, value in conditions.items():
            if value == 'null':
                mask = mask & (edf[col] == 'null')
            else:
                mask = mask & (edf[col] == value)
        matching_indices = edf[mask].index.tolist()
        for idx in matching_indices:
            # If a row matches multiple keys, decide how to handle duplicates.
            idx_mapping[idx] = ens_key
    return idx_mapping,combined_dict 


def ecmwf_filter_scan_grib(ecmwf_s3url):
    # Scan the GRIB file only once.
    esc = scan_grib(ecmwf_s3url)
    print(f"Completed scan_grib for {ecmwf_s3url}")
    
    # Get a mapping from index to ensemble key.
    idx_mapping, _ = ecmwf_idx_df_create_with_keys(ecmwf_s3url)
    print(f"Found {len(idx_mapping)} matching indices")
    
    # Build the groups list, attaching the ensemble key to each group.
    groups = []
    for i, ens_key in idx_mapping.items():
        group = esc[i]
        group['number'] = ens_key  # Tag the group with its ensemble member key.
        groups.append(group)
    return groups, idx_mapping


def update_tree_with_ens_members(grib_tree_store, idx_mapping):
    """
    Update the grib_tree_store's refs keys to include ensemble member information.
    """
    new_refs = {}
    for ref_key, ref_val in grib_tree_store['refs'].items():
        try:
            # This example assumes that the reference key contains an index 
            # that you can extract (e.g., "group_42/...")
            index = int(ref_key.split('_')[1].split('/')[0])
            ens_key = idx_mapping.get(index, "unknown")
        except Exception:
            ens_key = "unknown"
        new_key = f"ens_{ens_key}/{ref_key}"
        new_refs[new_key] = ref_val
    grib_tree_store['refs'] = new_refs
    return grib_tree_store


def ecmwf_filter_build_grib_tree(ecmwf_files: List[str]) -> Tuple[dict, dict, dict]:
    print("Building Grib Tree")
    
    # Combine groups and idx_mappings from all files.
    all_groups = []
    combined_idx_mapping = {}
    for eurl in ecmwf_files:
        groups, idx_mapping = ecmwf_filter_scan_grib(eurl)
        all_groups.extend(groups)
        # Merge the mappings; if keys overlap, later files will override earlier ones.
        combined_idx_mapping.update(idx_mapping)
    
    # Build the GRIB tree store from the combined groups.
    grib_tree_store = grib_tree(all_groups)
    
    # Update the GRIB tree store's references with ensemble member information.
    updated_tree = update_tree_with_ens_members(grib_tree_store, combined_idx_mapping)
    
    # Optionally, create a deflated version of the tree.
    deflated_tree = copy.deepcopy(updated_tree)
    strip_datavar_chunks(deflated_tree)
    
    print(f"Original references: {len(updated_tree['refs'])}")
    print(f"Stripped references: {len(deflated_tree['refs'])}")
    return grib_tree_store, updated_tree, deflated_tree



date_str='20240229'

ecmwf_files=[f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-0h-enfo-ef.grib2",
            f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-3h-enfo-ef.grib2"]


grib_tree_store, ecmwf_grib_tree_store, deflated_ecmwf_grib_tree_store=ecmwf_filter_build_grib_tree(ecmwf_files)
