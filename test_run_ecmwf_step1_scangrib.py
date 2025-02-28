import fsspec
import pandas as pd
import numpy as np
import copy
import json 

import datatree 
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

def dict_to_store(d=None):
    """Create a simple dictionary-based store."""
    if d is None:
        return {}
    return d

def translate_refs_serializable(refs: dict) -> dict:
    """Process references to ensure they're JSON serializable."""
    for key in list(refs.keys()):
        value = refs[key]
        # Convert bytes to strings
        if isinstance(value, bytes):
            refs[key] = value.decode('utf-8')
        # Convert lists of bytes to lists of strings
        elif isinstance(value, list) and value and isinstance(value[0], bytes):
            refs[key] = [v.decode('utf-8') if isinstance(v, bytes) else v for v in value]
    return refs


def ensemble_grib_tree(
    message_groups: Iterable[Dict],
    remote_options=None,
) -> Dict:
    """
    Modified version of grib_tree that properly handles ensemble dimensions.
    """
    filters = ["stepType", "typeOfLevel"]
    zarr_store_dict = {}  # Use a regular dict instead of zarr.storage.MemoryStore
    zroot = zarr.group()  # Create a temporary root group

    aggregations: Dict[str, List] = defaultdict(list)
    aggregation_dims: Dict[str, Set] = defaultdict(set)
    ensemble_info: Dict[str, Set] = defaultdict(set)

    unknown_counter = 0
    for msg_ind, group in enumerate(message_groups):
        assert group["version"] == 1

        ensemble_member = None
        if ".zattrs" in group["refs"]:
            try:
                root_attrs = json.loads(group["refs"][".zattrs"])
                if "ensemble_member" in root_attrs:
                    ensemble_member = root_attrs["ensemble_member"]
            except json.JSONDecodeError:
                pass

        for key in group["refs"]:
            if key == "number/0" or key.endswith("/number/0"):
                try:
                    val = group["refs"][key]
                    if isinstance(val, str):
                        val_bytes = val.encode('latin1')
                        try:
                            arr = np.frombuffer(val_bytes, dtype=np.int64)
                            if len(arr) == 1:
                                ensemble_member = int(arr[0])
                        except:
                            pass
                except:
                    pass

        try:
            gattrs = json.loads(group["refs"][".zattrs"])
            coordinates = gattrs["coordinates"].split(" ")
        except (KeyError, json.JSONDecodeError):
            print(f"Warning: Issue with attributes for message {msg_ind}")
            continue

        vname = None
        for key in group["refs"]:
            name = key.split("/")[0]
            if name not in [".zattrs", ".zgroup"] and name not in coordinates:
                vname = name
                break

        if vname is None:
            print(f"Warning: No data variable found for message {msg_ind}")
            continue

        if vname == "unknown":
            print(f"Warning: Dropping unknown variable in message {msg_ind}")
            unknown_counter += 1
            continue

        try:
            dattrs = json.loads(group["refs"][f"{vname}/.zattrs"])
        except (KeyError, json.JSONDecodeError):
            print(f"Warning: Issue with variable attributes for {vname} in message {msg_ind}")
            continue

        gfilters = {}
        for key in filters:
            attr_val = dattrs.get(f"GRIB_{key}")
            if attr_val is None:
                continue
            if attr_val == "unknown":
                print(f"Warning: Found 'unknown' attribute value for key {key} in var {vname}")
            gfilters[key] = attr_val

        # Build group path
        path_parts = [vname]
        for key, value in gfilters.items():
            if value:
                path_parts.append(value)
        
        # Add ensemble part if applicable
        if ensemble_member is not None:
            path_parts.append(f"ensemble_{ensemble_member}")
            parent_path = "/".join(path_parts[:-1])
            ensemble_info[parent_path].add(ensemble_member)
        
        # Join to form full path
        full_path = "/".join(path_parts)
        
        # Store group in aggregations
        aggregations[full_path].append(group)

        # Track level coordinate values
        for key, entry in group["refs"].items():
            name = key.split("/")[0]
            if name == gfilters.get("typeOfLevel") and key.endswith("0"):
                if isinstance(entry, list):
                    entry = tuple(entry)
                aggregation_dims[full_path].add(entry)

    concat_dims = ["time", "step"]
    identical_dims = ["longitude", "latitude"]
    
    # Create result container
    result_refs = {'.zgroup': json.dumps({'zarr_format': 2})}
    
    # Process each path
    for path in aggregations.keys():
        path_parts = path.split("/")
        catdims = concat_dims.copy()
        idims = identical_dims.copy()

        # Handle levels
        level_dimension_value_count = len(aggregation_dims.get(path, ()))
        if len(path_parts) >= 3:  # Has a level type
            level_group_name = path_parts[2]
            if level_dimension_value_count == 0:
                print(f"Path {path} has no coordinate value for level {level_group_name}")
            elif level_dimension_value_count == 1:
                idims.append(level_group_name)
            elif level_dimension_value_count > 1:
                catdims.insert(3, level_group_name)

        # Handle ensemble
        if len(path_parts) >= 4 and path_parts[3].startswith("ensemble_"):
            parent_path = "/".join(path_parts[:-1])
            if parent_path in ensemble_info and len(ensemble_info[parent_path]) > 1:
                if "number" not in catdims:
                    catdims.append("number")

        print(f"{path} calling MultiZarrToZarr with idims {idims} and catdims {catdims}")

        try:
            mzz = MultiZarrToZarr(
                aggregations[path],
                remote_options=remote_options,
                concat_dims=catdims,
                identical_dims=idims,
            )
            group = mzz.translate()

            # Copy references with path prefix
            for key, value in group["refs"].items():
                if key not in [".zattrs", ".zgroup"]:
                    result_refs[f"{path}/{key}"] = value
                elif key == ".zattrs":
                    # Add path-specific attributes
                    result_refs[f"{path}/.zattrs"] = value
                    
            # Add a .zgroup marker for this path
            result_refs[f"{path}/.zgroup"] = json.dumps({'zarr_format': 2})
            
        except Exception as e:
            print(f"Error processing path {path}: {e}")
            import traceback
            traceback.print_exc()

    # Build the final result
    result = {
        "refs": result_refs,
        "version": 1
    }

    return result


def ecmwf_filter_scan_grib(ecmwf_s3url):
    """
    Scan ECMWF GRIB file and add ensemble information to the Zarr references.
    
    Returns a list of groups with ensemble information and the index mapping.
    """
    # Scan the GRIB file
    esc_groups = scan_grib(ecmwf_s3url)
    print(f"Completed scan_grib for {ecmwf_s3url}, found {len(esc_groups)} messages")
    
    # Get a mapping from index to ensemble key
    idx_mapping, _ = ecmwf_idx_df_create_with_keys(ecmwf_s3url)
    print(f"Found {len(idx_mapping)} matching indices")
    
    # Build the modified groups list
    modified_groups = []
    for i, group in enumerate(esc_groups):
        if i in idx_mapping:
            ens_key = idx_mapping[i]
            
            # Extract ensemble number from the key
            if 'ens' in ens_key:
                ens_number = int(ens_key.split('ens')[-1])
            else:
                ens_number = -1  # Control member
            
            # Create a deep copy of the group
            mod_group = copy.deepcopy(group)
            refs = mod_group['refs']
            
            # Find the main data variables
            data_vars = []
            for key in refs:
                if key.endswith('/.zattrs'):
                    var_name = key.split('/')[0]
                    if not var_name.startswith('.'):
                        try:
                            attrs = json.loads(refs[key])
                            # Check if this is a coordinate
                            if '_ARRAY_DIMENSIONS' in attrs and len(attrs['_ARRAY_DIMENSIONS']) > 0:
                                # This might be a data variable
                                if var_name not in ['latitude', 'longitude', 'number', 'time', 'step', 'valid_time']:
                                    data_vars.append(var_name)
                        except json.JSONDecodeError:
                            print(f"Error decoding {key}")
            
            # Update root attributes
            if '.zattrs' in refs:
                try:
                    root_attrs = json.loads(refs['.zattrs'])
                    # Add ensemble information
                    root_attrs['ensemble_member'] = ens_number
                    root_attrs['ensemble_key'] = ens_key
                    
                    # Update coordinates to include number if needed
                    if 'coordinates' in root_attrs:
                        coords = root_attrs['coordinates'].split()
                        if 'number' not in coords:
                            coords.append('number')
                            root_attrs['coordinates'] = ' '.join(coords)
                    
                    refs['.zattrs'] = json.dumps(root_attrs)
                except json.JSONDecodeError:
                    print(f"Error updating root attributes for group {i}")
            
            # Update data variable attributes
            for var_name in data_vars:
                attr_key = f"{var_name}/.zattrs"
                if attr_key in refs:
                    try:
                        var_attrs = json.loads(refs[attr_key])
                        var_attrs['ensemble_member'] = ens_number
                        var_attrs['ensemble_key'] = ens_key
                        refs[attr_key] = json.dumps(var_attrs)
                    except json.JSONDecodeError:
                        print(f"Error updating attributes for {var_name}")
            
            # Check if number coordinate exists, add if needed
            has_number = False
            for key in refs:
                if key == 'number/.zattrs' or key.endswith('/number/.zattrs'):
                    has_number = True
                    break
            
            if not has_number:
                print(f"Adding number coordinate for group {i}, ensemble {ens_number}")
                
                # Add number coordinate
                import numpy as np
                
                refs['number/.zarray'] = json.dumps({
                    "chunks": [],
                    "compressor": None, 
                    "dtype": "<i8",  # Use int64 to match existing format
                    "fill_value": None,
                    "filters": None,
                    "order": "C",
                    "shape": [],
                    "zarr_format": 2
                })
                
                refs['number/.zattrs'] = json.dumps({
                    "_ARRAY_DIMENSIONS": [],
                    "long_name": "ensemble member numerical id",
                    "standard_name": "realization",
                    "units": "1"
                })
                
                # Properly encode the number value
                ens_num_array = np.array(ens_number, dtype=np.int64)
                refs['number/0'] = ens_num_array.tobytes().decode('latin1')
            
            modified_groups.append(mod_group)
    
    return modified_groups, idx_mapping

def ensemble_grib_tree(
    message_groups: Iterable[Dict],
    remote_options=None,
) -> Dict:
    """
    Modified version of grib_tree that properly handles ensemble dimensions.
    """
    filters = ["stepType", "typeOfLevel"]
    zarr_store_dict = {}  # Use a regular dict instead of zarr.storage.MemoryStore
    zroot = zarr.group()  # Create a temporary root group

    aggregations: Dict[str, List] = defaultdict(list)
    aggregation_dims: Dict[str, Set] = defaultdict(set)
    ensemble_info: Dict[str, Set] = defaultdict(set)

    unknown_counter = 0
    for msg_ind, group in enumerate(message_groups):
        assert group["version"] == 1

        ensemble_member = None
        if ".zattrs" in group["refs"]:
            try:
                root_attrs = json.loads(group["refs"][".zattrs"])
                if "ensemble_member" in root_attrs:
                    ensemble_member = root_attrs["ensemble_member"]
            except json.JSONDecodeError:
                pass

        for key in group["refs"]:
            if key == "number/0" or key.endswith("/number/0"):
                try:
                    val = group["refs"][key]
                    if isinstance(val, str):
                        val_bytes = val.encode('latin1')
                        try:
                            arr = np.frombuffer(val_bytes, dtype=np.int64)
                            if len(arr) == 1:
                                ensemble_member = int(arr[0])
                        except:
                            pass
                except:
                    pass

        try:
            gattrs = json.loads(group["refs"][".zattrs"])
            coordinates = gattrs["coordinates"].split(" ")
        except (KeyError, json.JSONDecodeError):
            print(f"Warning: Issue with attributes for message {msg_ind}")
            continue

        vname = None
        for key in group["refs"]:
            name = key.split("/")[0]
            if name not in [".zattrs", ".zgroup"] and name not in coordinates:
                vname = name
                break

        if vname is None:
            print(f"Warning: No data variable found for message {msg_ind}")
            continue

        if vname == "unknown":
            print(f"Warning: Dropping unknown variable in message {msg_ind}")
            unknown_counter += 1
            continue

        try:
            dattrs = json.loads(group["refs"][f"{vname}/.zattrs"])
        except (KeyError, json.JSONDecodeError):
            print(f"Warning: Issue with variable attributes for {vname} in message {msg_ind}")
            continue

        gfilters = {}
        for key in filters:
            attr_val = dattrs.get(f"GRIB_{key}")
            if attr_val is None:
                continue
            if attr_val == "unknown":
                print(f"Warning: Found 'unknown' attribute value for key {key} in var {vname}")
            gfilters[key] = attr_val

        # Build group path
        path_parts = [vname]
        for key, value in gfilters.items():
            if value:
                path_parts.append(value)
        
        # Add ensemble part if applicable
        if ensemble_member is not None:
            path_parts.append(f"ensemble_{ensemble_member}")
            parent_path = "/".join(path_parts[:-1])
            ensemble_info[parent_path].add(ensemble_member)
        
        # Join to form full path
        full_path = "/".join(path_parts)
        
        # Store group in aggregations
        aggregations[full_path].append(group)

        # Track level coordinate values
        for key, entry in group["refs"].items():
            name = key.split("/")[0]
            if name == gfilters.get("typeOfLevel") and key.endswith("0"):
                if isinstance(entry, list):
                    entry = tuple(entry)
                aggregation_dims[full_path].add(entry)

    concat_dims = ["time", "step"]
    identical_dims = ["longitude", "latitude"]
    
    # Create result container
    result_refs = {'.zgroup': json.dumps({'zarr_format': 2})}
    
    # Process each path
    for path in aggregations.keys():
        path_parts = path.split("/")
        catdims = concat_dims.copy()
        idims = identical_dims.copy()

        # Handle levels
        level_dimension_value_count = len(aggregation_dims.get(path, ()))
        if len(path_parts) >= 3:  # Has a level type
            level_group_name = path_parts[2]
            if level_dimension_value_count == 0:
                print(f"Path {path} has no coordinate value for level {level_group_name}")
            elif level_dimension_value_count == 1:
                idims.append(level_group_name)
            elif level_dimension_value_count > 1:
                catdims.insert(3, level_group_name)

        # Handle ensemble
        if len(path_parts) >= 4 and path_parts[3].startswith("ensemble_"):
            parent_path = "/".join(path_parts[:-1])
            if parent_path in ensemble_info and len(ensemble_info[parent_path]) > 1:
                if "number" not in catdims:
                    catdims.append("number")

        print(f"{path} calling MultiZarrToZarr with idims {idims} and catdims {catdims}")

        try:
            mzz = MultiZarrToZarr(
                aggregations[path],
                remote_options=remote_options,
                concat_dims=catdims,
                identical_dims=idims,
            )
            group = mzz.translate()

            # Copy references with path prefix
            for key, value in group["refs"].items():
                if key not in [".zattrs", ".zgroup"]:
                    result_refs[f"{path}/{key}"] = value
                elif key == ".zattrs":
                    # Add path-specific attributes
                    result_refs[f"{path}/.zattrs"] = value
                    
            # Add a .zgroup marker for this path
            result_refs[f"{path}/.zgroup"] = json.dumps({'zarr_format': 2})
            
        except Exception as e:
            print(f"Error processing path {path}: {e}")
            import traceback
            traceback.print_exc()

    # Build the final result
    result = {
        "refs": result_refs,
        "version": 1
    }

    return result




def ecmwf_filter_scan_grib(ecmwf_s3url):
    """
    Scan ECMWF GRIB file and add ensemble information to the Zarr references.
    
    Returns a list of groups with ensemble information and the index mapping.
    """
    # Scan the GRIB file
    esc_groups = scan_grib(ecmwf_s3url)
    print(f"Completed scan_grib for {ecmwf_s3url}, found {len(esc_groups)} messages")
    
    # Get a mapping from index to ensemble key
    idx_mapping, _ = ecmwf_idx_df_create_with_keys(ecmwf_s3url)
    print(f"Found {len(idx_mapping)} matching indices")
    
    # Build the modified groups list
    modified_groups = []
    for i, group in enumerate(esc_groups):
        if i in idx_mapping:
            ens_key = idx_mapping[i]
            
            # Extract ensemble number from the key
            if 'ens' in ens_key:
                ens_number = int(ens_key.split('ens')[-1])
            else:
                ens_number = -1  # Control member
            
            # Create a deep copy of the group
            mod_group = copy.deepcopy(group)
            refs = mod_group['refs']
            
            # Find the main data variables
            data_vars = []
            for key in refs:
                if key.endswith('/.zattrs'):
                    var_name = key.split('/')[0]
                    if not var_name.startswith('.'):
                        try:
                            attrs = json.loads(refs[key])
                            # Check if this is a coordinate
                            if '_ARRAY_DIMENSIONS' in attrs and len(attrs['_ARRAY_DIMENSIONS']) > 0:
                                # This might be a data variable
                                if var_name not in ['latitude', 'longitude', 'number', 'time', 'step', 'valid_time']:
                                    data_vars.append(var_name)
                        except json.JSONDecodeError:
                            print(f"Error decoding {key}")
            
            # Update root attributes
            if '.zattrs' in refs:
                try:
                    root_attrs = json.loads(refs['.zattrs'])
                    # Add ensemble information
                    root_attrs['ensemble_member'] = ens_number
                    root_attrs['ensemble_key'] = ens_key
                    
                    # Update coordinates to include number if needed
                    if 'coordinates' in root_attrs:
                        coords = root_attrs['coordinates'].split()
                        if 'number' not in coords:
                            coords.append('number')
                            root_attrs['coordinates'] = ' '.join(coords)
                    
                    refs['.zattrs'] = json.dumps(root_attrs)
                except json.JSONDecodeError:
                    print(f"Error updating root attributes for group {i}")
            
            # Update data variable attributes
            for var_name in data_vars:
                attr_key = f"{var_name}/.zattrs"
                if attr_key in refs:
                    try:
                        var_attrs = json.loads(refs[attr_key])
                        var_attrs['ensemble_member'] = ens_number
                        var_attrs['ensemble_key'] = ens_key
                        refs[attr_key] = json.dumps(var_attrs)
                    except json.JSONDecodeError:
                        print(f"Error updating attributes for {var_name}")
            
            # Check if number coordinate exists, add if needed
            has_number = False
            for key in refs:
                if key == 'number/.zattrs' or key.endswith('/number/.zattrs'):
                    has_number = True
                    break
            
            if not has_number:
                print(f"Adding number coordinate for group {i}, ensemble {ens_number}")
                
                # Add number coordinate
                import numpy as np
                
                refs['number/.zarray'] = json.dumps({
                    "chunks": [],
                    "compressor": None, 
                    "dtype": "<i8",  # Use int64 to match existing format
                    "fill_value": None,
                    "filters": None,
                    "order": "C",
                    "shape": [],
                    "zarr_format": 2
                })
                
                refs['number/.zattrs'] = json.dumps({
                    "_ARRAY_DIMENSIONS": [],
                    "long_name": "ensemble member numerical id",
                    "standard_name": "realization",
                    "units": "1"
                })
                
                # Properly encode the number value
                ens_num_array = np.array(ens_number, dtype=np.int64)
                refs['number/0'] = ens_num_array.tobytes().decode('latin1')
            
            modified_groups.append(mod_group)
    
    return modified_groups, idx_mapping


def organize_ensemble_tree(original_tree):
    """
    Organize the tree to include ensemble information as a dimension.
    This works by reorganizing the .zattrs files to include 'number' as a dimension.
    """
    # Create a deep copy of the original tree
    ensemble_tree = copy.deepcopy(original_tree)
    
    # Find all .zattrs files in the tree
    attrs_keys = [k for k in ensemble_tree['refs'] if k.endswith('/.zattrs')]
    
    # Process each .zattrs file
    for key in attrs_keys:
        try:
            # Parse the attributes
            attrs = json.loads(ensemble_tree['refs'][key])
            
            # Check if this is a data variable (has _ARRAY_DIMENSIONS)
            if '_ARRAY_DIMENSIONS' in attrs:
                # Add 'number' as a dimension if not already present
                if 'number' not in attrs['_ARRAY_DIMENSIONS']:
                    # Add number dimension - insert after time and step but before spatial dims
                    # Usually dimensions order is [time, step, ...spatial_dims]
                    dimensions = attrs['_ARRAY_DIMENSIONS']
                    if 'time' in dimensions and 'step' in dimensions:
                        step_index = dimensions.index('step')
                        dimensions.insert(step_index + 1, 'number')
                    else:
                        # Add at the beginning if no time/step
                        dimensions.insert(0, 'number')
                        
                    # Update the attributes
                    attrs['_ARRAY_DIMENSIONS'] = dimensions
                    ensemble_tree['refs'][key] = json.dumps(attrs)
            
            # Check if this is a root attribute file
            if key == '.zattrs':
                # Add number to coordinates if not present
                if 'coordinates' in attrs:
                    coords = attrs['coordinates'].split()
                    if 'number' not in coords:
                        coords.append('number')
                        attrs['coordinates'] = ' '.join(coords)
                        ensemble_tree['refs'][key] = json.dumps(attrs)
        except json.JSONDecodeError:
            print(f"Error parsing attributes for {key}")
    
    # Ensure we have a 'number' variable definition in the tree
    if 'number/.zattrs' not in ensemble_tree['refs']:
        # Add number coordinate definition
        ensemble_tree['refs']['number/.zattrs'] = json.dumps({
            "_ARRAY_DIMENSIONS": [],
            "long_name": "ensemble member numerical id",
            "standard_name": "realization",
            "units": "1"
        })
        
        ensemble_tree['refs']['number/.zarray'] = json.dumps({
            "chunks": [51],
            "compressor": None, 
            "dtype": "<i8",
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": [51],
            "zarr_format": 2
        })
        
        # Add number values array for all ensemble members
        import numpy as np
        # Create array with control member (-1) and members 0-49
        numbers = np.arange(-1, 50, dtype=np.int64)
        ensemble_tree['refs']['number/0'] = numbers.tobytes().decode('latin1')
    
    # If there are zarr arrays with variables, we might need to update shape/chunks
    array_keys = [k for k in ensemble_tree['refs'] if k.endswith('/.zarray')]
    for key in array_keys:
        try:
            # Get the variable name from the path
            var_name = key.split('/')[0]
            
            # Skip coordinates
            if var_name in ['latitude', 'longitude', 'time', 'step', 'valid_time', 'number']:
                continue
                
            # Get corresponding attributes
            attr_key = key.replace('/.zarray', '/.zattrs')
            if attr_key in ensemble_tree['refs']:
                attrs = json.loads(ensemble_tree['refs'][attr_key])
                
                # If number dimension was added, update array shape/chunks
                if '_ARRAY_DIMENSIONS' in attrs and 'number' in attrs['_ARRAY_DIMENSIONS']:
                    array_def = json.loads(ensemble_tree['refs'][key])
                    
                    # Find position of 'number' in dimensions
                    number_index = attrs['_ARRAY_DIMENSIONS'].index('number')
                    
                    # Update shape - insert 51 at the correct position
                    shape = array_def['shape']
                    if number_index < len(shape):
                        # If shape array is long enough, insert at correct position
                        shape.insert(number_index, 51)
                    elif number_index == len(shape):
                        # If number dimension is at the end, append
                        shape.append(51)
                    else:
                        # This shouldn't happen, but handle it anyway
                        print(f"Warning: Number dimension index {number_index} exceeds shape length {len(shape)}")
                        
                    # Update chunks if present
                    if 'chunks' in array_def and array_def['chunks']:
                        chunks = array_def['chunks']
                        if number_index < len(chunks):
                            chunks.insert(number_index, 51)
                        elif number_index == len(chunks):
                            chunks.append(51)
                    
                    # Update the array definition
                    ensemble_tree['refs'][key] = json.dumps(array_def)
        except json.JSONDecodeError:
            print(f"Error parsing array definition for {key}")
    
    return ensemble_tree





def analyze_grib_tree_output(original_tree, ensembe_tree):
    """
    Analyze and compare outputs from different grib_tree functions
    """
    # Count references by path prefix
    def count_by_prefix(refs_dict):
        prefix_counts = {}
        for key in refs_dict:
            # Extract first part of the path
            parts = key.split('/')
            if len(parts) > 0:
                prefix = parts[0]
                if prefix not in prefix_counts:
                    prefix_counts[prefix] = 0
                prefix_counts[prefix] += 1
        
        return prefix_counts
    
    # Count references by group level
    def count_by_level(refs_dict):
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for key in refs_dict:
            # Count levels in the path
            level = key.count('/')
            if level in level_counts:
                level_counts[level] += 1
            else:
                level_counts[5] += 1  # Group anything deeper than level 4
        
        return level_counts
    
    # Look for ensemble-related entries
    def find_ensemble_refs(refs_dict):
        ensemble_refs = []
        for key in refs_dict:
            if 'number' in key or 'ensemble' in key:
                ensemble_refs.append(key)
        
        return ensemble_refs

    # Analyze original tree
    orig_prefix_counts = count_by_prefix(original_tree['refs'])
    orig_level_counts = count_by_level(original_tree['refs'])
    orig_ensemble_refs = find_ensemble_refs(original_tree['refs'])
    
    # Analyze ensemble tree
    ens_prefix_counts = count_by_prefix(ensembe_tree['refs'])
    ens_level_counts = count_by_level(ensembe_tree['refs'])
    ens_ensemble_refs = find_ensemble_refs(ensembe_tree['refs'])
    
    # Print analysis
    print("=== ORIGINAL TREE ANALYSIS ===")
    print(f"Total references: {len(original_tree['refs'])}")
    print("\nReferences by variable prefix:")
    for prefix, count in sorted(orig_prefix_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prefix}: {count}")
    
    print("\nReferences by path depth:")
    for level, count in orig_level_counts.items():
        print(f"  Level {level}: {count}")
    
    print(f"\nEnsemble-related references: {len(orig_ensemble_refs)}")
    if orig_ensemble_refs:
        print("  Examples:")
        for ref in orig_ensemble_refs[:5]:  # Show up to 5 examples
            print(f"    {ref}")
    
    print("\n=== ENSEMBLE TREE ANALYSIS ===")
    print(f"Total references: {len(ensembe_tree['refs'])}")
    print("\nReferences by variable prefix:")
    for prefix, count in sorted(ens_prefix_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prefix}: {count}")
    
    print("\nReferences by path depth:")
    for level, count in ens_level_counts.items():
        print(f"  Level {level}: {count}")
    
    print(f"\nEnsemble-related references: {len(ens_ensemble_refs)}")
    if ens_ensemble_refs:
        print("  Examples:")
        for ref in ens_ensemble_refs[:5]:  # Show up to 5 examples
            print(f"    {ref}")
    
    # Compare structure of a sample variable
    print("\n=== SAMPLE VARIABLE COMPARISON ===")
    # Find a common variable prefix
    common_prefixes = set(orig_prefix_counts.keys()) & set(ens_prefix_counts.keys())
    if common_prefixes:
        sample_var = next(iter(common_prefixes))
        print(f"Sample variable: {sample_var}")
        
        # Get all paths for this variable
        orig_var_paths = [p for p in original_tree['refs'] if p.startswith(f"{sample_var}/")]
        ens_var_paths = [p for p in ensembe_tree['refs'] if p.startswith(f"{sample_var}/")]
        
        print(f"Original tree paths: {len(orig_var_paths)}")
        for path in sorted(orig_var_paths)[:5]:  # Show up to 5 examples
            print(f"  {path}")
            
        print(f"Ensemble tree paths: {len(ens_var_paths)}")
        for path in sorted(ens_var_paths)[:5]:  # Show up to 5 examples
            print(f"  {path}")
    
    return {
        "original": {
            "total": len(original_tree['refs']),
            "by_prefix": orig_prefix_counts,
            "by_level": orig_level_counts,
            "ensemble_refs": len(orig_ensemble_refs)
        },
        "ensemble": {
            "total": len(ensembe_tree['refs']),
            "by_prefix": ens_prefix_counts,
            "by_level": ens_level_counts,
            "ensemble_refs": len(ens_ensemble_refs)
        }
    }


def improved_ensemble_grib_tree_v2(
    message_groups: Iterable[Dict],
    remote_options=None,
    debug_output=False
) -> Dict:
    """
    Improved version with better debugging and closer alignment to grib_tree structure.
    
    Parameters
    ----------
    message_groups: iterable[dict]
        a collection of zarr store like dictionaries as produced by scan_grib
    remote_options: dict
        remote options to pass to MultiZarrToZarr
    debug_output: bool
        If True, prints detailed debugging information

    Returns
    -------
    dict: A zarr store like dictionary with proper ensemble support
    """
    # Hard code the filters in the correct order for the group hierarchy
    filters = ["stepType", "typeOfLevel"]

    zarr_store = {}
    
    # Track information by path
    aggregations = defaultdict(list)
    ensemble_dimensions = defaultdict(set)
    level_dimensions = defaultdict(set)
    path_counts = defaultdict(int)

    # Process each message group and determine paths
    for msg_ind, group in enumerate(message_groups):
        if "version" not in group or group["version"] != 1:
            if debug_output:
                print(f"Skipping message {msg_ind}: Invalid version")
            continue

        # Extract ensemble member information
        ensemble_member = None
        try:
            # Check various potential locations for ensemble info
            if ".zattrs" in group["refs"]:
                root_attrs = json.loads(group["refs"][".zattrs"])
                if "ensemble_member" in root_attrs:
                    ensemble_member = root_attrs["ensemble_member"]
            
            # Look for number variable which typically holds ensemble number
            if ensemble_member is None:
                for key in group["refs"]:
                    if key == "number/0" or key.endswith("/number/0"):
                        val = group["refs"][key]
                        if isinstance(val, str):
                            try:
                                arr = np.frombuffer(val.encode('latin1'), dtype=np.int64)
                                if len(arr) == 1:
                                    ensemble_member = int(arr[0])
                                    break
                            except:
                                pass
        except Exception as e:
            if debug_output:
                print(f"Warning: Error extracting ensemble information for msg {msg_ind}: {e}")
        
        # Try to extract coordinates from the root attributes
        try:
            gattrs = json.loads(group["refs"][".zattrs"])
            coordinates = gattrs["coordinates"].split(" ")
        except Exception as e:
            if debug_output:
                print(f"Warning: Issue with attributes for message {msg_ind}: {e}")
            continue

        # Find the data variable
        vname = None
        for key in group["refs"]:
            name = key.split("/")[0]
            if name not in [".zattrs", ".zgroup"] and name not in coordinates:
                vname = name
                break

        if vname is None or vname == "unknown":
            if debug_output:
                print(f"Warning: No valid data variable found for message {msg_ind}")
            continue

        # Extract attributes for this variable
        try:
            dattrs = json.loads(group["refs"][f"{vname}/.zattrs"])
        except Exception as e:
            if debug_output:
                print(f"Warning: Issue with variable attributes for {vname} in message {msg_ind}: {e}")
            continue

        # Build path based on filter attributes
        gfilters = {}
        for key in filters:
            attr_val = dattrs.get(f"GRIB_{key}")
            if attr_val and attr_val != "unknown":
                gfilters[key] = attr_val

        # Start with variable name
        path_parts = [vname]
        
        # Add filter values to path
        for key, value in gfilters.items():
            if value:
                path_parts.append(value)
        
        # The base path excludes ensemble information
        base_path = "/".join(path_parts)
        
        # Add group to aggregations
        group_copy = copy.deepcopy(group)
        if ensemble_member is not None:
            group_copy["ensemble_member"] = ensemble_member
        
        aggregations[base_path].append(group_copy)
        path_counts[base_path] += 1
        
        # Track ensemble dimension
        if ensemble_member is not None:
            ensemble_dimensions[base_path].add(ensemble_member)
        
        # Track level information
        for key, entry in group["refs"].items():
            name = key.split("/")[0]
            if name == gfilters.get("typeOfLevel") and key.endswith("0"):
                if isinstance(entry, list):
                    entry = tuple(entry)
                level_dimensions[base_path].add(entry)
    
    # Print diagnostics for paths if debug is enabled
    if debug_output:
        print(f"Found {len(aggregations)} unique paths from {len(message_groups)} messages")
        for path, groups in sorted(aggregations.items(), key=lambda x: len(x[1]), reverse=True):
            ensemble_count = len(ensemble_dimensions.get(path, set()))
            level_count = len(level_dimensions.get(path, set()))
            print(f"  {path}: {len(groups)} groups, {ensemble_count} ensemble members, {level_count} levels")
    
    # Process each path with MultiZarrToZarr
    result_refs = {'.zgroup': json.dumps({'zarr_format': 2})}
    
    for path, groups in aggregations.items():
        # Get dimensions for this path
        catdims = ["time", "step"]  # Always concatenate time and step
        idims = ["longitude", "latitude"]  # Latitude and longitude are always identical
        
        # Handle level dimensions
        level_count = len(level_dimensions.get(path, set()))
        level_name = path.split("/")[-1] if "/" in path else None
        
        if level_count == 1:
            # Single level - treat as identical dimension
            if level_name and level_name not in idims:
                idims.append(level_name)
        elif level_count > 1:
            # Multiple levels - treat as concat dimension
            if level_name and level_name not in catdims:
                catdims.append(level_name)
        
        # Handle ensemble dimension
        ensemble_count = len(ensemble_dimensions.get(path, set()))
        if ensemble_count > 1 and "number" not in catdims:
            catdims.append("number")
            # Sort groups by ensemble number for consistent processing
            groups.sort(key=lambda g: g.get("ensemble_member", 0))
        
        if debug_output:
            print(f"Processing {path} with concat_dims={catdims}, identical_dims={idims}")
        
        try:
            # Create aggregation
            mzz = MultiZarrToZarr(
                groups,
                remote_options=remote_options,
                concat_dims=catdims,
                identical_dims=idims,
            )
            
            # Get result and store references
            group_result = mzz.translate()
            
            # Add each reference with proper path prefix
            for key, value in group_result["refs"].items():
                if key == ".zattrs" or key == ".zgroup":
                    # Root attributes for this group
                    result_refs[f"{path}/{key}"] = value
                else:
                    # Data or other references
                    result_refs[f"{path}/{key}"] = value
            
        except Exception as e:
            if debug_output:
                print(f"Error processing path {path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Convert all byte values to strings for compatibility
    result_refs = {
        key: (val.decode('utf-8') if isinstance(val, bytes) else val)
        for key, val in result_refs.items()
    }
    
    return {
        "refs": result_refs,
        "version": 1
    }


def fixed_ensemble_grib_tree(
    message_groups: Iterable[Dict],
    remote_options=None,
    debug_output=False
) -> Dict:
    """
    Build a hierarchical data model from a set of scanned grib messages with proper ensemble support
    and correct zarr path structure.
    
    This function handles ensemble dimensions correctly while maintaining the proper zarr structure
    needed by datatree.
    
    Parameters
    ----------
    message_groups: iterable[dict]
        a collection of zarr store like dictionaries as produced by scan_grib
    remote_options: dict
        remote options to pass to MultiZarrToZarr
    debug_output: bool
        If True, prints detailed debugging information

    Returns
    -------
    dict: A zarr store like dictionary with proper ensemble support
    """
    # Hard code the filters in the correct order for the group hierarchy
    filters = ["stepType", "typeOfLevel"]

    # Use a regular dictionary for storage
    zarr_store = {'.zgroup': json.dumps({'zarr_format': 2})}
    zroot = zarr.group()
    
    # Track information by path
    aggregations = defaultdict(list)
    ensemble_dimensions = defaultdict(set)
    level_dimensions = defaultdict(set)
    path_counts = defaultdict(int)

    # Process each message group and determine paths
    for msg_ind, group in enumerate(message_groups):
        if "version" not in group or group["version"] != 1:
            if debug_output:
                print(f"Skipping message {msg_ind}: Invalid version")
            continue

        # Extract ensemble member information
        ensemble_member = None
        try:
            # Check various potential locations for ensemble info
            if ".zattrs" in group["refs"]:
                root_attrs = json.loads(group["refs"][".zattrs"])
                if "ensemble_member" in root_attrs:
                    ensemble_member = root_attrs["ensemble_member"]
            
            # Look for number variable which typically holds ensemble number
            if ensemble_member is None:
                for key in group["refs"]:
                    if key == "number/0" or key.endswith("/number/0"):
                        val = group["refs"][key]
                        if isinstance(val, str):
                            try:
                                arr = np.frombuffer(val.encode('latin1'), dtype=np.int64)
                                if len(arr) == 1:
                                    ensemble_member = int(arr[0])
                                    break
                            except:
                                pass
        except Exception as e:
            if debug_output:
                print(f"Warning: Error extracting ensemble information for msg {msg_ind}: {e}")
        
        # Try to extract coordinates from the root attributes
        try:
            gattrs = json.loads(group["refs"][".zattrs"])
            coordinates = gattrs["coordinates"].split(" ")
        except Exception as e:
            if debug_output:
                print(f"Warning: Issue with attributes for message {msg_ind}: {e}")
            continue

        # Find the data variable
        vname = None
        for key in group["refs"]:
            name = key.split("/")[0]
            if name not in [".zattrs", ".zgroup"] and name not in coordinates:
                vname = name
                break

        if vname is None or vname == "unknown":
            if debug_output:
                print(f"Warning: No valid data variable found for message {msg_ind}")
            continue

        # Extract attributes for this variable
        try:
            dattrs = json.loads(group["refs"][f"{vname}/.zattrs"])
        except Exception as e:
            if debug_output:
                print(f"Warning: Issue with variable attributes for {vname} in message {msg_ind}: {e}")
            continue

        # Build path based on filter attributes
        gfilters = {}
        for key in filters:
            attr_val = dattrs.get(f"GRIB_{key}")
            if attr_val and attr_val != "unknown":
                gfilters[key] = attr_val

        # Start with variable name
        path_parts = [vname]
        
        # Add filter values to path
        for key, value in gfilters.items():
            if value:
                path_parts.append(value)
        
        # The base path excludes ensemble information
        base_path = "/".join(path_parts)
        
        # Add group to aggregations
        group_copy = copy.deepcopy(group)
        if ensemble_member is not None:
            group_copy["ensemble_member"] = ensemble_member
        
        aggregations[base_path].append(group_copy)
        path_counts[base_path] += 1
        
        # Track ensemble dimension
        if ensemble_member is not None:
            ensemble_dimensions[base_path].add(ensemble_member)
        
        # Track level information
        for key, entry in group["refs"].items():
            name = key.split("/")[0]
            if name == gfilters.get("typeOfLevel") and key.endswith("0"):
                if isinstance(entry, list):
                    entry = tuple(entry)
                level_dimensions[base_path].add(entry)
    
    # Print diagnostics for paths if debug is enabled
    if debug_output:
        print(f"Found {len(aggregations)} unique paths from {len(message_groups)} messages")
        for path, groups in sorted(aggregations.items(), key=lambda x: len(x[1]), reverse=True):
            ensemble_count = len(ensemble_dimensions.get(path, set()))
            level_count = len(level_dimensions.get(path, set()))
            print(f"  {path}: {len(groups)} groups, {ensemble_count} ensemble members, {level_count} levels")
    
    # Process each path with MultiZarrToZarr and ensure proper hierarchical structure
    for path, groups in aggregations.items():
        # Build groups for each level in the hierarchy
        path_parts = path.split("/")
        current_path = ""
        for i, part in enumerate(path_parts):
            prev_path = current_path
            
            if current_path:
                current_path = f"{current_path}/{part}"
            else:
                current_path = part
            
            # Add .zgroup for this level if not already present
            if f"{current_path}/.zgroup" not in zarr_store:
                zarr_store[f"{current_path}/.zgroup"] = json.dumps({'zarr_format': 2})
            
            # Add .zattrs for this level
            if f"{current_path}/.zattrs" not in zarr_store:
                # Add appropriate attributes based on the level
                attrs = {}
                
                # Add filter-specific attributes
                if i == 1 and len(path_parts) > 1:  # stepType level
                    attrs["stepType"] = path_parts[i]
                if i == 2 and len(path_parts) > 2:  # typeOfLevel level
                    attrs["typeOfLevel"] = path_parts[i]
                
                zarr_store[f"{current_path}/.zattrs"] = json.dumps(attrs)
        
        # Get dimensions for this path
        catdims = ["time", "step"]  # Always concatenate time and step
        idims = ["longitude", "latitude"]  # Latitude and longitude are always identical
        
        # Handle level dimensions
        level_count = len(level_dimensions.get(path, set()))
        level_name = path_parts[-1] if len(path_parts) > 0 else None
        
        if level_count == 1:
            # Single level - treat as identical dimension
            if level_name and level_name not in idims:
                idims.append(level_name)
        elif level_count > 1:
            # Multiple levels - treat as concat dimension
            if level_name and level_name not in catdims:
                catdims.append(level_name)
        
        # Handle ensemble dimension
        ensemble_count = len(ensemble_dimensions.get(path, set()))
        if ensemble_count > 1 and "number" not in catdims:
            catdims.append("number")
            # Sort groups by ensemble number for consistent processing
            groups.sort(key=lambda g: g.get("ensemble_member", 0))
        
        if debug_output:
            print(f"Processing {path} with concat_dims={catdims}, identical_dims={idims}")
        
        try:
            # Create aggregation
            mzz = MultiZarrToZarr(
                groups,
                remote_options=remote_options,
                concat_dims=catdims,
                identical_dims=idims,
            )
            
            # Get result and store references
            group_result = mzz.translate()
            
            # Add each reference with proper path prefix
            for key, value in group_result["refs"].items():
                if key == ".zattrs" or key == ".zgroup":
                    # Don't overwrite existing group metadata
                    if f"{path}/{key}" not in zarr_store:
                        zarr_store[f"{path}/{key}"] = value
                else:
                    # Data or other references
                    zarr_store[f"{path}/{key}"] = value
            
        except Exception as e:
            if debug_output:
                print(f"Error processing path {path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Convert all byte values to strings for compatibility
    zarr_store = {
        key: (val.decode('utf-8') if isinstance(val, bytes) else val)
        for key, val in zarr_store.items()
    }
    
    return {
        "refs": zarr_store,
        "version": 1
    }
date_str='20240229'
ecmwf_files=[f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-0h-enfo-ef.grib2",
            f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-3h-enfo-ef.grib2"]


# Process all files
all_groups = []

for eurl in ecmwf_files[1:2]:
    try:
        groups, idx_mapping = ecmwf_filter_scan_grib(eurl)
        all_groups.extend(groups)
    except Exception as e:
        print(f"Error processing {eurl}: {e}")
        import traceback
        traceback.print_exc()

if not all_groups:
    raise ValueError("No valid groups were found")

# Build tree using the original grib_tree function
try:
    # Use kerchunk's original grib_tree function
    original_tree = grib_tree(all_groups)
    print(f"Built original tree with {len(original_tree['refs'])} references")
    # Create a modified version with ensemble information
    modified_tree = organize_ensemble_tree(original_tree)
    print(f"Created ensemble tree with {len(modified_tree['refs'])} references")
except Exception as e:
    print(f"Error building trees: {e}")
    import traceback
    traceback.print_exc()
    raise


# First analyze both outputs
original_tree = grib_tree(all_groups)
ensemble_tree = improved_ensemble_grib_tree_v2(all_groups, debug_output=True)

# Compare the structures
analysis = analyze_grib_tree_output(original_tree, ensemble_tree)

# Check if we can open with datatree
gfs_dt = datatree.open_datatree(
    fsspec.filesystem("reference", fo=ensemble_tree).get_mapper(""), 
    engine="zarr", 
    consolidated=False
)

# The key test: can we access ensemble members?
print(gfs_dt.keys())  # Check for variables
var_node = gfs_dt['variable_name']  # Pick one variable
print(var_node.dims)  # Should include 'number' dimension if ensemble data is present

