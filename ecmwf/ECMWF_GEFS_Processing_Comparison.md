# ECMWF and GEFS Processing Comparison: Three-Stage GRIB Index Kerchunk Method

## Executive Summary

This document compares the GEFS (Global Ensemble Forecast System) and ECMWF (European Centre for Medium-Range Weather Forecasts) processing pipelines, focusing on the three distinct operations in the GRIB Index Kerchunk method and identifying missing steps in the ECMWF implementation.

## Three Distinct Operations in GEFS Processing

### Operation 1: Scan GRIB to Parquet with Filter (scan_grib → filter_build_grib_tree)

**Location**: `run_day_gefs_ensemble_full.py:176-179` and `gefs_util.py:445-463`

**Purpose**: Initial GRIB file scanning and filtering to build a hierarchical tree structure

**Process Flow**:
```python
# In process_single_ensemble_member (line 176-179)
_, deflated_gefs_grib_tree_store = filter_build_grib_tree(gefs_files, forecast_dict)

# In gefs_util.py filter_build_grib_tree (lines 445-463)
sg_groups = [group for gurl in gefs_files for group in filter_gefs_scan_grib(gurl, tofilter_gefs_var_dict)]
gefs_grib_tree_store = grib_tree(sg_groups)
deflated_gefs_grib_tree_store = copy.deepcopy(gefs_grib_tree_store)
strip_datavar_chunks(deflated_gefs_grib_tree_store)
```

**Key Features**:
- Scans GRIB files using kerchunk's `scan_grib`
- Filters variables based on `forecast_dict`
- Builds GRIB tree structure
- Strips unnecessary data chunks for efficiency
- Creates deflated store with reduced memory footprint

### Operation 2: Create Mapped Index from GCS Pre-built Parquet (cs_create_mapped_index)

**Location**: `run_day_gefs_ensemble_full.py:188-192` and `gefs_util.py:634-664`

**Purpose**: Use pre-processed GCS parquet files as mapping templates for current date

**Process Flow**:
```python
# In process_single_ensemble_member (lines 188-192)
gefs_kind = cs_create_mapped_index(
    axes, gcs_bucket_name, target_date_str, member,
    gcp_service_account_json=gcp_service_account_json,
    reference_date_str=reference_date_str
)

# In gefs_util.py cs_create_mapped_index (lines 491-564)
# For each forecast hour:
# 1. Read fresh GRIB index (.idx) from target date
idxdf = parse_grib_idx(basename=fname, storage_options=storage_options)

# 2. Read pre-built mapping from GCS (reference date)
deduped_mapping = pd.read_parquet(gcs_mapping_path, filesystem=gcs_fs)

# 3. Map fresh index to template structure
mapped_index = map_from_index(datestr, deduped_mapping, idxdf_filtered)
```

**Key Features**:
- Leverages existing parquet mappings from GCS
- Combines fresh GRIB index with template structure
- Processes asynchronously in batches for efficiency
- Falls back to direct processing if GCS unavailable
- 85x faster than full scan_grib approach

### Operation 3: Prepare Zarr Store and Process Unique Groups

**Location**: `run_day_gefs_ensemble_full.py:194-196` and `gefs_util.py:263-332`

**Purpose**: Create final zarr store with time dimensions and coordinate variables

**Process Flow**:
```python
# In process_single_ensemble_member (lines 194-196)
zstore, chunk_index = prepare_zarr_store(deflated_gefs_grib_tree_store, gefs_kind)
updated_zstore = process_unique_groups(zstore, chunk_index, time_dims, time_coords,
                                     times, valid_times, steps)

# In gefs_util.py prepare_zarr_store (lines 263-278)
zarr_ref_store = deflated_gefs_grib_tree_store
chunk_index = gefs_kind
zstore = copy.deepcopy(zarr_ref_store["refs"])

# In process_unique_groups (lines 281-332)
unique_groups = chunk_index.set_index(["varname", "stepType", "typeOfLevel"]).index.unique()
# For each group:
store_coord_var(f"{base_path}/time", zstore, ...)
store_coord_var(f"{base_path}/valid_time", zstore, ...)
store_coord_var(f"{base_path}/step", zstore, ...)
store_data_var(f"{base_path}/{varname}", zstore, ...)
```

**Key Features**:
- Creates zarr reference store structure
- Processes unique variable groups
- Stores coordinate variables (time, valid_time, step)
- Stores data variables with proper dimensions
- Handles multiple forecast levels

## ECMWF Current Implementation Status

### What's Implemented

1. **Basic GRIB Scanning** (`run_day_ecmwf_ensemble_full.py:100-186`)
   - Only scans 0h and 3h GRIB files
   - Uses `ecmwf_filter_scan_grib` and `fixed_ensemble_grib_tree`
   - Missing complete time coverage (only 2 of 85 steps)

2. **GCS Parquet Loading** (`ecmwf_util.py:574-651`)
   - Function `load_ecmwf_parquet_from_gcs` exists
   - Can load pre-processed parquets from GCS
   - Supports authentication and member-specific paths

3. **Member Processing with GCS** (`ecmwf_util.py:694-756`)
   - Function `process_ecmwf_member_with_gcs_parquets` implemented
   - Can use reference date parquets as templates

### What's Missing or Incomplete

## Missing Steps for ECMWF Implementation

### 1. Complete Time Step Processing (Critical Gap)

**Issue**: ECMWF only processes 0h and 3h, missing 83 time steps

**GEFS Approach**:
```python
# GEFS processes all needed hours
for hour in [0, 3, 6, 9, ..., 240]:  # All required hours
    fname = f"...f{hour:03d}"
    # Process each hour
```

**Required ECMWF Fix**:
```python
# Need to process all ECMWF_FORECAST_HOURS
ECMWF_FORECAST_HOURS_3H = list(range(0, 145, 3))  # 0-144h at 3h intervals
ECMWF_FORECAST_HOURS_6H = list(range(150, 361, 6))  # 150-360h at 6h intervals
ECMWF_FORECAST_HOURS = ECMWF_FORECAST_HOURS_3H + ECMWF_FORECAST_HOURS_6H  # 85 steps total

# Process ALL hours, not just [0, 3]
for hour in ECMWF_FORECAST_HOURS:
    url = f"...{hour}h-enfo-ef.grib2"
    # Process with index-based method for efficiency
```

### 2. Index-Based Processing (Efficiency Gap)

**Issue**: ECMWF uses expensive scan_grib for all processing

**GEFS Approach** (gefs_util.py:491-564):
```python
# Uses GRIB index files for efficiency
async def process_single_gefs_file():
    # 1. Parse index file (fast)
    idxdf = parse_grib_idx(basename=fname)
    # 2. Read mapping template from GCS
    deduped_mapping = pd.read_parquet(gcs_mapping_path)
    # 3. Combine for complete reference
    mapped_index = map_from_index(datestr, deduped_mapping, idxdf)
```

**Required ECMWF Implementation**:
```python
def process_ecmwf_with_index(date_str, hour, member):
    # 1. Parse ECMWF index file
    idx_url = f"s3://ecmwf-forecasts/{date_str}/00z/ifs/0p25/enfo/{date_str}000000-{hour}h-enfo-ef.grib2.idx"
    idx_df = parse_grib_idx(idx_url)

    # 2. Filter for specific member
    member_idx = idx_df[idx_df['number'] == member]

    # 3. Create references without full GRIB scan
    refs = create_references_from_index(member_idx)

    return refs
```

### 3. Complete Zarr Store Building (Structure Gap)

**Issue**: ECMWF's zarr store creation is incomplete, missing proper time dimension handling

**GEFS Approach** (gefs_util.py:281-332):
```python
def process_unique_groups(zstore, chunk_index, time_dims, time_coords, ...):
    # Properly handles all time steps and dimensions
    for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
        # Store all coordinate variables
        store_coord_var(f"{base_path}/time", ...)
        store_coord_var(f"{base_path}/valid_time", ...)
        store_coord_var(f"{base_path}/step", ...)
        # Store data with correct dimensions
        store_data_var(f"{base_path}/{varname}", ...)
```

**Required ECMWF Fix**:
```python
def process_ecmwf_complete_timesteps(zstore, all_hours_index, ...):
    # Must handle 85 time steps, not just 2
    time_dims["valid_time"] = 85  # Not 2!

    # Process each unique variable group
    for var_group in unique_groups:
        # Ensure all 85 steps are included
        for hour in ECMWF_FORECAST_HOURS:
            # Add references for each hour
            add_hour_references(zstore, hour, var_group)
```

### 4. Batch Processing for All Hours (Performance Gap)

**Issue**: ECMWF doesn't efficiently process multiple time steps

**GEFS Approach** (gefs_util.py:562-632):
```python
async def process_gefs_files_in_batches(axes, gcs_bucket_name, ...):
    # Processes files in parallel batches
    sem = asyncio.Semaphore(max_concurrent)
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Process in batches for efficiency
        for batch_start in range(0, len(dtaxes), batch_size):
            tasks = [process_single_gefs_file(...) for idx in batch_indices]
            batch_results = await asyncio.gather(*tasks)
```

**Required ECMWF Implementation**:
```python
async def process_ecmwf_all_hours_batch(date_str, member, forecast_hours):
    # Process all 85 hours in parallel batches
    async with asyncio.Semaphore(10) as sem:
        tasks = []
        for hour in forecast_hours:
            task = process_ecmwf_hour_with_index(date_str, hour, member, sem)
            tasks.append(task)

        # Gather all results
        all_refs = await asyncio.gather(*tasks)

    # Combine into single zarr store
    return combine_hour_references(all_refs)
```

### 5. Member Extraction Logic (Data Isolation Gap)

**Issue**: ECMWF's member extraction from combined store is primitive

**GEFS Approach**: Each member processed individually with proper filtering

**Required ECMWF Fix** (based on ecmwf_util.py:326-353):
```python
def extract_ecmwf_member_properly(combined_store, member_number):
    # ECMWF stores all members in single GRIB
    # Need proper filtering by 'number' attribute

    member_refs = {}
    for key, value in combined_store['refs'].items():
        # Parse zarr path to check member
        if is_member_reference(key, member_number):
            member_refs[key] = value

    # Ensure complete time coverage
    validate_time_steps(member_refs, expected=85)

    return member_refs
```

## Recommended Implementation Path for ECMWF

### Phase 1: Fix Time Coverage (Highest Priority)
```python
# 1. Update process_ecmwf_files_for_scan_grib to handle all hours
critical_hours = ECMWF_FORECAST_HOURS  # All 85 hours, not just [0, 3]

# 2. Use index-based method for hours beyond 3h
for hour in ECMWF_FORECAST_HOURS[2:]:  # Skip 0h, 3h (already scanned)
    refs = process_with_index_method(hour)
    combine_with_scan_grib_base(refs)
```

### Phase 2: Implement Index-Based Processing
```python
# Copy GEFS's async index processing approach
# Adapt for ECMWF's URL structure and member encoding
from gefs_util import process_gefs_files_in_batches
# Modify for ECMWF:
- Change URL patterns
- Adjust member filtering logic
- Handle ECMWF's variable intervals (3h then 6h)
```

### Phase 3: Complete Zarr Store Building
```python
# Ensure proper dimensions
time_dims = {"valid_time": 85}  # Not 2!

# Process all unique groups with complete time coverage
for group in unique_groups:
    process_all_timesteps(group, ECMWF_FORECAST_HOURS)
```

### Phase 4: Optimize with GCS Templates
```python
# Use existing GCS parquets as templates
# Only update time references for new dates
if reference_parquets_exist:
    use_template_structure()
else:
    build_from_scratch()
```

## Performance Comparison

| Aspect | GEFS Implementation | ECMWF Current | ECMWF Required |
|--------|-------------------|---------------|----------------|
| Time Steps Processed | 81 (10 days) | 2 (0h, 3h) | 85 (15 days) |
| Processing Method | Index + GCS templates | scan_grib only | Index + GCS needed |
| Efficiency | 3-5 minutes | 4-6 hours (if complete) | 5-10 minutes possible |
| Parallel Processing | Yes (async batches) | No | Yes (needed) |
| GCS Integration | Full (read/write) | Partial (read only) | Full (needed) |
| Member Handling | Individual processing | Combined extraction | Needs improvement |

## Critical Files to Modify

1. **ecmwf_util.py**:
   - Add `process_ecmwf_with_index()` function
   - Update `calculate_ecmwf_time_dimensions()` for 85 steps
   - Add async batch processing functions

2. **run_day_ecmwf_ensemble_full.py**:
   - Update `process_ecmwf_files_for_scan_grib()` to handle all hours
   - Implement index-based processing option
   - Fix member extraction logic

3. **New File Needed**: `ecmwf_index_processor.py`
   - Port GEFS index processing logic
   - Adapt for ECMWF URL structure
   - Handle ECMWF's member encoding

## Validation Checklist

- [ ] All 85 time steps present in parquet files
- [ ] All 51 ensemble members properly extracted
- [ ] Variables have correct dimensions: (time=1, step=85, lat=721, lon=1440)
- [ ] Index-based processing reduces time from hours to minutes
- [ ] GCS parquet templates can be reused for different dates
- [ ] Zarr store can be opened with xarray.open_datatree()
- [ ] Regional subset extraction works correctly

## Conclusion

The ECMWF implementation has the framework in place but critically lacks:
1. **Complete time coverage** (only 2 of 85 steps)
2. **Index-based processing** for efficiency
3. **Proper zarr store construction** with all dimensions

By adopting GEFS's three-operation approach (scan_grib → mapped_index → zarr_store) and implementing the missing components, ECMWF processing can achieve similar efficiency and completeness. The priority should be fixing time coverage, as this is the most critical gap affecting data completeness.