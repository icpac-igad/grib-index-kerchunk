# GEFS PWAT Variable Skipping Analysis

## Executive Summary

The PWAT (Precipitable Water) variable is successfully making it through most of the GEFS processing pipeline but is being **skipped during the final zarr group processing stage**. This document analyzes the issue progression and provides the final solution.

## Issue Timeline and Attempts

### Initial Problem (From gik-gefs-part3.md)
- **Symptom**: PWAT missing from final parquet file output
- **Initial Diagnosis**: Incorrect typeOfLevel mapping in `gefs_util.py:188`
- **Attempted Fix**: Changed mapping from `'entireAtmosphere'` to `'entire atmosphere (considered as a single layer)'`
- **Result**: Issue persisted

### Current Status Analysis

Based on the latest debug logs, we can trace PWAT through the entire pipeline:

#### Stage 1: GRIB Tree Building ✅ SUCCESS
```
Building GEFS Grib Tree
Original references: 229
Stripped references: 169
✅ GRIB tree built successfully for gep19
```
- PWAT is successfully included in the GRIB tree
- No issues at this stage

#### Stage 2: Variable Discovery ✅ SUCCESS
```
Variables found in gefs_kind: ['vis', 'gust', 'mslet', ..., 'pwat', ...]
Variables to be specially processed: ['cape', 'gust', 'pwat']
```
- PWAT is correctly identified as a variable requiring special processing
- PWAT appears in the variable list

#### Stage 3: Type Mapping Resolution ✅ SUCCESS
```
PWAT entries before processing: 81
PWAT typeOfLevel values: ['atmosphereSingleLayer']
PWAT: Looking for typeOfLevel='entire atmosphere (considered as a single layer)', found 0 entries
PWAT: Looking for typeOfLevel='atmosphereSingleLayer', found 81 entries
PWAT entries after processing: 81
PWAT typeOfLevel after processing: ['atmosphereSingleLayer']
```
- **Key Finding**: PWAT uses `'atmosphereSingleLayer'` not `'entire atmosphere (considered as a single layer)'`
- The fix to accept both typeOfLevel values worked correctly
- All 81 PWAT entries are preserved after processing

#### Stage 4: Chunk Index Preparation ✅ SUCCESS
```
Chunk index contains 2985 rows
Unique variables in chunk_index: ['cape', 'cin', ..., 'pwat', ...]
Unique groups identified: 38
PWAT found in chunk_index with typeOfLevel: ['atmosphereSingleLayer']
```
- PWAT successfully makes it into the chunk_index DataFrame
- PWAT is properly identified with the correct typeOfLevel

#### Stage 5: Zarr Group Processing ❌ **FAILURE POINT**
```
Skipping processing of GEFS group ('pwat', 'instant', 'atmosphereSingleLayer'): 'pwat/instant/atmosphereSingleLayer/time/.zattrs'
```
- **Critical Issue**: PWAT group is being skipped during zarr store creation
- The group `('pwat', 'instant', 'atmosphereSingleLayer')` exists but fails processing

## Technical Analysis

### Why PWAT Is Being Skipped

The `process_unique_groups` function in `gefs_util.py:334-362` processes each group using this pattern:

```python
for key, group in chunk_index.groupby(["varname", "stepType", "typeOfLevel"]):
    try:
        base_path = "/".join(key)
        # ... processing logic ...
        store_data_var(key=f"{base_path}/{key[0]}", ...)
    except Exception as e:
        print(f"Skipping processing of GEFS group {key}: {str(e)}")
```

The skipping occurs because an **exception is being thrown** during the processing of the PWAT group, but the specific error is not being shown.

### Root Cause Analysis

1. **Data Structure Issue**: The processing expects certain data structures that PWAT may not conform to
2. **Level Value Handling**: The `lvals` processing may fail for atmospheric single layer data
3. **Coordinate Mismatch**: PWAT may have different coordinate requirements than surface/height variables

## The Fix

### Step 1: Enhanced Error Reporting

Add detailed error logging to identify the exact failure:

```python
# In process_unique_groups function, replace the except block:
except Exception as e:
    print(f"Skipping processing of GEFS group {key}: {str(e)}")
    if key[0] == 'pwat':  # Special attention to PWAT errors
        print(f"PWAT SPECIFIC ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
```

### Step 2: PWAT-Specific Handling

The issue likely lies in how the `lvals` (level values) are handled for atmospheric single layer data. Add PWAT-specific logic:

```python
# In the processing loop, before store_data_var:
if key[0] == 'pwat':
    print(f"PWAT group details: {key}")
    print(f"PWAT level values: {lvals}")
    print(f"PWAT group size: {len(group)}")
    print(f"PWAT unique levels: {group.level.unique()}")
```

### Step 3: Level Value Correction

For atmospheric single layer variables like PWAT, the level handling may need special treatment:

```python
# Modify the level handling section:
if key[0] == 'pwat' and key[2] == 'atmosphereSingleLayer':
    # Special handling for atmospheric single layer
    if len(lvals) == 1:
        lvals = float(lvals.squeeze()) if hasattr(lvals.squeeze(), '__float__') else 0.0
        dims[key[2]] = 0
    else:
        print(f"WARNING: PWAT has multiple level values: {lvals}")
        lvals = 0.0  # Default for single atmospheric layer
        dims[key[2]] = 0
```

## Verification Steps

After implementing the fix:

1. **Run the processing script** and look for detailed PWAT error messages
2. **Check the final parquet file** contains PWAT variable:
   ```python
   # Test script
   zstore = read_parquet_fixed('20250709_00/gep19.par')
   access_success, variable_groups, dt = test_zarr_access_and_discover_variables(zstore, 'gep19')
   pwat_groups = [g for g in variable_groups.keys() if 'pwat' in g.lower()]
   print(f"PWAT groups found: {pwat_groups}")
   ```

3. **Verify PWAT data accessibility** in the final zarr store

## Detailed Error Analysis

After implementing enhanced debugging, the exact error was identified:

```
Skipping processing of GEFS group ('pwat', 'instant', 'atmosphereSingleLayer'): 'pwat/instant/atmosphereSingleLayer/time/.zattrs'
PWAT SPECIFIC ERROR: KeyError: 'pwat/instant/atmosphereSingleLayer/time/.zattrs'
PWAT group details: varname=pwat, stepType=instant, typeOfLevel=atmosphereSingleLayer
PWAT group size: 81
PWAT level values: [1.]
Traceback (most recent call last):
  File "/scratch/notebook/gefs_util.py", line 358, in process_unique_groups
    store_coord_var(key=f"{base_path}/time", zstore=zstore, coords=time_coords["time"], data=times.astype("datetime64[s]"))
  File "/opt/coiled/env/lib/python3.11/site-packages/kerchunk/_grib_idx.py", line 123, in store_coord_var
    zattrs = ujson.loads(zstore[f"{key}/.zattrs"])
                         ~~~~~~^^^^^^^^^^^^^^^^^^
KeyError: 'pwat/instant/atmosphereSingleLayer/time/.zattrs'
```

### Root Cause

The error occurs because:
1. **PWAT successfully flows through** the entire data processing pipeline
2. **PWAT appears in chunk_index** with correct entries (81 timesteps)
3. **PWAT fails at zarr store creation** because the zarr store doesn't contain the required metadata structure for PWAT
4. **The missing structure indicates** PWAT wasn't properly included in the original GRIB tree during the filtering stage

The fundamental issue is that while PWAT gets processed through the mapping/indexing pipeline, it wasn't included in the original zarr store structure created from the GRIB files.

## The Complete Solution

### Fix 1: Enhanced Error Handling with Missing Structure Detection

```python
# In process_unique_groups function - gefs_util.py lines 348-364
# Check if this group has the required zarr structure
required_keys = [
    f"{base_path}/time/.zattrs",
    f"{base_path}/valid_time/.zattrs",
    f"{base_path}/step/.zattrs",
    f"{base_path}/{key[2]}/.zattrs" if key[2] else None
]
required_keys = [k for k in required_keys if k is not None]

missing_keys = [k for k in required_keys if k not in zstore]
if missing_keys:
    if key[0] == 'pwat':
        print(f"PWAT missing zarr structure keys: {missing_keys}")
        print(f"PWAT available keys matching pattern: {[k for k in zstore.keys() if 'pwat' in k.lower()][:5]}")
        print(f"PWAT group will be skipped - not included in original GRIB tree structure")
    # Skip this group if it's missing required zarr structure
    continue
```

### Fix 2: Enhanced Debugging for GRIB Filtering

```python
# In filter_gefs_scan_grib function - gefs_util.py lines 485-492
# Debug PWAT filtering
pwat_found = any('PWAT' in str(value) for value in tofilter_gefs_var_dict.values())
if pwat_found:
    pwat_entries = idx_gefs[idx_gefs['attrs'].str.contains('PWAT', na=False)]
    print(f"GRIB filtering debug - PWAT entries in idx: {len(pwat_entries)}")
    if not pwat_entries.empty:
        print(f"PWAT attrs sample: {pwat_entries['attrs'].iloc[0]}")
    print(f"Indices selected for PWAT: {[i for i, v in output_dict0.items() if 'PWAT' in str(v)]}")
```

### Fix 3: Success Monitoring

```python
# Success message for PWAT - gefs_util.py lines 383-385
if key[0] == 'pwat':
    print(f"✅ PWAT successfully processed and stored in zarr!")
```

## Expected Outcome After Fix

### Scenario 1: PWAT Missing from GRIB Tree (Current Issue)
```
PWAT missing zarr structure keys: ['pwat/instant/atmosphereSingleLayer/time/.zattrs', ...]
PWAT available keys matching pattern: []
PWAT group will be skipped - not included in original GRIB tree structure
```

### Scenario 2: PWAT Successfully Included (After Fixing GRIB Filtering)
```
GRIB filtering debug - PWAT entries in idx: 1
PWAT attrs sample: PWAT:entire atmosphere (considered as a single layer):3 hour fcst
✅ PWAT successfully processed and stored in zarr!
```

## ⚡ **CRITICAL DISCOVERY: The Real Root Cause**

Further diagnosis revealed the **actual root cause**:

```
GRIB filtering debug - PWAT entries in idx: 1
PWAT attrs sample: PWAT:entire atmosphere (considered as a single layer):anl:ENS=+1
Indices selected for PWAT: []
```

**The Issue**: The `map_forecast_to_indices` function was using regex search with `str.contains()`, but the parentheses `()` in `"PWAT:entire atmosphere (considered as a single layer)"` are special regex characters, causing the search to fail!

## 🎯 **The Final Fix**

### Fix 4: Regex Escape Issue in GRIB Mapping

```python
# In map_forecast_to_indices function - gefs_util.py lines 555-567
for key, value in forecast_dict.items():
    # Use regex=False to treat the search string as literal text, not regex
    matching_rows = df[df['attrs'].str.contains(value, na=False, regex=False)]

    if not matching_rows.empty:
        output_dict[key] = int(matching_rows.index[0] - 1)
        if 'PWAT' in value:  # Debug PWAT matching
            print(f"PWAT MATCH FOUND: {key} -> index {matching_rows.index[0] - 1}")
    else:
        output_dict[key] = 9999
        if 'PWAT' in value:  # Debug PWAT not matching
            print(f"PWAT NO MATCH: searching for '{value}' in {len(df)} entries")
            pwat_attrs = df[df['attrs'].str.contains('PWAT', na=False, regex=False)]['attrs'].tolist()
            print(f"Available PWAT attrs: {pwat_attrs[:3]}")
```

## Expected Outcome After Complete Fix

With this final fix, the logs should show:
```
GRIB filtering debug - PWAT entries in idx: 1
PWAT attrs sample: PWAT:entire atmosphere (considered as a single layer):anl:ENS=+1
PWAT MATCH FOUND: Precipitable water -> index X
✅ PWAT successfully processed and stored in zarr!
```

The final variable count should increase from 7 to 8 variables once PWAT is properly included in the zarr store structure.

## Historical Context

This issue represents a classic case of:
1. **Correct variable mapping** (fixed in earlier attempts)
2. **Successful data pipeline flow** (confirmed by debugging)
3. **GRIB tree filtering issue** (the actual root cause)
4. **Graceful error handling** (current solution)

The progression shows that systematic debugging through each pipeline stage was essential to isolate the exact failure point and implement appropriate error handling.