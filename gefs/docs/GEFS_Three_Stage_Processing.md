# GEFS Three-Stage Processing: Grib-Index-Kerchunk Method

## Overview

This document details the three distinct processing stages (routines) used in GEFS ensemble data processing using the grib-index-kerchunk method. These stages transform raw GRIB files into efficient parquet-based zarr stores.

**CRITICAL**: Stage 2 requires a one-time expensive preprocessing step (Stage 0) that creates reusable GCS parquet mapping templates.

---

## The Three Processing Stages

### Stage 0: ONE-TIME GCS Parquet Mapping Creation (PREPROCESSING)

**⚠️ IMPORTANT**: This is the **most expensive** step but only runs ONCE per ensemble member. The output parquet files can be reused for ANY date!

**Purpose**: Create parquet mapping templates in GCS that describe GRIB data structure

**Location**: `gefs_index_preprocessing_fixed.py` and `run_gefs_preprocessing.py`

**When to Run**:
- **ONCE** per ensemble member to create reusable templates
- Only needs to be re-run if GRIB structure changes (rare)

**Process**:
```bash
# Run once for all members (gep01-gep30)
python run_gefs_preprocessing.py

# OR run for individual member
python gefs_index_preprocessing_fixed.py \
  --date 20241112 \
  --run 00 \
  --member gep01 \
  --bucket gik-gefs-aws-tf
```

**What it does**:
```python
# For each forecast hour (0, 3, 6, ..., 240):
# 1. Parse GRIB index file
idxdf = parse_grib_idx(basename=fname, storage_options=storage_options)

# 2. Build complete idx-grib mapping (EXPENSIVE!)
grib_mapping = build_idx_grib_mapping(
    basename=fname,
    mapper=mapper,
    storage_options=storage_options
)

# 3. Deduplicate and save to GCS
deduped_mapping = grib_mapping.loc[~grib_mapping["attrs"].duplicated(keep="first"), :]
gcs_path = f"gs://{bucket}/gefs/{member}/gefs-time-{date}-{member}-rt{hour:03d}.parquet"
deduped_mapping.to_parquet(gcs_path, filesystem=gcs_fs)
```

**Output Structure in GCS**:
```
gs://gik-gefs-aws-tf/gefs/
├── gep01/
│   ├── gefs-time-20241112-gep01-rt000.parquet
│   ├── gefs-time-20241112-gep01-rt003.parquet
│   ├── gefs-time-20241112-gep01-rt006.parquet
│   └── ... (81 files, one per forecast hour)
├── gep02/
│   ├── gefs-time-20241112-gep02-rt000.parquet
│   └── ...
└── ... (gep03-gep30)
```

**Processing Time**:
- Per forecast hour: 15-30 seconds
- Per member (81 hours): 20-40 minutes
- All 30 members: 10-20 hours (one-time cost!)

**Why Expensive?**:
- Must scan actual GRIB files to build complete byte offset mappings
- Creates detailed metadata for every variable in every GRIB file
- But: These mappings are **reusable across any date with same structure!**

---

### Stage 1: Scan GRIB to Ensemble-wise Parquet Files

**Purpose**: Scan GRIB files and create deflated GRIB tree store with variable references

**Location**: `gefs_util.py:520-538` (filter_build_grib_tree)

**Used in**: `run_day_gefs_ensemble_full.py:176-179`

**Process**:
```python
# Step 1.1: Scan GRIB files with variable filtering
sg_groups = [
    group
    for gurl in gefs_files
    for group in filter_gefs_scan_grib(gurl, forecast_dict)
]

# Step 1.2: Build hierarchical GRIB tree
gefs_grib_tree_store = grib_tree(sg_groups)

# Step 1.3: Strip data chunks to reduce memory
deflated_gefs_grib_tree_store = copy.deepcopy(gefs_grib_tree_store)
strip_datavar_chunks(deflated_gefs_grib_tree_store)
```

**Input**:
- GRIB2 files from S3: `s3://noaa-gefs-pds/gefs.{date}/{run}/atmos/pgrb2sp25/{member}.t{run}z.pgrb2s.0p25.f{hour:03d}`
- Variable filter dictionary (forecast_dict)

**Output**:
- Deflated GRIB tree store (in-memory dictionary)
- Contains zarr-like references to filtered variables

**Processing Time**:
- Per GRIB file: 30-60 seconds
- Typically processes just 0h and 3h files (~2 minutes total)

**Key Features**:
- Uses kerchunk's `scan_grib` to extract GRIB metadata
- Filters only requested variables (reduces data size)
- Creates hierarchical zarr-like structure
- Strips unnecessary data chunks to reduce memory

---

### Stage 2: IDX Files + GCS Parquet Templates → Mapped Index

**⚡ KEY STAGE**: This is where the **85x speedup** happens!

**Purpose**: Combine fresh GRIB index files with pre-built GCS parquet templates to create complete mapped index

**Location**: `gefs_util.py:573-643` (process_single_gefs_file) and `gefs_util.py:646-689` (process_gefs_files_in_batches)

**Used in**: `run_day_gefs_ensemble_full.py:188-192` (cs_create_mapped_index)

**⚠️ CRITICAL CONCEPT**: Why Fresh IDX Files Are Essential

GRIB files store binary data with **variable byte offsets** that change between dates. The `.idx` index file contains the exact byte positions (offset, length) where each variable is located in the GRIB file.

**What Changes Between Dates**:
- ✅ Binary offsets (where data starts in the file)
- ✅ Binary lengths (size of each variable's data)
- ✅ URI (file path with date)

**What Stays the Same**:
- ✅ Variable structure (varname, stepType, typeOfLevel)
- ✅ Dimensions (latitude, longitude, levels)
- ✅ Attribute mappings

**Why We Need BOTH**:
1. **Fresh IDX file (target date)**: Provides correct binary positions for TODAY's data
2. **GCS template (reference date)**: Provides variable structure and metadata

**❌ CRITICAL ERROR TO AVOID**:
Using old binary positions (offset/length) from a reference date's IDX file will cause you to read **WRONG DATA** from the target date's GRIB file! The offsets are different, so you'll get garbage or mismatched variables.

**Process**:
```python
async def process_single_gefs_file():
    # Step 2.1: Read fresh GRIB index file for TARGET date
    # ⚠️ CRITICAL: Index file contains byte positions for THIS specific date
    # These positions are UNIQUE to this date's GRIB file!
    fname = f"s3://noaa-gefs-pds/gefs.{target_date}/00/.../f{hour:03d}"
    idxdf = parse_grib_idx(basename=fname, storage_options={"anon": True})
    # idxdf contains: attrs, offset, length, grib_uri (date-specific!)

    # Step 2.2: Read pre-built parquet mapping from GCS (REFERENCE date)
    # This was created in Stage 0 (preprocessing)
    # Contains: varname, stepType, typeOfLevel, dims (date-independent!)
    gcs_path = f"gs://{bucket}/gefs/{member}/gefs-time-{ref_date}-{member}-rt{hour:03d}.parquet"
    deduped_mapping = pd.read_parquet(gcs_path, filesystem=gcs_fs)
    # deduped_mapping contains: attrs, varname, step, level, etc. (structure!)

    # Step 2.3: Merge fresh binary positions + template structure
    # ⚠️ CRITICAL MERGE: map_from_index() combines:
    #   - Fresh offset/length/uri from idxdf (target date)
    #   - Variable structure from deduped_mapping (reference template)
    # Merge key: "attrs" (e.g., "TMP:2 m above ground:3 hour fcst")
    mapped_index = map_from_index(target_date, deduped_mapping, idxdf_filtered)
    # Output: varname, step, level, offset, length, uri (complete + correct!)

    return mapped_index

# Process all forecast hours in batches
async def process_gefs_files_in_batches():
    for batch in batches:
        tasks = [process_single_gefs_file(hour) for hour in batch]
        results = await asyncio.gather(*tasks)

    # Combine all hours into single DataFrame
    gefs_kind = pd.concat(all_results, ignore_index=True)
    return gefs_kind
```

**Input**:
- Fresh GRIB index files (.idx) for **target date** (lightweight, ~KB each)
- Pre-built parquet mappings from GCS for **reference date** (from Stage 0)
- Target date (e.g., 20250918) - date you want to process
- Reference date (e.g., 20241112) - date with pre-built mappings

**Output**:
- Complete mapped index DataFrame (gefs_kind)
- Columns: varname, stepType, typeOfLevel, latitude, longitude, time, valid_time, step, uri, offset, length
- Contains ALL forecast hours combined
- **Crucially**: Has fresh binary positions (offset/length) from target date + variable structure from template

**The map_from_index() Merge Explained** (`_grib_idx.py:864-924`):

```python
def map_from_index(run_time, mapping, idxdf):
    """
    Merges fresh IDX binary positions with GCS template structure.

    Key operation:
    idxdf.merge(mapping, on="attrs", how="left")

    Input from idxdf (fresh, target date):
      - attrs: "TMP:2 m above ground:3 hour fcst"
      - offset: 523847  ← UNIQUE to target date!
      - length: 234567  ← UNIQUE to target date!
      - grib_uri: s3://...gefs.20250918/.../f003 ← Target date path!

    Input from mapping (template, reference date):
      - attrs: "TMP:2 m above ground:3 hour fcst"
      - varname: "t2m"
      - stepType: "instant"
      - typeOfLevel: "heightAboveGround"
      - level: 2
      - (old uri, offset, length are DROPPED!)

    Output (merged):
      - varname: "t2m"         ← From template
      - stepType: "instant"    ← From template
      - level: 2               ← From template
      - offset: 523847         ← From fresh idx (target date!)
      - length: 234567         ← From fresh idx (target date!)
      - uri: s3://.../20250918 ← From fresh idx (target date!)
    ```

    This merge ensures we read from the CORRECT binary positions in the target date's GRIB file!

**Why This is Fast**:
1. **Index files are tiny**: ~2-5 KB vs GRIB files at 100+ MB
   - IDX file = text list of byte positions (~100 lines)
   - GRIB file = binary weather data (~200 MB)
2. **No GRIB scanning**: Reads pre-built structure from GCS parquets
   - Avoids expensive `scan_grib()` which must decode GRIB messages
3. **Only updates byte offsets**: Target date has different offsets but same structure
   - Variable names, dimensions, levels stay the same across dates
   - Only binary positions change (due to compression, encoding differences)
4. **Async batch processing**: 10-20 files processed in parallel
   - Parse multiple idx files simultaneously
   - Reduce I/O wait time

**Why We Can't Skip Fresh IDX Files**:
❌ **WRONG**: Use GCS template's offset/length directly
- Template has offsets for 20241112, but we're reading 20250918!
- Binary positions are DIFFERENT between dates
- Result: Read wrong bytes → garbage data or wrong variables

✅ **CORRECT**: Parse fresh idx + merge with template structure
- Fresh idx gives correct positions for 20250918
- Template gives variable metadata (names, levels, dims)
- Result: Read correct bytes → correct data!

**Processing Time**:
- Per index file: ~0.5 seconds (vs 30-60s for scan_grib)
- 81 forecast hours in batches: 2-3 minutes
- **85x faster than scanning GRIB files!**

**⚠️ CRITICAL: Avoid Cross-System Template Errors**

**Common Error**: Using GFS bucket templates for GEFS data (or vice versa)

This will fail because:
1. **Different GRIB structures**: GFS and GEFS have different variable encodings
2. **Different attrs strings**: Merge key won't match
   - GFS: "TMP:2 m above ground:anl"
   - GEFS: "TMP:2 m above ground:3 hour fcst:ens mean"
3. **Different file layouts**: Ensemble members vs deterministic

**Result of mixing systems**:
```python
# WRONG: Using GFS template for GEFS data
gcs_path = "gs://bucket/gfs/gfs-time-20241112-rt003.parquet"  # ❌ GFS template
idxdf = parse_grib_idx("s3://noaa-gefs-pds/gefs.20250918...")  # GEFS idx

mapped_index = map_from_index(date, gfs_template, gefs_idx)
# → Merge fails! No matching "attrs" between GFS and GEFS
# → Empty or incomplete mapped_index
# → Missing variables in final output
```

**Correct approach**:
```python
# ✅ CORRECT: GEFS template for GEFS data
gcs_path = "gs://bucket/gefs/gep01/gefs-time-20241112-gep01-rt003.parquet"
idxdf = parse_grib_idx("s3://noaa-gefs-pds/gefs.20250918/.../gep01...f003")

mapped_index = map_from_index(date, gefs_template, gefs_idx)
# → Successful merge on matching "attrs"
# → Complete mapped_index with all variables
```

**Key Principle**:
> **Templates MUST come from the same forecast system (GEFS, GFS, ECMWF) as the target data!**
>
> Binary positions change between dates (same system) ✓
>
> But variable structure changes between systems (GFS ≠ GEFS) ✗

**Key Function Calls**:
```python
# In run_day_gefs_ensemble_full.py
gefs_kind = cs_create_mapped_index(
    axes,
    gcs_bucket_name='gik-fmrc',
    target_date_str='20250918',     # Date to process
    member='gep01',
    gcp_service_account_json='credentials.json',
    reference_date_str='20241112'   # Date with pre-built mappings
)
```

**📊 Stage 2 Summary - The Critical Merge**:

```
┌─────────────────────────────────────────────────────────────────┐
│                  WHY STAGE 2 WORKS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Fresh IDX File (Target Date 20250918)     GCS Template (Ref)  │
│  ════════════════════════════════════     ═══════════════════  │
│  attrs: "TMP:2m:3h fcst"                  attrs: "TMP:2m:3h"   │
│  offset: 523847  ← DATE SPECIFIC!         varname: "t2m"       │
│  length: 234567  ← DATE SPECIFIC!         stepType: "instant"  │
│  uri: .../20250918/...f003                level: 2             │
│                                                                 │
│         ↓                                        ↓              │
│         └──────────── MERGE ON "attrs" ─────────┘              │
│                            ↓                                    │
│                  Complete Mapped Index                         │
│                  ═══════════════════════                        │
│                  varname: "t2m"        ← From template         │
│                  level: 2              ← From template         │
│                  offset: 523847        ← From fresh idx!       │
│                  length: 234567        ← From fresh idx!       │
│                  uri: .../20250918/... ← From fresh idx!       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ✅ CORRECT: Uses fresh binary positions for target date       │
│  ✅ FAST: Avoids expensive scan_grib (85x speedup!)            │
│  ✅ REUSABLE: Template works for any date (same structure)     │
│                                                                 │
│  ❌ NEVER: Use template's offset/length directly               │
│  ❌ NEVER: Mix GFS templates with GEFS data (attrs mismatch)   │
└─────────────────────────────────────────────────────────────────┘
```

---

### Stage 3: Merge and Create Final Zarr Parquet with All Timesteps

**Purpose**: Combine Stage 1 (deflated GRIB tree) and Stage 2 (mapped index) to create final zarr store

**Location**: `gefs_util.py:263-332` (prepare_zarr_store + process_unique_groups)

**Used in**: `run_day_gefs_ensemble_full.py:194-196`

**Process**:
```python
# Step 3.1: Prepare zarr store from Stage 1 and Stage 2
zstore, chunk_index = prepare_zarr_store(
    deflated_gefs_grib_tree_store,  # From Stage 1
    gefs_kind                        # From Stage 2
)

# Step 3.2: Process unique variable groups with time dimensions
updated_zstore = process_unique_groups(
    zstore,
    chunk_index,
    time_dims={'valid_time': 81},    # All forecast hours
    time_coords=time_coords,
    times=times,
    valid_times=valid_times,
    steps=steps
)

# Step 3.3: Save as parquet file
create_parquet_file_fixed(updated_zstore, f"{member}.parquet")
```

**Input**:
- Deflated GRIB tree store (from Stage 1)
- Mapped index DataFrame (from Stage 2)
- Time dimension info (time_dims, time_coords, times, valid_times, steps)

**Output**:
- Final zarr reference store (dictionary)
- Saved as parquet file: `{member}.parquet`
- Can be opened with xarray.open_datatree()

**What it does**:
```python
# For each unique variable group:
unique_groups = chunk_index.groupby(['varname', 'stepType', 'typeOfLevel']).groups

for group in unique_groups:
    base_path = f"/{varname}/{stepType}/{typeOfLevel}"

    # Store coordinate variables
    store_coord_var(f"{base_path}/time", zstore, times)
    store_coord_var(f"{base_path}/valid_time", zstore, valid_times)
    store_coord_var(f"{base_path}/step", zstore, steps)

    # Store data variable with proper chunking
    store_data_var(f"{base_path}/{varname}", zstore, chunk_index, time_dims)
```

**Processing Time**:
- Per member: 30-60 seconds
- Creates complete zarr structure with all 81 timesteps

**Key Features**:
- Processes unique variable groups
- Creates proper zarr hierarchy: `/variable/stepType/level/`
- Stores coordinate variables: time, valid_time, step
- Stores data variables with correct dimensions
- Handles multiple forecast levels (surface, pressure, etc.)

---

## Complete Workflow with GCS Templates

### For GEFS Member gep01, Target Date 20250918, Reference Date 20241112

```python
# ========== STAGE 0: ONE-TIME PREPROCESSING (ALREADY DONE!) ==========
# Run ONCE per member to create GCS templates
# python gefs_index_preprocessing_fixed.py --date 20241112 --member gep01 --bucket gik-gefs-aws-tf
# Creates: gs://gik-gefs-aws-tf/gefs/gep01/gefs-time-20241112-gep01-rt*.parquet

# ========== STAGE 1: SCAN GRIB (2 minutes) ==========
gefs_files = [
    "s3://noaa-gefs-pds/gefs.20250918/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f000",
    "s3://noaa-gefs-pds/gefs.20250918/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003"
]

_, deflated_store = filter_build_grib_tree(gefs_files, forecast_dict)
print(f"Stage 1 complete: {len(deflated_store['refs'])} references")

# ========== STAGE 2: IDX + GCS TEMPLATES (2-3 minutes) ==========
gefs_kind = cs_create_mapped_index(
    axes,
    gcs_bucket_name='gik-gefs-aws-tf',
    target_date_str='20250918',        # NEW date to process
    member='gep01',
    reference_date_str='20241112'      # Reference with templates
)
print(f"Stage 2 complete: {len(gefs_kind)} mapped entries for all hours")

# ========== STAGE 3: CREATE ZARR STORE (30-60 seconds) ==========
zstore, chunk_index = prepare_zarr_store(deflated_store, gefs_kind)
final_zstore = process_unique_groups(zstore, chunk_index, time_dims,
                                     time_coords, times, valid_times, steps)

create_parquet_file_fixed(final_zstore, "gep01.parquet")
print("Stage 3 complete: gep01.parquet created")

# Total time: ~5 minutes (vs 4-6 hours without GCS templates!)
```

---

## Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│        STAGE 0: ONE-TIME GCS PREPROCESSING (20-40 min/member)      │
│                                                                    │
│  For reference date 20241112:                                      │
│  For each hour (0, 3, 6, ..., 240):                               │
│    1. parse_grib_idx() → Parse index                              │
│    2. build_idx_grib_mapping() → Build complete mapping (SLOW!)   │
│    3. Save to GCS: gs://bucket/gefs/gep01/gefs-time-{date}-*.par  │
│                                                                    │
│  Output: 81 parquet files per member in GCS (REUSABLE!)           │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
                    (Use for ANY future date!)
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│              STAGE 1: SCAN GRIB (2 minutes)                        │
│                                                                    │
│  Target date 20250918:                                             │
│  GRIB Files (0h, 3h) → filter_gefs_scan_grib()                    │
│         ↓                                                          │
│  grib_tree() → Build hierarchical structure                        │
│         ↓                                                          │
│  strip_datavar_chunks() → Deflated store                          │
│         ↓                                                          │
│  Output: deflated_gefs_grib_tree_store                            │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│         STAGE 2: IDX + GCS TEMPLATES (2-3 minutes, 85x faster!)   │
│         ⚠️ CRITICAL: Fresh IDX binary positions required!         │
│                                                                    │
│  Target date 20250918:                                             │
│  For each hour (0, 3, ..., 240) in batches:                       │
│    1. parse_grib_idx(target_date) → Fresh index (FAST! ~0.5s)    │
│       ↳ Gets: offset, length, uri for 20250918 ← DATE SPECIFIC!  │
│    2. read GCS template(ref_date) → Structure template            │
│       ↳ Gets: varname, level, dims from 20241112 ← REUSABLE!     │
│    3. map_from_index() → MERGE on "attrs"                         │
│       ↳ Combines: fresh positions + template structure           │
│         ↓                                                          │
│  Async batch processing (10-20 files in parallel)                 │
│         ↓                                                          │
│  Output: gefs_kind DataFrame (all 81 hours mapped)                │
│          with CORRECT binary positions for target date!           │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│           STAGE 3: CREATE FINAL ZARR STORE (30-60 seconds)         │
│                                                                    │
│  deflated_store + gefs_kind + time_dims                           │
│         ↓                                                          │
│  prepare_zarr_store() → Initialize zarr structure                  │
│         ↓                                                          │
│  process_unique_groups() → Add coordinates & data vars             │
│         ↓                                                          │
│  create_parquet_file_fixed() → Save parquet                       │
│         ↓                                                          │
│  Output: gep01.parquet (xarray-compatible)                        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Performance Comparison

### With GCS Templates (Stage 0 already done)
- **Stage 1**: 2 minutes (scan 2 GRIB files)
- **Stage 2**: 2-3 minutes (81 index files + GCS templates)
- **Stage 3**: 30-60 seconds
- **Total per member**: ~5 minutes
- **30 members**: 15-20 minutes (parallel batches)

### Without GCS Templates (scan_grib for everything)
- **Scan all 81 GRIB files**: 40-50 minutes per member
- **Process and merge**: 10-15 minutes
- **Total per member**: 50-65 minutes
- **30 members**: 25-32 hours!

### Stage 0 Preprocessing (One-Time Cost)
- **Per member**: 20-40 minutes (builds 81 parquet files)
- **All 30 members**: 10-20 hours (one-time!)
- **Benefit**: Reusable for ANY future date!

---

## Key Takeaways

1. **Stage 0 is expensive but ONE-TIME**: Build GCS templates once, reuse forever
2. **Stage 2 is the magic**: 85x faster by using index files + GCS templates
3. **GCS templates are date-independent**: Structure is same, only byte offsets change
4. **Async batching is critical**: Process 10-20 files in parallel
5. **Total speedup**: 5 minutes vs 50+ minutes per member (10x faster!)

---

## Missing in ECMWF Implementation

| Component | GEFS | ECMWF Status |
|-----------|------|--------------|
| **Stage 0 Preprocessing** | ✅ Done | ❌ Missing |
| **GCS Template Storage** | ✅ Implemented | ❌ Missing |
| **Index-based Processing** | ✅ Stage 2 | ❌ Uses scan_grib only |
| **Async Batch Processing** | ✅ Stage 2 | ❌ Sequential only |
| **Complete Time Coverage** | ✅ 81 steps | ❌ Only 2 steps (0h, 3h) |
| **Reusable Mappings** | ✅ Cross-date | ❌ No templates |

---

## Usage Example

```python
# Read parquet and open with xarray
zstore = read_parquet_fixed("gep01.parquet")
fs = fsspec.filesystem("reference", fo=zstore,
                      remote_protocol='s3',
                      remote_options={'anon': True})
mapper = fs.get_mapper("")
dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

# Access precipitation data
tp_data = dt['/tp/accum/surface'].ds['tp']

# Extract East Africa region at 12-hour forecast
ea_tp = tp_data.sel(
    latitude=slice(15, -12),
    longitude=slice(25, 52),
    step=12
).compute()
```

---

## Verification Checklist: Is Your Stage 2 Working Correctly?

Use this checklist to verify that your Stage 2 implementation correctly uses fresh IDX files:

### ✅ Correct Implementation Signs:

1. **Different dates in logs**:
   ```
   [INFO] Parsing fresh idx from: s3://.../gefs.20250918/.../f003
   [INFO] Reading GCS template from: gs://bucket/gefs/gep01/gefs-time-20241112-gep01-rt003.parquet
   ```
   ✓ Target date (20250918) ≠ Reference date (20241112)

2. **Successful merge output**:
   ```python
   print(mapped_index[['varname', 'uri', 'offset', 'length']].head())
   #    varname  uri                              offset    length
   # 0  t2m      s3://.../gefs.20250918/...f003  523847    234567
   ```
   ✓ URI contains target date (20250918)
   ✓ Offset/length are date-specific values

3. **Complete variable coverage**:
   ```python
   print(f"Variables in mapped_index: {mapped_index['varname'].nunique()}")
   # Variables in mapped_index: 8
   ```
   ✓ All expected variables present (matches forecast_dict)

### ❌ Error Signs (Incorrect Implementation):

1. **Missing fresh IDX parsing**:
   ```python
   # WRONG: Directly using GCS template without fresh idx
   mapped_index = pd.read_parquet(gcs_template_path)  # ❌ Old offsets!
   ```
   → You'll read wrong binary positions → garbage data

2. **Empty or incomplete merge**:
   ```python
   print(len(mapped_index))
   # 0  ← No matches! Wrong template system (GFS vs GEFS?)
   ```
   → Check: Are you using GFS template for GEFS data?

3. **Same date for target and reference**:
   ```python
   gefs_kind = cs_create_mapped_index(...,
       target_date_str='20241112',
       reference_date_str='20241112'
   )
   ```
   → Works, but you're not testing template reusability!

4. **URI contains wrong date**:
   ```python
   print(mapped_index['uri'].iloc[0])
   # s3://.../gefs.20241112/...  ← Wrong! Should be target date!
   ```
   → Fresh IDX was not used correctly

### 🔍 Debug: Check the Merge

```python
# Verify map_from_index() is working correctly
print("From fresh IDX (target date):")
print(idxdf[['attrs', 'offset', 'length', 'grib_uri']].head(3))

print("\nFrom GCS template (reference date):")
print(deduped_mapping[['attrs', 'varname', 'level']].head(3))

print("\nAfter merge:")
print(mapped_index[['attrs', 'varname', 'offset', 'length', 'uri']].head(3))

# Check:
# 1. attrs column matches between idxdf and deduped_mapping ✓
# 2. offset/length come from idxdf (fresh!) ✓
# 3. varname/level come from deduped_mapping (template) ✓
# 4. uri contains target date, not reference date ✓
```

### 🎯 Key Test: Process Different Dates

The ultimate test: Process multiple dates with the same GCS template

```bash
# Create template once for reference date
python gefs_index_preprocessing_fixed.py --date 20241112 --member gep01

# Process different dates using same template
python run_day_gefs_ensemble_full.py --target-date 20250101 --reference-date 20241112
python run_day_gefs_ensemble_full.py --target-date 20250201 --reference-date 20241112
python run_day_gefs_ensemble_full.py --target-date 20250301 --reference-date 20241112
```

If all three dates produce valid output with correct data, your Stage 2 is working correctly! 🎉
