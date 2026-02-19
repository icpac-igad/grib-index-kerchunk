# ECMWF Three-Stage Processing: Grib-Index-Kerchunk Method

## Overview

This document details the three distinct processing stages (routines) used in ECMWF ensemble data processing using the grib-index-kerchunk method. These stages transform raw GRIB files into efficient parquet-based zarr stores, following the same pattern as GEFS but adapted for ECMWF's unique data structure.

**CRITICAL**: Stage 2 requires a one-time expensive preprocessing step (Stage 0) that creates reusable GCS parquet mapping templates.

**Key Differences from GEFS**:
- ECMWF has 51 ensemble members (control + 50 perturbed) vs GEFS's 30
- ECMWF uses different time intervals (3h for 0-144h, then 6h for 150-360h)
- ECMWF stores all members in single GRIB files (need to filter by member number)
- Total of 85 time steps vs GEFS's 81

---

## The Three Processing Stages

### Stage 0: ONE-TIME GCS Parquet Mapping Creation (PREPROCESSING)

**⚠️ IMPORTANT**: This is the **most expensive** step but only runs ONCE per ensemble member. The output parquet files can be reused for ANY date!

**Purpose**: Create parquet mapping templates in GCS that describe ECMWF GRIB data structure

**Location**: `ecmwf_index_preprocessing.py`

**When to Run**:
- **ONCE** per ensemble member to create reusable templates
- Only needs to be re-run if GRIB structure changes (rare)

**Process**:
```bash
# Run once for single member
python ecmwf_index_preprocessing.py \
  --date 20240529 \
  --member ens01 \
  --bucket gik-ecmwf-aws-tf

# OR run for all 51 members
python ecmwf_index_preprocessing.py \
  --date 20240529 \
  --all-members \
  --bucket gik-ecmwf-aws-tf
```

**What it does**:
```python
# For each forecast hour (0, 3, 6, ..., 360):
# 1. Parse GRIB index file
idxdf = parse_grib_idx(basename=fname, storage_options=storage_options)

# 2. Filter for specific member (ECMWF-specific)
# Member number: 0 for control, 1-50 for perturbed
idxdf = idxdf[idxdf['attrs'].str.contains(f"number={member_number}")]

# 3. Build complete idx-grib mapping (EXPENSIVE!)
grib_mapping = build_idx_grib_mapping(
    basename=fname,
    mapper=mapper,
    storage_options=storage_options,
    validate=True
)

# 4. Deduplicate and save to GCS
deduped_mapping = grib_mapping.loc[~grib_mapping["attrs"].duplicated(keep="first"), :]
gcs_path = f"gs://{bucket}/ecmwf/{member}/ecmwf-time-{date}-{member}-rt{hour:03d}.parquet"
deduped_mapping.to_parquet(gcs_path, filesystem=gcs_fs)
```

**Output Structure in GCS**:
```
gs://gik-ecmwf-aws-tf/ecmwf/
├── control/
│   ├── ecmwf-time-20240529-control-rt000.parquet
│   ├── ecmwf-time-20240529-control-rt003.parquet
│   ├── ecmwf-time-20240529-control-rt006.parquet
│   └── ... (85 files, one per forecast hour)
├── ens01/
│   ├── ecmwf-time-20240529-ens01-rt000.parquet
│   └── ... (85 files)
├── ens02/
└── ... (ens03-ens50)
```

**Processing Time**:
- Per forecast hour: 20-40 seconds
- Per member (85 hours): 30-50 minutes
- All 51 members: 25-40 hours (one-time cost!)

**Why Expensive?**:
- Must scan actual GRIB files to build complete byte offset mappings
- ECMWF GRIB files are larger (~500MB-1GB per timestep)
- Creates detailed metadata for every variable in every GRIB file
- But: These mappings are **reusable across any date with same structure!**

---

### Stage 1: Scan GRIB to Ensemble-wise Parquet Files

**Purpose**: Scan GRIB files and create deflated GRIB tree store with variable references

**Location**: `ecmwf_ensemble_par_creator_efficient.py`

**Used in**: `run_day_ecmwf_ensemble_full.py` (to be created)

**Process**:
```python
# Step 1.1: Scan ECMWF GRIB files with filtering
ecmwf_groups = []
for hour in forecast_hours:
    url = f"s3://ecmwf-forecasts/{date}/00z/ifs/0p25/enfo/{date}000000-{hour}h-enfo-ef.grib2"
    groups = scan_grib(url, filter=forecast_dict, storage_options={"anon": True})

    # Filter for specific member
    member_groups = [g for g in groups if check_member(g, member_number)]
    ecmwf_groups.extend(member_groups)

# Step 1.2: Build hierarchical GRIB tree
ecmwf_grib_tree_store = grib_tree(ecmwf_groups)

# Step 1.3: Strip data chunks to reduce memory
deflated_ecmwf_grib_tree_store = copy.deepcopy(ecmwf_grib_tree_store)
strip_datavar_chunks(deflated_ecmwf_grib_tree_store)
```

**Input**:
- GRIB2 files from S3: `s3://ecmwf-forecasts/{date}/00z/ifs/0p25/enfo/{date}000000-{hour}h-enfo-ef.grib2`
- Variable filter dictionary (ECMWF_FORECAST_DICT)
- Member number to filter (0 for control, 1-50 for perturbed)

**Output**:
- Deflated GRIB tree store (in-memory dictionary)
- Contains zarr-like references to filtered variables for specific member

**Processing Time**:
- Per GRIB file: 45-90 seconds (larger than GEFS files)
- Typically processes just 0h, 3h, 6h files for testing (~3-5 minutes)
- Full 85 hours: 60-120 minutes per member

**Key Features**:
- Handles ECMWF's combined member format (all members in one file)
- Filters by member number during scanning
- Uses kerchunk's `scan_grib` to extract GRIB metadata
- Creates hierarchical zarr-like structure

---

### Stage 2: IDX Files + GCS Parquet Templates → Mapped Index

**⚡ KEY STAGE**: This is where the **85x speedup** happens!

**Purpose**: Combine fresh GRIB index files with pre-built GCS parquet templates to create complete mapped index

**Location**: `ecmwf_index_processor.py` (enhanced version)

**Used in**: `run_day_ecmwf_ensemble_full.py`

**⚠️ CRITICAL CONCEPT**: Same as GEFS - Fresh IDX Files Are Essential

**Process**:
```python
async def process_single_ecmwf_file(hour, member):
    # Step 2.1: Read fresh GRIB index file for TARGET date
    fname = f"s3://ecmwf-forecasts/{target_date}/00z/ifs/0p25/enfo/{target_date}000000-{hour}h-enfo-ef.grib2"
    idxdf = parse_grib_idx(basename=fname, storage_options={"anon": True})

    # Filter for specific member
    member_number = 0 if member == "control" else int(member.replace("ens", ""))
    idxdf_filtered = idxdf[idxdf['attrs'].str.contains(f"number={member_number}")]

    # Step 2.2: Read pre-built parquet mapping from GCS (REFERENCE date)
    gcs_path = f"gs://{bucket}/ecmwf/{member}/ecmwf-time-{ref_date}-{member}-rt{hour:03d}.parquet"
    deduped_mapping = pd.read_parquet(gcs_path, filesystem=gcs_fs)

    # Step 2.3: Merge fresh binary positions + template structure
    mapped_index = map_from_index(
        run_time=pd.Timestamp(target_date),
        mapping=deduped_mapping,
        idxdf=idxdf_filtered
    )

    return mapped_index

# Process all 85 forecast hours in batches
async def process_ecmwf_files_in_batches(member):
    # Process in batches of 10 for efficiency
    for batch in batches:
        tasks = [process_single_ecmwf_file(hour, member) for hour in batch]
        results = await asyncio.gather(*tasks)

    # Combine all hours into single DataFrame
    ecmwf_kind = pd.concat(all_results, ignore_index=True)
    return ecmwf_kind
```

**Input**:
- Fresh GRIB index files (.index) for **target date** (lightweight, ~KB each)
- Pre-built parquet mappings from GCS for **reference date** (from Stage 0)
- Target date (e.g., 20250101) - date you want to process
- Reference date (e.g., 20240529) - date with pre-built mappings
- Member to process (control, ens01-ens50)

**Output**:
- Complete mapped index DataFrame (ecmwf_kind)
- Columns: varname, stepType, typeOfLevel, latitude, longitude, time, valid_time, step, uri, offset, length
- Contains ALL 85 forecast hours combined
- **Crucially**: Has fresh binary positions (offset/length) from target date + variable structure from template

**Processing Time**:
- Per index file: ~0.5-1 seconds
- 85 forecast hours in batches: 3-5 minutes
- **85x faster than scanning GRIB files!**

**ECMWF-Specific Considerations**:
- Must filter index by member number (all members in same file)
- Different time intervals (3h then 6h) handled automatically
- Larger index files than GEFS (more variables/levels)
- Template attrs format: includes "number=X" for member identification

---

### Stage 3: Merge and Create Final Zarr Parquet with All Timesteps

**Purpose**: Combine Stage 1 (deflated GRIB tree) and Stage 2 (mapped index) to create final zarr store

**Location**: `ecmwf_util.py` (enhanced functions) + test implementation in `test_three_stage_ecmwf_simple.py`

**Used in**: `run_day_ecmwf_ensemble_full.py`

**Process**:
```python
# Step 3.1: Generate ECMWF-specific time dimensions
axes = generate_ecmwf_axes(target_date)
# Handles variable time intervals: 3h (0-144h) then 6h (150-360h)

time_dims = {'valid_time': 85}  # All 85 forecast hours
time_coords = {
    'valid_time': 'valid_time',
    'time': 'time',
    'step': 'step'
}

# Step 3.2: Prepare zarr store from Stage 1 and Stage 2
zstore, chunk_index = prepare_zarr_store(
    deflated_ecmwf_grib_tree_store,  # From Stage 1
    ecmwf_kind                        # From Stage 2
)

# Step 3.3: Process unique variable groups with ECMWF dimensions
updated_zstore = process_unique_groups(
    zstore,
    chunk_index,
    time_dims={'valid_time': 85},    # All 85 forecast hours
    time_coords=time_coords,
    times=times,
    valid_times=valid_times,
    steps=steps
)

# Step 3.4: Save as parquet file
create_parquet_file_fixed(updated_zstore, f"{member}.parquet")
```

**Input**:
- Deflated GRIB tree store (from Stage 1)
- Mapped index DataFrame (from Stage 2)
- ECMWF time dimension info (85 steps with variable intervals)

**Output**:
- Final zarr reference store (dictionary)
- Saved as parquet file: `{member}.parquet` (e.g., `control.parquet`, `ens01.parquet`)
- Can be opened with xarray.open_datatree()

**Processing Time**:
- Per member: 45-90 seconds
- Creates complete zarr structure with all 85 timesteps

**ECMWF-Specific Features**:
- Handles variable time intervals correctly
- Processes 51 ensemble members
- Larger spatial grid (0.25° global = 721x1440 points)
- More vertical levels than GEFS

---

## Complete Workflow with GCS Templates

### For ECMWF Member ens01, Target Date 20250101, Reference Date 20240529

```python
# ========== STAGE 0: ONE-TIME PREPROCESSING (ALREADY DONE!) ==========
# Run ONCE per member to create GCS templates
# python ecmwf_index_preprocessing.py --date 20240529 --member ens01 --bucket gik-ecmwf-aws-tf
# Creates: gs://gik-ecmwf-aws-tf/ecmwf/ens01/ecmwf-time-20240529-ens01-rt*.parquet

# ========== STAGE 1: SCAN GRIB (3-5 minutes for testing) ==========
ecmwf_files = [
    "s3://ecmwf-forecasts/20250101/00z/ifs/0p25/enfo/20250101000000-0h-enfo-ef.grib2",
    "s3://ecmwf-forecasts/20250101/00z/ifs/0p25/enfo/20250101000000-3h-enfo-ef.grib2",
    "s3://ecmwf-forecasts/20250101/00z/ifs/0p25/enfo/20250101000000-6h-enfo-ef.grib2"
]

# Scan and filter for member ens01 (number=1)
_, deflated_store = filter_build_ecmwf_grib_tree(ecmwf_files, forecast_dict, member="ens01")
print(f"Stage 1 complete: {len(deflated_store['refs'])} references")

# ========== STAGE 2: IDX + GCS TEMPLATES (3-5 minutes) ==========
ecmwf_kind = cs_create_ecmwf_mapped_index(
    axes,
    gcs_bucket_name='gik-ecmwf-aws-tf',
    target_date_str='20250101',        # NEW date to process
    member='ens01',
    reference_date_str='20240529'      # Reference with templates
)
print(f"Stage 2 complete: {len(ecmwf_kind)} mapped entries for all hours")

# ========== STAGE 3: CREATE ZARR STORE (45-90 seconds) ==========
zstore, chunk_index = prepare_zarr_store(deflated_store, ecmwf_kind)
final_zstore = process_unique_groups(zstore, chunk_index, time_dims,
                                     time_coords, times, valid_times, steps)

create_parquet_file_fixed(final_zstore, "ens01.parquet")
print("Stage 3 complete: ens01.parquet created")

# Total time: ~8-10 minutes (vs 1-2 hours without GCS templates!)
```

---

## Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│        STAGE 0: ONE-TIME GCS PREPROCESSING (30-50 min/member)      │
│                                                                    │
│  For reference date 20240529:                                      │
│  For each hour (0, 3, 6, ..., 360) - 85 total:                   │
│    1. parse_grib_idx() → Parse index                              │
│    2. Filter by member number (ECMWF-specific)                    │
│    3. build_idx_grib_mapping() → Build complete mapping (SLOW!)   │
│    4. Save to GCS: gs://bucket/ecmwf/{member}/ecmwf-time-*.par    │
│                                                                    │
│  Output: 85 parquet files per member in GCS (REUSABLE!)           │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
                    (Use for ANY future date!)
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│              STAGE 1: SCAN GRIB (3-5 minutes for test)             │
│                                                                    │
│  Target date 20250101:                                             │
│  GRIB Files → scan_grib() with member filter                      │
│         ↓                                                          │
│  grib_tree() → Build hierarchical structure                        │
│         ↓                                                          │
│  strip_datavar_chunks() → Deflated store                          │
│         ↓                                                          │
│  Output: deflated_ecmwf_grib_tree_store                           │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│         STAGE 2: IDX + GCS TEMPLATES (3-5 minutes, 85x faster!)   │
│         ⚠️ CRITICAL: Fresh IDX binary positions required!         │
│                                                                    │
│  Target date 20250101:                                             │
│  For each hour (0, 3, ..., 360) in batches - 85 total:            │
│    1. parse_grib_idx(target_date) → Fresh index (FAST! ~0.5s)    │
│       ↳ Filter by member number                                   │
│       ↳ Gets: offset, length, uri for 20250101 ← DATE SPECIFIC!  │
│    2. read GCS template(ref_date) → Structure template            │
│       ↳ Gets: varname, level, dims from 20240529 ← REUSABLE!     │
│    3. map_from_index() → MERGE on "attrs"                         │
│       ↳ Combines: fresh positions + template structure           │
│         ↓                                                          │
│  Async batch processing (10-20 files in parallel)                 │
│         ↓                                                          │
│  Output: ecmwf_kind DataFrame (all 85 hours mapped)               │
│          with CORRECT binary positions for target date!           │
└────────────────────────────────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│           STAGE 3: CREATE FINAL ZARR STORE (45-90 seconds)         │
│                                                                    │
│  deflated_store + ecmwf_kind + time_dims (85 steps)              │
│         ↓                                                          │
│  prepare_zarr_store() → Initialize zarr structure                  │
│         ↓                                                          │
│  process_unique_groups() → Add coordinates & data vars             │
│         ↓                                                          │
│  create_parquet_file_fixed() → Save parquet                       │
│         ↓                                                          │
│  Output: {member}.parquet (xarray-compatible)                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Performance Comparison

### With GCS Templates (Stage 0 already done)
- **Stage 1**: 3-5 minutes (scan 3 test GRIB files)
- **Stage 2**: 3-5 minutes (85 index files + GCS templates)
- **Stage 3**: 45-90 seconds
- **Total per member**: ~8-10 minutes
- **51 members**: 30-40 minutes (parallel batches)

### Without GCS Templates (scan_grib for everything)
- **Scan all 85 GRIB files**: 60-120 minutes per member
- **Process and merge**: 15-20 minutes
- **Total per member**: 75-140 minutes
- **51 members**: 64-120 hours!

### Stage 0 Preprocessing (One-Time Cost)
- **Per member**: 30-50 minutes (builds 85 parquet files)
- **All 51 members**: 25-40 hours (one-time!)
- **Benefit**: Reusable for ANY future date!

---

## Key Differences: ECMWF vs GEFS

| Aspect | GEFS | ECMWF |
|--------|------|-------|
| **Ensemble Members** | 30 (gep01-gep30) | 51 (control + ens01-ens50) |
| **Time Steps** | 81 (uniform 3h) | 85 (3h then 6h intervals) |
| **GRIB Structure** | Separate files per member | All members in single file |
| **Member Identification** | File name (gep01.grib2) | attrs field (number=1) |
| **File Size** | ~200MB per timestep | ~500MB-1GB per timestep |
| **S3 Bucket** | noaa-gefs-pds | ecmwf-forecasts |
| **Grid Resolution** | 0.25° (721x1440) | 0.25° (721x1440) |
| **Index Format** | Text (.idx) | JSON (.index) |

---

## Implementation Files

### Core Components

1. **Stage 0 - Preprocessing**:
   - `ecmwf_index_preprocessing.py` - Create GCS templates
   - `run_ecmwf_preprocessing.py` - Batch runner for all members

2. **Stage 1 - GRIB Scanning**:
   - `ecmwf_ensemble_par_creator_efficient.py` - Efficient GRIB scanning
   - `fmrc_utils.py` - Utility functions for FMRC processing
   - `run_ecmwf_scangrib_fmrc.py` - Runner for Stage 1

3. **Stage 2 - Index Processing**:
   - `ecmwf_index_processor.py` - Index-based fast processing
   - `ecmwf_par_to_ensemble_members.py` - Member extraction utilities

4. **Stage 3 - Zarr Creation**:
   - Implemented in `ecmwf_util.py` and test files
   - Uses kerchunk utilities for zarr store creation

5. **Utilities**:
   - `ecmwf_util.py` - ECMWF-specific utility functions
   - `ecmwf_gcs_uploader.py` - GCS upload utilities

6. **Testing**:
   - `test_three_stage_ecmwf_simple.py` - Simple test routine for learning
   - `test_run_ecmwf_step1_scangrib.py` - Test Stage 1
   - `test_fixed_efficient.py` - Test efficient processing

---

## Usage Example

```python
# Read parquet and open with xarray
zstore = read_parquet_fixed("ens01.parquet")
fs = fsspec.filesystem("reference", fo=zstore,
                      remote_protocol='s3',
                      remote_options={'anon': True})
mapper = fs.get_mapper("")
dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

# Access temperature data
t2m_data = dt['/2t/instant/surface'].ds['2t']

# Extract Europe region at 24-hour forecast
europe_t2m = t2m_data.sel(
    latitude=slice(70, 35),
    longitude=slice(-10, 40),
    step=24
).compute()

# Access precipitation (accumulated)
tp_data = dt['/tp/accum/surface'].ds['tp']
```

---

## Verification Checklist

### ✅ Correct Implementation Signs:

1. **Different dates in logs**:
   ```
   [INFO] Parsing fresh idx from: s3://.../ecmwf-forecasts/20250101/.../3h-enfo-ef.grib2
   [INFO] Reading GCS template from: gs://bucket/ecmwf/ens01/ecmwf-time-20240529-ens01-rt003.parquet
   ```

2. **Member filtering working**:
   ```python
   print(f"Member filter: number={member_number}")
   print(f"Filtered entries: {len(idxdf_filtered)} from {len(idxdf)} total")
   ```

3. **Complete time coverage**:
   ```python
   print(f"Time steps in mapped_index: {mapped_index['step'].nunique()}")
   # Should be 85 for complete ECMWF forecast
   ```

### ❌ Error Signs:

1. **Missing member filtering**:
   - Processing all members together (51x too much data)
   - Wrong member extracted

2. **Time interval confusion**:
   - Missing hours after 144h
   - Wrong step size for late hours

3. **Template mismatch**:
   - Using GEFS templates for ECMWF (different attrs format)
   - Wrong bucket or path structure

---

## Next Steps

1. **Run Stage 0** for all ensemble members:
   ```bash
   python ecmwf_index_preprocessing.py --date 20240529 --all-members --bucket gik-ecmwf-aws-tf
   ```

2. **Test with single member**:
   ```bash
   python test_three_stage_ecmwf_simple.py
   ```

3. **Scale to production**:
   - Create `run_day_ecmwf_ensemble_full.py` for full processing
   - Implement parallel processing for all 51 members
   - Add monitoring and error recovery

4. **Optimize further**:
   - Cache frequently used templates locally
   - Implement incremental updates for new forecast runs
   - Add data validation and quality checks

---

## Key Takeaways

1. **Stage 0 is expensive but ONE-TIME**: Build GCS templates once, reuse forever
2. **Stage 2 is the magic**: 85x faster by using index files + GCS templates
3. **Member filtering is critical**: ECMWF stores all members together
4. **Time intervals vary**: Handle 3h and 6h intervals correctly
5. **Total speedup**: 8-10 minutes vs 75+ minutes per member (8-10x faster!)
6. **Scale matters**: 51 members × 85 timesteps = massive data volume