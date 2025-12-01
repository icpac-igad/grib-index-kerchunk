## Files Required for Implementation

### Core Processing Scripts

| File | Purpose | Stage |
|------|---------|-------|
| `utils_ecmwf_step1_scangrib.py` | `scan_grib` wrapper with ensemble member extraction and `fixed_ensemble_grib_tree()` | Stage 1 |
| `ecmwf_util.py` | ECMWF-specific utilities, time dimension handling, GCS parquet loading | All |
| `ecmwf_index_processor.py` | Fast index-based processing (~85x faster than scan_grib) | Stage 2 |
| `ecmwf_ensemble_par_creator_efficient_multidate.py` | Multi-date ensemble parquet creation | Stage 1 |
| `ecmwf_three_stage_multidate.py` | Complete three-stage pipeline orchestration | All |

### Validation and Comparison Scripts

| File | Purpose |
|------|---------|
| `compare_ecmwf_opendata_vs_gik.py` | Side-by-side validation of GIK vs ECMWF Open Data API |
| `read_stage3_aifs_all_timesteps.py` | Extract all 85 timesteps from Stage 3 parquet |

### Documentation

| File | Purpose |
|------|---------|
| `ECMWF_Three_Stage_Processing.md` | Detailed three-stage processing documentation |
| `GIK_VALIDATION_AND_VERIFICATION.md` | Validation methodology and results |

---

## Overview: The GIK Method for ECMWF

The **Grib-Index-Kerchunk (GIK)** method creates zarr-compatible parquet reference files that enable efficient cloud-native access to ECMWF ensemble forecast data stored on AWS S3 (`s3://ecmwf-forecasts/`).

### Data Source
```
s3://ecmwf-forecasts/{date}/{run}z/ifs/0p25/enfo/{date}{run}0000-{hour}h-enfo-ef.grib2
```

### Key Characteristics
- **51 ensemble members**: Control + 50 perturbed members (vs GEFS's 30)
- **85 forecast timesteps**: 3h intervals (0-144h) + 6h intervals (150-360h)
- **0.25° global resolution**: 721 × 1440 grid points
- **All members in single GRIB**: Requires member filtering during processing

---

## Three-Stage Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│     STAGE 0: ONE-TIME GCS PREPROCESSING (30-50 min/member)           │
│                                                                      │
│  For reference date (e.g., 20240529):                                │
│  For each hour (0, 3, 6, ..., 360) - 85 total:                       │
│    1. parse_grib_idx() → Parse JSON index                            │
│    2. Filter by member number (ECMWF-specific)                       │
│    3. build_idx_grib_mapping() → Build complete mapping (SLOW!)      │
│    4. Save to GCS: gs://bucket/ecmwf/{member}/*.parquet              │
│                                                                      │
│  Output: 85 parquet files per member in GCS (REUSABLE!)              │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
                  (Use for ANY future date!)
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│            STAGE 1: SCAN GRIB (3-5 minutes for test)                 │
│                                                                      │
│  Target date (e.g., 20251125):                                       │
│  GRIB Files → scan_grib() with member filter                         │
│         ↓                                                            │
│  fixed_ensemble_grib_tree() → Build hierarchical structure           │
│         ↓                                                            │
│  strip_datavar_chunks() → Deflated store                             │
│         ↓                                                            │
│  Output: deflated_ecmwf_grib_tree_store                              │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│       STAGE 2: IDX + GCS TEMPLATES (3-5 minutes, 85x faster!)        │
│                                                                      │
│  Target date (e.g., 20251125):                                       │
│  For each hour (0, 3, ..., 360) in batches - 85 total:               │
│    1. parse_grib_idx(target_date) → Fresh index (FAST! ~0.5s)        │
│       ↳ Gets: offset, length, uri for target date                    │
│    2. read GCS template(ref_date) → Structure template               │
│       ↳ Gets: varname, level, dims from reference date               │
│    3. map_from_index() → MERGE on "attrs"                            │
│       ↳ Combines: fresh positions + template structure               │
│         ↓                                                            │
│  Output: ecmwf_kind DataFrame (all 85 hours mapped)                  │
│          with CORRECT binary positions for target date!              │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│         STAGE 3: CREATE FINAL ZARR STORE (45-90 seconds)             │
│                                                                      │
│  deflated_store + ecmwf_kind + time_dims (85 steps)                  │
│         ↓                                                            │
│  prepare_zarr_store() → Initialize zarr structure                    │
│         ↓                                                            │
│  process_unique_groups() → Add coordinates & data vars               │
│         ↓                                                            │
│  create_parquet_file_fixed() → Save parquet                          │
│         ↓                                                            │
│  Output: {member}.parquet (xarray-compatible)                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## How scan_grib is Used

### Stage 1: Initial GRIB Scanning

The `scan_grib` function from kerchunk is the foundation for building the zarr reference structure:

```python
from kerchunk.grib2 import scan_grib, grib_tree
from kerchunk._grib_idx import strip_datavar_chunks

def ecmwf_filter_scan_grib(ecmwf_s3url):
    """
    Scan an ECMWF GRIB file, add ensemble information to the Zarr references,
    and return a list of modified groups along with an index mapping.
    """
    # Use anonymous access for S3
    storage_options = {"anon": True}

    # Core kerchunk scan_grib call
    esc_groups = scan_grib(ecmwf_s3url, storage_options=storage_options)
    print(f"Completed scan_grib for {ecmwf_s3url}, found {len(esc_groups)} messages")

    # Parse ECMWF JSON index file to map ensemble members
    idx_mapping, _ = ecmwf_idx_df_create_with_keys(ecmwf_s3url)

    # Enrich groups with ensemble member information
    modified_groups = []
    for i, group in enumerate(esc_groups):
        if i in idx_mapping:
            ens_key = idx_mapping[i]
            ens_number = int(ens_key.split('ens')[-1]) if 'ens' in ens_key else -1

            mod_group = copy.deepcopy(group)
            refs = mod_group['refs']

            # Add ensemble member metadata to root attributes
            if '.zattrs' in refs:
                root_attrs = json.loads(refs['.zattrs'])
                root_attrs['ensemble_member'] = ens_number
                root_attrs['ensemble_key'] = ens_key
                refs['.zattrs'] = json.dumps(root_attrs)

            # Add 'number' coordinate for ensemble dimension
            refs['number/.zarray'] = json.dumps({
                "chunks": [],
                "compressor": None,
                "dtype": "<i8",
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

            modified_groups.append(mod_group)

    return modified_groups, idx_mapping
```

### Building the Ensemble Tree

The `fixed_ensemble_grib_tree` function aggregates scanned groups into a hierarchical zarr structure:

```python
def fixed_ensemble_grib_tree(message_groups, remote_options=None, debug_output=False):
    """
    Build a hierarchical data model from scanned grib messages
    with proper ensemble support.
    """
    if remote_options is None:
        remote_options = {"anon": True}

    # Hierarchy filters
    filters = ["stepType", "typeOfLevel"]

    zarr_store = {'.zgroup': json.dumps({'zarr_format': 2})}
    aggregations = defaultdict(list)
    ensemble_dimensions = defaultdict(set)

    for msg_ind, group in enumerate(message_groups):
        # Extract ensemble member information
        ensemble_member = None
        if ".zattrs" in group["refs"]:
            root_attrs = json.loads(group["refs"][".zattrs"])
            if "ensemble_member" in root_attrs:
                ensemble_member = root_attrs["ensemble_member"]

        # Build path: variable/stepType/typeOfLevel
        vname = extract_variable_name(group)
        path_parts = [vname]
        for key in filters:
            attr_val = dattrs.get(f"GRIB_{key}")
            if attr_val:
                path_parts.append(attr_val)

        base_path = "/".join(path_parts)
        aggregations[base_path].append(group)

        if ensemble_member is not None:
            ensemble_dimensions[base_path].add(ensemble_member)

    # Use MultiZarrToZarr to combine groups
    for path, groups in aggregations.items():
        catdims = ["time", "step"]
        if len(ensemble_dimensions.get(path, set())) > 1:
            catdims.append("number")

        mzz = MultiZarrToZarr(
            groups,
            remote_options=remote_options,
            concat_dims=catdims,
            identical_dims=["longitude", "latitude"],
        )

        group_result = mzz.translate()
        for key, value in group_result["refs"].items():
            zarr_store[f"{path}/{key}"] = value

    return {"refs": zarr_store, "version": 1}
```

### The 85x Speed Improvement: Index-Based Processing

The key innovation is using lightweight `.index` files instead of full GRIB scanning:

```python
def parse_grib_index(idx_url: str, member_filter: Optional[str] = None) -> List[Dict]:
    """
    Parse ECMWF GRIB index file (JSON format) - ~0.5 seconds per file!
    vs ~45-90 seconds for scan_grib
    """
    fs = fsspec.filesystem("s3", anon=True)

    entries = []
    with fs.open(idx_url, 'r') as f:
        for line_num, line in enumerate(f):
            entry_data = json.loads(line.strip())

            # Extract ensemble member
            member_num = int(entry_data.get('number', 0))
            member = 'control' if member_num == 0 else f'ens{member_num:02d}'

            # Filter by member if specified
            if member_filter and member != member_filter:
                continue

            entry = {
                'byte_offset': entry_data['_offset'],
                'byte_length': entry_data['_length'],
                'variable': entry_data.get('param', ''),
                'level': entry_data.get('levtype', ''),
                'step': entry_data.get('step', '0'),
                'member': member,
            }
            entries.append(entry)

    return entries
```

---

## Performance Comparison

### Processing Times

| Method | Per Timestep | Full 85 Hours | Per Member | All 51 Members |
|--------|--------------|---------------|------------|----------------|
| **scan_grib** | 45-90 sec | 60-120 min | 75-140 min | 64-120 hours |
| **GIK (Index)** | 0.5-1 sec | 3-5 min | 8-10 min | 30-40 min |
| **Speedup** | **~85x** | **~20x** | **~10x** | **~100x** |

### Data Transfer

| Method | Data Downloaded | Network I/O |
|--------|-----------------|-------------|
| **scan_grib** | Full GRIB files (500MB-1GB each) | ~42-85 GB/member |
| **GIK (Index)** | Index files only (~KB each) | ~7 MB/member |
| **Reduction** | **99.99%** | **~6000x less** |

---

## Validation Results

### Comparison: ECMWF Open Data API vs GIK Stage 3

The `compare_ecmwf_opendata_vs_gik.py` script validates that both methods produce **byte-for-byte identical data**:

```
================================================================================
VERIFICATION SUMMARY
================================================================================

  Step  |  TP Max |Diff|  |  TP Correlation  |  TP Result
  ------|-----------------|------------------|------------
     6h |   0.0000000000  |    1.0000000000  |  EXACT MATCH
    12h |   0.0000000000  |    1.0000000000  |  EXACT MATCH
    24h |   0.0000000000  |    1.0000000000  |  EXACT MATCH

================================================================================
FINAL RESULT: ALL STEPS PASSED

Conclusion:
  - Stage 3 tp IS the original cumulative precipitation field
  - The GIK method is semantically equivalent to ECMWF Open Data
================================================================================
```

### Data Integrity Verification

| Aspect | Status | Evidence |
|--------|--------|----------|
| Data Integrity | **VERIFIED** | Zero differences in cumulative tp |
| Cumulative Semantics | **VERIFIED** | tp at 6h, 12h, 24h match exactly |
| Grid Alignment | **VERIFIED** | Shape (721, 1440) matches |
| Model Run Matching | **VERIFIED** | Same initialization time |

---

## Key Differences: GEFS vs ECMWF

| Aspect | GEFS | ECMWF |
|--------|------|-------|
| **Ensemble Members** | 30 (gep01-gep30) | 51 (control + ens01-ens50) |
| **Time Steps** | 81 (uniform 3h) | 85 (3h then 6h intervals) |
| **GRIB Structure** | Separate files per member | All members in single file |
| **Member Identification** | File name (gep01.grib2) | attrs field (number=1) |
| **File Size** | ~200MB per timestep | ~500MB-1GB per timestep |
| **S3 Bucket** | noaa-gefs-pds | ecmwf-forecasts |
| **Index Format** | Text (.idx) | JSON (.index) |

### ECMWF-Specific Handling

```python
# ECMWF uses JSON index files
def s3_parse_ecmwf_grib_idx(fs, basename, suffix="index"):
    fname = f"{basename.rsplit('.', 1)[0]}.{suffix}"

    with fs.open(fname, "r") as f:
        for idx, line in enumerate(f):
            # JSON format instead of text
            data = json.loads(line.strip())

            offset = data.get("_offset", 0)
            length = data.get("_length", 0)
            ens_number = data.get("number", -1)  # -1 for control

            # ... process entry
```

---

## Usage Examples

### Complete Three-Stage Processing

```python
from ecmwf_three_stage_multidate import process_single_date

# Process a single date through all three stages
success, output_dir = process_single_date(
    date_str='20251125',
    run='00',
    max_members=None  # Process all 51 members
)

# Output: ecmwf_three_stage_20251125_00z/
#   ├── stage2_control_merged.parquet
#   ├── stage2_ens_01_merged.parquet
#   ├── ...
#   ├── stage3_control_final.parquet
#   └── stage3_ens_50_final.parquet
```

### Reading Final Parquet with xarray

```python
import xarray as xr
import fsspec
import pandas as pd

# Read parquet to zarr store
def read_parquet_to_refs(parquet_file):
    df = pd.read_parquet(parquet_file)
    zstore = {}
    for _, row in df.iterrows():
        key, value = row['key'], row['value']
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        zstore[key] = value
    return zstore

# Open with xarray
zstore = read_parquet_to_refs("stage3_control_final.parquet")
fs = fsspec.filesystem("reference", fo=zstore,
                       remote_protocol='s3',
                       remote_options={'anon': True})
mapper = fs.get_mapper("")
dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)

# Access 2m temperature
t2m = dt['/2t/instant/surface'].ds['2t']

# Regional subset
europe = t2m.sel(
    latitude=slice(70, 35),
    longitude=slice(-10, 40),
    step=24
).compute()
```

### Extracting All 85 Timesteps

```bash
# Using read_stage3_aifs_all_timesteps.py
python read_stage3_aifs_all_timesteps.py \
    --member control \
    --variable 2t \
    --output control_2t_all85.npz
```

```python
# Load extracted data
import numpy as np
data = np.load('control_2t_all85.npz')

print(data['data'].shape)        # (85, 721, 1440)
print(data['forecast_hours'])    # [0, 3, 6, ..., 360]
print(data['latitude'].shape)    # (721,)
print(data['longitude'].shape)   # (1440,)
```

---






# ECMWF Ensemble Processing Pipeline

## Overview

Complete pipeline for processing ECMWF (European Centre for Medium-Range Weather Forecasts) ensemble forecast data using the **grib-index-kerchunk method** for efficient GRIB data access without full file scanning.

## Key Features

- **51 ensemble members**: 1 control + 50 perturbed members
- **85 forecast timesteps**: 0-144h at 3h intervals (49 steps) + 150-360h at 6h intervals (36 steps)
- **High resolution**: 0.25° (~25 km)
- **Fast daily processing**: Reuses mapping templates from one-time preprocessing

## The Two-Step Architecture

This pipeline uses a **two-step process** that separates expensive one-time preprocessing from fast daily processing:

### Step 1: One-Time Expensive Preprocessing
**Purpose**: Create reusable parquet mapping files that describe the GRIB data structure

**Scripts**:
- `ecmwf_index_preprocessing.py` - Creates GCS parquet templates from scanning GRIB files
- `ecmwf_ensemble_par_creator_v2.py` - Downloads and processes GCS parquet files (alternative approach)

**When to run**: ONCE per ensemble member to create mapping templates

**What it creates**: Parquet files in GCS at `gs://bucket/ecmwf/{member}/ecmwf-time-{date}-{member}-rt{hour}.parquet`

**Why expensive**: Scans actual GRIB files using `scan_grib` and `build_idx_grib_mapping` to build complete index mappings for all 85 forecast hours

**Key benefit**: These parquet files can be reused across different dates with the same ECMWF structure

#### Usage:
```bash
# Process single member
python ecmwf_index_preprocessing.py --date 20240529 --member ens01 --bucket gik-ecmwf-aws-tf

# Process all 51 members
python ecmwf_index_preprocessing.py --date 20240529 --bucket gik-ecmwf-aws-tf --all-members
```

---

### Step 2: Fast Daily Processing
**Purpose**: Efficiently process new forecast dates by scanning each GRIB file once and extracting all ensemble members

**Script**: `ecmwf_ensemble_par_creator_efficient.py`

**When to run**: For each new forecast date you want to process

**Processing approach**:
1. Scans each GRIB file ONCE using `ecmwf_filter_scan_grib`
2. Extracts all 51 ensemble members simultaneously using `fixed_ensemble_grib_tree`
3. Creates comprehensive parquet with all ensemble data
4. Optionally extracts individual member-specific parquet files

**Efficiency gain**: Instead of scanning files 51 times (once per member), scans each file just ONCE

#### Usage:
```bash
# Configure in the script:
# - date_str: Date to process (e.g., '20251103')
# - run: Run hour (e.g., '00', '12')
# - target_members: List of ensemble members to extract

python ecmwf_ensemble_par_creator_efficient.py
```

**Output structure**:
```
ecmwf_{date}_{run}_efficient/
├── comprehensive/
│   └── ecmwf_{date}_{run}z_ensemble_all.parquet  # All members
└── members/
    ├── control/
    │   └── control.parquet
    ├── ens_01/
    │   └── ens_01.parquet
    ├── ens_02/
    │   └── ens_02.parquet
    └── ...
```

---

### Step 3: Generate PKL Files for AIFS
**Purpose**: Convert ensemble member parquet files to PKL format for AI weather model input

**Script**: `read_par_manifest_array/test_levels/aifs-etl.py`

**When to run**: After Step 2 completes to prepare data for AI-FS (Artificial Intelligence Forecasting System)

**What it does**:
- Reads parquet files with hybrid references (base64 + S3 byte ranges)
- Extracts meteorological variables at multiple pressure levels
- Handles GRIB2 decoding using cfgrib/eccodes
- Uses obstore for fast S3 data fetching
- Converts geopotential height (gh) to geopotential (z)
- Saves as PKL file ready for AI model input

**Variables extracted**:
- **Surface**: 10u, 10v, 2t, 2d, msl, sp, skt, tcw
- **Fixed fields**: lsm
- **Pressure levels**: gh, t, u, v, w, q at 13 levels (1000-50 hPa)

#### Usage:
```bash
# Edit parquet file path in script:
# parquet_file = "ecmwf_20250728_18_efficient/members/ens_01/ens_01.parquet"

python read_par_manifest_array/test_levels/aifs-etl.py
```

**Output**:
- `ecmwf_pkl_from_parquet/input_state_member_001_phase1.pkl`

---

## Complete Workflow Example

```bash
# 1. One-time preprocessing (run once)
python ecmwf_index_preprocessing.py \
    --date 20240529 \
    --member ens01 \
    --bucket gik-ecmwf-aws-tf

# 2. Fast daily processing (run for each new date)
# Edit date_str in script to target date
python ecmwf_ensemble_par_creator_efficient.py

# 3. Generate PKL for AIFS (optional)
# Edit parquet_file path in script
python read_par_manifest_array/test_levels/aifs-etl.py
```

---

## How Fast Daily Processing Works

1. **Efficient single-pass scanning**: Uses `ecmwf_filter_scan_grib` to scan each GRIB file once
2. **Simultaneous member extraction**: `fixed_ensemble_grib_tree` processes all 51 members together
3. **Avoids redundant scanning**: Old approach required 51 × 85 = 4,335 file scans; new approach requires only 85 scans
4. **Result**: **51× faster** processing for daily runs!

---

## Data Source

ECMWF forecast data is accessed from AWS S3:
- **Bucket**: `s3://ecmwf-forecasts/`
- **Format**: `{date}/{run}z/ifs/0p25/enfo/{date}{run}0000-{hour}h-enfo-ef.grib2`
- **Access**: Anonymous (no credentials required)

Example URL:
```
s3://ecmwf-forecasts/20240529/00z/ifs/0p25/enfo/20240529000000-0h-enfo-ef.grib2
```

---

## Technical Details

### ECMWF Ensemble Configuration
- **Control member**: 1 (number = -1 in code)
- **Perturbed members**: 50 (number = 1-50)
- **Total members**: 51

### Forecast Hours
- **0-144h**: 3-hourly intervals (49 timesteps)
- **150-360h**: 6-hourly intervals (36 timesteps)
- **Total**: 85 forecast hours

### Variables in Parquet
- Surface fields: u10, v10, t2m, d2m, msl, sp, skt, tcw, lsm
- Pressure levels: gh, t, u, v, w, q
- Levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

---

## Environment Setup

### Coiled Environment (Recommended for large-scale processing)

```bash
# Start Coiled notebook
coiled notebook start --name dask-thresholds --vm-type n2-standard-2 \
    --software itt-jupyter-env-v20250318 --workspace=geosfm

# Install dependencies in notebook
pip install "kerchunk<=0.2.7"
pip install "zarr<=3.0"
```

Upload the coiled data service account and run the processing scripts.

---

## References

- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [ECMWF on AWS](https://registry.opendata.aws/ecmwf-forecasts/)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [Zarr v3 Specification](https://zarr.readthedocs.io/)

## Acknowledgements

This work was funded in part by:

1. Hazard modeling, impact estimation, climate storylines for event catalogue
   on drought and flood disasters in the Eastern Africa (E4DRR) project.
   https://icpac-igad.github.io/e4drr/ United Nations | Complex Risk Analytics
   Fund (CRAF'd)
2. The Strengthening Early Warning Systems for Anticipatory Action (SEWAA)
   Project. https://cgan.icpac.net/
