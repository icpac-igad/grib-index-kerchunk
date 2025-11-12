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
