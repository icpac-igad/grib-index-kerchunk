# GEFS Data Processing Pipeline

## Overview
This repository contains a complete GEFS (Global Ensemble Forecast System) data processing pipeline that converts GRIB data to Zarr format and generates ensemble statistics.

## Processing Workflow

### Step 1: Initial Ensemble Processing
Start with the full ensemble processing script:

```bash
python run_day_gefs_ensemble_full.py
```

This script handles the initial preprocessing and creates parquet mapping files that describe the GRIB data structure.

### Step 2: Individual Ensemble Member Processing
Use the single ensemble member processing script with a for loop to process all 30 ensemble members:

```bash
python run_single_gefs_to_zarr.py
```

**Example for loop to process all ensemble members:**
```bash
for i in $(seq -f "%02g" 1 30); do
    python run_single_gefs_to_zarr.py 20250909 18 gep$i
done
```

This converts the parquet files into actual streamed or downloaded files in local Zarr format.

### Step 3: Ensemble Statistics Generation
After processing all ensemble members, generate NetCDF files with ensemble mean and standard deviation:

```bash
python process_ensemble_by_variable.py
```

This script creates NetCDF files and calculates ensemble statistics (mean and standard deviation).

## Environment Setup

### Option 1: Coiled Environment (Cloud-based)
For cloud processing, you can use the Coiled environment setup:

```bash
coiled notebook start --name dask-thresholds --vm-type n2-standard-2 --software gik-zarr2 --workspace=geosfm
```

Upload the required files:
1. gefs_utils.py 
2. run_day_gefs_ensemble_full.py 
3. run_gefs_24h_accumulation.py 
4. ea_ghcf_simple.geojson

### Option 2: Local Micromamba Environment (Linux)

#### Install Micromamba (Linux)
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
micromamba self-update -c conda-forge
```

#### Create Environment
```bash
micromamba create --name gik-zarrv2 --file vz_gik_env.yaml
micromamba activate gik-zarrv2
```

### Option 3: Complete Environment Setup Guide
For detailed environment setup instructions covering Linux, Windows, and Coiled cloud environments, see the comprehensive documentation in the [docs/micromamba-env-setup.md](docs/micromamba-env-setup.md) file.

## GEFS Processing Architecture: Technical Details

### The Two-Step Process

#### Step 1: One-Time Expensive Preprocessing
- **Purpose**: Create parquet mapping files that describe the GRIB data structure
- **When**: Run ONCE per ensemble member to create the mapping templates
- **What it creates**: Parquet files in GCS at `gs://bucket/gefs/{member}/gefs-time-{date}-{member}-rt{hour}.parquet`
- **Why expensive**: Scans actual GRIB files to build index mappings
- **Reusability**: These parquet files can be reused across different dates!

#### Step 2: Fast Daily Processing
- **Purpose**: Process new date's GEFS data using existing parquet structure + new GRIB index
- **When**: Run daily for each new forecast date
- **How it works**:
  1. Uses existing parquet mapping files from GCS (created in Step 1)
  2. Reads GRIB index (.idx) files directly from new date's S3 data
  3. Combines existing mapping structure with new index data
  4. Avoids expensive GRIB scanning by reusing parquet mappings



