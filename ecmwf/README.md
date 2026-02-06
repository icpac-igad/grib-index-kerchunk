# ECMWF GIK-cGAN Integration

This folder contains standalone scripts for processing ECMWF ensemble data using the Grib-Index-Kerchunk (GIK) method and preparing input data for cGAN rainfall downscaling.

**Source Repository:** [grib-index-kerchunk](https://github.com/icpac-igad/grib-index-kerchunk)
**Source Commit:** `a487612ece68f2ce7e0c278a2fa1d5170349877e`

## Overview

The workflow consists of two main phases:

```
Phase 1: Create Parquet Reference Files (GIK Pipeline)
    ECMWF S3 GRIB -> Index Processing + Template -> Stage3 Parquet Files

Phase 2: Stream cGAN Variables
    Stage3 Parquet -> Parallel S3 Fetch -> GRIB Decode -> NetCDF for cGAN
```

## Directory Structure

```
cgan_ecmwf/
├── README.md                           # This file
├── ECMWF_cGAN_INFERENCE_WORKFLOW.md    # Detailed workflow documentation
├── GRIBBERISH_EXPERIMENT_REPORT.md     # scan_grib replacement experiment
├── run_ecmwf_tutorial.py               # Main entry point for Phase 1
├── stream_cgan_variables.py            # Phase 2: Local data streaming
├── stream_cgan_variables_coiled_simple.py  # Phase 2: Coiled parallel streaming
├── test_gribberish_vs_scangrib.py      # Benchmark: .index vs scan_grib
└── gik_ecmwf/                          # Core GIK processing modules
    ├── __init__.py
    ├── ecmwf_util.py                   # Core utilities and variable definitions
    ├── ecmwf_ensemble_par_creator_efficient.py  # Stage 1: GRIB scanning (legacy)
    ├── ecmwf_three_stage_multidate.py  # Three-stage pipeline orchestration
    ├── ecmwf_index_processor.py        # Stage 2: Index-based processing
    └── utils_ecmwf_step1_scangrib.py   # GRIB scanning utilities
```

## Installation

```bash
# Core dependencies
pip install kerchunk zarr xarray pandas numpy fsspec s3fs requests pyarrow

# GRIB processing
pip install cfgrib eccodes gribberish

# NetCDF output
pip install netCDF4 h5netcdf

# Optional: For Dask/Coiled parallelization
pip install dask distributed coiled gcsfs
```

## GCS Bucket Configuration

The pipeline uses two GCS buckets:

| Bucket | Purpose | Contains |
|--------|---------|----------|
| `gik-ecmwf-aws-tf` | **Parquet output** (for Coiled streaming) | `run_par_ecmwf/YYYYMMDD_HHz/stage3_*_final.parquet` |
| `gik-fmrc` | Template storage (internal) | `v2ecmwf_fmrc/ens_*/ecmwf-*.par` |

**Important:** When running streaming scripts, always use `gik-ecmwf-aws-tf` as the GCS parquet bucket:
```bash
--gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/YYYYMMDD_00z
```

Environment variables (optional, set in `.env` file):
```bash
GCS_BUCKET=gik-ecmwf-aws-tf          # Parquet upload bucket
GCS_PARQUET_PREFIX=run_par_ecmwf      # Prefix path within bucket
GCS_SERVICE_ACCOUNT_FILE=coiled-data.json  # Service account key (optional)
```

## Quick Start

### Phase 1: Create Parquet Reference Files

```bash
cd cgan_ecmwf/

# Recommended: Template fast-path (skips GRIB scanning, ~31 min)
python run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan

# Upload parquets to GCS (needed for Coiled streaming)
python run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --upload-gcs

# Legacy: Full pipeline with GRIB scanning (~110 min)
python run_ecmwf_tutorial.py --date 20260206 --run-stage1

# With limited members for testing
python run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --max-members 5
```

**Output:** `ecmwf_three_stage_YYYYMMDD_HHz/` directory with `stage3_*.parquet` files

### Phase 2: Stream cGAN Variables

**Local streaming** (no cloud cluster needed):
```bash
# Stream 3 members for quick test
python stream_cgan_variables.py \
    --parquet-dir ecmwf_three_stage_20260206_00z \
    --max-members 3

# Stream all 51 members
python stream_cgan_variables.py \
    --parquet-dir ecmwf_three_stage_20260206_00z

# Custom timesteps
python stream_cgan_variables.py \
    --parquet-dir ecmwf_three_stage_20260206_00z \
    --steps "30,36,42,48,54,60"
```

**Coiled parallel streaming** (requires GCS parquets + Coiled account):
```bash
# Test mode (3 members, 3 workers)
python stream_cgan_variables_coiled_simple.py --test \
    --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z

# Full production run
python stream_cgan_variables_coiled_simple.py \
    --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z \
    --n-workers 20
```

**Output:** `cgan_output/IFS_YYYYMMDD_HHZ_cgan.nc`

## Script Details

### 1. `run_ecmwf_tutorial.py`

Main entry point for the GIK three-stage pipeline.

**Stages:**
1. **Stage 1** (~1 min with `--skip-grib-scan`, ~73 min legacy): Build zarr structure from template or scan GRIB files
2. **Stage 2** (~30 min): Index-based processing using pre-built templates, 51 members sequential
3. **Stage 3** (~2 sec): Create final zarr-compatible parquet files

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--date` | 20260106 | Target date (YYYYMMDD) |
| `--run` | 00 | Model run hour (00 or 12) |
| `--skip-grib-scan` | False | **Recommended.** Skip GRIB scanning, build Stage 1 from template (~1 min vs ~73 min) |
| `--run-stage1` | False | Legacy: Run Stage 1 GRIB scanning |
| `--max-members` | None | Limit ensemble members |
| `--hours` | "0,3" | Forecast hours for Stage 1 |
| `--upload-gcs` | False | Upload final parquets to GCS |
| `--gcs-bucket` | gik-ecmwf-aws-tf | GCS bucket for upload |
| `--gcs-prefix` | run_par_ecmwf | GCS prefix path |

### 2. `stream_cgan_variables.py`

Local streaming: extracts 12 cGAN input variables with ensemble mean and standard deviation.

**Variables Extracted:**
| Variable | ECMWF Name | Description |
|----------|------------|-------------|
| tp | tp | Total Precipitation |
| t2m | 2t | 2-meter Temperature |
| sp | sp | Surface Pressure |
| ssr | ssr | Surface Solar Radiation |
| ssrd | ssrd | Surface Solar Radiation Downwards |
| tcw | tcw | Total Cloud Water |
| tcwv | tcwv | Total Column Water Vapour |
| tcc | tcc | Total Cloud Cover |
| u700 | u | U-wind at 700 hPa |
| v700 | v | V-wind at 700 hPa |
| sf | sf | Snowfall |
| ro | ro | Runoff |

### 3. `stream_cgan_variables_coiled_simple.py`

Coiled parallel streaming: same as above but distributes work across cloud workers.

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--gcs-parquet-path` | gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z | GCS parquet path |
| `--parquet-dir` | None | Local parquet directory (alternative to GCS) |
| `--n-workers` | 5 | Number of Coiled workers |
| `--members-per-batch` | 1 | Members per worker batch |
| `--max-members` | None | Limit ensemble members |
| `--workspace` | gcp-sewaa-nka | Coiled workspace name |
| `--test` | False | Test mode: 3 members, 3 workers |

### 4. `gik_ecmwf/` Module

Core processing modules for the GIK pipeline.

| File | Purpose |
|------|---------|
| `ecmwf_util.py` | Variable definitions, forecast hours, grid specs |
| `ecmwf_ensemble_par_creator_efficient.py` | Stage 1 GRIB scanning (legacy) |
| `ecmwf_three_stage_multidate.py` | Pipeline orchestration + template fast-path |
| `ecmwf_index_processor.py` | Stage 2 fast index processing |
| `utils_ecmwf_step1_scangrib.py` | GRIB scanning utilities |

## Output Format

### NetCDF File Structure

```
IFS_20260206_00Z_cgan.nc
├── Dimensions:
│   ├── time: 1 (initialization)
│   ├── valid_time: 9 (forecast hours)
│   ├── latitude: 157 (ICPAC region)
│   └── longitude: 145
├── Coordinates:
│   ├── time: 2026-02-06T00:00:00
│   ├── valid_time: [36h, 39h, ..., 60h]
│   ├── latitude: [25.0, ..., -14.0]
│   └── longitude: [19.0, ..., 55.0]
└── Variables (for each of 12 fields):
    ├── {var}_ensemble_mean: (1, 9, 157, 145)
    └── {var}_ensemble_standard_deviation: (1, 9, 157, 145)
```

## Performance

### With `--skip-grib-scan` (recommended)

| Stage | Operation | Time | Notes |
|-------|-----------|------|-------|
| 1 | Template loading + validation | ~1 min | Replaces 73 min GRIB scanning |
| 2 | Index processing (51 members) | ~30 min | Sequential; parallelizable |
| 3 | Final parquet merge | ~2 sec | Fast dict merge |
| **Total** | | **~31 min** | **3.6x faster than legacy** |

### Legacy (with `--run-stage1`)

| Stage | Operation | Time | Notes |
|-------|-----------|------|-------|
| 1 | GRIB scanning (kerchunk) | ~73 min | S3 byte-range reads |
| 2 | Index processing (51 members) | ~36 min | Sequential |
| 3 | Final parquet merge | ~5 sec | |
| **Total** | | **~110 min** | |

### Data Streaming (Phase 2)

| Method | Time (51 members, 9 steps) | Notes |
|--------|---------------------------|-------|
| Local (`stream_cgan_variables.py`) | ~4 hours | Sequential S3 fetch |
| Coiled (`coiled_simple.py`, 20 workers) | ~15-30 min | Parallel across workers |

### Bottleneck Analysis

After the `--skip-grib-scan` optimization, **Stage 2 is the remaining bottleneck** at ~30 min (96% of total). Each of 51 members is processed sequentially (~35 sec each). Future Phase 2 parallelization with ProcessPoolExecutor (8 workers) would reduce this to ~5 min.

See `GRIBBERISH_EXPERIMENT_REPORT.md` for the full experiment report and risk analysis.

## Troubleshooting

### Common Issues

**Wrong GCS bucket for streaming:**
```
OSError: Forbidden: ... does not have storage.objects.list access
```
Make sure you use `gik-ecmwf-aws-tf` (not `gik-fmrc`) for parquet access:
```bash
--gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/YYYYMMDD_00z
```

**S3 Access Denied:**
```bash
export AWS_NO_SIGN_REQUEST=YES
```

**GRIB File Not Found:**
- Check date format (YYYYMMDD)
- Verify data exists on ECMWF S3: `s3://ecmwf-forecasts/`

**Memory Issues:**
- Reduce `--max-members`
- Process fewer timesteps with `--hours`

**Import Errors:**
- Ensure gik_ecmwf/ directory is in the same folder
- Check all dependencies are installed

**Parquets not on GCS after pipeline run:**
- The `--upload-gcs` flag is required for GCS upload (not automatic)
- Run: `python run_ecmwf_tutorial.py --date YYYYMMDD --skip-grib-scan --upload-gcs`

## References

- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [gribberish](https://github.com/mpiannucci/gribberish) - Fast GRIB decoding
- [Coiled](https://coiled.io/) - Managed Dask clusters

## License

This code is part of the ICPAC climate services infrastructure.
