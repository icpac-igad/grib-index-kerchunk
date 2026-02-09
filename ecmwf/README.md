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
├── plot_cgan_maps.py                   # Visualization: 4x3 panel maps
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

### Using `uv` (recommended)

Both main scripts include [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so `uv` will automatically create an isolated environment and install dependencies:

```bash
# No manual install needed — just run:
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan
uv run stream_cgan_variables.py --parquet-dir ecmwf_three_stage_20260206_00z
```

### Using `pip`

```bash
# Core dependencies
pip install kerchunk "zarr<3" xarray pandas numpy fsspec s3fs requests pyarrow

# GRIB processing
pip install cfgrib eccodes gribberish

# NetCDF output
pip install netCDF4 h5netcdf

# Environment file support
pip install python-dotenv

# Optional: For GCS upload (--upload-gcs)
pip install gcsfs

# Optional: For Dask/Coiled parallelization
pip install dask distributed coiled gcsfs
```

## Credentials & Service Account Requirements

### AWS S3 (ECMWF forecast data)

**No credentials needed.** Both scripts use anonymous S3 access to read ECMWF public forecast data:

```python
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
```

This is set automatically — no AWS account, keys, or configuration required.

### Google Cloud Storage (GCS)

GCS credentials are **only required** when using the `--upload-gcs` flag in `run_ecmwf_tutorial.py` (to upload parquet files for Coiled workers) or when using `stream_cgan_variables_coiled_simple.py`.

| Script | GCS Required? | When? |
|--------|--------------|-------|
| `run_ecmwf_tutorial.py` | Only with `--upload-gcs` | Uploading parquets to GCS |
| `stream_cgan_variables.py` | **No** | Reads local parquets, fetches from S3 |
| `stream_cgan_variables_coiled_simple.py` | **Yes** | Reads parquets from GCS |

**Setting up GCS credentials:**

1. **Service account JSON file** (recommended for CI/automation):
   ```bash
   # Set in .env file or environment
   GCS_SERVICE_ACCOUNT_FILE=coiled-data.json
   ```
   The file must have `storage.objects.create` and `storage.objects.list` permissions on the target bucket.

2. **Application Default Credentials** (recommended for local development):
   ```bash
   gcloud auth application-default login
   ```
   If no service account file is found, scripts automatically fall back to ADC.

3. **Environment variables** (copy `.env.example` to `.env`):
   ```bash
   cp .env.example .env
   # Edit .env with your values:
   #   GCS_BUCKET=gik-ecmwf-aws-tf
   #   GCS_PARQUET_PREFIX=run_par_ecmwf
   #   GCS_SERVICE_ACCOUNT_FILE=coiled-data.json
   ```

### Summary: What you need for each workflow

| Workflow | AWS Creds | GCS Creds | Service Account File |
|----------|-----------|-----------|---------------------|
| Phase 1: Create parquets locally | None | None | None |
| Phase 1 + upload to GCS | None | Required | Optional (ADC works) |
| Phase 2: Local streaming | None | None | None |
| Phase 2: Coiled streaming | None | Required | Recommended |

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

Environment variables (set in `.env` file — see `.env.example`):
```bash
GCS_BUCKET=gik-ecmwf-aws-tf              # Parquet bucket
GCS_PARQUET_PREFIX=run_par_ecmwf          # Prefix path within bucket
GCS_SERVICE_ACCOUNT_FILE=coiled-data.json # Service account JSON (falls back to ADC)
```

## Quick Start

### Phase 1: Create Parquet Reference Files

```bash
cd cgan_ecmwf/

# Recommended: Template fast-path + parallel Stage 2 (~10.5 min)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 8

# Upload parquets to GCS (needed for Coiled streaming — requires GCS credentials)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 8 --upload-gcs

# Sequential processing (no parallelization, ~31 min)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 1

# Legacy: Full pipeline with GRIB scanning (~110 min)
uv run run_ecmwf_tutorial.py --date 20260206 --run-stage1

# With limited members for testing
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --max-members 5
```

**Output:** `ecmwf_three_stage_YYYYMMDD_HHz/` directory with `stage3_*.parquet` files

### Phase 2: Stream cGAN Variables

**GCS streaming** (reads parquets from GCS, requires `.env` with service account):
```bash
# Stream by date — builds GCS path from .env (GCS_BUCKET/GCS_PARQUET_PREFIX)
uv run stream_cgan_variables.py --date 20260207

# Explicit GCS path
uv run stream_cgan_variables.py \
    --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260207_00z

# Quick test: 1 member, 1 step
uv run stream_cgan_variables.py --date 20260207 --max-members 1 --steps "48"

# Custom timesteps with more parallelism
uv run stream_cgan_variables.py --date 20260207 \
    --steps "36,39,42,45,48,51,54,57,60" --parallel-fetches 8
```

**Local streaming** (reads parquets from local directory, no credentials required):
```bash
uv run stream_cgan_variables.py \
    --parquet-dir ecmwf_three_stage_20260206_00z \
    --parallel-fetches 8
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
2. **Stage 2** (~8.8 min with 8 workers, ~30 min sequential): Index-based processing using pre-built templates, 51 members parallelized via `ProcessPoolExecutor`
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
| `--parallel-workers` | 8 | Number of parallel workers for Stage 2 (set to 1 for sequential) |
| `--upload-gcs` | False | Upload final parquets to GCS |
| `--gcs-bucket` | gik-ecmwf-aws-tf | GCS bucket for upload |
| `--gcs-prefix` | run_par_ecmwf | GCS prefix path |

### 2. `stream_cgan_variables.py`

Local or GCS streaming with parallel S3 fetches: extracts 12 cGAN input variables with ensemble mean and standard deviation. Uses `ThreadPoolExecutor` at two levels (members + timestep fetches) for ~10x speedup over sequential. Reads GCS/bucket config from `.env` via `python-dotenv`.

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--date` | None | Date (YYYYMMDD). Builds GCS path from `.env` bucket/prefix |
| `--run` | 00 | Model run hour |
| `--gcs-parquet-path` | None | Explicit GCS path (overrides `--parquet-dir` and `--date`) |
| `--parquet-dir` | ecmwf_three_stage_20260203_00z | Local directory (fallback when no GCS path) |
| `--steps` | 36,39,42,45,48,51,54,57,60 | Comma-separated forecast hours |
| `--max-members` | 51 | Maximum number of ensemble members |
| `--parallel-fetches` | 8 | Number of concurrent member streams |
| `--output-dir` | cgan_output | Output directory for NetCDF file |

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

### With `--skip-grib-scan --parallel-workers 8` (recommended)

| Stage | Operation | Time | Notes |
|-------|-----------|------|-------|
| 1 | Template loading + validation | ~1.1 min | Replaces 73 min GRIB scanning |
| 2 | Index processing (51 members, 8 workers) | ~8.8 min | `ProcessPoolExecutor` parallelization |
| 3 | Final parquet merge | ~3.8 sec | Fast dict merge |
| **Total** | | **~10.5 min** | **10.5x faster than legacy** |

### With `--skip-grib-scan --parallel-workers 1` (sequential)

| Stage | Operation | Time | Notes |
|-------|-----------|------|-------|
| 1 | Template loading + validation | ~1 min | Replaces 73 min GRIB scanning |
| 2 | Index processing (51 members) | ~30 min | Sequential, single-threaded |
| 3 | Final parquet merge | ~2 sec | Fast dict merge |
| **Total** | | **~31 min** | **3.6x faster than legacy** |

### Legacy (with `--run-stage1`)

| Stage | Operation | Time | Notes |
|-------|-----------|------|-------|
| 1 | GRIB scanning (kerchunk) | ~73 min | S3 byte-range reads |
| 2 | Index processing (51 members) | ~36 min | Sequential |
| 3 | Final parquet merge | ~5 sec | |
| **Total** | | **~110 min** | |

### Stage 2 Parallelization Details

Stage 2 processes 51 ensemble members (control + ens01–ens50). Each member is independent — it fetches `.index` files from ECMWF S3, parses byte offsets, and merges with the local template to produce a parquet file with ~6,685 references.

With `--parallel-workers N`, Stage 2 uses Python's `ProcessPoolExecutor` to process N members concurrently:

```bash
# 8 workers (default, recommended for most machines)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 8

# 16 workers (for machines with more cores and bandwidth)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 16

# Sequential fallback (no parallelization)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 1
```

The bottleneck is S3 I/O (fetching 85 `.index` files per member), so the optimal worker count depends on network bandwidth rather than CPU cores. With 8 workers, Stage 2 drops from ~30 min to ~8.8 min (3.4x speedup).

### Data Streaming (Phase 2)

| Method | Time (51 members, 9 steps) | Notes |
|--------|---------------------------|-------|
| Local sequential | ~4 hours | `--parallel-fetches 1` |
| **Local threaded (default)** | **~24 min** | **`--parallel-fetches 8` (ThreadPoolExecutor)** |
| Coiled (`coiled_simple.py`, 20 workers) | ~15-30 min | Distributed cloud VMs |

### Local Streaming Parallelization Details

`stream_cgan_variables.py` uses two levels of `ThreadPoolExecutor` parallelism:

1. **Member level**: `--parallel-fetches N` members are streamed concurrently (default: 8)
2. **Fetch level**: Within each member, all 9 timestep S3 fetches run in parallel threads

```bash
# Default: 8 concurrent member streams (~24 min for 51 members)
uv run stream_cgan_variables.py \
    --parquet-dir ecmwf_three_stage_20260206_00z \
    --parallel-fetches 8

# Sequential fallback (~4 hours)
uv run stream_cgan_variables.py \
    --parquet-dir ecmwf_three_stage_20260206_00z \
    --parallel-fetches 1
```

Threads are used (not processes) because the bottleneck is S3 I/O latency — Python's GIL releases during network I/O, so threads achieve full parallelism with minimal memory overhead. This makes it effective even on small machines (2 CPUs, 8 GB RAM).

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
- Run: `uv run run_ecmwf_tutorial.py --date YYYYMMDD --skip-grib-scan --upload-gcs`

## References

- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [gribberish](https://github.com/mpiannucci/gribberish) - Fast GRIB decoding
- [Coiled](https://coiled.io/) - Managed Dask clusters

## License

This code is part of the ICPAC climate services infrastructure.
