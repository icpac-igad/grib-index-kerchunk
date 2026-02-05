# ECMWF GIK-cGAN Integration

This folder contains standalone scripts for processing ECMWF ensemble data using the Grib-Index-Kerchunk (GIK) method and preparing input data for cGAN rainfall downscaling.

**Source Repository:** [grib-index-kerchunk](https://github.com/icpac-igad/grib-index-kerchunk)
**Source Commit:** `a487612ece68f2ce7e0c278a2fa1d5170349877e`

## Overview

The workflow consists of two main phases:

```
Phase 1: Create Parquet Reference Files (GIK Pipeline)
    ECMWF S3 GRIB → Scan GRIB → Index Processing → Stage3 Parquet Files

Phase 2: Stream cGAN Variables
    Stage3 Parquet → Parallel S3 Fetch → GRIB Decode → NetCDF for cGAN
```

## Directory Structure

```
cgan_ecmwf/
├── README.md                           # This file
├── ECMWF_cGAN_INFERENCE_WORKFLOW.md    # Detailed workflow documentation
├── run_ecmwf_tutorial.py               # Main entry point for Phase 1
├── stream_cgan_variables.py            # Phase 2: Data streaming for cGAN
└── gik_ecmwf/                          # Core GIK processing modules
    ├── __init__.py
    ├── ecmwf_util.py                   # Core utilities and variable definitions
    ├── ecmwf_ensemble_par_creator_efficient.py  # Stage 1: GRIB scanning
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
pip install dask distributed coiled
```

## Quick Start

### Phase 1: Create Parquet Reference Files

```bash
cd cgan_ecmwf/

# Run full pipeline for a specific date (takes ~90 minutes for Stage 1)
python run_ecmwf_tutorial.py --date 20260203 --run-stage1

# With limited members for testing (faster)
python run_ecmwf_tutorial.py --date 20260203 --run-stage1 --max-members 5

# Skip Stage 1 if parquet files already exist
python run_ecmwf_tutorial.py --date 20260203
```

**Output:** `ecmwf_three_stage_YYYYMMDD_HHz/` directory with `stage3_*.parquet` files

### Phase 2: Stream cGAN Variables

```bash
# Stream all 51 members (takes ~4 hours)
python stream_cgan_variables.py --parquet-dir ecmwf_three_stage_20260203_00z

# With limited members for testing
python stream_cgan_variables.py --parquet-dir ecmwf_three_stage_20260203_00z --max-members 5

# Custom timesteps
python stream_cgan_variables.py --steps "30,36,42,48,54,60"
```

**Output:** `cgan_output/IFS_YYYYMMDD_HHZ_cgan.nc` (~10 MB)

## Script Details

### 1. `run_ecmwf_tutorial.py`

Main entry point for the GIK three-stage pipeline.

**Purpose:** Creates parquet reference files from ECMWF GRIB data on S3.

**Stages:**
1. **Stage 1** (~90 min): Scan GRIB files using kerchunk to build hierarchical zarr structure
2. **Stage 2** (~5 min): Fast index-based processing using pre-built templates
3. **Stage 3** (~2 min): Create final zarr-compatible parquet files

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--date` | 20260106 | Target date (YYYYMMDD) |
| `--run` | 00 | Model run hour (00 or 12) |
| `--run-stage1` | False | Run Stage 1 GRIB scanning |
| `--max-members` | None | Limit ensemble members |
| `--hours` | "0,3" | Forecast hours for Stage 1 |

### 2. `stream_cgan_variables.py`

Streams ECMWF ensemble data and creates NetCDF files for cGAN inference.

**Purpose:** Extracts 12 cGAN input variables with ensemble mean and standard deviation.

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

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--parquet-dir` | ecmwf_three_stage_20260203_00z | Stage3 parquet directory |
| `--steps` | 36,39,42,45,48,51,54,57,60 | Forecast hours |
| `--max-members` | 51 | Number of ensemble members |
| `--output-dir` | cgan_output | Output directory |

### 3. `gik_ecmwf/` Module

Core processing modules for the GIK pipeline.

| File | Purpose |
|------|---------|
| `ecmwf_util.py` | Variable definitions, forecast hours, grid specs |
| `ecmwf_ensemble_par_creator_efficient.py` | Stage 1 GRIB scanning |
| `ecmwf_three_stage_multidate.py` | Pipeline orchestration |
| `ecmwf_index_processor.py` | Stage 2 fast index processing |
| `utils_ecmwf_step1_scangrib.py` | GRIB scanning utilities |

## Output Format

### NetCDF File Structure

```
IFS_20260203_00Z_cgan.nc
├── Dimensions:
│   ├── time: 1 (initialization)
│   ├── valid_time: 9 (forecast hours)
│   ├── latitude: 157 (ICPAC region)
│   └── longitude: 145
├── Coordinates:
│   ├── time: 2026-02-03T00:00:00
│   ├── valid_time: [36h, 39h, ..., 60h]
│   ├── latitude: [25.0, ..., -14.0]
│   └── longitude: [19.0, ..., 55.0]
└── Variables (for each of 12 fields):
    ├── {var}_ensemble_mean: (1, 9, 157, 145)
    └── {var}_ensemble_standard_deviation: (1, 9, 157, 145)
```

## Performance

| Phase | Operation | Time (51 members) | Notes |
|-------|-----------|-------------------|-------|
| 1 | Stage 1 GRIB scanning | ~90 min | Per forecast hour |
| 1 | Stage 2 Index processing | ~5 min | Uses templates |
| 1 | Stage 3 Final parquet | ~2 min | Fast merge |
| 2 | Data streaming | ~4 hours | Sequential S3 fetch |

### Bottleneck Analysis

The main bottleneck is **Phase 2 data streaming** (~4 hours for 51 members):
- Sequential S3 fetches for each member/variable/timestep
- Network latency dominates processing time
- Currently ~14,400 seconds for 12 variables × 51 members × 9 timesteps

## Future: Dask/Coiled Parallelization

See `ECMWF_cGAN_INFERENCE_WORKFLOW.md` for detailed plans on parallelizing the workflow using Dask and Coiled to reduce Phase 2 from ~4 hours to ~15-30 minutes.

## Troubleshooting

### Common Issues

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

## References

- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [gribberish](https://github.com/mpiannucci/gribberish) - Fast GRIB decoding
- [Coiled](https://coiled.io/) - Managed Dask clusters

## License

This code is part of the ICPAC climate services infrastructure.
