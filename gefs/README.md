# GEFS East Africa Ensemble Processing Pipeline

## Overview

Complete pipeline for processing GEFS (Global Ensemble Forecast System) ensemble data for the East Africa region using the **Grib-Index-Kerchunk (GIK) method** for efficient GRIB data access without full file scanning.

- **30 ensemble members** (gep01-gep30)
- **81 forecast timesteps** (0-240h at 3h intervals)
- **0.25° global grid** (721x1440), subset to East Africa (lat -12..15, lon 25..52)
- **Data source**: `s3://noaa-gefs-pds/` (anonymous access)

## Three-Stage Pipeline

The pipeline creates lightweight parquet reference files containing `[url, byte_offset, byte_length]` triplets pointing into remote GRIB files on AWS S3.

### Stage 1: Build Zarr Metadata Structure (Deflated Store)

Creates the zarr variable/dimension/chunk schema — identical across all 30 members.

| Mode | Method | Time | Command |
|------|--------|------|---------|
| **Template** (recommended) | `build_gefs_deflated_store_from_template()` | ~0.05s | Automatic when template exists |
| **scan_grib** (fallback) | `scan_grib(f000, f003) → grib_tree()` | ~5.5s | Automatic fallback |

The pre-built template `gefs-deflated-store-template-20241112.parquet` (26 KB, 915 entries) is loaded once and reused for all 30 members — **112x speedup** per invocation, **3300x total** for the full ensemble.

### Stage 2: Index-Based Reference Building

Reads `.idx` files from S3 (one per GRIB file), extracts byte offsets, and merges with mapping parquets from the template archive (`gik-fmrc-gefs-20241112.tar.gz`). I/O-bound, ~25-30s per member.

### Stage 3: Final Zarr Store Assembly

Merges the deflated store with fresh byte-range references to produce one parquet file per ensemble member.

## Quick Start

### Daily Ensemble Processing

```bash
# Process all 30 members for a date (uses template-based Stage 1)
uv run run_day_gefs_ensemble_full.py --date 20250106

# Process with member limit for testing
uv run run_day_gefs_ensemble_full.py --date 20250106 --max-members 5
```

### Single Member to Zarr

```bash
# Convert one member's parquet to zarr (cfgrib)
uv run run_single_gefs_to_zarr.py 20250106 00 gep01 --region east_africa

# Same with gribberish (~80x faster decoding)
uv run run_single_gefs_to_zarr_gribberish.py 20250106 00 gep01 --region east_africa
```

### Ensemble Statistics

```bash
# Concatenate members and compute mean/std/probabilities
uv run process_ensemble_by_variable.py zarr_stores/20250106_00/

# 24h rainfall accumulation and exceedance probabilities
uv run run_gefs_24h_accumulation.py
```

## GIK vs Herbie Validation

The GIK pipeline has been validated against [Herbie](https://herbie.readthedocs.io/) to confirm bit-identical data extraction. Herbie uses cfgrib to download and decode GRIB data conventionally; GIK streams only the needed bytes via byte-range reads.

**Result: r=1.0, RMSE=0.0, MAE=0.0** across 11,881 grid points over East Africa.

```bash
# Step 1: Create GIK TP NetCDF via gribberish streaming
uv run create_gik_tp_netcdf.py --date 20250106 --max-members 3

# Step 2: Create Herbie reference TP NetCDF
uv run fetch_tp_herbie_gefs.py --date 20250106 --max-members 3

# Step 3: Compare and generate PNG plots
uv run compare_gik_herbie_gefs.py --dates 20250106 \
    --gik-dir gik_tp_gefs_output --herbie-dir herbie_tp_gefs_output
```

Output in `gik_vs_herbie_gefs/`:
- `compare_gefs_YYYYMMDD.png` — Side-by-side maps (GIK | Herbie | Difference) for mean and spread
- `scatter_gik_vs_herbie_gefs.png` — Grid-point scatter plot
- `comparison_stats_gefs.json` — Numerical metrics

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `gefs_util.py` | Core utilities — template loading, scan_grib, zarr store ops | Imported by other scripts |
| `run_day_gefs_ensemble_full.py` | Daily ensemble processing (all 30 members) | `uv run run_day_gefs_ensemble_full.py --date YYYYMMDD` |
| `run_gefs_preprocessing.py` | One-time template/mapping creation | `uv run run_gefs_preprocessing.py` |
| `create_gik_tp_netcdf.py` | GIK TP extraction with gribberish streaming | `uv run create_gik_tp_netcdf.py --max-members 3` |
| `fetch_tp_herbie_gefs.py` | Herbie-based TP fetching for validation | `uv run fetch_tp_herbie_gefs.py --date YYYYMMDD` |
| `compare_gik_herbie_gefs.py` | GIK vs Herbie comparison PNGs | `uv run compare_gik_herbie_gefs.py --dates YYYYMMDD` |
| `run_single_gefs_to_zarr.py` | Single member parquet-to-zarr (cfgrib) | `uv run run_single_gefs_to_zarr.py DATE RUN MEMBER` |
| `run_single_gefs_to_zarr_gribberish.py` | Single member parquet-to-zarr (gribberish, ~80x faster) | `uv run run_single_gefs_to_zarr_gribberish.py DATE RUN MEMBER` |
| `process_ensemble_by_variable.py` | Ensemble concatenation + statistics | `uv run process_ensemble_by_variable.py ZARR_DIR/` |
| `run_gefs_24h_accumulation.py` | 24h rainfall accumulation + probability maps | `uv run run_gefs_24h_accumulation.py` |
| `plot_ensemble_east_africa.py` | Visualization of ensemble outputs | `uv run plot_ensemble_east_africa.py` |

## Key Files

| File | Description |
|------|-------------|
| `gefs-deflated-store-template-20241112.parquet` | Pre-built Stage 1 template (26 KB, 915 zarr metadata entries) |
| `gik-fmrc-gefs-20241112.tar.gz` | Template archive with per-member mapping parquets (from [HuggingFace](https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/blob/main/gik-fmrc-gefs-20241112.tar.gz)) |
| `gik_tp_gefs_output/` | GIK-produced TP NetCDF files |
| `herbie_tp_gefs_output/` | Herbie-produced TP NetCDF files |
| `gik_vs_herbie_gefs/` | Comparison PNG plots and statistics |
| `docs/GEFS_SCAN_GRIB_TO_TEMPLATE.md` | Detailed documentation of the scan_grib to template transformation |

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Stage 1 (template load) | 0.05s | Load once, reuse for all 30 members |
| Stage 1 (scan_grib fallback) | 5.5s per call | 30 x 5.5s = 165s for all members |
| Stage 2 (index reading) | ~25-30s per member | I/O-bound, parallelizable |
| Stage 3 (zarr assembly) | ~1s per member | CPU-bound |
| gribberish decode | ~25ms per chunk | vs ~2000ms for cfgrib (80x faster) |
| Full 30-member ensemble | ~3-5 min | Template-based, single machine |

## Public HuggingFace Dataset

The full 00z backfill (2020-09-25 → 2025-12-31, ~57,780 per-member parquets,
30 members) is mirrored to **[`E4DRR/gik-gefs-par`](https://huggingface.co/datasets/E4DRR/gik-gefs-par)**
in two layouts:

| Layout | HF path | Best for |
|--------|---------|----------|
| Catalog index | `run_par_gefs_agg/catalog.parquet` (773 KB) | Discover what's available |
| Monthly aggregate | `run_par_gefs_agg/monthly_agg/{Y}/{MM}_00z.parquet` (~250 MB each) | Bulk reads of one month |
| Per-member parquet | `run_par_gefs/{Y}/{MM}/{YYYYMMDD}/00z/{YYYYMMDD}00z-gepNN.parquet` | Single-member access |

`upload_parquets_to_hf.py` builds the catalog and monthly aggregates;
see [`lithops-cr-gik-gefs/SETUP_NEW_MACHINE.md`](lithops-cr-gik-gefs/SETUP_NEW_MACHINE.md#11-publishing-to-huggingface)
for the publishing workflow.

### Open a monthly aggregate and pull a single date

The monthly aggregate concatenates every member's parquet for one month
into a single file, with `date` and `source_file` columns added for
provenance. Each row is one byte-range reference; one date contributes
~30 members × 60 references = ~1,800 rows.

```python
import pandas as pd

# Open one month's aggregate directly from HuggingFace (no auth needed for public repo)
URL = "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet"
df = pd.read_parquet(URL)
print(f"{len(df):,} rows, columns: {list(df.columns)}")

# Filter to a single date
day = df[df.date == "20240615"]
print(f"  2024-06-15: {len(day)} rows from "
      f"{day.source_file.nunique()} member parquets")

# Inspect what's there per member (gep01..gep30)
day = day.assign(member=day.source_file.str.extract(r"-(gep\d+)")[0])
print(day.groupby("member").size().head())
```

### Open a single date's data as a virtual zarr

The per-member parquets are kerchunk byte-range references that point
back into the original NOAA GEFS GRIBs on `s3://noaa-gefs-pds/`. To
materialise one member of one date as an xarray dataset:

```python
from huggingface_hub import hf_hub_download
import xarray as xr

# Download one member's reference parquet (~280 KB)
ref_path = hf_hub_download(
    repo_id="E4DRR/gik-gefs-par",
    repo_type="dataset",
    filename="run_par_gefs/2024/06/20240615/00z/2024061500z-gep01.parquet",
)

# Open as a virtual zarr — kerchunk fetches only the bytes you index
ds = xr.open_dataset(
    "reference://", engine="zarr",
    backend_kwargs={
        "consolidated": False,
        "storage_options": {
            "fo": ref_path,
            "remote_protocol": "s3",
            "remote_options": {"anon": True},  # NOAA bucket is public
        },
    },
)
print(ds)        # 81 timesteps × 30 vars × 721×1440 grid
print(ds.tp)     # Total precipitation, lazy-loaded
```

To pull every member for one date, iterate the catalog:

```python
catalog = pd.read_parquet(
    "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/catalog.parquet"
)
day_files = catalog[catalog.date == "20240615"]
print(f"{len(day_files)} parquet files for 2024-06-15:")
print(day_files[["member", "hf_path", "size_bytes"]].head())
```

## Tutorial

Self-contained tutorials in `../tutorial/gefs/`:

```bash
# Parquet creation for all 30 members
uv run tutorial/gefs/run_gefs_tutorial.py

# Data streaming + probability maps
uv run tutorial/gefs/run_gefs_data_streaming.py
```

The tutorial downloads the template archive from HuggingFace automatically.

## Documentation

- `docs/GEFS_SCAN_GRIB_TO_TEMPLATE.md` — Template-based Stage 1 transformation
- `docs/GEFS_Complete_Documentation.md` — Full technical documentation
- `docs/GEFS_Three_Stage_Processing.md` — Three-stage pipeline details
- `docs/GRIBBERISH_VS_CFGRIB_ANALYSIS.md` — Decoder comparison

## References

- [NOAA GEFS on AWS](https://registry.opendata.aws/noaa-gefs/)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [Zarr Documentation](https://zarr.readthedocs.io/)
- [Herbie Documentation](https://herbie.readthedocs.io/)
- [Gribberish](https://github.com/mpiannucci/gribberish)

## Acknowledgements

This work was funded in part by:

1. Hazard modeling, impact estimation, climate storylines for event catalogue
   on drought and flood disasters in the Eastern Africa (E4DRR) project.
   https://icpac-igad.github.io/e4drr/ United Nations | Complex Risk Analytics
   Fund (CRAF'd)
2. The Strengthening Early Warning Systems for Anticipatory Action (SEWAA)
   Project. https://cgan.icpac.net/
