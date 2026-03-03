# ECMWF GIK Tutorial

Stream ECMWF ensemble forecast data directly from AWS S3 without downloading
full GRIB files. Three self-contained scripts, two pathways.

## Overview

| Script | What it does | Runtime |
|--------|-------------|---------|
| `process1_make_virtual_manifest.py` | Create parquet reference files for any date | ~5-15 min |
| `process2_open_virtual_dataset.py` | Open parquets as lazy xarray (dask.delayed) | ~10s setup |
| `process3_open_materialized_dataset.py` | Open parquets as in-memory xarray | ~30s/member |

**Two pathways** to access ECMWF data:

1. **Use pre-built catalog** (process2 or process3) — parquets are downloaded
   from the [E4DRR/gik-ecmwf-par](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par)
   HuggingFace dataset. Fastest way to get started.

2. **Create your own** (process1 → process2 or process3) — generate parquet
   reference files for any date, then open them. Use this for dates not in
   the HuggingFace catalog.

## Quick Start

```bash
cd tutorial/ecmwf

# Pathway 1: Use pre-built catalog (fastest)
uv run process2_open_virtual_dataset.py                           # lazy, 1 member
uv run process2_open_virtual_dataset.py --members 5 --load-step 24  # load step 24
uv run process3_open_materialized_dataset.py --members 3          # eager, 3 members

# Pathway 2: Create your own parquets
uv run process1_make_virtual_manifest.py --date 20260301 --max-members 3
```

## What Each Script Does

### process1: Create Virtual Manifest

Runs the three-stage GIK pipeline to produce parquet reference files:

- **Stage 1** (~5s): Loads zarr metadata structure from a HuggingFace template
  archive (no GRIB scanning needed)
- **Stage 2** (~5-15 min): Reads `.index` files from S3 to extract byte offsets,
  merges with the template to build full zarr references
- **Stage 3** (~2s): Writes one parquet file per ensemble member

```bash
# 3 members (quick demo)
uv run process1_make_virtual_manifest.py --date 20260301

# All 51 members
uv run process1_make_virtual_manifest.py --date 20260301 --max-members 51 --run 00
```

Output: `output_parquet/2026030100z-control.parquet`, etc.

### process2: Open as Lazy Virtual Dataset

Builds an xarray Dataset backed by `dask.delayed` — zero bytes fetched from S3
until you call `.load()` or `.sel().compute()`. Ideal for interactive exploration
where you only need specific slices.

```bash
uv run process2_open_virtual_dataset.py --date 20250101 --members 5
uv run process2_open_virtual_dataset.py --load-step 24   # verify by loading 1 step
```

### process3: Open as Materialized Dataset

Eagerly fetches all data via parallel S3 byte-range reads + gribberish decoding.
Produces a fully in-memory xarray Dataset ready for analysis, plotting, or
export to NetCDF.

```bash
uv run process3_open_materialized_dataset.py --date 20250101 --members 3
```

## Performance

### Parquet Creation (process1)

| Stage | Time | What happens |
|-------|------|-------------|
| Stage 1 | ~5s | Load zarr structure from HuggingFace template |
| Stage 2 | ~5-15 min | Read 85 `.index` files per member from S3 |
| Stage 3 | ~2s | Write parquet files to disk |

### Data Access (process2 / process3)

| Script | Setup | Per-member fetch | Notes |
|--------|-------|-----------------|-------|
| process2 (lazy) | ~10s | ~2s per step on demand | No data until `.load()` |
| process3 (eager) | ~10s | ~30s (85 steps, 8 threads) | Full dataset in memory |

### Decoding Speed

| Decoder | Time per chunk | Notes |
|---------|---------------|-------|
| gribberish (Rust) | ~25 ms | Direct byte buffer |
| cfgrib (Python) | ~2000 ms | Temp file + eccodes |
| Speedup | **~80x** | |

## ECMWF Data Characteristics

| Property | Value |
|----------|-------|
| S3 bucket | `s3://ecmwf-forecasts/` |
| Ensemble members | 51 (1 control + 50 perturbed) |
| Grid resolution | 0.25 degree (~25 km) |
| Grid size | 721 x 1440 (lat x lon) |
| Forecast hours | 85 total |
| Time resolution | 3h (0-144h), 6h (150-360h) |
| Index format | JSON-lines (`.index` files) |
| Access | Anonymous (`anon=True`) |

## Dependencies

All scripts use PEP 723 inline metadata and can be run directly with `uv run`.

**process1** (no kerchunk, no gribberish):
```
pandas numpy pyarrow fsspec s3fs requests
```

**process2** (lazy/virtual):
```
numpy pandas xarray fsspec s3fs pyarrow gribberish dask huggingface_hub
```

**process3** (materialized):
```
numpy pandas xarray fsspec s3fs pyarrow gribberish huggingface_hub
```

### Installing manually

```bash
pip install pandas numpy pyarrow fsspec s3fs requests xarray gribberish dask huggingface_hub
```

## Troubleshooting

**"Index file not found"** — The ECMWF forecast for that date/run may not be
published yet, or may have been removed (data is retained for ~1 year).
Try a recent date.

**Slow S3 fetches** — ECMWF GRIB files are large (~3 GB each) and byte-range
reads from deep offsets can be slow. The parallel fetching in process3 (8 threads)
mitigates this.

**gribberish decode errors** — Some GRIB messages may fail to decode with
gribberish. The scripts handle this gracefully (NaN fill or skip).

## References

- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [ECMWF on AWS](https://registry.opendata.aws/ecmwf-forecasts/)
- [gribberish](https://github.com/mpiannucci/gribberish)
- [Kerchunk](https://fsspec.github.io/kerchunk/)
- [HuggingFace Catalog: E4DRR/gik-ecmwf-par](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par)

## Acknowledgements

This work was funded by:
1. **E4DRR Project** — UN Complex Risk Analytics Fund (CRAF'd)
   https://icpac-igad.github.io/e4drr/
2. **SEWAA Project** — Strengthening Early Warning Systems for Anticipatory Action
   https://cgan.icpac.net/
