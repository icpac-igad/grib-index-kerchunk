# GIK vs Herbie: ECMWF Ensemble Data Streaming Comparison

**Author**: ICPAC GIK Team | **Date**: February 2026 | **Project**: E4DRR / SEWAA

---

## Overview

This document compares two methods for accessing ECMWF IFS ensemble forecast data
from AWS S3 for downstream analysis (cGAN rainfall downscaling over East Africa):

1. **GIK (GRIB Index Kerchunk)** -- Pre-built parquet reference files with
   gribberish decoding
2. **Herbie** -- Python library for NWP data access with cfgrib decoding

Both methods produce numerically identical results (Pearson r > 0.9999 across all
tested dates), but differ significantly in decode speed, scalability, and
suitability for multi-variable operational pipelines.

---

## Validation Results

### Methodology

- **Variable tested**: Total Precipitation (tp)
- **Ensemble members**: All 51 (1 control + 50 perturbed)
- **Forecast lead time**: T+48h
- **Region**: ICPAC East Africa (lat -14 to 25, lon 19 to 55)
- **Metrics**: Pearson correlation, RMSE, MAE on ensemble mean and spread

### 2024 Validation (10 dates, March--December)

| Date | Mean r | Spread r | Mean RMSE | Mean MAE |
|------|--------|----------|-----------|----------|
| 2024-03-21 | 0.999984 | 0.999970 | 6.13e-05 | 2.77e-05 |
| 2024-04-04 | 0.999972 | 0.999961 | 7.36e-05 | 4.33e-05 |
| 2024-05-01 | 0.999967 | 0.999877 | 1.80e-04 | 5.21e-05 |
| 2024-06-24 | 0.999966 | 0.999925 | 4.77e-05 | 1.85e-05 |
| 2024-07-09 | 0.999945 | 0.999947 | 5.50e-05 | 2.33e-05 |
| 2024-08-08 | 0.999957 | 0.999864 | 7.36e-05 | 3.51e-05 |
| 2024-09-08 | 0.999951 | 0.999941 | 5.32e-05 | 2.39e-05 |
| 2024-10-05 | 0.999947 | 0.999961 | 4.87e-05 | 2.26e-05 |
| 2024-11-24 | 0.999968 | 0.999923 | 6.75e-05 | 3.26e-05 |
| 2024-12-04 | 0.999978 | 0.999969 | 6.02e-05 | 2.41e-05 |

**All 10 dates**: Mean r > 0.99994, Spread r > 0.99986

### 2025 Validation (11 dates, January--November)

| Date | Mean r | Spread r | Mean RMSE | Mean MAE |
|------|--------|----------|-----------|----------|
| 2025-01-21 | 0.999977 | 0.999948 | 5.74e-05 | 2.40e-05 |
| 2025-02-04 | 0.999987 | 0.999961 | 6.08e-05 | 2.07e-05 |
| 2025-03-01 | 0.999929 | 0.999921 | 3.29e-05 | 1.14e-05 |
| 2025-04-24 | 0.999957 | 0.999941 | 4.70e-05 | 2.29e-05 |
| 2025-05-09 | 0.999973 | 0.999963 | 7.46e-05 | 3.69e-05 |
| 2025-06-08 | 0.999962 | 0.999951 | 4.05e-05 | 1.64e-05 |
| 2025-07-08 | 0.999967 | 0.999940 | 6.16e-05 | 2.71e-05 |
| 2025-08-05 | 0.999959 | 0.999905 | 6.80e-05 | 2.72e-05 |
| 2025-09-24 | 0.999941 | 0.999901 | 5.92e-05 | 2.62e-05 |
| 2025-10-04 | 0.999953 | 0.999884 | 6.59e-05 | 2.78e-05 |
| 2025-11-22 | 0.999960 | 0.999917 | 4.76e-05 | 1.85e-05 |

**All 11 dates**: Mean r > 0.99992, Spread r > 0.99988

### Conclusion

The tiny differences (RMSE ~5e-05 m, relative diff <1%) arise from floating-point
representation differences between the gribberish (Rust) and cfgrib (Python/eccodes)
decoders. Both methods decode the exact same GRIB bytes from the exact same S3
objects. The data is **functionally identical**.

---

## Data Transfer Comparison

### How Both Methods Access Data

Both GIK and Herbie use **byte-range reads** against the same ECMWF GRIB files on
`s3://ecmwf-forecasts/`. Neither downloads full 3--4 GB GRIB files.

```
ECMWF S3 Structure:
  s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/ifs/0p25/enfo/
    {YYYYMMDDHH}0000-{H}h-enfo-ef.grib2      (~3-4 GB per timestep, 51 members)
    {YYYYMMDDHH}0000-{H}h-enfo-ef.grib2.index (JSON-lines index, ~1 MB)
```

**Each GRIB message** (one variable, one member, one timestep) is ~800 KB.
Both methods read only the specific messages needed.

### Single Variable (tp) Transfer Size

For total precipitation at 9 timesteps across 51 members:

| | GIK | Herbie |
|---|---|---|
| GRIB messages fetched | 51 members x 9 steps = 459 | 51 members x 9 steps = 459 |
| Bytes per message | ~835 KB | ~817 KB |
| Total GRIB data | ~384 MB | ~375 MB |
| Index overhead | ~0 (parquet pre-built) | ~9 MB (.index files) |
| **Total transfer** | **~384 MB** | **~384 MB** |

The ~2% size difference comes from how each method determines message boundaries:
GIK uses `_offset` + `_length` from the `.index` file exactly, while Herbie
infers length from consecutive offsets, occasionally including a few extra bytes.

### Key Insight: Both Download the Full Global Grid

Neither method performs server-side spatial subsetting. Each ~800 KB GRIB message
contains the **full 721 x 1440 global grid** (0.25 deg). The ICPAC East Africa
subset (157 x 145 grid points) is only **2.2%** of the global grid.

```
Full global grid:    721 x 1440 = 1,038,240 points
ICPAC subset:        157 x 145  =    22,765 points  (2.2%)
```

Spatial subsetting happens **after** download and decode, in client-side numpy
slicing. This is a fundamental property of the GRIB format -- byte-range reads
can target individual messages but cannot subset within a message.

---

## Decode Speed: gribberish vs cfgrib

The critical performance difference between GIK and Herbie is not data transfer
but **GRIB decoding**:

| Decoder | Used by | Time per chunk | Implementation |
|---------|---------|---------------|----------------|
| gribberish | GIK | ~25 ms | Rust, operates on byte buffers in memory |
| cfgrib | Herbie | ~2,000 ms | Python/eccodes, writes temp file to disk |
| **Speedup** | | **~80x** | |

### Why gribberish Is Faster

1. **No disk I/O**: gribberish decodes directly from a byte buffer in memory.
   cfgrib must write GRIB bytes to a temporary file, then eccodes reads that file.
2. **Rust vs Python**: gribberish is compiled Rust; eccodes is a C library called
   through Python bindings with per-call overhead.
3. **No metadata overhead**: gribberish's `parse_grib_array()` returns a flat
   numpy array directly. cfgrib constructs a full xarray Dataset with coordinates,
   attributes, and dimension metadata for every single chunk.

### Impact on Single-Variable Processing

For 459 GRIB messages (51 members x 9 timesteps):

| | GIK (gribberish) | Herbie (cfgrib) |
|---|---|---|
| Decode time | 459 x 25ms = **~11 seconds** | 459 x 2000ms = **~15 minutes** |
| I/O time (S3) | ~5 minutes | ~5 minutes |
| **Total** | **~5-6 minutes** | **~20 minutes** |

---

## Multi-Variable Scaling: The GIK Advantage

The ICPAC cGAN downscaling pipeline requires **12 variables** across 9 timesteps
for all 51 ensemble members:

### Surface Variables (10)
| Variable | ECMWF Name | Description |
|----------|-----------|-------------|
| tp | tp | Total Precipitation |
| t2m | 2t | 2-meter Temperature |
| sp | sp | Surface Pressure |
| ssr | ssr | Surface Solar Radiation |
| ssrd | ssrd | Surface Solar Radiation Downwards |
| sf | sf | Snowfall |
| ro | ro | Runoff |
| tcw | tcw | Total Cloud Water |
| tcwv | tcwv | Total Column Water Vapour |
| tcc | tcc | Total Cloud Cover |

### Pressure Level Variables (2)
| Variable | Level | Description |
|----------|-------|-------------|
| u700 | 700 hPa | U-wind component |
| v700 | 700 hPa | V-wind component |

### Multi-Variable Transfer and Decode Time

| | GIK (gribberish) | Herbie (cfgrib) |
|---|---|---|
| Messages per variable | 459 | 459 |
| Total messages (12 vars) | 5,508 | 5,508 |
| Total GRIB data | ~4.5 GB | ~4.5 GB |
| Decode time | 5,508 x 25ms = **~2.3 min** | 5,508 x 2000ms = **~3 hours** |
| I/O time (parallel) | ~15 min | ~15 min |
| **Total (single machine)** | **~24 min** | **~4 hours** |

The 80x decode speedup becomes increasingly dominant as the number of variables
grows. For 12 variables, GIK completes in ~24 minutes on a single machine with
8-thread parallelism, while Herbie would take ~4 hours.

---

## Dask Cluster Scaling with Coiled

### The Parallelism Architecture

GIK's parquet reference files are inherently parallelizable. Each member's parquet
is a self-contained lookup table mapping zarr keys to `[s3_url, offset, length]`
triplets. Workers need no coordination -- they independently read their assigned
parquet, fetch GRIB bytes from S3, decode, and return results.

Two Coiled Dask implementations exist:

### 1. Simple Version (`stream_cgan_variables_coiled_simple.py`)

```
Local Client
  │
  ├─► Coiled Worker 1 → read parquet → fetch GRIB → decode → return numpy array
  ├─► Coiled Worker 2 → read parquet → fetch GRIB → decode → return numpy array
  ├─► ...
  └─► Coiled Worker N → read parquet → fetch GRIB → decode → return numpy array
  │
  ▼
Local: aggregate arrays → compute ensemble mean/std → write NetCDF
```

- Workers return decoded numpy arrays directly to the client
- No intermediate storage needed
- Best for moderate-scale runs (5--20 workers)
- Parquets read from GCS (`gs://gik-ecmwf-aws-tf/run_par_ecmwf/`)

### 2. Icechunk Version (`stream_cgan_variables_coiled.py`)

```
Local Client
  │
  ├─► Worker 1 → read parquet → fetch GRIB → decode → write to Icechunk branch_0
  ├─► Worker 2 → read parquet → fetch GRIB → decode → write to Icechunk branch_1
  ├─► ...
  └─► Worker N → read parquet → fetch GRIB → decode → write to Icechunk branch_N
  │
  ▼
Local: read all branches → aggregate → compute statistics → write NetCDF
```

- Each worker writes to a unique Icechunk branch (avoids write conflicts)
- Intermediate data persisted in GCS via Icechunk
- Better for large-scale production (20--50 workers)
- Supports retry on failure without re-processing successful batches

### Estimated Processing Times with Coiled

| Configuration | Members | Workers | Time per member | Total |
|---|---|---|---|---|
| Single machine, 8 threads | 51 | 1 | ~24 min | ~24 min |
| Coiled, 10 workers | 51 | 10 | ~3 min | ~5 min |
| Coiled, 20 workers | 51 | 20 | ~3 min | ~3 min |
| Coiled, 51 workers | 51 | 51 | ~3 min | ~3 min |

With Coiled, the bottleneck shifts from compute to I/O. Each worker processes one
member independently in ~3 minutes (reading 12 vars x 9 steps = 108 GRIB messages,
~90 MB). With 20 workers, the full 51-member ensemble completes in ~3 minutes
plus aggregation overhead.

### Why Herbie Cannot Scale the Same Way

Herbie is designed as a **single-user, single-machine library**. To parallelize
Herbie across a Dask cluster, you would need to:

1. Install eccodes and cfgrib on every worker (complex C library dependency)
2. Manage temporary files for each decode operation on each worker
3. Handle eccodes' global state and file locking across parallel processes
4. Accept the ~2 second per-chunk decode overhead on every worker

A "microservices" approach (running Herbie on multiple cloud functions) is
theoretically possible but introduces significant complexity:
- Each function needs a full eccodes/cfgrib installation (~200 MB container)
- No shared state -- each function re-downloads `.index` files independently
- cfgrib's temp-file approach creates disk pressure on serverless containers
- Orchestration overhead for coordinating 51+ independent function invocations

By contrast, GIK + gribberish on Coiled:
- Pure Python + Rust wheels, no system library dependencies
- No temporary files, no disk I/O during decode
- Each worker reads a single parquet file from GCS (~1 MB) and streams GRIB bytes
- Standard `pip install gribberish` on workers via Coiled's `package_sync=True`

---

## Architecture Comparison Summary

| Aspect | GIK + gribberish | Herbie + cfgrib |
|--------|-----------------|----------------|
| **Data access** | Byte-range reads via parquet references | Byte-range reads via .index parsing |
| **Index format** | Pre-built parquet (key → [url,offset,len]) | .index JSON-lines (parsed at runtime) |
| **Decoder** | gribberish (Rust, in-memory) | cfgrib (Python/eccodes, temp files) |
| **Decode speed** | ~25 ms/chunk | ~2,000 ms/chunk |
| **Data transfer** | ~384 MB (tp only) / ~4.5 GB (12 vars) | ~384 MB (tp only) / ~4.5 GB (12 vars) |
| **Spatial subsetting** | Post-decode numpy slicing | Post-decode numpy slicing |
| **Dependencies** | numpy, pandas, fsspec, gribberish | herbie, cfgrib, eccodes (system lib) |
| **Dask parallelism** | Native (parquet per member) | Difficult (eccodes state, temp files) |
| **Operational overhead** | One-time parquet creation (~5 min/date) | None (reads .index at runtime) |
| **Offline capability** | Parquets can be cached/archived | Requires live S3 access to .index files |

---

## When to Use Each Method

### Use GIK when:
- Processing multiple variables (>3) across the full ensemble
- Running operational pipelines on a schedule
- Scaling to cloud-based Dask clusters
- Decode speed is a bottleneck (large ensemble, many variables)
- Parquet reference files are already built (production pipeline)

### Use Herbie when:
- Quick ad-hoc exploration of a single variable
- Working on a local machine without GCS access to parquets
- No pre-built parquets available for the target date
- Simplicity is more important than performance

---

## Parquet Reference Files on HuggingFace

Pre-built parquet reference files for all ECMWF dates are available at:

**Dataset**: `E4DRR/gik-ecmwf-par`

```
run_par_ecmwf/
  2024/{MM}/{YYYYMMDD}/00z/{YYYYMMDD}00z-{member}.parquet
  2025/{MM}/{YYYYMMDD}/00z/{YYYYMMDD}00z-{member}.parquet
  2026/{MM}/{YYYYMMDD}/00z/{YYYYMMDD}00z-{member}.parquet
```

| Year | Parquets | Approximate Size |
|------|----------|-----------------|
| 2024 | ~15,606 | ~2.2 GB |
| 2025 | ~18,615 | ~2.6 GB |
| 2026 | ~1,581 | ~0.2 GB |
| **Total** | **~35,802** | **~5.0 GB** |

Each parquet file is ~140 KB and contains the zarr store structure + byte-range
references for one ensemble member for one forecast date. These are the files
consumed by `stream_cgan_variables_coiled_simple.py` and
`stream_cgan_variables_coiled.py`.

---

## Validation Scripts

The comparison was validated using:

- `validate_gik_vs_herbie_2024.py` -- 10 random dates (March--December 2024)
- `validate_gik_vs_herbie_2025.py` -- 11 dates (January--November 2025)

Results (comparison plots, scatter plots, NetCDF files, and JSON stats) are
uploaded to `E4DRR/gik-ecmwf-par` under `validation_gik_vs_herbie_2024/` and
`validation_gik_vs_herbie_2025/`.

---

## References

- ICPAC GIK Repository: `github.com/icpac-igad/grib-index-kerchunk`
- ECMWF Open Data on AWS: `s3://ecmwf-forecasts/`
- gribberish: Rust-based GRIB2 decoder
- Herbie: Python library for NWP data access
- Coiled: Managed Dask clusters on cloud infrastructure
- Icechunk: Version-controlled array storage
