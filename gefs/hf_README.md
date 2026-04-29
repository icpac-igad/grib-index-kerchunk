---
license: apache-2.0
tags:
  - weather
  - climate
  - ensemble-forecasts
  - gefs
  - noaa
  - grib
  - zarr
  - kerchunk
  - byte-range
  - streaming
---

# GIK-GEFS-PAR: Virtual Reference Parquets for NOAA GEFS Ensemble Forecasts

Lightweight parquet reference files that turn **24+ GB of daily NOAA GEFS GRIB
data into ~250 MB of monthly virtual references**, enabling Dask-based
parallel analysis without downloading the raw GRIBs.

## What These Parquets Do

Each reference contains `[zarr_key, [s3_url, byte_offset, byte_length]]`
tuples pointing into NOAA GEFS GRIB files on AWS S3 (`s3://noaa-gefs-pds/`).
Instead of downloading full GRIB files, analysis code performs targeted
**byte-range reads** — fetching only the variables, members, and timesteps
needed (typically **2–5%** of the original data).

```
Without GIK:  Download ~24 GB of GRIB files per day  →  then process
With GIK:     Read ~10 MB of parquet refs + stream    →  direct analysis
```

## Scale: Parquets vs Source GRIB Data

### Source GRIB Scale (NOAA GEFS 00z only)

| Parameter | Value |
|-----------|-------|
| Timesteps per run | 81 (0–240 h at 3 h intervals) |
| Members per timestep | 30 (gep01 – gep30) |
| Grid resolution | 0.25° global (721 × 1440) |
| Daily GRIB volume (00z, all members) | **~24 GB** |

### Annual GRIB Data Referenced

| Year | Coverage | Days | Total GRIB Referenced |
|------|----------|------|-----------------------|
| 2020 | Sep 25 – Dec 31 | 98 | ~2.4 TB |
| 2021 | Jan 1 – Dec 31 | 365 | ~8.9 TB |
| 2022 | Jan 1 – Dec 31 | 365 | ~8.9 TB |
| 2023 | Jan 1 – Dec 31 | 365 | ~8.9 TB |
| 2024 | Jan 1 – Dec 31 | 366 | ~8.9 TB |
| 2025 | Jan 1 – Dec 31 | 365 | ~8.9 TB |
| **Total** | | **1,924 days** | **~47 TB** |

### Parquet Files in This Dataset

| Layout | Files | Size | Compression vs source GRIB |
|--------|-------|------|----------------------------|
| Per-member kerchunk parquets | 57,780 (1,924 dates × 30 members) | ~15 GB on GCS, ~280 KB each |
| Monthly aggregates (this repo) | 64 (one per year-month) | ~14 GB total, ~250 MB each | **~3,000×** |
| Catalog index | 1 | 791 KB |

NOAA GEFS realtime/reforecast archive on `s3://noaa-gefs-pds/` starts on
**2020-09-25** — earlier dates have no upstream data.

## Dataset Structure

```
catalog.parquet                                    # 791 KB, 57,780-row index
run_par_gefs_agg/
  monthly_agg/
    {YYYY}/{MM}_00z.parquet                        # one optimised parquet per month
```

Each monthly aggregate is **sorted by `(date, member)`** and written with
**`row_group_size=60`** (one row group per date). With PyArrow's predicate
pushdown a single-date filter reads only ~5–10 MB of a 250 MB file via
HuggingFace's range-read support.

## Catalog / Index

The `catalog.parquet` at the repo root indexes every reference parquet
that ever lived in GCS — useful for discovering coverage without listing
the full repo tree.

| Column | Example | Description |
|--------|---------|-------------|
| `year` | `2024` | Forecast year |
| `month` | `06` | Forecast month |
| `date` | `20240615` | Forecast date (YYYYMMDD) |
| `run` | `00z` | Run hour |
| `member` | `gep01` | Ensemble member name (gep01–gep30) |
| `filename` | `2024061500z-gep01.parquet` | Source parquet filename |
| `hf_path` | `run_par_gefs/2024/06/20240615/00z/...` | Original GCS layout path |
| `size_bytes` | `287232` | Source parquet size |

## Quick Start: Filter the monthly aggregate by date + member

This is the canonical access pattern. PyArrow + HF range reads + parquet
predicate pushdown means a single-date query reads only the relevant
~5–10 MB of a 250 MB file.

```python
import pandas as pd

# Read only the rows for one date + one member — pyarrow skips non-matching row groups
df = pd.read_parquet(
    "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet",
    filters=[("date", "=", "20240615"), ("member", "=", "gep01")],
)
print(f"{len(df)} reference rows for 2024-06-15 gep01")
print(df[["key", "member", "date"]].head())
```

To pull all 30 members for one date (still <100 MB read):

```python
df = pd.read_parquet(
    "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet",
    filters=[("date", "=", "20240615")],
)
print(f"{df.member.nunique()} members, {len(df)} rows total")
```

## Open one date as a lazy xarray Dataset (virtual, on-demand)

Each (date, member) yields **two rows** in the aggregate: one with
`key="refs"` whose `value` is a ~1.5 MB Python dict literal containing
the entire kerchunk zstore (byte-range refs as Python lists, zarr
metadata as JSON strings), and one with `key="version"`. Use
`ast.literal_eval` — the blob uses single quotes so `json.loads` will
fail.

The pattern below builds the **full 30-member ensemble** as a
dask-backed `xarray.Dataset` where parquet parsing happens instantly
and S3 byte-range reads to NOAA's public bucket only fire when you call
`.load()` or `.compute()`.

```python
import ast, base64
import dask, dask.array as da
import fsspec, gribberish
import numpy as np, pandas as pd, xarray as xr

GEFS_GRID = (721, 1440)
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)            # GEFS uses 0..360 longitude
STEPS = [3, 6, 12, 24, 48, 72, 120, 168, 240]       # forecast hours

# 1. Read all 30 members for one date in one HF call
#    (~30 MB pulled via parquet filter pushdown + HF range reads)
df = pd.read_parquet(
    "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet",
    filters=[("date", "=", "20240615")],
)

def member_zstore(sub):
    blob = sub[sub["key"] == "refs"].iloc[0]["value"]
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    return ast.literal_eval(blob)

# 2. Lazy chunk fetcher — runs only when xarray pulls data
s3 = fsspec.filesystem("s3", anon=True)

def lazy_chunk(ref):
    @dask.delayed
    def _fetch():
        url, off, ln = ref[0], ref[1], ref[2]
        with s3.open(url, "rb") as f:
            f.seek(off)
            raw = f.read(ln)
        return gribberish.parse_grib_array(raw, 0).reshape(GEFS_GRID).astype(np.float32)
    return da.from_delayed(_fetch(), shape=GEFS_GRID, dtype=np.float32)

# 3. Build the (member, step) lazy stack — no S3 reads here, just metadata
member_arrays, member_names = [], []
for member, sub in df.groupby("member"):
    zs = member_zstore(sub)
    val = zs.get("tp/accum/surface/step/0", "")
    if isinstance(val, str) and val.startswith("base64:"):
        step_hours = np.frombuffer(base64.b64decode(val[7:]), dtype="<f8")
    else:
        step_hours = np.arange(0, 243, 3, dtype=float)
    tp_refs = {
        int(k.rsplit("/", 1)[1].split(".")[0]): v
        for k, v in zs.items()
        if k.startswith("tp/accum/surface/tp/") and isinstance(v, list)
    }
    chunks = []
    for h in STEPS:
        idx = int(np.argmin(np.abs(step_hours - h)))
        ref = tp_refs.get(idx)
        chunks.append(lazy_chunk(ref) if ref else
                      da.full(GEFS_GRID, np.nan, dtype=np.float32))
    member_arrays.append(da.stack(chunks, axis=0))
    member_names.append(member)

ds = xr.Dataset(
    {"tp": (["member", "step", "latitude", "longitude"],
            da.stack(member_arrays, axis=0))},
    coords={"member": member_names, "step": STEPS,
            "latitude": GEFS_LATS, "longitude": GEFS_LONS},
)
print(ds)
# <xarray.Dataset>  Dimensions: member=30, step=9, latitude=721, longitude=1440
# tp: dask.array<...>  ← zero bytes in memory, all delayed

# 4. Fetch only what you need — each .load() triggers parallel S3 byte-range reads
step48 = ds.tp.sel(step=48).load()                       # all 30 members, T+48h
ea = ds.tp.sel(step=48,
               latitude=slice(15, -12),                  # GEFS lat is 90..-90
               longitude=slice(25, 52)).load()           # East Africa subset (~5 MB total)
```

## Variables Available

NOAA GEFS 0.25° outputs ~70 variables across the surface and pressure
levels. Common ones referenced by these parquets include:

| Surface Variables | Pressure Level Variables |
|-------------------|--------------------------|
| tp (Total Precipitation) | u/v wind at standard levels |
| t2m (2 m Temperature) | Temperature at standard levels |
| sp (Surface Pressure) | Geopotential at standard levels |
| dswrf (Downward Shortwave Radiation) | Specific humidity |
| tcc (Total Cloud Cover) | |

See [NOAA GEFS variables documentation](https://www.nco.ncep.noaa.gov/pmb/products/gens/) for the full list.

## How It Works

The **Grib-Index-Kerchunk (GIK)** method applies the same principle as
video streaming to weather data:

| Video Streaming | Weather Data Streaming (GIK) |
|-----------------|------------------------------|
| Video split into segments | GRIB split into variable/member/timestep chunks |
| Manifest (`.m3u8`) lists segment URLs + byte ranges | Parquet lists GRIB URLs + byte ranges |
| Player fetches only visible segments | Analysis code fetches only needed variables |
| Full video never downloaded | Full GRIB never downloaded |

## Validation

GIK parquet-derived data was validated against
[Herbie](https://herbie.readthedocs.io/) (independent NOAA GEFS access
library) over the East Africa region (lat -12..15, lon 25..52):

- **Pearson r = 1.0**, RMSE = 0.0, MAE = 0.0 across 11,881 grid points
- Results bit-identical because both paths use the same source GRIB bytes;
  GIK skips the full-file download and decode step.

## Project

Developed by [ICPAC](https://www.icpac.net/) (IGAD Climate Prediction
and Applications Centre) for continuous climate risk monitoring over
East Africa.

- **Funding**: E4DRR (UN CRAF'd) and SEWAA projects
- **Repository**: [icpac-igad/grib-index-kerchunk](https://github.com/icpac-igad/grib-index-kerchunk)
- **Paired dataset**: [`E4DRR/gik-ecmwf-par`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par) (ECMWF IFS ensemble; same architecture)
- **Method documentation**: see [`gefs/README.md`](https://github.com/icpac-igad/grib-index-kerchunk/blob/main/gefs/README.md) and [`gefs/lithops-cr-gik-gefs/BACKFILL.md`](https://github.com/icpac-igad/grib-index-kerchunk/blob/main/gefs/lithops-cr-gik-gefs/BACKFILL.md)
