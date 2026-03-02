---
license: apache-2.0
tags:
  - weather
  - climate
  - ensemble-forecasts
  - ecmwf
  - grib
  - zarr
  - kerchunk
  - byte-range
  - streaming
---

# GIK-ECMWF-PAR: Virtual Reference Parquets for ECMWF IFS Ensemble Forecasts

Lightweight parquet reference files that turn **3--4 GB ECMWF GRIB files into ~140 KB virtual datasets**, enabling Dask-based parallel analysis without downloading the raw data.

## What These Parquets Do

Each parquet file contains a table of `[zarr_key, [s3_url, byte_offset, byte_length]]` references pointing into ECMWF IFS ensemble GRIB files on AWS S3 (`s3://ecmwf-forecasts/`). Instead of downloading full GRIB files, analysis code performs targeted **byte-range reads** fetching only the specific variables, members, and timesteps needed -- typically **2--5% of the original data**.

```
Without GIK:  Download 340 GB of GRIB files per day  --> then process
With GIK:     Read 7 MB of parquets + stream ~5 GB   --> direct analysis
```

## Scale: Parquets vs Source GRIB Data

### Source GRIB Scale (ECMWF IFS 00z only)

| Parameter | Value |
|-----------|-------|
| Timesteps per run | 85 (0--144h at 3h, 150--360h at 6h) |
| Members per timestep | 51 (1 control + 50 perturbed) |
| File size per timestep | ~3--4 GB (all 51 members in one GRIB) |
| **GRIB data per day (00z)** | **~85 x 4 GB = ~340 GB** |

### Annual GRIB Data Referenced

| Year | Coverage | Days | Daily GRIB (00z) | **Total GRIB Referenced** |
|------|----------|------|-------------------|---------------------------|
| 2024 | March 1 -- December 31 | ~306 | ~340 GB | **~101 TB** |
| 2025 | January 1 -- December 31 | 365 | ~340 GB | **~124 TB** |
| 2026 | January 1 -- February 20 | 51 | ~340 GB | **~17 TB** |
| **Total** | | **~722 days** | | **~242 TB** |

### Parquet Reference Files in This Dataset

| Year | Parquets | Size | Compression vs GRIB |
|------|----------|------|---------------------|
| 2024 | ~15,606 | ~2.2 GB | 2.2 GB references ~101 TB (**46,000x**) |
| 2025 | ~18,615 | ~2.6 GB | 2.6 GB references ~124 TB (**48,000x**) |
| 2026 | ~1,581 | ~0.2 GB | 0.2 GB references ~17 TB (**85,000x**) |
| **Total** | **~35,802** | **~5.0 GB** | **5 GB references ~242 TB** |

Each parquet is ~140 KB and represents **one ensemble member for one forecast date**.

## Dataset Structure

```
run_par_ecmwf/
  {YYYY}/{MM}/{YYYYMMDD}/00z/
    {YYYYMMDD}00z-control.parquet     # Control member
    {YYYYMMDD}00z-ens01.parquet       # Ensemble member 1
    {YYYYMMDD}00z-ens02.parquet       # Ensemble member 2
    ...
    {YYYYMMDD}00z-ens50.parquet       # Ensemble member 50

validation_gik_vs_herbie_2024/        # Validation: 10 dates, r > 0.9999
validation_gik_vs_herbie_2025/        # Validation: 11 dates, r > 0.9999
```

## Catalog / Index

A lightweight **catalog.parquet** (1.7 MB) at the repo root indexes all 144,228 parquet files
across the dataset. Use it to discover available dates, runs, and members without listing the
full repo tree.

**Download**: [`catalog.parquet`](catalog.parquet)

| Column | Example | Description |
|--------|---------|-------------|
| `year` | `2024` | Forecast year |
| `month` | `03` | Forecast month |
| `date` | `20240301` | Forecast date (YYYYMMDD) |
| `run` | `00z` | Run hour |
| `member` | `control` | Ensemble member name |
| `filename` | `2024030100z-control.parquet` | Parquet filename |
| `hf_path` | `run_par_ecmwf/2024/03/20240301/00z/...` | Full path in this repo |
| `size_bytes` | `107520` | File size in bytes |

### Quick Start with the Catalog

```python
import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Load the catalog (1.7 MB, indexes all 144k+ files)
catalog_path = hf_hub_download(
    repo_id="E4DRR/gik-ecmwf-par", repo_type="dataset",
    filename="catalog.parquet"
)
catalog = pd.read_parquet(catalog_path)

# 2. Explore what's available
print(catalog.groupby(["year", "month"]).size())       # files per month
print(catalog["run"].unique())                          # ['00z','06z','12z','18z']
print(catalog["member"].nunique())                      # 51

# 3. Filter for a specific date + run
subset = catalog[(catalog["date"] == "20250101") & (catalog["run"] == "00z")]
print(subset[["member", "filename", "size_bytes"]])

# 4. Download a specific parquet using its hf_path
row = subset.iloc[0]
parquet_path = hf_hub_download(
    repo_id="E4DRR/gik-ecmwf-par", repo_type="dataset",
    filename=row["hf_path"]
)
```

### Open Parquets as xarray Dataset

Each parquet contains zarr-style `[key, value]` references into remote GRIB files on S3.
Below is a minimal example that loads all 51 ensemble members for a single date and
streams one variable (total precipitation) into an xarray Dataset.

```python
import json
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

# ── 1. Pick a date from the catalog ──────────────────────────────
catalog = pd.read_parquet(
    hf_hub_download("E4DRR/gik-ecmwf-par", "catalog.parquet", repo_type="dataset")
)
date = "20250101"
subset = catalog[(catalog["date"] == date) & (catalog["run"] == "00z")]

# ── 2. Helper: parquet → zstore dict ─────────────────────────────
def parquet_to_zstore(parquet_path):
    """Read a GIK parquet and return a {zarr_key: value} dict."""
    df = pd.read_parquet(parquet_path)
    zstore = {}
    for _, row in df.iterrows():
        val = row["value"]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        if isinstance(val, str) and val[0] in ("[", "{"):
            val = json.loads(val)
        zstore[row["key"]] = val
    return zstore

# ── 3. Helper: fetch one GRIB chunk from S3 ──────────────────────
s3 = fsspec.filesystem("s3", anon=True)

def fetch_step(ref):
    """Fetch raw GRIB bytes from S3 using [url, offset, length]."""
    url, offset, length = ref[0], ref[1], ref[2]
    if not url.endswith(".grib2"):
        url += ".grib2"
    with s3.open(url, "rb") as f:
        f.seek(offset)
        return f.read(length)

# ── 4. Load all 51 members for one variable (tp) ─────────────────
STEPS = list(range(0, 145, 3)) + [150, 156, 162, 168]  # 53 lead times
GRID = (721, 1440)  # ECMWF 0.25 deg global

try:
    import gribberish
    decode = lambda b: gribberish.parse_grib_array(b, 0).reshape(GRID)
except ImportError:
    # Fallback: cfgrib (slower)
    import tempfile, os
    def decode(b):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
        tmp.write(b); tmp.close()
        ds = xr.open_dataset(tmp.name, engine="cfgrib")
        arr = ds[list(ds.data_vars)[0]].values.copy(); ds.close()
        os.unlink(tmp.name)
        return arr

members_data = {}
for _, row in subset.iterrows():
    pq_path = hf_hub_download(
        "E4DRR/gik-ecmwf-par", row["hf_path"], repo_type="dataset"
    )
    zstore = parquet_to_zstore(pq_path)

    # Find tp references for each lead time
    data = np.full((len(STEPS), *GRID), np.nan, dtype=np.float32)
    member_key = row["member"].replace("_", "")

    step_refs = []
    for i, h in enumerate(STEPS):
        for pattern in [
            f"step_{h:03d}/tp/sfc/{member_key}/0.0.0",
            f"step_{h:03d}/tp/surface/{member_key}/0.0.0",
        ]:
            if pattern in zstore and isinstance(zstore[pattern], list):
                step_refs.append((i, zstore[pattern]))
                break

    # Parallel S3 fetches (8 threads, I/O bound)
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(fetch_step, ref): i for i, ref in step_refs}
        for fut in futs:
            try:
                data[futs[fut]] = decode(fut.result())
            except Exception:
                pass

    members_data[row["member"]] = data

# ── 5. Assemble xarray Dataset ───────────────────────────────────
ds = xr.Dataset(
    {"tp": (["member", "step", "latitude", "longitude"],
            np.stack([members_data[m] for m in sorted(members_data)]))},
    coords={
        "member":    sorted(members_data.keys()),
        "step":      STEPS,
        "latitude":  np.linspace(90, -90, 721),
        "longitude": np.linspace(-180, 179.75, 1440),
    },
)
print(ds)
# <xarray.Dataset>  Dimensions: member=51, step=53, latitude=721, longitude=1440
```

### Open Parquets as Lazy xarray Dataset (Virtual / On-Demand)

Same parquets, but opened **lazily with dask** — parquets are parsed instantly,
S3 byte-range reads happen **only when you call `.load()`**. This lets you build
a full 51-member dataset in seconds and then selectively materialize slices.

```python
import json
import dask
import dask.array as da
import fsspec
import gribberish
import numpy as np
import pandas as pd
import xarray as xr
from huggingface_hub import hf_hub_download

# ── 1. Setup ─────────────────────────────────────────────────────
catalog = pd.read_parquet(
    hf_hub_download("E4DRR/gik-ecmwf-par", "catalog.parquet", repo_type="dataset")
)
date, var = "20250101", "tp"
subset = catalog[(catalog["date"] == date) & (catalog["run"] == "00z")]

STEPS = list(range(0, 145, 3)) + [150, 156, 162, 168]  # 53 lead times
GRID = (721, 1440)
s3 = fsspec.filesystem("s3", anon=True)

def parquet_to_zstore(path):
    zstore = {}
    for _, row in pd.read_parquet(path).iterrows():
        val = row["value"]
        if isinstance(val, bytes): val = val.decode("utf-8")
        if isinstance(val, str) and len(val) > 0 and val[0] in ("[", "{"):
            val = json.loads(val)
        zstore[row["key"]] = val
    return zstore

# ── 2. Delayed fetcher: executes only when .load() is called ─────
def make_lazy_chunk(ref):
    @dask.delayed
    def _fetch():
        url, offset, length = ref[0], ref[1], ref[2]
        if not url.endswith(".grib2"): url += ".grib2"
        with s3.open(url, "rb") as f:
            f.seek(offset)
            return gribberish.parse_grib_array(f.read(length), 0) \
                .reshape(GRID).astype(np.float32)
    return da.from_delayed(_fetch(), shape=GRID, dtype=np.float32)

# ── 3. Build lazy dataset (instant — no S3 reads) ────────────────
member_arrays, member_names = [], []
for _, row in subset.iterrows():
    pq = hf_hub_download("E4DRR/gik-ecmwf-par", row["hf_path"], repo_type="dataset")
    zs = parquet_to_zstore(pq)
    mk = row["member"].replace("_", "")
    steps = []
    for h in STEPS:
        ref = None
        for p in [f"step_{h:03d}/{var}/sfc/{mk}/0.0.0",
                  f"step_{h:03d}/{var}/surface/{mk}/0.0.0"]:
            if p in zs and isinstance(zs[p], list):
                ref = zs[p]; break
        steps.append(make_lazy_chunk(ref) if ref else
                     da.full(GRID, np.nan, dtype=np.float32))
    member_arrays.append(da.stack(steps, axis=0))
    member_names.append(row["member"])

ds = xr.Dataset(
    {var: (["member", "step", "latitude", "longitude"],
           da.stack(member_arrays, axis=0))},
    coords={"member": member_names, "step": STEPS,
            "latitude": np.linspace(90, -90, 721),
            "longitude": np.linspace(-180, 179.75, 1440)},
)
print(ds)
# <xarray.Dataset>  Dimensions: member=51, step=53, latitude=721, longitude=1440
# tp: dask.array<...>  ← zero bytes in memory, all delayed

# ── 4. Fetch only what you need ──────────────────────────────────
# Each .load() triggers S3 byte-range reads (~2 MB per member per step)
step24 = ds["tp"].sel(step=24).load()                     # 1 step, all members
ea = ds["tp"].sel(step=24, latitude=slice(25, -14),
                  longitude=slice(19, 55)).load()          # East Africa subset
```

### Coverage Summary (from catalog)

| Year | Months | Dates | Files | Total Size |
|------|--------|-------|-------|------------|
| 2024 | Mar--Dec | 306 | ~62,424 | ~6.5 GB |
| 2025 | Jan--Dec | 365 | ~75,504 | ~7.8 GB |
| 2026 | Jan--Feb | 51 | ~7,344 | ~0.8 GB |
| **Total** | | **720** | **144,228** | **~17.8 GB** |

## How It Works

The **Grib-Index-Kerchunk (GIK)** method applies the same principle as video streaming to weather data:

| Video Streaming | Weather Data Streaming (GIK) |
|-----------------|------------------------------|
| Video split into segments | GRIB split into variable/member/timestep chunks |
| Manifest (.m3u8) lists segment URLs + byte ranges | Parquet lists GRIB URLs + byte ranges |
| Player fetches only visible segments | Analysis code fetches only needed variables |
| Full video never downloaded | Full GRIB (3--4 GB) never downloaded |

## Usage with Dask

These parquets are designed for **parallel analysis on Dask clusters**. Each member's parquet is independent -- workers read their assigned parquet, fetch GRIB bytes via S3 byte-range reads, and decode using gribberish (Rust, ~25 ms/chunk).

| Approach | 12 vars x 51 members x 9 steps | Time |
|----------|-------------------------------|------|
| Download full GRIBs | ~340 GB transfer | Hours |
| GIK + single machine (8 threads) | ~4.5 GB byte-range reads | ~24 min |
| GIK + Coiled Dask (20 workers) | ~4.5 GB distributed reads | ~3 min |

## Variables Available

All ECMWF IFS ensemble variables are referenced, including:

| Surface Variables | Pressure Level Variables |
|-------------------|--------------------------|
| tp (Total Precipitation) | u/v wind at multiple levels |
| 2t (2m Temperature) | Temperature at multiple levels |
| sp (Surface Pressure) | Geopotential at multiple levels |
| ssr, ssrd (Solar Radiation) | Specific humidity |
| tcw, tcwv (Cloud Water) | |
| tcc (Total Cloud Cover) | |
| sf (Snowfall), ro (Runoff) | |

## Validation

GIK parquet-based data was validated against [Herbie](https://herbie.readthedocs.io/) (independent ECMWF access library) across 21 dates spanning 2024--2025:

- **Pearson r > 0.9999** for both ensemble mean and spread
- **RMSE < 2e-04 m** for total precipitation
- Tiny differences arise from decoder floating-point representation (gribberish vs cfgrib)

## Project

Developed by [ICPAC](https://www.icpac.net/) (IGAD Climate Prediction and Applications Centre) for continuous climate risk monitoring over East Africa.

- **Funding**: E4DRR (UN CRAF'd) and SEWAA projects
- **Repository**: [icpac-igad/grib-index-kerchunk](https://github.com/icpac-igad/grib-index-kerchunk)
- **Method documentation**: See `ecmwf/docs/gik_vs_herbie_comparison.md` in the repository
