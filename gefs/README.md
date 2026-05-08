# GEFS East Africa Ensemble Processing Pipeline

## Overview

Complete pipeline for processing GEFS (Global Ensemble Forecast System) ensemble data for the East Africa region using the **Grib-Index-Kerchunk (GIK) method** for efficient GRIB data access without full file scanning.

- **30 ensemble members** (gep01-gep30)
- **81 forecast timesteps** (0-240h at 3h intervals)
- **0.25° global grid** (721x1440), subset to East Africa (lat -12..15, lon 25..52)
- **Data source**: `s3://noaa-gefs-pds/` (anonymous access)

## Three-Stage Pipeline

The pipeline creates lightweight parquet reference files containing `[url, byte_offset, byte_length]` triplets pointing into remote GRIB files on AWS S3.

### Stage 1: Template (one-time, frozen at reference date 2024-11-12)

The production "template" is two artefacts, both baked into the Cloud Run
Docker image and also published to HuggingFace:

| Artefact | Size | How it was built (one-time, on `20241112`) | Used for |
|----------|------|--------------------------------------------|----------|
| `gik-fmrc-gefs-20241112.tar.gz` | ~10 MB | `gefs/dev-test/gefs_index_preprocessing_fixed.py` running `build_idx_grib_mapping` over **every (member, timestep) = 30 × 81 = 2,430 GRIB files** | Stage 2 byte-range merging |
| `gefs-deflated-store-template-20241112.parquet` | 26 KB | One-time `scan_grib(f000, f003) → grib_tree → strip_datavar_chunks` | Stage 3 zarr metadata skeleton (convenience — equivalent zstore is also present inside any `rt000` slice of the tar.gz) |

**At runtime** (`lithops-cr-gik-gefs/run_lithops_gefs.py`), Stage 1 is just
`pd.read_parquet(...)` on the 26 KB file (~0.05 s). **No `scan_grib` is
called in the production Lithops path.** The single-machine driver
`run_day_gefs_ensemble_full.py` (now in `dev-test/`) has a `scan_grib`
fallback for environments without the template, but it's not on the
production path.

### Stage 2: Index-Based Reference Building

Reads `.idx` files from S3 (one per GRIB file), extracts byte offsets, and merges with the per-(member, timestep) mapping parquets that were pre-built into the template archive (`gik-fmrc-gefs-20241112.tar.gz`). I/O-bound, ~25-30s per member.

### Stage 3: Final Zarr Store Assembly

Merges the deflated store with fresh byte-range references to produce one parquet file per ensemble member.

## Quick Start

### Production-scale parquet generation (Lithops Cloud Run)

For multi-date backfills or ongoing daily generation, use the Cloud
Run path — see [`lithops-cr-gik-gefs/`](lithops-cr-gik-gefs/) for the
full setup including IAM, Dockerfile, deployment, and the
[`BACKFILL.md`](lithops-cr-gik-gefs/BACKFILL.md) ops guide.

### Single Member to Zarr (consumer side)

```bash
# Convert one member's parquet to zarr using gribberish (~80x faster than cfgrib)
uv run run_single_gefs_to_zarr_gribberish.py 20250106 00 gep01 --region east_africa
```

### Build a TP NetCDF for one date

```bash
uv run create_gik_tp_netcdf.py --date 20250106 --max-members 30
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

## Scripts on the main path (v1.0)

| Script | Step | Purpose |
|--------|------|---------|
| `gefs_util.py` | shared | Core utilities — template loading, scan_grib, zarr store ops |
| `run_gefs_preprocessing.py` | 1 | One-time template / mapping creation |
| `lithops-cr-gik-gefs/` | 3 | Cloud Run + Lithops orchestration for production |
| `create_gik_tp_netcdf.py` | 4 | GIK TP extraction + gribberish streaming → NetCDF |
| `run_single_gefs_to_zarr_gribberish.py` | 4 | Single member → zarr (gribberish, ~80× faster than cfgrib) |
| `compare_gik_herbie_gefs.py` | validation | GIK vs Herbie comparison PNGs |
| `fetch_tp_herbie_gefs.py` | validation | Herbie-based TP fetching for validation |
| `validate_gik_vs_herbie_2022_2026.py` | validation | Multi-year random-date Herbie comparison (local pipeline) |
| `validate_hf_gik_vs_herbie.py` | validation | Same comparison but pulling refs from `E4DRR/gik-gefs-par` on HF |
| `upload_parquets_to_hf.py` | publish | Mirror per-member parquets + monthly aggregates + catalog to HF |
| `hf_README.md` | docs | Dataset card on `E4DRR/gik-gefs-par` |

The single-machine ensemble driver, cfgrib-decoder variants, ensemble
analytics scripts, and bundled template parquet were moved to
[`dev-test/`](dev-test/) in the v1.0 cleanup — see
[`../RELEASE_CLEANUP_PLAN.md`](../RELEASE_CLEANUP_PLAN.md) §4.

## Key folders

| Folder | Description |
|--------|-------------|
| `lithops-cr-gik-gefs/` | Cloud Run + Lithops setup (Dockerfile, lithops_config, BACKFILL.md, COST_ANALYSIS.md) |
| `gik_vs_herbie_gefs/` | Comparison PNG plots and statistics |
| `docs/` | Method documentation including `GEFS_SCAN_GRIB_TO_TEMPLATE.md` |
| `dev-test/` | Archived experiments, single-machine drivers, bundled template parquet |

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

### Open a monthly aggregate and pull a single date (with predicate pushdown)

Each monthly aggregate is sorted by `(date, member)` and written with
`row_group_size=60` (one row group per date). PyArrow's predicate
pushdown skips non-matching row groups, so a single-date filter against
a 44 MB monthly file reads only ~1.5 MB via HuggingFace range requests.

```python
import pandas as pd

URL = "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet"

# Predicate pushdown — pyarrow only reads the row groups matching the filter
df = pd.read_parquet(
    URL,
    filters=[("date", "=", "20240615"), ("member", "=", "gep01")],
)
print(f"{len(df)} rows, columns: {list(df.columns)}")
# Schema: key, value, date, source_file, member
```

To pull all 30 members for one date:

```python
df = pd.read_parquet(URL, filters=[("date", "=", "20240615")])
print(f"{df.member.nunique()} members, {len(df)} rows total")
```

For coverage discovery without downloading any aggregate, use the
catalog at the repo root:

```python
catalog = pd.read_parquet("hf://datasets/E4DRR/gik-gefs-par/catalog.parquet")
print(f"{len(catalog):,} files, dates {catalog.date.min()}..{catalog.date.max()}")
```

### Open a single date as a lazy xarray Dataset

Each (date, member) gives 2 rows in the aggregate: `key="refs"` (a
~1.5 MB Python-dict-literal containing the full kerchunk zstore) and
`key="version"`. Use `ast.literal_eval` (not `json.loads` — the blob
uses single quotes), then build a dask-delayed array per chunk so S3
byte-range reads only fire on `.load()`:

```python
import ast, base64
import dask, dask.array as da
import fsspec, gribberish
import numpy as np, pandas as pd, xarray as xr

GEFS_GRID = (721, 1440)
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)
STEPS = [3, 6, 12, 24, 48, 72, 120, 168, 240]

# Read all 30 members for one date in one HF call
df = pd.read_parquet(
    "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet",
    filters=[("date", "=", "20240615")],
)

def member_zstore(sub):
    blob = sub[sub["key"] == "refs"].iloc[0]["value"]
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    return ast.literal_eval(blob)

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

member_arrays, member_names = [], []
for member, sub in df.groupby("member"):
    zs = member_zstore(sub)
    val = zs.get("tp/accum/surface/step/0", "")
    step_hours = (np.frombuffer(base64.b64decode(val[7:]), dtype="<f8")
                  if isinstance(val, str) and val.startswith("base64:")
                  else np.arange(0, 243, 3, dtype=float))
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
# tp: dask.array<...>   ← zero bytes in memory, all delayed

# Fetch only what you need — each .load() triggers parallel S3 byte-range reads
ea = ds.tp.sel(step=48,
               latitude=slice(15, -12),
               longitude=slice(25, 52)).load()           # ~5 MB total for all 30 members
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
