# Grib-Index-Kerchunk (GIK) — Cloud-Native Weather Data Streaming

[![v1.0](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/icpac-igad/grib-index-kerchunk/releases/tag/v1.0.0)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)


CASE STUDY: [preprint](https://drive.google.com/file/d/1-BUnCVNU5EWdytSlZrria5N6e4cK_H8v/view?usp=drive_link)

GIK turns multi-GB GRIB files into ~140 KB virtual references — enabling
xarray analysis and Dask-cluster ensemble work without ever downloading the
raw GRIBs. Based on the
[dynamic-Grib-chunking method](https://github.com/asascience-open/nextgen-dmac/commit/6b3286627070c36127ec97b7dbbb88b0ab481f06).

```
Without GIK:  Download tens of GB of GRIB files per day  →  then process
With GIK:     Read ~10 MB of parquet refs + stream slices →  direct analysis
```

The trick: `kerchunk.scan_grib` is replaced by reading the published `.idx`
sidecar files alongside every GRIB on S3 and merging them with a one-time
template — so generating a new day's references takes seconds, not the
~25 minutes it would take to scan every member's worth of GRIB files.

---

## The 4-step pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Template      ONE-TIME per model on a reference date:            │
│                  run build_idx_grib_mapping over EVERY (member,     │
│                  timestep) → tar.gz of mapping parquets             │
│                  (GEFS: 30×81 parquets, ECMWF: 51×85 parquets).     │
│                  Published to HF, baked into the Docker image.      │
│                  Frozen at 2024-11-12 (GEFS) / 2024-05-29 (ECMWF).  │
├─────────────────────────────────────────────────────────────────────┤
│ 2. Index         For each new date+member: fetch the .idx (~few KB),│
│                  look up byte ranges, merge with the template's     │
│                  mapping parquets — NO scan_grib at runtime.        │
├─────────────────────────────────────────────────────────────────────┤
│ 3. Build parquet Combine step 1 (zarr metadata) + step 2 (byte-range│
│   & save         refs) → write one parquet per member-date.         │
│                  Production: Lithops on Cloud Run (lithops-cr-gik-*)│
├─────────────────────────────────────────────────────────────────────┤
│ 4. Stream →      Read the parquet, fetch only the byte ranges you   │
│   xarray         need from S3, decode with gribberish (~80× faster  │
│                  than cfgrib), wrap in xarray Dataset / Dask array. │
│                  Validators: Herbie + dynamical-earth zarr.         │
└─────────────────────────────────────────────────────────────────────┘
```

> **Note on Step 1.** Earlier descriptions of this repo claimed Step 1
> was a `scan_grib(f000, f003)` two-file scan. That is **not** what
> produces the production templates. The tar.gz baked into the Docker
> image is the output of a full per-timestep `build_idx_grib_mapping`
> sweep (`gefs/dev-test/gefs_index_preprocessing_fixed.py` and
> `ecmwf/ecmwf_index_preprocessing.py`). A separate 27 KB GEFS
> `gefs-deflated-store-template-20241112.parquet` was made one-time by
> a 2-file `scan_grib` and is bundled alongside the tar.gz, but it is
> a convenience artefact only — the same zarr-metadata skeleton can be
> reconstructed from any single `rt000` parquet inside the tar.gz, which
> is what the ECMWF Lithops path does.

### Weather Data vs Video Streaming — same idea

| Aspect | Video (HTML5) | Weather (GIK) |
|--------|---------------|---------------|
| Download | Full file for playback | Full GRIB for analysis |
| Streaming | Adaptive segments on demand | Byte-range refs on demand |
| Manifest | `.m3u8` listing segments | Parquet listing GRIB byte ranges |
| Efficiency | No full download | Fetch 2–5% of source data |
| Scaling | Across devices/networks | Dask cluster / Lithops Cloud Run |

---

## Supported Products

| Product | Source | Members | Timesteps | Grid | Status |
|---------|--------|---------|-----------|------|--------|
| **GEFS** | `s3://noaa-gefs-pds/` | 30 ensemble (gep01–gep30) | 81 (0–240 h at 3 h) | 0.25° (721×1440) | **Production** |
| **ECMWF IFS** | `s3://ecmwf-forecasts/` | 51 (1 control + 50 ens) | 85 (0–360 h) | 0.25° (721×1440) | **Production** |
| GFS | `s3://noaa-gfs-bdp-pds/` | Deterministic | 240 h | 0.25° | Legacy (`gfs/`) |
| CFS | `s3://noaa-cfs-pds/` | Seasonal | Variable | 1.0° | Experimental (`cfs/`) |

Production datasets are mirrored to public HuggingFace repos:

| HF dataset | Coverage | Files |
|------------|----------|-------|
| [`E4DRR/gik-gefs-par`](https://huggingface.co/datasets/E4DRR/gik-gefs-par) | GEFS 00z, 2020-09-25 → 2025-12-31 | 64 monthly aggregates + per-member mirror + catalog |
| [`E4DRR/gik-ecmwf-par`](https://huggingface.co/datasets/E4DRR/gik-ecmwf-par) | ECMWF IFS, 2024–2026 (all run hours) | 144,228 per-member parquets + catalog |

---

## Repository layout (v1.0)

```
grib-index-kerchunk/
├── README.md                 ← you are here
├── RELEASE_CLEANUP_PLAN.md   ← v1.0 file-categorisation record
├── LICENSE                   ← Apache 2.0
│
├── ecmwf/                    ← ECMWF IFS pipeline (production)
│   ├── README.md             ← step-by-step usage
│   ├── hf_README.md          ← dataset card on HF
│   ├── lithops-cr-gik-ecmwf/ ← step 3 Cloud Run deployment
│   ├── docs/                 ← method docs (gik vs herbie etc.)
│   └── dev-test/             ← archived experiments + older variants
│
├── gefs/                     ← NOAA GEFS pipeline (production)
│   ├── README.md             ← step-by-step usage
│   ├── hf_README.md          ← dataset card on HF
│   ├── lithops-cr-gik-gefs/  ← step 3 Cloud Run deployment
│   ├── docs/                 ← method docs
│   └── dev-test/             ← archived experiments + older variants
│
├── tutorial/                 ← self-contained walk-throughs
│   ├── ecmwf/
│   └── gefs/
│
├── gfs/                      ← legacy GFS pipeline (pre-v1.0)
├── cfs/                      ← experimental CFS pipeline
└── devops/                   ← (private) deployment tooling
```

---

## Quick start — read one date for one member

Once a date's parquet has been published to HF (or sits on your GCS bucket),
fetching it as an xarray Dataset is ~3 seconds + ~5 MB of S3 byte-range
reads:

```python
import ast, fsspec, gribberish
import numpy as np, pandas as pd

GEFS_GRID = (721, 1440)

# 1. Pull one date+member's refs in a single HF call
df = pd.read_parquet(
    "hf://datasets/E4DRR/gik-gefs-par/run_par_gefs_agg/monthly_agg/2024/06_00z.parquet",
    filters=[("date", "=", "20240615"), ("member", "=", "gep01")],
)
zstore = ast.literal_eval(
    df[df["key"] == "refs"].iloc[0]["value"].decode("utf-8")
)

# 2. Find TP byte-range refs by step index
tp_refs = {
    int(k.rsplit("/", 1)[1].split(".")[0]): v
    for k, v in zstore.items()
    if k.startswith("tp/accum/surface/tp/") and isinstance(v, list)
}

# 3. Stream T+48h — ~3 MB pulled from NOAA's public S3
s3 = fsspec.filesystem("s3", anon=True)
url, off, ln = tp_refs[16]                 # step idx 16 ≈ 48 h
with s3.open(url, "rb") as f:
    f.seek(off)
    raw = f.read(ln)
tp_48h = gribberish.parse_grib_array(raw, 0).reshape(GEFS_GRID)
```

For the lazy-Dask ensemble pattern (30 members × 9 steps as one
`xr.Dataset`), see [`gefs/README.md`](gefs/README.md) §"Open a single
date as a lazy xarray Dataset" or [`gefs/hf_README.md`](gefs/hf_README.md).

---

## Validation

GIK references produce **bit-identical** APCP / TP fields to
[Herbie](https://herbie.readthedocs.io/) (independent GRIB access library)
because both paths read the same source bytes from S3 — GIK just skips the
full GRIB download:

| Pipeline | Validator script | Result |
|----------|------------------|--------|
| GEFS local pipeline | `gefs/validate_gik_vs_herbie_2022_2026.py` | r=1.0, RMSE=0 across 50+ random dates |
| GEFS HF aggregate (consumer-side) | `gefs/validate_hf_gik_vs_herbie.py` | r=1.0, RMSE=0 |
| ECMWF | `ecmwf/validate_gik_vs_herbie_2025.py` | r > 0.9999 |

---

## How to cite / contribute

GIK is developed by [ICPAC](https://www.icpac.net/) (IGAD Climate
Prediction and Applications Centre) for continuous climate-risk
monitoring over East Africa. Funded by the **E4DRR** (UN CRAF'd) and
**SEWAA** projects.

- Method paper / tutorials: see `tutorial/`
- Issues / PRs welcome: <https://github.com/icpac-igad/grib-index-kerchunk>
- Licensed under Apache 2.0 — see [`LICENSE`](LICENSE)

---

## Where things moved in v1.0

The `v1.0.0` cleanup archived all experimental, superseded, or
application-specific scripts to `<datasource>/dev-test/`. See
[`RELEASE_CLEANUP_PLAN.md`](RELEASE_CLEANUP_PLAN.md) for the full
inventory and per-file rationale.
