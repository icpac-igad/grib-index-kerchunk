# ECMWF GIK-cGAN Integration

This folder contains standalone scripts for processing ECMWF ensemble data using the Grib-Index-Kerchunk (GIK) method and preparing input data for cGAN rainfall downscaling, as well as a materialized Icechunk store pipeline for long-term archival of total precipitation (tp) over East Africa.

**Source Repository:** [grib-index-kerchunk](https://github.com/icpac-igad/grib-index-kerchunk)
**Source Commit:** `a487612ece68f2ce7e0c278a2fa1d5170349877e`

## Overview

The workflow consists of three main phases:

```
Phase 1: Create Parquet Reference Files (GIK Pipeline via Lithops)
    ECMWF S3 GRIB -> Index Processing + Template -> Stage3 Parquet Files -> HuggingFace

Phase 2: Materialized Icechunk Store (Data Streaming)
    HuggingFace Parquets -> Coiled/Dask Workers -> S3 GRIB Byte-Range Reads
    -> Decode with gribberish -> Subset to East Africa -> Icechunk Store on GCS

Phase 3: Inspection, Visualisation & Forecast Maps
    Icechunk Store -> Daily Forecast Maps (PNG) -> Animated GIFs
```

## Directory Structure

```
ecmwf/
├── README.md                           # This file
├── ECMWF_cGAN_INFERENCE_WORKFLOW.md    # Detailed workflow documentation
├── GRIBBERISH_EXPERIMENT_REPORT.md     # scan_grib replacement experiment
│
├── # ── Phase 1: GIK Parquet Creation ──
├── run_ecmwf_tutorial.py               # Entry point for Phase 1 pipeline
├── stream_cgan_variables.py            # Phase 2 (cGAN): Local data streaming
├── stream_cgan_variables_coiled_simple.py  # Phase 2 (cGAN): Coiled parallel streaming
├── plot_cgan_maps.py                   # Visualization: 4x3 panel maps
├── test_gribberish_vs_scangrib.py      # Benchmark: .index vs scan_grib
├── gik_ecmwf/                          # Core GIK processing modules
│   ├── __init__.py
│   ├── ecmwf_util.py                   # Core utilities and variable definitions
│   ├── ecmwf_ensemble_par_creator_efficient.py  # Stage 1: GRIB scanning (legacy)
│   ├── ecmwf_three_stage_multidate.py  # Three-stage pipeline orchestration
│   ├── ecmwf_index_processor.py        # Stage 2: Index-based processing
│   └── utils_ecmwf_step1_scangrib.py   # GRIB scanning utilities
│
├── # ── Phase 1: Lithops Cloud Deployment ──
├── lithops-cr-gik-ecmwf/              # Lithops Cloud Run deployment
│   ├── run_lithops_ecmwf.py           # Self-contained Lithops script (~960 lines)
│   ├── lithops_config.yaml            # Lithops backend config
│   ├── run_backfill_06z.sh            # Backfill scripts for different run hours
│   ├── run_backfill_12z.sh
│   ├── run_backfill_18z.sh
│   ├── OPERATIONS_GUIDE.md            # Operations and monitoring guide
│   └── SETUP_NEW_MACHINE.md           # Machine setup instructions
│
├── # ── Phase 2: Icechunk Materialized Store ──
├── ecmwf_ea_tp_icechunk.py            # Init / Fill / Verify Icechunk store
├── coiled-data.json                    # GCS service account for Icechunk + Coiled
│
├── # ── Phase 3: Inspection & Visualisation ──
├── inspect_icechunk_store.py           # Daily forecast maps + animated GIFs
├── ea_ghcf_simple.geojson              # East Africa (GHACOF) country boundaries
│
├── # ── Supporting Files ──
├── compare_gik_herbie.py               # GIK vs Herbie comparison
├── fetch_tp_herbie.py                  # Herbie-based data fetching
├── cmorph_east_africa_icechunk.py      # CMORPH Icechunk store (reference)
└── docs/                               # Architecture docs, analysis reports
```

---

## Phase 1: GIK Parquet Creation

### Overview

Creates lightweight parquet reference files containing `[url, byte_offset, byte_length]` triplets pointing into remote GRIB files on S3. These parquets are the foundation for both the cGAN streaming pipeline and the Icechunk materialized store.

### Running Locally

```bash
# Recommended: Template fast-path + parallel Stage 2 (~10.5 min)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 8

# Upload parquets to GCS (needed for Coiled streaming)
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan --parallel-workers 8 --upload-gcs
```

**Output:** `ecmwf_three_stage_YYYYMMDD_HHz/` directory with `stage3_*.parquet` files

### Running via Lithops (Production)

The production deployment uses **Lithops** with Google Cloud Run to process multiple dates in parallel. Each date is processed independently on a separate Cloud Run container.

```
Local machine
    │
    ▼
Lithops FunctionExecutor(backend='gcp_cloudrun')
    │   serializes process_ecmwf_date() via cloudpickle
    │   uploads to GCS
    │
    ├──► Cloud Run Worker 1 → process date 2026-02-01 → upload parquets to HuggingFace
    ├──► Cloud Run Worker 2 → process date 2026-02-02 → upload parquets to HuggingFace
    └──► Cloud Run Worker N → process date 2026-02-0N → upload parquets to HuggingFace
```

The Lithops script (`lithops-cr-gik-ecmwf/run_lithops_ecmwf.py`) is **fully self-contained** (~960 lines) with zero imports from the GIK codebase. This is a hard requirement of cloudpickle serialization.

```bash
cd lithops-cr-gik-ecmwf/

# Process last 7 days
uv run run_lithops_ecmwf.py --days-back 7

# Backfill a specific date range
uv run run_lithops_ecmwf.py --start-date 20240301 --end-date 20240401

# Backfill specific run hours
bash run_backfill_06z.sh
bash run_backfill_12z.sh
```

See `lithops-cr-gik-ecmwf/OPERATIONS_GUIDE.md` for monitoring, troubleshooting, and operational procedures.
See `lithops-cr-gik-ecmwf/SETUP_NEW_MACHINE.md` for setting up a new deployment machine.

**Parquet output destination:** Parquets are uploaded to HuggingFace (`E4DRR/gik-ecmwf-par`) and/or GCS (`gs://gik-ecmwf-aws-tf/run_par_ecmwf/`), organised as:

```
E4DRR/gik-ecmwf-par/run_par_ecmwf/{YYYY}/{MM}/{YYYYMMDD}/00z/
├── {YYYYMMDD}00z-control.parquet
├── {YYYYMMDD}00z-ens_01.parquet
├── ...
└── {YYYYMMDD}00z-ens_50.parquet
```

---

## Phase 2: Materialized Icechunk Store

### Overview

The Icechunk store materializes ECMWF IFS ensemble total precipitation (tp) over East Africa into a versioned, Git-like Zarr store on GCS. It reads the parquet reference files from Phase 1 (hosted on HuggingFace), uses Coiled/Dask workers to stream raw GRIB bytes from S3, decodes with gribberish, subsets to the East Africa bounding box, and writes into the Icechunk store with per-date commits.

### Prerequisites

- **Phase 1 parquets must exist** on HuggingFace (`E4DRR/gik-ecmwf-par`) for the target date range. Run the Lithops pipeline first if they are missing.
- **GCS service account** (`coiled-data.json`) with read/write access to `gs://gik-ecmwf-aws-tf/`.
- **Coiled account** configured (the fill step launches a Coiled cluster on GCP `europe-west1`).

### Store Architecture

```
gs://gik-ecmwf-aws-tf/ecmwf-ea-ic-store/   (Icechunk repository)

Dataset:
    Dimensions:  (init_date: 671, member: 51, lead_time: 53, lat: 157, lon: 145)
    Coordinates:
      * init_date  datetime64    2024-03-01 ... 2025-12-31
      * member     <U10          'control', 'ens_01', ..., 'ens_50'
      * lead_time  timedelta64   0h, 3h, 6h, ..., 144h, 150h, 156h, 162h, 168h
      * lat        float64       25.0, 24.75, ..., -14.0
      * lon        float64       19.0, 19.25, ..., 55.0
    Variables:
        tp         (init_date, member, lead_time, lat, lon) float32  ~165 GB
    Attributes:
        title:        ECMWF IFS Ensemble TP — East Africa Subset
        institution:  ICPAC
        region:       East Africa (ICPAC)
        lat_range:    -14 to 25
        lon_range:    19 to 55
```

### Key Design Decisions

| Feature | Detail |
|---|---|
| **Per-date commits** | Each date is committed individually to Icechunk after all 51 members are written. This prevents orphaned chunks on crash/OOM and enables automatic resume. |
| **Per-member Dask tasks** | One Dask task per (date, member) = 51 tasks per date. Each task returns ~4.6 MB. Gives Dask full visibility for load balancing across workers. |
| **Resume detection** | Parses commit messages (`"fill date {idx} ({date_str}): N/51 members"`) to find completed dates and skips them on restart. |
| **gribberish decoding** | ~25ms/chunk vs ~2000ms for cfgrib. Falls back to cfgrib if gribberish fails. |
| **Worker region** | Coiled cluster runs in `europe-west1` (near ECMWF S3 bucket) for low-latency S3 reads. |

### Script: `ecmwf_ea_tp_icechunk.py`

Three subcommands: `init`, `fill`, `verify`.

#### Step 1: Initialise the Store

Creates an empty template Icechunk store with the correct dimensions, coordinates, and chunk layout. Data arrays are filled with NaN (uses `compute=False` / GLAD pattern).

```bash
uv run ecmwf_ea_tp_icechunk.py init \
    --start-date 20240301 --end-date 20251231
```

| Argument | Default | Description |
|---|---|---|
| `--start-date` | `20240301` | First date (YYYYMMDD) |
| `--end-date` | `20251231` | Last date (YYYYMMDD) |
| `--gcs-bucket` | `gik-ecmwf-aws-tf` | GCS bucket |
| `--gcs-prefix` | `ecmwf-ea-ic-store` | GCS prefix |
| `--service-account` | `coiled-data.json` | GCS service account JSON |
| `--local` | None | Local path (overrides GCS) |

#### Step 2: Fill the Store

Launches a Coiled cluster on GCP, distributes (date, member) tasks to workers, collects results, and writes to Icechunk with per-date commits.

```bash
# Fill a single month (recommended for testing)
uv run ecmwf_ea_tp_icechunk.py fill \
    --start-date 20240301 --end-date 20240401 \
    --n-workers 50

# Fill full date range (production)
uv run ecmwf_ea_tp_icechunk.py fill \
    --start-date 20240301 --end-date 20251231 \
    --n-workers 50
```

| Argument | Default | Description |
|---|---|---|
| `--start-date` | `20240301` | First date to fill |
| `--end-date` | `20251231` | Last date to fill |
| `--n-workers` | `20` | Number of Coiled workers |
| `--commit-batch` | `20` | Dates per processing batch (cluster is recreated between batches) |

**Cluster configuration:**

| Setting | Value |
|---|---|
| VM type | `e2-standard-4` (4 vCPU, 16 GB RAM) |
| Region | `europe-west1` |
| Package sync | Enabled (auto-syncs local environment) |
| Idle timeout | 30 minutes |

**Performance (measured on 32-date test run):**

| Metric | Value |
|---|---|
| Dates | 32 (20240301–20240401) |
| Workers | 50 |
| Tasks per batch | 32 dates x 51 members = 1,632 |
| Total time | 20.6 minutes |
| Per-date time | ~38 seconds |
| Failures | 0 |

**Resume behaviour:** If the fill is interrupted (OOM, timeout, network), simply re-run the same command. It will parse commit history, find completed dates, and resume from the next unfilled date.

#### Step 3: Verify the Store

Inspects the store, reports dimensions, and spot-checks a few dates for non-NaN data.

```bash
uv run ecmwf_ea_tp_icechunk.py verify
```

| Argument | Default | Description |
|---|---|---|
| `--spot-check` / `--no-spot-check` | `True` | Sample random dates to verify data |

---

## Phase 3: Inspection & Forecast Visualisation

### Overview

The inspection script reads the Icechunk store, identifies filled dates, and generates daily 7-day forecast maps showing ensemble mean total precipitation over East Africa with country boundary overlays from `ea_ghcf_simple.geojson`. Outputs are organised into date-wise folders with animated GIFs.

### Script: `inspect_icechunk_store.py`

```bash
# Generate maps for first 5 filled dates (default)
uv run inspect_icechunk_store.py

# Generate maps for all filled dates
uv run inspect_icechunk_store.py --max-dates 0

# Custom output directory
uv run inspect_icechunk_store.py --max-dates 10 --out-dir my_plots
```

| Argument | Default | Description |
|---|---|---|
| `--max-dates` | `5` | Max dates to plot (0 = all) |
| `--out-dir` | `icechunk_inspect_plots` | Output directory |
| `--geojson` | `ea_ghcf_simple.geojson` | Country boundary GeoJSON |
| `--gcs-bucket` | `gik-ecmwf-aws-tf` | GCS bucket |
| `--gcs-prefix` | `ecmwf-ea-ic-store` | GCS prefix |
| `--service-account` | `coiled-data.json` | GCS service account |
| `--local` | None | Local store path (overrides GCS) |

### Output Structure

For each filled init date, the script produces 7 daily forecast PNGs (one per forecast day) and an animated GIF:

```
icechunk_inspect_plots/
├── 20240301_00z/
│   ├── 20240301_00z_day1_000-024h.png    # Day 1 forecast (0-24h)
│   ├── 20240301_00z_day2_024-048h.png    # Day 2 forecast (24-48h)
│   ├── 20240301_00z_day3_048-072h.png    # Day 3 forecast (48-72h)
│   ├── 20240301_00z_day4_072-096h.png    # Day 4 forecast (72-96h)
│   ├── 20240301_00z_day5_096-120h.png    # Day 5 forecast (96-120h)
│   ├── 20240301_00z_day6_120-144h.png    # Day 6 forecast (120-144h)
│   ├── 20240301_00z_day7_144-168h.png    # Day 7 forecast (144-168h)
│   └── 20240301_00z_forecast.gif         # Animated 7-day forecast
├── 20240302_00z/
│   └── ...
└── ecmwf_ea_tp_all_dates_forecast.gif    # Combined GIF (all dates, all days)
```

### What Each Plot Shows

Each PNG displays:
- **Ensemble mean TP** (mm) for the given forecast day, computed as the sum of 3-hourly precipitation across all lead times in that day, averaged over all 51 ensemble members
- **Country boundaries** from `ea_ghcf_simple.geojson` (11 GHACOF countries: BDI, DJI, ERI, ETH, KEN, RWA, SDN, SOM, SSD, TZA, UGA)
- **Title** with init date, run hour (00Z), forecast day number, hour range, valid date range, and maximum TP value
- **Colorbar** in mm with adaptive scaling (98th percentile)

### Day Breakdown

| Day | Lead Time Range | Steps (3h) | Steps (6h) |
|---|---|---|---|
| Day 1 | 0–24h | 0, 3, 6, 9, 12, 15, 18, 21, 24 | — |
| Day 2 | 24–48h | 24, 27, 30, 33, 36, 39, 42, 45, 48 | — |
| Day 3 | 48–72h | 48, 51, 54, 57, 60, 63, 66, 69, 72 | — |
| Day 4 | 72–96h | 72, 75, 78, 81, 84, 87, 90, 93, 96 | — |
| Day 5 | 96–120h | 96, 99, 102, 105, 108, 111, 114, 117, 120 | — |
| Day 6 | 120–144h | 120, 123, 126, 129, 132, 135, 138, 141, 144 | — |
| Day 7 | 144–168h | 144 | 150, 156, 162, 168 |

Day 7 has fewer steps because lead times switch from 3-hourly to 6-hourly after 144h.

### Creating the Combined GIF

After generating per-date GIFs, merge them into a single chronological animation:

```bash
uv run --with Pillow python3 -c "
from pathlib import Path
from PIL import Image

out_dir = Path('icechunk_inspect_plots')
gif_paths = sorted(out_dir.glob('*_00z/*_forecast.gif'))

all_frames = []
for gif_path in gif_paths:
    img = Image.open(gif_path)
    try:
        while True:
            all_frames.append(img.copy().convert('RGB'))
            img.seek(img.tell() + 1)
    except EOFError:
        pass

all_frames[0].save(
    out_dir / 'ecmwf_ea_tp_all_dates_forecast.gif',
    save_all=True, append_images=all_frames[1:],
    duration=600, loop=0,
)
print(f'Merged {len(all_frames)} frames from {len(gif_paths)} dates')
"
```

### Performance

| Metric | Value |
|---|---|
| Time per date | ~20 seconds (7 PNGs + 1 GIF) |
| 32 dates | ~11 minutes |
| PNG size | ~170 KB each |
| Per-date GIF | ~500 KB |
| Combined GIF (32 dates, 224 frames) | ~20 MB |

### GeoJSON: `ea_ghcf_simple.geojson`

Contains simplified country boundary polygons for the 11 GHACOF (Greater Horn of Africa Climate Outlook Forum) countries:

| ISO Code | Country |
|---|---|
| BDI | Burundi |
| DJI | Djibouti |
| ERI | Eritrea |
| ETH | Ethiopia |
| KEN | Kenya |
| RWA | Rwanda |
| SDN | Sudan |
| SOM | Somalia |
| SSD | South Sudan |
| TZA | Tanzania |
| UGA | Uganda |

---

## End-to-End Workflow

The complete pipeline from ECMWF GRIB data to forecast visualisations:

```
Step 1: Create parquets (Lithops, runs daily in production)
    └─► cd lithops-cr-gik-ecmwf/
        uv run run_lithops_ecmwf.py --days-back 7
        Output: parquets on HuggingFace (E4DRR/gik-ecmwf-par)

Step 2: Initialise Icechunk store (one-time setup)
    └─► uv run ecmwf_ea_tp_icechunk.py init \
            --start-date 20240301 --end-date 20251231

Step 3: Fill store from HuggingFace parquets (via Coiled)
    └─► uv run ecmwf_ea_tp_icechunk.py fill \
            --start-date 20240301 --end-date 20251231 \
            --n-workers 50
        Output: materialized data in gs://gik-ecmwf-aws-tf/ecmwf-ea-ic-store/

Step 4: Verify store contents
    └─► uv run ecmwf_ea_tp_icechunk.py verify

Step 5: Generate forecast maps and GIFs
    └─► uv run inspect_icechunk_store.py --max-dates 0
        Output: icechunk_inspect_plots/ (PNGs + GIFs)
```

---

## cGAN Data Streaming (Phase 2 Alternative)

In addition to the Icechunk store, the parquet reference files from Phase 1 can be used directly for cGAN rainfall downscaling input. This streams 12 variables (not just tp) for a specific set of forecast timesteps.

### Local Streaming

```bash
# Stream by date (reads parquets from HuggingFace/GCS)
uv run stream_cgan_variables.py --date 20260207

# From local parquet directory
uv run stream_cgan_variables.py \
    --parquet-dir ecmwf_three_stage_20260206_00z \
    --parallel-fetches 8

# Quick test: 1 member, 1 step
uv run stream_cgan_variables.py --date 20260207 --max-members 1 --steps "48"
```

### Coiled Parallel Streaming

```bash
# Full production run
python stream_cgan_variables_coiled_simple.py \
    --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z \
    --n-workers 20

# Test mode (3 members, 3 workers)
python stream_cgan_variables_coiled_simple.py --test \
    --gcs-parquet-path gs://gik-ecmwf-aws-tf/run_par_ecmwf/20260206_00z
```

**Output:** `cgan_output/IFS_YYYYMMDD_HHZ_cgan.nc`

### cGAN Variables Extracted

| Variable | ECMWF Name | Description |
|---|---|---|
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

---

## Installation

### Using `uv` (recommended)

All main scripts include [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so `uv` will automatically create an isolated environment and install dependencies:

```bash
# No manual install needed — just run:
uv run ecmwf_ea_tp_icechunk.py init --start-date 20240301 --end-date 20251231
uv run inspect_icechunk_store.py --max-dates 5
uv run run_ecmwf_tutorial.py --date 20260206 --skip-grib-scan
```

### Using `pip`

```bash
# Core dependencies
pip install kerchunk "zarr<3" xarray pandas numpy fsspec s3fs requests pyarrow

# GRIB processing
pip install cfgrib eccodes gribberish

# Icechunk store
pip install icechunk

# Visualisation
pip install matplotlib cartopy Pillow shapely

# Coiled/Dask (for fill step)
pip install dask distributed coiled gcsfs bokeh>=3.1.0

# NetCDF output (cGAN streaming)
pip install netCDF4 h5netcdf
```

## Credentials & Service Account Requirements

### AWS S3 (ECMWF forecast data)

**No credentials needed.** All scripts use anonymous S3 access:

```python
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
```

### Google Cloud Storage (Icechunk + Coiled)

A GCS service account (`coiled-data.json`) is required for:
- Reading/writing the Icechunk store on `gs://gik-ecmwf-aws-tf/ecmwf-ea-ic-store/`
- Launching Coiled clusters on GCP
- Uploading parquets to GCS (optional)

| Script | GCS Required? | When? |
|---|---|---|
| `ecmwf_ea_tp_icechunk.py` | **Yes** | All subcommands (init/fill/verify) |
| `inspect_icechunk_store.py` | **Yes** | Reads from GCS Icechunk store |
| `run_ecmwf_tutorial.py` | Only with `--upload-gcs` | Uploading parquets to GCS |
| `stream_cgan_variables.py` | **No** | Reads local parquets, fetches from S3 |

## Troubleshooting

### Common Issues

**Coiled cluster timeout / connection error:**
- Ensure region is `europe-west1` (not a zone like `europe-west1-d`)
- Check that `bokeh>=3.1.0` is in the script dependencies
- Verify Coiled account has GCP permissions

**Fill interrupted (OOM, network):**
- Simply re-run the same `fill` command. Resume detection will skip completed dates.
- Per-date commits ensure no data is lost from partially completed batches.

**Wrong GCS bucket for streaming:**
```
OSError: Forbidden: ... does not have storage.objects.list access
```
Use `gik-ecmwf-aws-tf` (not `gik-fmrc`) for parquet access.

**S3 Access Denied:**
```bash
export AWS_NO_SIGN_REQUEST=YES
```

**Memory Issues during fill:**
- Reduce `--n-workers` to lower the concurrent task count
- Peak memory on coordinator is ~245 MB per date (51 members x 4.6 MB, freed after each date commit)

## HuggingFace Catalog & Data Access Testing

The HuggingFace dataset (`E4DRR/gik-ecmwf-par`) includes a **catalog.parquet** (1.7 MB) at the repo root that indexes all 144,228+ parquet files. Two test scripts demonstrate how to use the catalog to access data.

### Upload & Catalog Management

```bash
# Upload individual parquets from GCS to HuggingFace
uv run upload_parquets_to_hf.py --sync --dry-run            # preview missing dates
uv run upload_parquets_to_hf.py --sync                       # upload missing dates

# Build/refresh the catalog index on HuggingFace
uv run upload_parquets_to_hf.py --catalog                    # full catalog (all years)
uv run upload_parquets_to_hf.py --catalog --year 2024 --dry-run  # preview
```

### Materialized Streaming: `test_catalog_xarray.py`

Downloads parquets from HuggingFace, fetches GRIB bytes from S3 via byte-range reads, decodes with gribberish, and assembles a fully materialized xarray Dataset.

```bash
# Single member (control) — ~30s
uv run test_catalog_xarray.py --date 20250101 --members 1

# 5 members — ~2.5 min
uv run test_catalog_xarray.py --date 20250101 --members 5

# All 51 members — ~25 min
uv run test_catalog_xarray.py --date 20250101 --members 51
```

**Output:** `xarray.Dataset` with dimensions `(member, step=53, latitude=721, longitude=1440)`, all data in memory.

### Lazy / Virtual Streaming: `test_catalog_virtual.py`

Builds a **dask-backed lazy xarray Dataset** — parquets are parsed instantly but no GRIB data is fetched from S3 until `.load()` is called on a slice. This lets you open all 51 members in seconds and selectively materialize only the data you need.

```bash
# Open 3 members lazily (no S3 reads) — ~5s
uv run test_catalog_virtual.py --date 20250101 --members 3

# Open all 51 members lazily — ~30s
uv run test_catalog_virtual.py --date 20250101 --members 51

# Open lazily + fetch one lead time to verify — ~15s
uv run test_catalog_virtual.py --date 20250101 --members 3 --load-step 24

# Different variable
uv run test_catalog_virtual.py --date 20250101 --var 2t --load-step 0
```

**Output:** `xarray.Dataset` with `dask.array` (zero bytes in memory). Call `.load()` on any slice to trigger S3 byte-range reads (~2 MB per member per step).

| Approach | 1 Member | 51 Members | Data in Memory |
|----------|----------|------------|----------------|
| Materialized (`test_catalog_xarray.py`) | ~30s | ~25 min | All steps, all data |
| Lazy (`test_catalog_virtual.py`) | ~2s | ~30s | 0 bytes until `.load()` |
| Lazy + `.sel(step=24).load()` | ~10s | ~30s + ~30s | 1 step only |

---

## References

- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
- [Kerchunk Documentation](https://fsspec.github.io/kerchunk/)
- [Icechunk](https://icechunk.io/) - Git-like versioned Zarr stores
- [gribberish](https://github.com/mpiannucci/gribberish) - Fast GRIB decoding
- [Coiled](https://coiled.io/) - Managed Dask clusters
- [Lithops](https://lithops-cloud.github.io/) - Serverless computing framework

## License

This code is part of the ICPAC climate services infrastructure, developed under the E4DRR (UN CRAF'd) and SEWAA projects.
