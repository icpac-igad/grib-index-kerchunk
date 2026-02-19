# CLAUDE.md - Grib-Index-Kerchunk (GIK) Project Context

## What This Project Does

This repository implements the **Grib-Index-Kerchunk (GIK) method** for streaming
ensemble weather forecast data directly from cloud storage (AWS S3) without
downloading full GRIB files. It creates lightweight parquet reference files
containing `[url, byte_offset, byte_length]` triplets that point into remote GRIB
files, enabling on-demand byte-range reads of only the data actually needed.

The project is developed by **ICPAC** (IGAD Climate Prediction and Applications
Centre) for continuous climate risk monitoring over East Africa, funded by the
**E4DRR** (UN CRAF'd) and **SEWAA** projects.

### The Streaming Analogy

The GIK method applies the same principle as video streaming to weather data:

| Video Streaming (HTML5/HLS) | Weather Data Streaming (GIK) |
|---|---|
| Video split into small segments | GRIB file split into variable/level/timestep chunks |
| Manifest file (.m3u8) lists segment URLs + byte ranges | Parquet reference file lists GRIB URLs + byte ranges |
| Player fetches only visible segments on demand | Analysis code fetches only needed variables on demand |
| Full video never downloaded | Full GRIB file (3-8 GB) never downloaded |
| Fetch per segment: ~2-5 MB | Fetch per chunk: ~2 MB |

Without GIK, processing 30 GEFS ensemble members requires downloading ~2,400 GRIB
files. With GIK, the same analysis streams only ~2-5% of the data via targeted
byte-range reads.

---

## Data Sources

### NOAA GEFS (Global Ensemble Forecast System)

- **S3 bucket**: `s3://noaa-gefs-pds/`
- **Path pattern**: `gefs.{YYYYMMDD}/{HH}/atmos/pgrb2sp25/{member}.t{HH}z.pgrb2s.0p25.f{FFF}`
- **Index files**: `.idx` (text format, one line per GRIB message with byte offset and variable description)
- **Members**: 30 ensemble (gep01-gep30), no control published in this path
- **Resolution**: 0.25 deg global
- **Forecast range**: 0-240h at 3h intervals (81 timesteps)
- **File size**: ~25 MB per member per timestep
- **Public access**: Anonymous (`anon=True`)

### ECMWF IFS (Integrated Forecasting System)

- **S3 bucket**: `s3://ecmwf-forecasts/`
- **Path pattern**: `{YYYYMMDD}/{HH}z/ifs/0p25/enfo/{YYYYMMDDHH}0000-{H}h-enfo-ef.grib2`
- **Index files**: `.index` (JSON-lines format, each line is a JSON object with `_offset`, `_length`, `param`, `number`, `step`, etc.)
- **Members**: 51 (1 control + 50 ensemble, encoded as `number` field in `.index`)
- **Resolution**: 0.25 deg global
- **Forecast range**: 0-144h at 3h, 150-360h at 6h (85 timesteps)
- **File size**: ~3-4 GB per timestep (all 51 members packed into one file)
- **Public access**: Anonymous (`anon=True`)

### Key Structural Difference

GEFS publishes **one file per member per timestep** with a companion `.idx` file.
ECMWF publishes **one file per timestep containing all 51 members** with a
companion `.index` file. This difference shapes the Stage 2 processing logic
for each product but the overall GIK method is the same.

---

## The Three-Stage Pipeline

Both GEFS and ECMWF use a three-stage pipeline to create parquet reference files.
The stages have evolved differently between the two products.

### Stage 1: Build Zarr Metadata Structure

**Purpose**: Create a "deflated store" -- the zarr variable/dimension/chunk schema
with all data references stripped out. This is the skeleton that defines what
variables exist, their dimensions, chunk layout, and coordinate metadata.

**ECMWF (current)**: Loads the zarr structure directly from a pre-built
HuggingFace template archive (`gik-fmrc-v2ecmwf_fmrc.tar.gz`). Takes ~5 seconds.
No GRIB scanning. The template was originally created by running `scan_grib` once
on a reference date (`20240529`) and archiving the result.

**GEFS (current)**: Still uses `kerchunk.grib2.scan_grib` to scan 2 GRIB files
(`f000` and `f003`) and build the tree via `grib_tree`. Takes ~30 seconds. This
is carried over from the original architecture. The pre-built template
(`gik-fmrc-gefs-20241112.tar.gz`) exists on HuggingFace and could replace
`scan_grib` in the same way ECMWF did, but the implementation has not been
refactored yet. The 2-file scan is fast enough (~30s) that it is not a bottleneck.

**Why only 2 files**: The zarr structure (variable names, dimension names, chunk
shapes, coordinate schemas) is identical across all timesteps of a forecast run.
Scanning `f000` and `f003` captures the full variable/dimension schema. The actual
byte-range references for all 81 timesteps come from Stage 2.

### Stage 2: Index-Based Reference Building

**Purpose**: Read the lightweight index files (`.idx` for GEFS, `.index` for
ECMWF) published alongside every GRIB file. These index files contain the byte
offset and length of every GRIB message. Combine with pre-built template parquets
to create per-member reference sets mapping every variable/timestep to its exact
byte range in the remote GRIB file.

**ECMWF (Lithops version)**: Reads `.index` files (JSON-lines) directly from S3.
Each line contains `{"_offset": N, "_length": M, "param": "tp", "number": 1, ...}`.
Parses all 85 timestep index files per member, extracts byte ranges, merges with
the Stage 1 template structure. Fully self-contained in `run_lithops_ecmwf.py`.

**GEFS**: Uses `kerchunk._grib_idx.parse_grib_idx` to read `.idx` files from S3,
then `map_from_index` to merge fresh byte positions with pre-built mapping parquets
from the template archive. Processing is per-member, per-timestep (81 iterations
per member). Uses `LocalTarGzMappingManager` to read template parquets from the
local tar.gz archive.

**Both products**: Stage 2 is the slowest stage (~5-15 minutes) because it makes
one S3 read per timestep per member to fetch the index file. This is inherently
I/O-bound and parallelized via `ThreadPoolExecutor`.

### Stage 3: Final Parquet Creation

**Purpose**: Merge the zarr metadata structure (Stage 1) with the fresh byte-range
references (Stage 2) to produce one parquet file per ensemble member. Each parquet
contains a DataFrame with columns `[key, value]` where `key` is a zarr path
(e.g., `tp/instant/surface/tp/0.0.0`) and `value` is either zarr metadata JSON
or a `[url, offset, length]` reference.

**Output**: One `.parquet` file per member (e.g., `2026020600z-control.parquet`,
`2026020600z-gep01.parquet`).

**Usage**: These parquet files are loaded by downstream scripts to construct
virtual zarr datasets. A zarr-aware reader (xarray + fsspec) resolves each
reference to a byte-range HTTP/S3 read, fetching only the exact bytes of the
GRIB messages needed.

---

## Repository Structure

```
grib-index-kerchunk/
├── ecmwf/                          # ECMWF IFS ensemble pipeline
│   ├── ecmwf_util.py               # Variable definitions, axis generation, forecast hours
│   ├── ecmwf_three_stage_multidate.py  # Three-stage pipeline orchestration
│   ├── ecmwf_index_processor.py    # Stage 2 index processing
│   ├── ecmwf_ensemble_par_creator_efficient.py  # Stage 1 GRIB scanning (legacy)
│   ├── utils_ecmwf_step1_scangrib.py  # Stage 1 scanning utilities
│   ├── run_ecmwf_tutorial.py       # Entry point for tutorial pipeline
│   ├── stream_cgan_variables.py    # Phase 2: stream data for cGAN input
│   ├── plot_cgan_maps.py           # Visualization of cGAN outputs
│   ├── compare_gik_herbie.py       # GIK vs Herbie comparison
│   ├── fetch_tp_herbie.py          # Herbie-based data fetching
│   ├── docs/                       # Architecture docs, analysis reports
│   └── dev-test/                   # Development and test scripts
│
├── gefs/                           # NOAA GEFS ensemble pipeline
│   ├── gefs_util.py                # Core utilities (scan_grib, parse_grib_idx, zarr store ops)
│   ├── run_gefs_preprocessing.py   # Step 0: one-time template creation
│   ├── run_single_gefs_to_zarr.py  # Single member parquet-to-zarr
│   ├── run_single_gefs_to_zarr_gribberish.py  # Same with gribberish (~80x faster)
│   ├── run_day_gefs_ensemble_full.py  # Daily ensemble processing
│   ├── process_ensemble_by_variable.py  # Ensemble concatenation + statistics
│   ├── run_gefs_24h_accumulation.py  # 24h rainfall accumulation + probability
│   ├── test_three_stage_gefs_simple.py  # Simple three-stage test
│   ├── docs/                       # GEFS-specific documentation
│   └── dev-test/                   # Development and test scripts
│
├── tutorial/                       # Self-contained tutorials
│   ├── ecmwf/
│   │   ├── run_ecmwf_tutorial.py   # ECMWF parquet creation tutorial
│   │   └── run_ecmwf_data_streaming.py  # ECMWF data streaming tutorial
│   ├── gefs/
│   │   ├── run_gefs_tutorial.py    # GEFS parquet creation tutorial
│   │   └── run_gefs_data_streaming.py  # GEFS data streaming + plots
│   └── VIRTUALIZARR_ICECHUNK_MIGRATION_ROADMAP.md
│
├── gfs/                            # Original GFS implementation (legacy)
├── cfs/                            # CFS seasonal forecast (in development)
├── devops/                         # Docker/environment configs
└── README.md
```

---

## Pre-Built Templates on HuggingFace

Both pipelines use pre-built template archives hosted on HuggingFace. These
contain the zarr metadata structure and/or mapping parquets from a reference date.

| Product | Template Archive | Reference Date | Contents |
|---|---|---|---|
| GEFS | `gik-fmrc-gefs-20241112.tar.gz` | 2024-11-12 | Per-member, per-timestep mapping parquets |
| ECMWF | `gik-fmrc-v2ecmwf_fmrc.tar.gz` | 2024-05-29 | Per-member zarr store parquets |

**Repository**: `https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/`

**Template validity**: Templates remain valid as long as the forecast model
configuration (grid resolution, variable set, dimension layout) does not change.
If NOAA or ECMWF changes their model grid, new templates must be regenerated by
running `scan_grib` on the updated GRIB files.

---

## Lithops Cloud Deployment

The production deployment uses **Lithops** with Google Cloud Run to process
multiple dates in parallel. Each date is processed independently on a separate
Cloud Run container.

### Architecture

```
Local machine (uv run run_lithops_{ecmwf,gefs}.py --days-back 7)
    │
    ▼
Lithops FunctionExecutor(backend='gcp_cloudrun')
    │   serializes process_{ecmwf,gefs}_date() via cloudpickle
    │   uploads to GCS
    │
    ├──► Cloud Run Worker 1 → process date 2026-02-01 → upload parquets to GCS
    ├──► Cloud Run Worker 2 → process date 2026-02-02 → upload parquets to GCS
    └──► Cloud Run Worker N → process date 2026-02-0N → upload parquets to GCS
```

### ECMWF Lithops (`run_lithops_ecmwf.py`)

- **Self-contained**: Zero imports from the GIK codebase. All three stages are
  implemented inline (~960 lines). This is a hard requirement of cloudpickle
  serialization -- local `.py` files cannot be captured for remote execution
  unless baked into the Docker image.
- **No `scan_grib`**: Stage 1 loads zarr structure from HuggingFace template.
- **No `kerchunk` dependency**: Reads `.index` files directly as JSON-lines.
- **Pre-flight validation**: Checks `.index` file availability before processing.
- **GCS output**: `gs://gik-ecmwf-aws-tf/run_par_ecmwf/{date}/{run}z/`

### GEFS Lithops (`run_lithops_gefs.py`)

- **Imports `gefs_util.py`**: Not self-contained. The `gefs_util.py` must be
  baked into the Docker image for cloudpickle deserialization to work on workers.
- **Uses `scan_grib`**: Stage 1 still scans 2 GRIB files (~30s) to build the
  zarr tree. Both GEFS `.idx` files and ECMWF `.index` files serve the same
  purpose (byte offsets only, no zarr structure), so `scan_grib` could be
  eliminated the same way ECMWF did. Kept for now because it is fast and the
  template-based alternative has not been implemented for the GEFS parquet format.
- **Depends on `kerchunk`**: Required for `scan_grib`, `grib_tree`,
  `parse_grib_idx`, `map_from_index`.
- **GCS output**: `gs://gik-gefs-aws-tf/run_par_gefs/{date}/{run}z/`

### Key Files in Lithops Deployments

Located in `cno-e4drr/devops/`:

```
lithops_cr_ecmwf_gik/
├── run_lithops_ecmwf.py      # Main script (self-contained, ~960 lines)
├── Dockerfile                # Cloud Run container image
├── cloudbuild.yaml           # GCP Cloud Build config
├── lithops_config.yaml       # Lithops backend config
├── run_backfill_00z.sh       # Backfill shell script
└── terraform/                # GCS bucket and IAM provisioning

lithops_cr_gefs_gik/
├── run_lithops_gefs.py       # Main script (~610 lines)
├── gefs_util.py              # Core utilities (~1100 lines, baked into Docker)
├── Dockerfile                # Cloud Run container image
├── cloudbuild.yaml           # GCP Cloud Build config
└── lithops_config.yaml       # Lithops backend config
```

---

## Downstream Usage

### Phase 2: Data Streaming

After the parquet reference files are created (Phase 1), downstream scripts use
them to stream forecast data into analysis-ready formats:

**ECMWF → cGAN rainfall downscaling** (`stream_cgan_variables.py`):
- Reads parquet reference files for each of 51 members
- Streams 12 variables (tp, t2m, sp, ssr, ssrd, tcw, tcwv, tcc, u700, v700, sf, ro)
  at 9 timesteps via parallel S3 byte-range reads
- Decodes GRIB chunks using gribberish (Rust-based, ~80x faster than cfgrib)
- Outputs NetCDF for cGAN inference (`IFS_YYYYMMDD_HHZ_cgan.nc`)

**GEFS → ensemble probability maps** (`run_gefs_data_streaming.py`):
- Reads parquet reference files for each of 30 members
- Streams target variables (t2m, tp, u10, v10, cape, sp)
- Decodes with gribberish
- Computes 24h rainfall exceedance probabilities (>1mm, >5mm, >10mm, >25mm, >50mm)
- Creates probability maps over East Africa using cartopy

### Gribberish Decoder

Both products use **gribberish** (a Rust-based GRIB2 decoder) for the data
streaming phase:
- ~25ms per chunk vs ~2000ms for cfgrib
- Works directly on byte buffers from S3 (no temp files)
- Falls back to cfgrib for chunks that fail decoding
- Not used during parquet creation (Phase 1), only during data streaming (Phase 2)

---

## Key Kerchunk Internals Used

From `kerchunk._grib_idx`:
- `parse_grib_idx(basename)`: Reads `.idx` files from S3, returns DataFrame with columns `[attrs, offset, length, ...]`
- `map_from_index(datestr, deduped_mapping, idxdf)`: Merges template mapping with fresh index positions
- `store_coord_var(key, zstore, coords, data)`: Writes coordinate variable metadata into zarr store
- `store_data_var(key, zstore, dims, coords, data, steps, times, lvals)`: Writes data variable metadata with chunk references
- `strip_datavar_chunks(store)`: Removes data chunk references, keeping only metadata (creates "deflated store")

From `kerchunk.grib2`:
- `scan_grib(url)`: Scans a GRIB file to extract zarr-compatible references (used in GEFS Stage 1)
- `grib_tree(groups)`: Builds hierarchical tree structure from scan_grib output

---

## Performance Benchmarks

### Parquet Creation (Phase 1)

| Product | Members | Stage 1 | Stage 2 | Stage 3 | Total |
|---|---|---|---|---|---|
| ECMWF (Lithops) | 51 | ~5s (template) | ~5-15 min | ~2s | ~5-15 min per date |
| GEFS (Lithops) | 30 | ~30s (scan_grib) | ~2-3 min | ~1 min | ~3-5 min per date |
| ECMWF (tutorial, legacy) | 51 | ~73 min (scan_grib) | ~35 min | ~5s | ~110 min |

### Data Streaming (Phase 2)

| Product | Members | Sequential | Parallel | Method |
|---|---|---|---|---|
| ECMWF (local, 12 vars) | 51 | ~4 hours | ~24 min (8 threads) | ThreadPoolExecutor |
| GEFS (local, 4 vars) | 30 | ~2 hours | ~14 min (8 threads) | ThreadPoolExecutor |

### Decoding Speed

| Decoder | Time per chunk | Notes |
|---|---|---|
| gribberish (Rust) | ~25 ms | Direct byte buffer, no temp files |
| cfgrib (Python/eccodes) | ~2000 ms | Writes temp file, calls eccodes |
| Speedup | **~80x** | |

---

## Architecture Evolution Summary

The project has progressed through distinct architectural phases:

1. **Original (GFS)**: Full `scan_grib` on every GRIB file, local processing,
   per-variable parquets. Worked but was slow.

2. **Two-step with templates (GEFS)**: One-time `scan_grib` to create reusable
   mapping templates in GCS. Daily runs only read `.idx` files and merge with
   templates. 10-100x faster for daily operations.

3. **Template-based Stage 1 (ECMWF)**: Replaced `scan_grib` entirely with
   HuggingFace template loading (~5s vs ~73 min). Pipeline became stateless
   with respect to GRIB scanning.

4. **Serverless cloud-native (Lithops)**: Self-contained single-file scripts
   deployed on Google Cloud Run. Date-level parallelism via `fexec.map()`.
   ECMWF fully self-contained; GEFS still requires `gefs_util.py` in the
   Docker image.

5. **Future: VirtualiZarr + Icechunk**: Planned migration to custom VirtualiZarr
   parsers with Icechunk persistence. Would replace parquet reference files with
   standardized ManifestStore, integrate with the broader Zarr v3 ecosystem, and
   reduce pipeline code from ~800 lines to ~100 lines.

---

## Common Patterns and Conventions

- **Anonymous S3 access**: `os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'` set at script start
- **PEP 723 inline metadata**: Lithops scripts use `# /// script` blocks for `uv run` compatibility
- **Date format**: `YYYYMMDD` strings throughout (e.g., `'20260206'`)
- **Run hour**: `'00'`, `'06'`, `'12'`, `'18'` as string
- **Parquet naming**: `{date}{run}z-{member}.parquet` (e.g., `2026020600z-gep01.parquet`)
- **GCS bucket layout**: `gs://{bucket}/{prefix}/{date}/{run}z/*.parquet`
- **Template reference dates**: GEFS uses `20241112`, ECMWF uses `20240529`
- **Variable filtering**: GEFS uses `"TMP:2 m above ground"` format (colon-separated name:level); ECMWF uses short names (`tp`, `t2m`, etc.)
