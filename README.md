# Grib-Index-Kerchunk (GIK) — Cloud-Native Weather Data Streaming

Based on the [dynamic-Grib-chunking method](https://github.com/asascience-open/nextgen-dmac/commit/6b3286627070c36127ec97b7dbbb88b0ab481f06), the GIK method uses Kerchunk scan_grib with
GRIB index files to significantly reduce the need to scan all GRIB files. This
method offers major advantages in reducing the costs involved in
scan_grib — whereas reading all the GRIB files in the FMRC to make references
(for example, GFS per run has 240 hours, GEFS involves 2400 GRIB files for 30
members, or ECMWF with 86 files of 4GB for 50 members) typically requires scanning every file, this approach needs only two files to scan to generate sample metadata.

Kerchunk facilitates the creation of references that can be converted into a virtual Zarr dataset, enabling Analysis Ready Cloud Optimized (ARCO) datasets and Cloud Native Operations (CNO).
The virtual Zarr dataset can be streamed using byte-range reads, supporting
real-time processing through scalable parallel processing.
This enables users to access and interact with the data, select
variables, and subset regions and timesteps — without downloading full GRIB files.

### Weather Data vs. Video Streaming

| **Aspect**               | **Video Streaming (HTML5)**                          | **Weather Data (GIK/Kerchunk)**                          |
|---------------------------|-----------------------------------------------------|-----------------------------------------------------|
| **Download Workflow**  | Full video download for playback                   | Full GRIB file download for analysis               |
| **Streaming Workflow**       | Stream segments on demand using adaptive bitrate   | Stream slices on demand using Kerchunk byte-range refs    |
| **Metadata Handling**     | Indexed file for frames, timecodes, and bitrates   | Indexed metadata for variables, timestamps, region (lat/lon) and ensemble members    |
| **Efficiency**            | Lower bandwidth; no full downloads needed          | Fetch only 2-5% of data via targeted byte-range reads                  |
| **Scalability**           | Scales easily across devices and networks          | Scales horizontally using Dask cluster / Lithops Cloud Run |

---

## Supported Products

| Product | Source | Members | Timesteps | Grid | Pipeline Status |
|---------|--------|---------|-----------|------|-----------------|
| **GEFS** | `s3://noaa-gefs-pds/` | 30 ensemble (gep01-gep30) | 81 (0-240h at 3h) | 0.25° (721x1440) | Production |
| **ECMWF IFS** | `s3://ecmwf-forecasts/` | 51 (1 control + 50 ensemble) | 85 (0-360h) | 0.25° (721x1440) | Production |
| **GFS** | `s3://noaa-gfs-bdp-pds/` | Deterministic | 240h | 0.25° | Legacy |
| **CFS** | `s3://noaa-cfs-pds/` | Seasonal | Variable | 1.0° | In development |

---

## GEFS Pipeline

### Three-Stage Pipeline

The GEFS pipeline creates lightweight parquet reference files containing `[url, byte_offset, byte_length]` triplets pointing into remote GRIB files on AWS S3. Each parquet enables on-demand byte-range reads of only the data actually needed.

**Stage 1 — Build Zarr Metadata Structure (Deflated Store)**

Creates the zarr variable/dimension/chunk schema. Two modes:

| Mode | Method | Time | When to use |
|------|--------|------|-------------|
| **Template** (recommended) | `build_gefs_deflated_store_from_template()` | ~0.05s | Normal operation |
| **scan_grib** (fallback) | `scan_grib(f000, f003) → grib_tree()` | ~5.5s | Template unavailable or model config changed |

The template (`gefs-deflated-store-template-20241112.parquet`, 26 KB, 915 entries) contains the complete zarr structure for all 39 GEFS variables. It is loaded once and reused for all 30 ensemble members.

**Stage 2 — Index-Based Reference Building**

Reads the `.idx` files published alongside every GRIB file on S3. These contain byte offsets for every GRIB message. Merges with pre-built mapping parquets from the template archive (`gik-fmrc-gefs-20241112.tar.gz`) to create per-member reference sets. This stage is I/O-bound (~25-30s per member).

**Stage 3 — Final Zarr Store Assembly**

Merges the deflated store (Stage 1) with the fresh byte-range references (Stage 2) to produce a complete zarr store. Output is one parquet file per ensemble member.

### GEFS Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `gefs/gefs_util.py` | Core utilities — scan_grib, template loading, zarr store ops | Imported by other scripts |
| `gefs/run_day_gefs_ensemble_full.py` | Daily ensemble processing (all 30 members) | `uv run run_day_gefs_ensemble_full.py --date 20250106` |
| `gefs/run_gefs_preprocessing.py` | One-time template creation | `uv run run_gefs_preprocessing.py` |
| `gefs/create_gik_tp_netcdf.py` | GIK TP extraction + gribberish streaming | `uv run create_gik_tp_netcdf.py --max-members 3` |
| `gefs/fetch_tp_herbie_gefs.py` | Herbie-based TP fetching for validation | `uv run fetch_tp_herbie_gefs.py --date 20250106 --max-members 3` |
| `gefs/compare_gik_herbie_gefs.py` | GIK vs Herbie comparison PNGs | `uv run compare_gik_herbie_gefs.py --dates 20250106` |
| `gefs/run_single_gefs_to_zarr.py` | Single member parquet-to-zarr | `uv run run_single_gefs_to_zarr.py` |
| `gefs/run_single_gefs_to_zarr_gribberish.py` | Same with gribberish (~80x faster) | `uv run run_single_gefs_to_zarr_gribberish.py` |
| `gefs/process_ensemble_by_variable.py` | Ensemble concatenation + statistics | `uv run process_ensemble_by_variable.py` |
| `gefs/run_gefs_24h_accumulation.py` | 24h rainfall accumulation + probability | `uv run run_gefs_24h_accumulation.py` |

### GEFS Template-Based Stage 1 (scan_grib Replacement)

The original pipeline called `scan_grib()` on 2 GRIB files per member (~5.5s each). Since the zarr structure is identical across all 30 members, this was replaced with a pre-built parquet template loaded in ~0.05s — a **112x speedup per invocation** and **3300x total** for the 30-member ensemble.

Key function in `gefs_util.py`:

```python
from gefs_util import build_gefs_deflated_store_from_template

# Load once, reuse for all 30 members
deflated_store = build_gefs_deflated_store_from_template(
    'gefs-deflated-store-template-20241112.parquet'
)
```

Optional variable filtering:
```python
deflated_store = build_gefs_deflated_store_from_template(
    'gefs-deflated-store-template-20241112.parquet',
    filter_vars={"Total Precipitation": "APCP:surface", "2 metre temperature": "TMP:2 m above ground"}
)
```

See `gefs/docs/GEFS_SCAN_GRIB_TO_TEMPLATE.md` for full documentation.

### GIK vs Herbie Validation

The GIK pipeline has been validated against Herbie (which uses cfgrib) to confirm data integrity. The validation uses gribberish (Rust-based GRIB decoder, ~80x faster than cfgrib) to stream TP data directly from S3 byte-range references.

**Results: bit-identical output** — r=1.0, RMSE=0.0, MAE=0.0 across 11,881 grid points over East Africa.

```bash
# Step 1: Create GIK TP NetCDF (gribberish streaming from S3)
uv run create_gik_tp_netcdf.py --date 20250106 --max-members 3

# Step 2: Create Herbie TP NetCDF (cfgrib reference)
uv run fetch_tp_herbie_gefs.py --date 20250106 --max-members 3

# Step 3: Compare and generate PNGs
uv run compare_gik_herbie_gefs.py --dates 20250106 \
    --gik-dir gik_tp_gefs_output --herbie-dir herbie_tp_gefs_output
```

Output PNGs in `gik_vs_herbie_gefs/`:
- `compare_gefs_20250106.png` — 3-column (GIK | Herbie | Diff) x 2-row (mean | spread) map
- `scatter_gik_vs_herbie_gefs.png` — Grid-point scatter plot with r=1.0

### GEFS Data Streaming (Phase 2)

After parquet reference files are created (Phase 1), data is streamed using gribberish:

```python
import gribberish
import fsspec

fs = fsspec.filesystem("s3", anon=True)

# ref = [url, offset, length] from the parquet zarr store
with fs.open(url, "rb") as f:
    f.seek(offset)
    grib_bytes = f.read(length)

# Decode: ~25ms per chunk (vs ~2000ms for cfgrib)
data = gribberish.parse_grib_array(grib_bytes, 0).reshape((721, 1440))
```

---

## GEFS Tutorial

Self-contained tutorials in `tutorial/gefs/`:

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_gefs_tutorial.py` | Parquet creation for all 30 members | `uv run run_gefs_tutorial.py` |
| `run_gefs_data_streaming.py` | Data streaming + probability maps | `uv run run_gefs_data_streaming.py` |

The tutorial downloads the template archive from HuggingFace and processes ensemble members using the three-stage pipeline. It uses the template-based Stage 1 (skipping scan_grib) when `gefs-deflated-store-template-20241112.parquet` is available.

**Template Source**: [HuggingFace — gik-fmrc-gefs-20241112.tar.gz](https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/blob/main/gik-fmrc-gefs-20241112.tar.gz)

---

## ECMWF IFS Pipeline

The ECMWF pipeline follows the same three-stage architecture but with key differences:

- **Stage 1**: Loads zarr structure from HuggingFace template (no scan_grib)
- **Stage 2**: Reads `.index` files (JSON-lines) instead of `.idx` (text)
- **Stage 3**: Same zarr store assembly
- **51 members** packed into one file per timestep (vs GEFS: one file per member per timestep)

See `ecmwf/` for the full ECMWF pipeline and `tutorial/ecmwf/` for tutorials.

---

## GFS (Legacy)

The original GFS implementation in `gfs/`:

1. Make virtual dataset for a day in parquet format using `run_day_gfs_gik.py`:
    1. Use kerchunk scan_grib to create metadata of GFS GRIB files
    2. Use the metadata mapping to build an index table from `.idx` files
    3. Combine index data with metadata to build FMRC slices

2. Read parquet file and stream into zarr using `run_day_stream_gfs_gik_to_zarr.py`:
    1. Parquet file with 15 variable references into zarr, stored in GCS

---

## Pre-Built Templates on HuggingFace

Both pipelines use pre-built template archives hosted on HuggingFace:

| Product | Template Archive | Reference Date | Contents |
|---------|------------------|----------------|----------|
| GEFS | `gik-fmrc-gefs-20241112.tar.gz` | 2024-11-12 | Per-member, per-timestep mapping parquets |
| ECMWF | `gik-fmrc-v2ecmwf_fmrc.tar.gz` | 2024-05-29 | Per-member zarr store parquets |

**Repository**: https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/

Templates remain valid as long as the forecast model configuration (grid resolution, variable set, dimension layout) does not change.

---

## Performance Benchmarks

### Parquet Creation (Phase 1)

| Product | Members | Stage 1 | Stage 2 | Stage 3 | Total |
|---------|---------|---------|---------|---------|-------|
| GEFS (template) | 30 | ~0.05s | ~2-3 min | ~1 min | ~3-5 min per date |
| GEFS (scan_grib) | 30 | ~165s (30x5.5s) | ~2-3 min | ~1 min | ~5-8 min per date |
| ECMWF (template) | 51 | ~5s | ~5-15 min | ~2s | ~5-15 min per date |

### Data Streaming (Phase 2)

| Product | Members | Method | Time |
|---------|---------|--------|------|
| GEFS (8 threads) | 30 | ThreadPoolExecutor + gribberish | ~14 min |
| ECMWF (8 threads) | 51 | ThreadPoolExecutor + gribberish | ~24 min |

### GRIB Decoding Speed

| Decoder | Time per chunk | Notes |
|---------|---------------|-------|
| gribberish (Rust) | ~25 ms | Direct byte buffer, no temp files |
| cfgrib (Python/eccodes) | ~2000 ms | Writes temp file, calls eccodes |
| **Speedup** | **~80x** | |

---

## Recent Changes

| Commit | Description |
|--------|-------------|
| `83da9d1` | Fix valid_time offset in GIK vs Herbie GEFS comparison (r=0.36 → r=1.0) |
| `aefba4c` | Fix template loader bug, add gribberish streaming script and comparison PNGs |
| `a32073c` | Add Herbie validation scripts for GEFS |
| `d733ccf` | Replace scan_grib with template-based Stage 1 for GEFS (112x speedup) |
| `20842f5` | Restructure ECMWF tutorial with self-contained scripts |

---

## Repository Structure

```
grib-index-kerchunk/
├── gefs/                           # NOAA GEFS ensemble pipeline
│   ├── gefs_util.py                # Core utilities (template loading, scan_grib, zarr ops)
│   ├── gefs-deflated-store-template-20241112.parquet  # Pre-built Stage 1 template
│   ├── run_day_gefs_ensemble_full.py   # Daily ensemble processing
│   ├── create_gik_tp_netcdf.py     # GIK TP extraction (gribberish streaming)
│   ├── fetch_tp_herbie_gefs.py     # Herbie TP fetching for validation
│   ├── compare_gik_herbie_gefs.py  # GIK vs Herbie comparison PNGs
│   ├── run_single_gefs_to_zarr.py  # Single member parquet-to-zarr
│   ├── run_single_gefs_to_zarr_gribberish.py  # Same with gribberish (~80x faster)
│   ├── process_ensemble_by_variable.py  # Ensemble concatenation + statistics
│   ├── run_gefs_24h_accumulation.py    # 24h rainfall accumulation
│   ├── gik_tp_gefs_output/        # GIK TP NetCDF output
│   ├── herbie_tp_gefs_output/      # Herbie TP NetCDF output
│   ├── gik_vs_herbie_gefs/         # Comparison PNGs and stats
│   └── docs/                       # GEFS documentation
│
├── ecmwf/                          # ECMWF IFS ensemble pipeline
│   ├── ecmwf_util.py               # Variable definitions, axis generation
│   ├── ecmwf_three_stage_multidate.py  # Three-stage pipeline
│   ├── validate_gik_vs_herbie_2024.py  # GIK vs Herbie validation (10 dates)
│   ├── stream_cgan_variables.py    # Phase 2: stream data for cGAN input
│   └── docs/                       # ECMWF documentation
│
├── tutorial/                       # Self-contained tutorials
│   ├── ecmwf/
│   │   ├── run_ecmwf_tutorial.py   # ECMWF parquet creation tutorial
│   │   └── run_ecmwf_data_streaming.py
│   └── gefs/
│       ├── run_gefs_tutorial.py    # GEFS parquet creation tutorial
│       └── run_gefs_data_streaming.py
│
├── gfs/                            # Original GFS implementation (legacy)
├── cfs/                            # CFS seasonal forecast (in development)
├── devops/                         # Docker/environment configs
└── README.md
```

---

**Developed by [ICPAC](https://www.icpac.net/)** (IGAD Climate Prediction and Applications Centre) for continuous climate risk monitoring over East Africa, funded by the **E4DRR** (UN CRAF'd) and **SEWAA** projects.
