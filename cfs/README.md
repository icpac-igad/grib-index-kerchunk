# CFS GIK Pipeline — Quick Start Guide

Create parquet reference files for NOAA CFS seasonal forecast data using the
GIK (Grib-Index-Kerchunk) method. These parquets contain `[url, byte_offset,
byte_length]` references that point directly into remote GRIB files on S3,
enabling on-demand data streaming without downloading full files.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (handles dependencies automatically via PEP 723 inline metadata)

No manual `pip install` needed — `uv run` resolves all dependencies.

---

## Two-Phase Workflow

### Phase 1: Template Creation (one-time)

Build a reusable zarr skeleton template from a reference CFS forecast date.
This captures the variable/dimension/chunk structure and is reused for all
subsequent runs.

### Phase 2: Parquet Generation (per forecast)

Parse `.idx` index files from S3 to get fresh byte-range offsets, merge with
the template, and save as parquet files — one per ensemble member.

---

## Phase 1: Create the Zarr Template

Script: `run_cfs_template_creation.py`

```bash
# Create zarr skeleton template for reference date 2025-11-01 00z
# Saves to local folder (scans 2 GRIB files via kerchunk, ~24 seconds)
uv run run_cfs_template_creation.py \
    --zarr-template \
    --init-date 20251101 \
    --run 00 \
    --local-dir ./cfs_test/template \
    --max-forecast-hours 48
```

Output: `cfs_test/template/cfs-zarr-template-20251101-00.parquet`
(1705 metadata keys covering all 68 CFS variable groups)

This template is valid for any CFS forecast date as long as the model
configuration does not change.

### What happens under the hood

1. Scans two GRIB files (`f000` + `f006`) using `kerchunk.grib2.scan_grib`
2. Builds the zarr tree via `grib_tree` with all variables and dimensions
3. Strips data chunk references (keeps only metadata — the "deflated store")
4. Saves as a parquet with columns `[key, value]`

### Optional: Create mapping parquets

Mapping parquets store the full `build_idx_grib_mapping` output per timestep.
These are used by the template creation preprocessing, not by the Lithops pipeline.

```bash
# Single date, local output
uv run run_cfs_template_creation.py \
    --init-date 20251101 \
    --run 00 \
    --local-dir ./cfs_templates \
    --max-forecast-hours 48

# Single date, upload to GCS
uv run run_cfs_template_creation.py \
    --init-date 20251101 \
    --run 00 \
    --bucket gik-fmrc \
    --max-forecast-hours 48

# Full month (all 124 members: 31 days x 4 runs)
uv run run_cfs_template_creation.py \
    --init-month 202511 \
    --local-dir ./cfs_templates \
    --max-forecast-hours 5160
```

---

## Phase 2: Generate Parquet Reference Files

Script: `lithops-cr-gik-cfs/run_lithops_cfs.py`

This script reads `.idx` files from `s3://noaa-cfs-pds/`, extracts byte
offsets for all variables, merges with the zarr template, and outputs a
final parquet. No kerchunk dependency at runtime.

### Local Testing (Sequential Mode)

```bash
# Single member, 48-hour forecast, local template + local output
uv run lithops-cr-gik-cfs/run_lithops_cfs.py \
    --init-date 20251101 \
    --run 00 \
    --sequential \
    --local-template cfs_test/template/cfs-zarr-template-20251101-00.parquet \
    --output-dir ./cfs_output \
    --max-forecast-hours 48 \
    --yes
```

Output: `cfs_output/2025110100z.parquet`
(2630 keys = 1705 metadata + 925 data refs for 103 variables x 9 timesteps)

### Cloud Run via Lithops

```bash
# Full 124-member ensemble for December 2025
# (init dates from November 2025: 31 days x 4 runs)
uv run lithops-cr-gik-cfs/run_lithops_cfs.py \
    --target-month 202512 \
    --max-workers 20

# Dry run — show members without executing
uv run lithops-cr-gik-cfs/run_lithops_cfs.py \
    --target-month 202512 \
    --dry-run

# Fewer forecast hours (e.g., 3 months instead of 6)
uv run lithops-cr-gik-cfs/run_lithops_cfs.py \
    --target-month 202512 \
    --max-workers 50 \
    --max-forecast-hours 2160
```

GCS output: `gs://gik-cfs-aws-tf/run_par_cfs/{target_month}/{init_date}/{run}z/`

---

## End-to-End Example (Local, ~30 seconds)

```bash
cd cfs/

# Step 1: Create template (one-time, ~24s)
uv run run_cfs_template_creation.py \
    --zarr-template \
    --init-date 20251101 --run 00 \
    --local-dir ./cfs_test/template \
    --max-forecast-hours 48

# Step 2: Generate parquet with byte-range references (~8s)
uv run lithops-cr-gik-cfs/run_lithops_cfs.py \
    --init-date 20251101 --run 00 \
    --sequential \
    --local-template cfs_test/template/cfs-zarr-template-20251101-00.parquet \
    --output-dir ./cfs_output \
    --max-forecast-hours 48 \
    --yes

# Step 3: Verify the output parquet
python -c "
import pandas as pd
df = pd.read_parquet('cfs_output/2025110100z.parquet')
data_refs = [k for k in df['key'] if k.startswith('step_')]
print(f'Total keys: {len(df)}')
print(f'Data references: {len(data_refs)}')
print(f'Metadata keys: {len(df) - len(data_refs)}')
"
```

---

## CLI Reference

### `run_cfs_template_creation.py`

| Flag | Description |
|------|-------------|
| `--init-date YYYYMMDD` | Single init date to process |
| `--init-month YYYYMM` | Process all dates+runs in a month (124 members) |
| `--run 00\|06\|12\|18` | Run hour (default: 00) |
| `--all-runs` | Process all 4 run hours |
| `--local-dir PATH` | Save output to local directory |
| `--bucket NAME` | Upload output to GCS bucket |
| `--zarr-template` | Create zarr skeleton template (instead of mapping parquets) |
| `--max-forecast-hours N` | Max forecast hours (default: 5160) |

### `lithops-cr-gik-cfs/run_lithops_cfs.py`

| Flag | Description |
|------|-------------|
| `--init-date YYYYMMDD` | Single init date (for testing) |
| `--target-month YYYYMM` | Full 124-member ensemble |
| `--run 00\|06\|12\|18` | Run hour (default: 00, with `--init-date`) |
| `--max-forecast-hours N` | Max forecast hours (default: 4320) |
| `--sequential` | Local execution, no Lithops |
| `--local-template PATH` | Use local zarr template parquet |
| `--output-dir PATH` | Save output locally instead of GCS |
| `--max-workers N` | Max Lithops Cloud Run workers (default: 10) |
| `--dry-run` | Show what would be processed |
| `--yes`, `-y` | Skip confirmation prompt |

---

## Output Format

Each parquet file contains a DataFrame with columns `[key, value]`:

- **Metadata keys** (from template): `.zattrs`, `.zarray`, `.zgroup` entries
  defining zarr structure for all 68 variable groups
- **Data reference keys** (from `.idx` parsing): `step_XXXX/variable/level/0.0.0`
  entries containing `[s3_url, byte_offset, byte_length]` triplets

### Variables Captured (103 total from CFS flux files)

All variables from the `.idx` files are captured, including:

| Variable | Level | Description |
|----------|-------|-------------|
| PRATE | surface | Precipitation rate |
| TMP | 2 m above ground | 2-meter temperature |
| TMAX | 2 m above ground | Maximum temperature |
| TMIN | 2 m above ground | Minimum temperature |
| DSWRF | surface | Downward shortwave radiation |
| USWRF | surface | Upward shortwave radiation |
| DLWRF | surface | Downward longwave radiation |
| ULWRF | surface | Upward longwave radiation |
| UGRD | 10 m above ground | 10-meter U wind |
| VGRD | 10 m above ground | 10-meter V wind |
| SOILW | 4 depth layers | Soil moisture |
| TCDC | 6 cloud layers | Total cloud cover |
| PRES | surface + cloud levels | Pressure |
| ... | ... | 103 variable/level combinations total |

---

## CFS Ensemble Structure

CFS uses a multi-date ensemble from the preceding month:

```
Target month: December 2025
Init dates:   November 1-30, 2025 (31 days)
Runs per day:  4 (00z, 06z, 12z, 18z)
Total members: 31 x 4 = 124
Forecast range: Up to 215 days (5160 hours), 6-hourly
Files per member: ~861 GRIB files
```

Each member is one `(init_date, run)` pair processed independently.

---

## File Structure

```
cfs/
├── README.md                              # This file
├── run_cfs_template_creation.py           # Phase 1: one-time template creation
├── lithops-cr-gik-cfs/
│   ├── run_lithops_cfs.py                 # Phase 2: parquet generation
│   ├── Dockerfile                         # Cloud Run container image
│   ├── cloudbuild.yaml                    # GCP Cloud Build config
│   └── lithops_config.yaml                # Lithops backend config
├── GIK_THREE_DATASET_COMPARISON.md        # CFS vs ECMWF vs GEFS comparison
├── cfs_test/template/                     # Template output directory
└── cfs_output/                            # Parquet output directory
```
