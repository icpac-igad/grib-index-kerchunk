# GEFS Parquet Generation - Lithops Cloud Run

Lithops-native Cloud Run deployment for GEFS (Global Ensemble Forecast System) ensemble forecast parquet generation using the GIK (Grib-Index-Kerchunk) three-stage pipeline.

**Project**: `e4drr-crafd`
**Region**: `us-central1` (Iowa - closest GCP region to NOAA GEFS S3 data in AWS `us-east-1`)
**GCS Output**: `gs://gik-gefs-aws-tf/run_par_gefs/`
**Service Account**: `gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com` (dedicated — NOT shared with ECMWF; see DEPLOYMENT_GUIDE.md)

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [How Lithops Works](#how-lithops-works)
3. [Architecture](#architecture)
4. [File Structure](#file-structure)
5. [Prerequisites](#prerequisites)
6. [Deployment](#deployment)
7. [Running](#running)
8. [Configuration Reference](#configuration-reference)
9. [Runtime Management](#runtime-management)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Cost Estimates](#cost-estimates)
12. [Troubleshooting](#troubleshooting)

---

## What This Does

Generates daily parquet reference files for GEFS ensemble weather forecasts (30 members: gep01-gep30) and uploads them to GCS. Each date produces 30 parquet files containing Kerchunk-style references that point back to the original GRIB2 data on NOAA's public S3 bucket (`s3://noaa-gefs-pds/` in AWS `us-east-1`).

The processing follows a three-stage GIK pipeline:

| Stage | What it does | Time |
|-------|-------------|------|
| **Stage 1** | Build GRIB tree structure from sample files | ~30s |
| **Stage 2** | Create mapped index from HuggingFace template tar.gz (~120 MB) for all 30 members | ~3 min |
| **Stage 3** | Process time dimensions and create final parquet files | ~1 min |
| **Upload** | Write 30 parquet files to GCS | ~1s |

**Output path**: `gs://gik-gefs-aws-tf/run_par_gefs/YYYYMMDD/00z/`

**Filename format**: `{date}{run}z-{member}.parquet` (e.g., `2026021600z-gep01.parquet`)

---

## How Lithops Works

[Lithops](https://github.com/lithops-cloud/lithops) is a Python multi-cloud serverless computing framework. Instead of writing a Flask/HTTP worker app and calling it from Python, Lithops **serializes your Python function directly** and runs it on cloud infrastructure.

### Key Concept: No Application Code in the Container

The Cloud Run container runs a **generic Lithops proxy** (Flask/gunicorn). Your actual processing function (`process_gefs_date()`) is:

1. Serialized locally using `cloudpickle` (captures the function + all its module-level dependencies, including `gefs_util.py`)
2. Uploaded to GCS as a pickle blob
3. Sent to Cloud Run workers via HTTP POST (just the GCS keys, not the function itself)
4. Workers download the pickle from GCS, deserialize, and execute

This means you can change your processing logic and re-run **without rebuilding the container**. The container only needs to be rebuilt when you add/remove Python packages.

### Execution Flow

```
Orchestrator (local machine)           GCS Bucket                Cloud Run Workers
────────────────────────               ──────────                ─────────────────
uv run run_lithops_gefs.py
  |
  +-- cloudpickle(process_gefs_date + gefs_util functions)
  |     |
  |     +---> upload func.pkl -------> gs://lithops-.../func_key
  |     +---> upload data.pkl -------> gs://lithops-.../data_key
  |
  +-- HTTP POST {func_key, data_key} ─────────────────────────> Worker 1 (date A)
  +-- HTTP POST {func_key, data_key} ─────────────────────────> Worker 2 (date B)
  +-- HTTP POST {func_key, data_key} ─────────────────────────> Worker N (date N)
  |                                                                  |
  |                                                           fetch func.pkl
  |                                                           fetch data.pkl
  |                                                           exec func(data)
  |                                                           upload result.pkl
  |                                                                  |
  +-- poll for results <----------- gs://lithops-.../result <--------+
  |
  fexec.get_result()
```

---

## Architecture

```
run_lithops_gefs.py (local, via uv run)
         |
         v
  lithops.FunctionExecutor(backend='gcp_cloudrun')
    /    |    \
   /     |     \     cloudpickle-serialized via GCS
  v      v      v
[CR 1]  [CR 2]  ... [CR N]    Lithops proxy containers (us-central1)
  |      |           |
  v      v           v
process_gefs_date() executes inside each container:
  Stage 1: Build GRIB tree structure
  Stage 2: Create mapped index from template (30 members)
  Stage 3: Process time dimensions + create parquet
  Upload: Write to GCS
  |      |           |
  v      v           v
gs://gik-gefs-aws-tf/run_par_gefs/YYYYMMDD/00z/
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestrator | `run_lithops_gefs.py` via `uv run` | Serialize function, dispatch to workers, collect results |
| Compute Backend | GCP Cloud Run (us-central1) | Serverless container execution |
| Runtime Image | `gcr.io/e4drr-crafd/gefs-lithops-runtime` | Lithops proxy + Python processing packages |
| Temp Storage | GCS (`lithops-us-central1-*`) | Lithops internal: function/data/result pickle blobs |
| Output Storage | GCS (`gik-gefs-aws-tf`) | Final parquet files |
| Data Source | NOAA S3 (`s3://noaa-gefs-pds/`, AWS us-east-1) | Public GRIB2 ensemble forecasts |
| Template Source | HuggingFace (`Nishadhka/gfs_s3_gik_refs`) | Pre-built GEFS template (~120 MB) |
| Build System | Cloud Build | Docker image creation (no local Docker needed) |
| Image Registry | GCR (`gcr.io/e4drr-crafd/`) | Docker image storage |
| IAM/Infra | Terraform (shared) | Service account, API enablement, IAM roles |

---

## File Structure

```
lithops_cr_gefs_gik/
├── run_lithops_gefs.py            # Orchestrator + processing function (PEP 723, uv run)
├── gefs_util.py                   # GEFS GIK utility functions (from grib-index-kerchunk)
├── Dockerfile                     # Lithops runtime image (NOT a Flask app)
├── cloudbuild.yaml                # Cloud Build config for GCR
├── lithops_config.yaml            # Lithops backend config (region, memory, workers)
├── README.md                      # This document
├── .dockerignore
└── .gitignore
```

### Key Files Explained

**`run_lithops_gefs.py`** - The main script. Contains:
- PEP 723 inline dependency metadata (runs with `uv run`, no venv needed)
- The `process_gefs_date()` function that Lithops serializes and sends to Cloud Run
- Imports from `gefs_util.py`: `LocalTarGzMappingManager`, `generate_axes`, `filter_build_grib_tree`, etc.
- GIK pipeline orchestration (Stage 1-3), parquet creation, GCS upload
- CLI argument parsing (`--date`, `--days-back`, `--start-date/--end-date`, `--sequential`, `--dry-run`, `--max-workers`)
- Lithops orchestration (`FunctionExecutor.map()`)

**`gefs_util.py`** (~47KB) - Core GEFS GIK processing utilities:
- `LocalTarGzMappingManager` class - manages template tar.gz file
- `generate_axes()` - generate time axes for GEFS
- `filter_build_grib_tree()` - Stage 1: scan GRIB structure
- `cs_create_mapped_index_local()` - Stage 2: create mapped index from template
- `prepare_zarr_store()` - Stage 3a: prepare zarr store
- `process_unique_groups()` - Stage 3b: process time dimensions
- `calculate_time_dimensions()` - calculate time coordinates

**`Dockerfile`** - A Lithops runtime image, NOT an application:
- Installs `lithops` via pip + all processing dependencies (pandas, pyarrow, fsspec, s3fs, gcsfs, kerchunk, xarray, zarr)
- Pre-downloads GEFS template (gik-fmrc-gefs-20241112.tar.gz, ~120MB) to `/opt/gefs_templates/`
- Copies `entry_point.py` from the installed lithops package as `lithopsproxy.py`
- Runs gunicorn serving the Lithops proxy (not your code)
- Built via Cloud Build, not local Docker

**`lithops_config.yaml`** - Tells Lithops how to connect:
- Backend: `gcp_cloudrun` in `us-central1`
- Storage: `gcp_storage` in `us-central1`
- Runtime image: `gcr.io/e4drr-crafd/gefs-lithops-runtime`
- Resources: 2 GB RAM, 2 vCPUs, 3600s timeout, 20 max workers
- Service account path: `service_account/gefs-lithops-deployer-key.json`

**`cloudbuild.yaml`** - Cloud Build config:
- Builds Docker image and pushes to `gcr.io/e4drr-crafd/gefs-lithops-runtime`
- Tags: `$BUILD_ID` (versioned) + `latest`
- Uses `E2_HIGHCPU_8` machine for faster builds

---

## Prerequisites

1. **GCP Setup**:
   - GCP project: `e4drr-crafd`
   - Dedicated service account `gefs-lithops-deployer` created via local `terraform/` (NOT shared with ECMWF)
   - Required APIs enabled: Cloud Run, Cloud Build, GCR, GCS
   - Service account key at `service_account/gefs-lithops-deployer-key.json`

2. **Local Machine**:
   - Python 3.10+ with `uv` installed
   - `gcloud` CLI configured for `e4drr-crafd` project
   - `lithops` installed: `pip install lithops`

3. **GCS Bucket**:
   - Create bucket: `gsutil mb -p e4drr-crafd -c STANDARD -l us-central1 gs://gik-gefs-aws-tf`
   - Or via Terraform/console

---

## Deployment

### Step 1: Service Account (Dedicated, Standalone)

This deployment uses its own dedicated SA — do NOT reuse the ECMWF SA.
Create it from the terraform module that lives inside this directory:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # edit project_id/buckets
terraform init
terraform apply

# Generate the key
gcloud iam service-accounts keys create \
  ../service_account/gefs-lithops-deployer-key.json \
  --iam-account=gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
  --project=e4drr-crafd
```

See `DEPLOYMENT_GUIDE.md` for the full step-by-step.

### Step 2: Build Runtime Image (Cloud Build)

Build the Docker image via Cloud Build (no local Docker needed):

```bash
cd lithops_cr_gefs_gik/

gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
```

**Build time**: ~20 minutes (downloads GEFS template during build)

**Output**: `gcr.io/e4drr-crafd/gefs-lithops-runtime:latest`

### Step 3: Deploy Lithops Runtime

Deploy the image as a Cloud Run service managed by Lithops:

```bash
lithops runtime deploy gcr.io/e4drr-crafd/gefs-lithops-runtime \
  -b gcp_cloudrun \
  -s gcp_storage \
  --config lithops_config.yaml
```

This creates a Cloud Run service in `us-central1` named `lithops-worker-<hash>`.

**Verify deployment**:
```bash
lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

---

## Running

All commands use `uv run` (no venv activation needed, thanks to PEP 723).

### Single Date

Process a single date (default run: 00z):

```bash
uv run run_lithops_gefs.py --date 20260216
```

Process specific run hour:

```bash
uv run run_lithops_gefs.py --date 20260216 --run 06
uv run run_lithops_gefs.py --date 20260216 --run 12
uv run run_lithops_gefs.py --date 20260216 --run 18
```

### Date Range / Batch

Last 7 days:

```bash
uv run run_lithops_gefs.py --days-back 7
```

Specific date range:

```bash
uv run run_lithops_gefs.py --start-date 20260201 --end-date 20260207
```

Custom parallelism:

```bash
uv run run_lithops_gefs.py --days-back 30 --max-workers 20
```

### Sequential Local Test

Run locally without Lithops (single-threaded, for debugging):

```bash
uv run run_lithops_gefs.py --date 20260216 --sequential
```

This downloads the template to `/tmp/` and processes locally.

### Dry Run

Show what would be processed without executing:

```bash
uv run run_lithops_gefs.py --days-back 30 --dry-run
```

---

## Configuration Reference

### lithops_config.yaml

```yaml
lithops:
    backend: gcp_cloudrun
    storage: gcp_storage

gcp:
    project_name: e4drr-crafd
    region: us-central1
    credentials_path: service_account/gefs-lithops-deployer-key.json

gcp_cloudrun:
    runtime: gcr.io/e4drr-crafd/gefs-lithops-runtime
    runtime_memory: 2048       # 2 GB (adjust based on testing)
    runtime_cpu: 2             # 2 vCPUs
    runtime_timeout: 3600      # 1 hour
    max_workers: 20
    min_workers: 0
    worker_processes: 1

gcp_storage:
    bucket: gik-gefs-aws-tf
    region: us-central1
```

### Environment Variables

Override defaults in `run_lithops_gefs.py`:

```bash
export GCS_BUCKET=gik-gefs-aws-tf
export GCS_PARQUET_PREFIX=run_par_gefs
export PARALLEL_WORKERS=4          # Threads for Stage 2 (per worker)
export GEFS_TEMPLATE_PATH=/opt/gefs_templates/gik-fmrc-gefs-20241112.tar.gz
```

---

## Runtime Management

### List Runtimes

```bash
lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

### Delete Runtime

```bash
lithops runtime delete gcr.io/e4drr-crafd/gefs-lithops-runtime \
  -b gcp_cloudrun \
  -s gcp_storage \
  --config lithops_config.yaml
```

This deletes the Cloud Run service but **not** the GCR image.

### Update Runtime (After Code Changes)

If you only changed `run_lithops_gefs.py` or `gefs_util.py`, **no rebuild needed**. Lithops serializes your code automatically.

If you changed dependencies in `Dockerfile`:

1. Rebuild image: `gcloud builds submit --config=cloudbuild.yaml ...`
2. Update runtime: `lithops runtime update ...` (or delete + redeploy)

---

## Performance Benchmarks

Based on ECMWF deployment (GEFS expected to be similar or faster due to fewer members):

| Dates | Workers | Time | Cost | Avg per date |
|-------|---------|------|------|--------------|
| 1 date | 1 | ~7 min | $0.01 | ~7 min |
| 7 dates | 7 | ~8 min | $0.05 | ~1 min |
| 30 dates | 20 | ~15 min | $0.20 | ~30s |

**Processing per date**: ~4-7 minutes (30 members × ~10-15s each)

---

## Cost Estimates

**Cloud Run (us-central1, 2GB/2vCPU)**:
- $0.00002400 per vCPU-second
- $0.00000250 per GB-second
- ~$0.01 per date (7 min × 2 vCPU × $0.00002400)

**GCS Storage**:
- Standard: $0.020/GB-month
- 30 parquet files @ ~50KB each = ~1.5MB per date
- Negligible cost

**Cloud Build**:
- Free tier: 120 build-minutes/day
- One-time build (~20 min): Free

**Monthly cost (30 dates/day)**:
- Processing: 30 dates × $0.01 = **$0.30/day** = **$9/month**
- Storage: 30 days × 1.5MB × $0.020/GB = **$0.001/month**
- **Total**: ~$9/month

---

## Troubleshooting

### Build Failures

**Error**: `Template download failed`
```bash
# Template URL may be down - check HuggingFace
curl -I https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs/resolve/main/gik-fmrc-gefs-20241112.tar.gz
```

**Fix**: Update `TEMPLATE_URL` in Dockerfile if template moved

### Runtime Deployment Failures

**Error**: `Permission denied`
```bash
# Check service account has Cloud Run Admin role
gcloud projects get-iam-policy e4drr-crafd \
  --flatten="bindings[].members" \
  --filter="bindings.members:gefs-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com"
```

### Processing Failures

**Error**: `GEFS data not available`
```bash
# Check S3 data exists
aws s3 ls s3://noaa-gefs-pds/gefs.20260216/00/atmos/pgrb2sp25/ --no-sign-request | head
```

**Fix**: GEFS data may not be available for recent dates (check NOAA availability)

**Error**: `Import error: gefs_util`
```bash
# Check gefs_util.py exists in same directory
ls -lh lithops_cr_gefs_gik/gefs_util.py
```

**Fix**: Ensure `gefs_util.py` is in the same directory as `run_lithops_gefs.py`

### Memory Issues

**Error**: `Cloud Run memory exceeded`
```bash
# Increase memory in lithops_config.yaml
runtime_memory: 4096  # 4 GB
```

Then redeploy runtime.

### Slow Processing

- **Check region**: `us-central1` is optimal for NOAA S3 (us-east-1)
- **Check workers**: Increase `max_workers` in lithops_config.yaml
- **Check S3 throttling**: NOAA may rate-limit requests

---

## GEFS Data Specifications

**S3 Bucket**: `s3://noaa-gefs-pds/` (AWS us-east-1, public)

**S3 Path Pattern**:
```
s3://noaa-gefs-pds/gefs.{date}/{run}/atmos/pgrb2sp25/{member}.t{run}z.pgrb2s.0p25.f{hour:03d}
```

**Example**:
```
s3://noaa-gefs-pds/gefs.20260216/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f000
s3://noaa-gefs-pds/gefs.20260216/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f003
...
s3://noaa-gefs-pds/gefs.20260216/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f240
```

**Ensemble Members**: gep01-gep30 (30 members, no control member)

**Forecast Hours**: 0-240h at 3h intervals (81 timesteps)

**Runs per day**: 4 (00z, 06z, 12z, 18z)

**Resolution**: 0.25° global grid

**Variables**: TMP (2m temp), APCP (precip), UGRD/VGRD (10m winds), and more

---

## Next Steps

1. **Automate Daily Runs**: Use Cloud Scheduler to trigger processing daily
2. **Add Variables**: Extend `FORECAST_VARIABLES` dict in `run_lithops_gefs.py`
3. **Data Validation**: Add parquet integrity checks
4. **Monitoring**: Set up Cloud Monitoring alerts for failures
5. **Cost Optimization**: Tune memory/CPU based on actual usage

---

## References

- [GEFS on AWS](https://registry.opendata.aws/noaa-gefs/)
- [Lithops Documentation](https://lithops-cloud.github.io/)
- [GIK Method](https://github.com/Unidata/grib-index-kerchunk)
- [GEFS Template (HuggingFace)](https://huggingface.co/datasets/Nishadhka/gfs_s3_gik_refs)
