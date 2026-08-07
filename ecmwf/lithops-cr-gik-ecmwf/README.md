# ECMWF Parquet Generation - Lithops Cloud Run

Lithops-native Cloud Run deployment for ECMWF ensemble forecast parquet generation using the GIK (Grib-Index-Kerchunk) three-stage pipeline.

**Project**: `e4drr-crafd`
**Region**: `europe-west3` (Frankfurt - co-located with ECMWF S3 data in AWS `eu-central-1`)
**GCS Output**: `gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/`
(the `run_par_ecmwf/` path without the version prefix is the **legacy, pre-per-level-fix** catalog — never write there)
**Service Account**: `ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com`

---

## Upstream sync — porting the working method from the private repo

> **Status: APPLIED 2026-08-07** from `cno-e4drr/devops/lithops_cr_ecmwf_gik/`
> (private). `run_lithops_ecmwf.py`, `Dockerfile` and `lithops_config.yaml` are
> now byte-identical to the private copies; 23 files were added. Verified: no
> key material, `terraform/`, `service_account/`, `*.orig` or
> `.ipynb_checkpoints/` crossed over.
>
> ⚠️ **One post-sync action required** — see *Credential path* below.

### ⚠️ Credential path — action required

`lithops_config.yaml:29` expects the deployer key at:

```
service_account/ecmwf-lithops-deployer-key.json
```

but in this repo the key currently sits at the directory root
(`ecmwf-lithops-deployer-key.json`). **A run will fail to authenticate until
these agree.** Pick one:

```bash
# (a) match the private layout -- covered by the root .gitignore rule
#     `**/service_account/*.json`
mkdir -p service_account && git mv --force 2>/dev/null; \
  mv ecmwf-lithops-deployer-key.json service_account/

# (b) or point the config at the current location
#     lithops_config.yaml: credentials_path: ecmwf-lithops-deployer-key.json
#     (already ignored by the root rule `**/*-key.json`)
```

Option (a) keeps this directory identical to the private one, which is why
`lithops_config.yaml` is shipped that way. After either, confirm the key is
still ignored:

```bash
git check-ignore -v <path-to-key>      # must print a matching rule
```

The authoritative deployment lives in the **private** `cno-e4drr` repo. This
public copy has drifted: **3 code files differ** and **24 files are absent**.
Two of the three differences are correctness traps, not cosmetics.

### Where the template is actually chosen — it is NOT `TEMPLATE_URL`

This is the part that is easy to get wrong. `TEMPLATE_URL` is **dead code on
Cloud Run**. The real chain is build-time:

```
cloudbuild.yaml  --substitutions=_TEMPLATE_ARTIFACT=<tarball>
   -> Dockerfile ARG TEMPLATE_ARTIFACT           (default: ...-50r1.tar.gz)
   -> curl into /opt/ecmwf_templates/<tarball>   (baked into the image)
   -> ENV ECMWF_TEMPLATE_PATH=/opt/ecmwf_templates/<tarball>
   -> ensure_template() returns ECMWF_TEMPLATE_PATH BEFORE reading TEMPLATE_URL
```

So **the image tag *is* the template**, and `lithops_config.yaml: runtime:`
picks the tag. Exporting `TEMPLATE_URL` cannot override an era.

| era | image tag | `_TEMPLATE_ARTIFACT` | `_REFERENCE_DATE` | `_RESOLUTION` | `_CONTROL_STREAM` |
|---|---|---|---|---|---|
| 0p4 | `:0p4` | `gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz` | `20230601` | `0p4` | `enfo` |
| 49r1 | `:49r1` | `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` | `20250515` | `0p25` | `enfo` |
| 50r1 | `:50r1` | `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` | `20260513` | `0p25` | `oper` |

`cloudbuild.yaml` here is **already identical** to the private copy, so the
per-era build substitutions are in place. Only the three files below need work.

### Files to update

| file | change | why it matters |
|---|---|---|
| `run_lithops_ecmwf.py` | `GCS_PARQUET_PREFIX` default `run_par_ecmwf` → **`v20260623_run_par_ecmwf`** | **Correctness.** The legacy path holds *pre-fix* pars whose pl chunk keys lost the level segment — 13 pressure levels collapsed onto one arbitrary message. Anyone running this copy without exporting the env var silently uses the broken catalog. |
| `run_lithops_ecmwf.py` | add **`forecast_hours_for_run(run)`**; use it at the two `ALL_FORECAST_HOURS` call sites | **Correctness + cost.** 00z/12z are 85 steps; 06z/18z stop at +144h (49 steps). Without it a 06z date issues 36 × 51 = **1,836 futile ranged GETs** against a throttling bucket. |
| `run_lithops_ecmwf.py` | PEP 723 pin `lithops==3.6.3` → **`3.6.4`** | Must match the deployed runtime. |
| `Dockerfile` | `lithops` → **`lithops==3.6.4`**; drop `namegenerator` | Client/runtime version parity — an unpinned install drifts from the client. |
| `lithops_config.yaml` | `runtime: …/ecmwf-lithops-runtime` → **`…:50r1`** (era tag) | **Switch 2.** An untagged runtime does not name an era; a mismatch against the exported `ECMWF_*` env is **silent** — it writes 0 files or 50, never an error. |

### Files added (23)

| group | files |
|---|---|
| **era/cycle tooling** | `era_check.py` (era reference + preflight + GCS verify), `fix_cycle_boundary_dates.sh` |
| **cycle runbooks** | `cycles/README.md`, `cycles/{06z,12z,18z}/PLAN.md`, `cycles/{06z,12z,18z}/RESULTS.md` |
| **wave drivers** | `run_cycle_waves.sh`, `run_cycle_herbie_gate.sh`, `rebake_50r1_00z.sh`, `run_jan2026_backfill.sh` |
| **build/deploy notes** | `.dockerignore`, `DEPLOYMENT_PLAN.md`, `DEPLOYMENT_SUCCESS.md`, `ECFLOW_50R1_OPERATIONALIZATION.md`, `ECMWF_00Z_BACKFILL_SUMMARY.md`, `FILENAME_STRUCTURE_UPDATE.md`, the three `*-DONE*.md` era build records |
| **misc** | `logs/.gitignore` |

### Never copy

`service_account/*.json` (live deployer key), `terraform/terraform.tfvars`,
`logs/*.log`, `**/.ipynb_checkpoints/`, `*.orig`. The repo-root `.gitignore`
covers the key patterns — verify with `git check-ignore -v <path>` before adding.

### Verify after syncing

```bash
export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf AWS_NO_SIGN_REQUEST=YES
export ECMWF_REFERENCE_DATE=20250515 ECMWF_RESOLUTION=0p25 ECMWF_CONTROL_STREAM=enfo
uv run era_check.py preflight --run 00 --date 20260319   # do both switches agree?
uv run era_check.py verify --run 00 --start 20260319 --end 20260319
```

`preflight` is the check that catches the silent era mismatch; it compares the
exported env against the `runtime:` tag in `lithops_config.yaml`.

---

## Table of Contents

0. [Upstream sync — porting the working method](#upstream-sync--porting-the-working-method-from-the-private-repo)
1. [What This Does](#what-this-does)
2. [How Lithops Works](#how-lithops-works)
3. [Architecture](#architecture)
4. [File Structure](#file-structure)
5. [Prerequisites](#prerequisites)
6. [Deployment](#deployment)
   - [Step 1: Service Account (Terraform)](#step-1-service-account-terraform)
   - [Step 2: Build Runtime Image (Cloud Build)](#step-2-build-runtime-image-cloud-build)
   - [Step 3: Deploy Lithops Runtime](#step-3-deploy-lithops-runtime)
7. [Running](#running)
   - [Single Date](#single-date)
   - [Date Range / Batch](#date-range--batch)
   - [Sequential Local Test](#sequential-local-test)
   - [Dry Run](#dry-run)
8. [Configuration Reference](#configuration-reference)
9. [Runtime Management](#runtime-management)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Cost Estimates](#cost-estimates)
12. [Troubleshooting](#troubleshooting)
13. [Legacy Files](#legacy-files)

---

## What This Does

Generates daily parquet reference files for ECMWF ensemble weather forecasts (51 members: control + ens01-ens50) and uploads them to GCS. Each date produces 51 parquet files containing Kerchunk-style references that point back to the original GRIB2 data on ECMWF's public S3 bucket (`s3://ecmwf-forecasts/` in AWS `eu-central-1`).

The processing follows a three-stage GIK pipeline:

| Stage | What it does | Time |
|-------|-------------|------|
| **Stage 1** | Load zarr template from HuggingFace tar.gz (~120 MB) | ~5s |
| **Stage 2** | Fetch `.index` files from ECMWF S3, parse GRIB references, merge with template for all 51 members (~6,685 refs each) | ~7 min |
| **Stage 3** | Create final parquet files from merged references | ~0.5s |
| **Upload** | Write 51 parquet files to GCS | ~1s |

**Output path**: `gs://gik-ecmwf-aws-tf/run_par_ecmwf/YYYYMMDD_00z/`

---

## How Lithops Works

[Lithops](https://github.com/lithops-cloud/lithops) is a Python multi-cloud serverless computing framework. Instead of writing a Flask/HTTP worker app and calling it from Python, Lithops **serializes your Python function directly** and runs it on cloud infrastructure.

### Key Concept: No Application Code in the Container

The Cloud Run container runs a **generic Lithops proxy** (Flask/gunicorn). Your actual processing function (`process_ecmwf_date()`) is:

1. Serialized locally using `cloudpickle` (captures the function + all its module-level dependencies)
2. Uploaded to GCS as a pickle blob
3. Sent to Cloud Run workers via HTTP POST (just the GCS keys, not the function itself)
4. Workers download the pickle from GCS, deserialize, and execute

This means you can change your processing logic and re-run **without rebuilding the container**. The container only needs to be rebuilt when you add/remove Python packages.

### Execution Flow

```
Orchestrator (local machine)           GCS Bucket                Cloud Run Workers
────────────────────────               ──────────                ─────────────────
uv run run_lithops_ecmwf.py
  |
  +-- cloudpickle(process_ecmwf_date)
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

### Why Lithops-Native (Not Flask Worker)

The original approach deployed a Flask/Gunicorn app to Cloud Run and had Lithops call it via HTTP. This **nullifies Lithops' purpose** because:

- You're manually managing the Cloud Run service, scaling, and deployment
- Lithops becomes a glorified HTTP client
- Function changes require container rebuilds

With Lithops-native, the framework manages the entire Cloud Run lifecycle. You just write a Python function and call `fexec.map()`.

---

## Architecture

```
run_lithops_ecmwf.py (local, via uv run)
         |
         v
  lithops.FunctionExecutor(backend='gcp_cloudrun')
    /    |    \
   /     |     \     cloudpickle-serialized via GCS
  v      v      v
[CR 1]  [CR 2]  ... [CR N]    Lithops proxy containers (europe-west3)
  |      |           |
  v      v           v
process_ecmwf_date() executes inside each container:
  Stage 1: Load zarr template from HuggingFace
  Stage 2: Fetch S3 .index files + merge refs (51 members)
  Stage 3: Create parquet files
  Upload: Write to GCS
  |      |           |
  v      v           v
gs://gik-ecmwf-aws-tf/run_par_ecmwf/YYYYMMDD_00z/
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestrator | `run_lithops_ecmwf.py` via `uv run` | Serialize function, dispatch to workers, collect results |
| Compute Backend | GCP Cloud Run (europe-west3) | Serverless container execution |
| Runtime Image | `gcr.io/e4drr-crafd/ecmwf-lithops-runtime` | Lithops proxy + Python processing packages |
| Temp Storage | GCS (`lithops-europe-west3-*`) | Lithops internal: function/data/result pickle blobs |
| Output Storage | GCS (`gik-ecmwf-aws-tf`) | Final parquet files |
| Data Source | ECMWF S3 (`s3://ecmwf-forecasts/`, AWS eu-central-1) | Public GRIB2 ensemble forecasts |
| Template Source | HuggingFace (`Nishadhka/gfs_s3_gik_refs`) | Pre-built zarr template (~120 MB) |
| Build System | Cloud Build | Docker image creation (no local Docker needed) |
| Image Registry | GCR (`gcr.io/e4drr-crafd/`) | Docker image storage |
| IAM/Infra | Terraform | Service account, API enablement, IAM roles |

---

## File Structure

```
lithops_cr_ecmwf_gik/
├── run_lithops_ecmwf.py          # Orchestrator + processing function (PEP 723, uv run)
├── Dockerfile                     # Lithops runtime image (NOT a Flask app)
├── cloudbuild.yaml                # Cloud Build config for GCR
├── lithops_config.yaml            # Lithops backend config (region, memory, workers)
├── README.md                      # This document
├── .dockerignore
├── .gitignore
├── service_account/               # Terraform: SA, APIs, IAM, Artifact Registry
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── terraform.tfvars.example
│   ├── .gitignore
│   └── ecmwf-lithops-deployer-key.json  (generated, git-ignored)
├── terraform/                     # Legacy (Lithops manages Cloud Run now)
│   ├── cloud_run.tf               # Not used - Lithops deploys Cloud Run service
│   └── ...
├── DEPLOYMENT_GUIDE.md            # Legacy (Flask-based approach)
├── DEPLOYMENT_PLAN.md             # Legacy (Flask-based approach)
└── DEPLOYMENT_SUCCESS.md          # Legacy (Flask-based approach)
```

### Key Files Explained

**`run_lithops_ecmwf.py`** (~950 lines) - The main script. Contains:
- PEP 723 inline dependency metadata (runs with `uv run`, no venv needed)
- The `process_ecmwf_date()` function that Lithops serializes and sends to Cloud Run
- All GIK pipeline helpers (template loading, S3 index parsing, reference merging, parquet creation, GCS upload)
- CLI argument parsing (`--date`, `--days-back`, `--start-date/--end-date`, `--sequential`, `--dry-run`, `--max-workers`)
- Lithops orchestration (`FunctionExecutor.map()`)

**`Dockerfile`** - A Lithops runtime image, NOT an application:
- Installs `lithops` via pip + all processing dependencies (pandas, pyarrow, fsspec, s3fs, gcsfs)
- Copies `entry_point.py` from the installed lithops package as `lithopsproxy.py`
- Runs gunicorn serving the Lithops proxy (not your code)
- Built via Cloud Build, not local Docker

**`lithops_config.yaml`** - Tells Lithops how to connect:
- Backend: `gcp_cloudrun` in `europe-west3`
- Storage: `gcp_storage` in `europe-west3`
- Runtime image: `gcr.io/e4drr-crafd/ecmwf-lithops-runtime`
- Resources: 2 GB RAM, 2 vCPUs, 3600s timeout, 20 max workers

**`cloudbuild.yaml`** - Cloud Build config:
- Builds Docker image and pushes to `gcr.io/e4drr-crafd/ecmwf-lithops-runtime`
- Tags: `$BUILD_ID` (versioned) + `latest`
- Uses `E2_HIGHCPU_8` machine for faster builds

---

## Prerequisites

- `gcloud` CLI authenticated with access to `e4drr-crafd` project
- `terraform` >= 1.5.0
- `uv` (Python package runner) - [install](https://docs.astral.sh/uv/)
- `lithops` CLI tool: `uv tool install lithops --with httplib2 --with google-auth --with google-cloud-storage --with google-api-python-client --with google-cloud-pubsub --with gcsfs`
- Service account key at `service_account/ecmwf-lithops-deployer-key.json`

---

## Deployment

### Step 1: Service Account (Terraform)

One-time setup. Creates the service account with IAM roles for Cloud Build, Cloud Run, GCS, and Artifact Registry.

```bash
cd lithops_cr_ecmwf_gik/service_account/

# Use admin account
gcloud config set account nkalladath@icpac.net

terraform init
terraform apply -var="project_id=e4drr-crafd"

# Extract the SA key
terraform output -raw service_account_key_private | base64 -d > ecmwf-lithops-deployer-key.json
```

**IAM roles granted:**
- `roles/cloudbuild.builds.editor` + `roles/cloudbuild.builds.builder` - Build images
- `roles/artifactregistry.admin` - Push/pull images
- `roles/run.admin` - Deploy Cloud Run services
- `roles/iam.serviceAccountUser` - Act as service account
- `roles/storage.admin` - Read/write GCS (Lithops temp + output)
- `roles/logging.logWriter` - Cloud Run logs

### Step 2: Build Runtime Image (Cloud Build)

Builds the Docker image and pushes to GCR. No local Docker required.

```bash
cd lithops_cr_ecmwf_gik/

# Activate the deployer SA
gcloud auth activate-service-account \
  --key-file=service_account/ecmwf-lithops-deployer-key.json

# Submit build
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
```

**Build time**: ~2 minutes
**Output**: `gcr.io/e4drr-crafd/ecmwf-lithops-runtime:${_ERA}` (per-era tag; default `50r1`). See "Three template eras" below for `--substitutions` per era.

### Step 3: Deploy Lithops Runtime

Creates the Cloud Run service from the built image. Lithops handles this via its CLI.

```bash
cd lithops_cr_ecmwf_gik/

lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
  -b gcp_cloudrun -s gcp_storage \
  --config lithops_config.yaml
```

This creates a Cloud Run service named `lithops-worker-363-*` in `europe-west3` with the configured resources (2 GB, 2 vCPUs). Lithops also creates a storage bucket (`lithops-europe-west3-*`) for temporary data.

**Note**: If you skip this step, Lithops will auto-deploy the runtime on the first `fexec.map()` call.

---

## Three template eras (per-era deploy)

ECMWF open data spans **4 schema eras**; each needs its own per-level template
+ runtime env. One Cloud Run image per era (select at build via Cloud Build
`--substitutions`; deploy the matching tag). See
`ecmwf/docs/2026-06-03-per-era-deploy-prep.md` (public repo) for the full
matrix and re-bake order.

| Era | TEMPLATE_ARTIFACT | REFERENCE_DATE | RESOLUTION | CONTROL_STREAM | Dates covered |
|---|---|---|---|---|---|
| 49r1 13-level | `gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz` | 20250515 | 0p25 | enfo | 2025-01-14 06z → 2026-05-12 00z (+9-level era as superset from 2024-02-29) |
| 50r1 14-level | `gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz` (default) | 20260513 | 0p25 | oper | 2026-05-12 06z → present |
| 0.4-beta 9-level | `gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz` | 20230601 | 0p4 | enfo | 2023-01-18 → 2024-02-28 |

Build a given era (49r1 example):
```bash
gcloud builds submit --config=cloudbuild.yaml --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com \
  --substitutions=_ERA=49r1,_TEMPLATE_ARTIFACT=gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz,_REFERENCE_DATE=20250515,_RESOLUTION=0p25,_CONTROL_STREAM=enfo
# -> gcr.io/e4drr-crafd/ecmwf-lithops-runtime:49r1 ; deploy that tag, then:
bash run_backfill_00z.sh --era 49r1 --from 2026-03 --to 2026-05   # MAM 2026 (cGAN)
```
**MAM 2026 is in the 49r1 era** (50r1 starts 2026-05-12 06z), so the 49r1 image
unblocks the cGAN MAM 2025+2026 windows; deploy it first.

## Running

All commands use `uv run` which automatically installs dependencies from the PEP 723 header.

### Single Date

```bash
cd lithops_cr_ecmwf_gik/

# Process one date on Cloud Run
uv run run_lithops_ecmwf.py --date 20260210
```

### Date Range / Batch

```bash
# Last 7 days
uv run run_lithops_ecmwf.py --days-back 7

# Specific range
uv run run_lithops_ecmwf.py --start-date 20240301 --end-date 20240331

# Control parallelism
uv run run_lithops_ecmwf.py --days-back 30 --max-workers 20
```

### Sequential Local Test

Runs the processing function on the local machine without Lithops or Cloud Run. Useful for debugging.

```bash
uv run run_lithops_ecmwf.py --date 20260210 --sequential
```

### Dry Run

Shows which dates would be processed without executing.

```bash
uv run run_lithops_ecmwf.py --days-back 30 --dry-run
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
    region: europe-west3
    credentials_path: service_account/ecmwf-lithops-deployer-key.json

gcp_cloudrun:
    runtime: gcr.io/e4drr-crafd/ecmwf-lithops-runtime
    runtime_memory: 2048       # 2 GB
    runtime_cpu: 2             # 2 vCPUs
    runtime_timeout: 3600      # 1 hour
    max_workers: 20
    min_workers: 0
    worker_processes: 1        # 1 process per container

gcp_storage:
    bucket: gik-ecmwf-aws-tf
    region: europe-west3
```

### Environment Variables (in Dockerfile / Cloud Run)

| Variable | Value | Purpose |
|----------|-------|---------|
| `AWS_NO_SIGN_REQUEST` | `YES` | Anonymous access to ECMWF public S3 data |
| `PORT` | `8080` | Gunicorn listen port (Cloud Run requirement) |
| `CONCURRENCY` | `1` | One worker process per container |
| `TIMEOUT` | `3600` | Gunicorn timeout (1 hour) |
| `ECMWF_REFERENCE_DATE` | era-specific | Template reference date (49r1=20250515, 50r1=20260513, 0p4=20230601) |
| `ECMWF_RESOLUTION` | `0p25`/`0p4` | Source path: ifs/0p25 vs 0p4-beta |
| `ECMWF_CONTROL_STREAM` | `enfo`/`oper` | Control member stream (50r1=oper; 49r1/0p4=enfo, bundled) |
| `TEMPLATE_URL` / `ECMWF_TEMPLATE_PATH` | era artifact | Per-era HF template (baked via Dockerfile build-arg) |

### Processing Constants (in run_lithops_ecmwf.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `GCS_BUCKET` | `gik-ecmwf-aws-tf` | Output bucket |
| `GCS_PARQUET_PREFIX` | `run_par_ecmwf` | Output path prefix |
| `S3_BUCKET` | `ecmwf-forecasts` | ECMWF source bucket (AWS eu-central-1) |
| `TEMPLATE_URL` | HuggingFace URL | Pre-built zarr template |
| `ALL_FORECAST_HOURS` | 0-144 (3h) + 150-360 (6h) = 85 steps | Forecast hours to process |

---

## Runtime Management

```bash
# List deployed runtimes
lithops runtime list -b gcp_cloudrun --config lithops_config.yaml

# Delete runtime (removes Cloud Run service)
lithops runtime delete gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
  -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Full cleanup (all runtimes + temp storage)
lithops clean -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml

# Rebuild after Dockerfile changes (add new Python packages)
# Step 1: Cloud Build
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=e4drr-crafd \
  --service-account=projects/e4drr-crafd/serviceAccounts/ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com

# Step 2: Delete old runtime + redeploy
lithops runtime delete gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
  -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml
lithops runtime deploy gcr.io/e4drr-crafd/ecmwf-lithops-runtime \
  -b gcp_cloudrun -s gcp_storage --config lithops_config.yaml
```

---

## Performance Benchmarks

Actual test results from processing date 20260210 (51 members):

| Configuration | Region | Memory | CPU | Total Time | Per-Date Cost |
|--------------|--------|--------|-----|------------|---------------|
| Sequential (local) | local machine | - | - | 40.9 min | $0.00 |
| Cloud Run (original) | us-central1 | 8 GB | 4 vCPU | 23.7 min | $0.165 |
| Cloud Run (optimized) | europe-west3 | 2 GB | 2 vCPU | **8.0 min** | **$0.026** |

The 3x speedup from us-central1 to europe-west3 is because Stage 2 (S3 `.index` file fetches) is network-bound. Moving Cloud Run to Frankfurt (same city as ECMWF's S3 bucket in AWS eu-central-1) eliminates cross-Atlantic latency.

The memory reduction from 8 GB to 2 GB works because the pipeline processes members sequentially within each worker - peak memory stays well under 2 GB.

---

## Cost Estimates

Based on europe-west3 pricing with 2 GB / 2 vCPU / 8 min per date:

| Resource | Rate | Per Date |
|----------|------|----------|
| vCPU-seconds | $0.00002400/s | 2 vCPU x 480s = $0.023 |
| Memory-GiB-seconds | $0.00000250/s | 2 GB x 480s = $0.0024 |
| Requests | $0.40/million | ~$0.00 |
| **Total per date** | | **~$0.026** |

### Batch Estimates

| Scenario | Dates | Cost | Wall Time (20 workers) |
|----------|-------|------|-----------------------|
| Single date | 1 | $0.03 | 8 min |
| Last 7 days | 7 | $0.18 | 8 min (1 batch) |
| Last 30 days | 30 | $0.78 | 16 min (2 batches) |
| Full backfill (2024-03-01 to 2026-02-11) | ~712 | ~$18.50 | ~5 hours |
| Daily automation (monthly) | 30 | $0.78/month | 8 min/day |

Scale to zero when idle: **$0.00/month** base cost.

---

## Troubleshooting

### View Cloud Run Logs

```bash
gcloud logging read \
  "resource.type=cloud_run_revision" \
  --project=e4drr-crafd \
  --limit=50 \
  --format="table(timestamp,severity,textPayload)"
```

### View Lithops Execution Logs

```bash
# Lithops writes detailed logs to /tmp/lithops-<user>/logs/
ls -la /tmp/lithops-*/logs/
cat /tmp/lithops-*/logs/<executor-id>.log
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Docker permission denied | `lithops runtime build` fails | Use Cloud Build (Step 2), not local Docker |
| Module not found on local | `ModuleNotFoundError: httplib2` | Add missing dep to PEP 723 header in run_lithops_ecmwf.py |
| Module not found on worker | `ModuleNotFoundError: <package>` | Add package to Dockerfile, rebuild via Cloud Build |
| Credentials error | `service_account_email` attribute error | Set `credentials_path` in lithops_config.yaml |
| Worker timeout | Function exceeds 3600s | Check network (is region co-located with S3 data?) |
| GCS permission denied | 403 on bucket write | Verify SA has `roles/storage.admin` (check main.tf) |
| ECMWF data unavailable | Index validation fails (0 messages) | Date may be outside ECMWF retention window (~2 years) |
| Lithops can't find config | Default config used | Ensure lithops_config.yaml is in the same directory as run_lithops_ecmwf.py |

### Check GCS Output

```bash
# List processed dates (requires a SA with storage.objects.list on the bucket)
# The ecmwf-lithops-deployer SA has objectAdmin (write) but may lack list
# Use a SA with broader read access, e.g.:
GOOGLE_APPLICATION_CREDENTIALS=/path/to/reader-sa.json \
  gsutil ls gs://gik-ecmwf-aws-tf/run_par_ecmwf/

# Or via Python:
uv run --with google-cloud-storage python3 -c "
from google.cloud import storage
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file('/path/to/reader-sa.json')
client = storage.Client(credentials=creds, project=creds.project_id)
for blob in client.list_blobs('gik-ecmwf-aws-tf', prefix='run_par_ecmwf/', delimiter='/'):
    pass
for prefix in sorted(client.list_blobs('gik-ecmwf-aws-tf', prefix='run_par_ecmwf/', delimiter='/').prefixes):
    print(prefix)
"
```

### Verify Deployment

```bash
# Check Cloud Run service exists
gcloud run services list --project=e4drr-crafd --region=europe-west3

# Check runtime is registered with Lithops
lithops runtime list -b gcp_cloudrun --config lithops_config.yaml
```

---

## Legacy Files

The following files are from the original Flask-based Cloud Run approach (Feb 2026) and are **no longer used**:

| File | Description | Status |
|------|-------------|--------|
| `DEPLOYMENT_GUIDE.md` | Flask worker deployment guide | Superseded by this README |
| `DEPLOYMENT_PLAN.md` | Flask worker deployment plan | Superseded by this README |
| `DEPLOYMENT_SUCCESS.md` | Flask deployment success report | Superseded by this README |
| `test_worker.py` | Flask HTTP endpoint tests | Not applicable (no Flask endpoints) |
| `test_single_date.py` | Flask HTTP endpoint tests | Not applicable |
| `terraform/cloud_run.tf` | Terraform Cloud Run service | Lithops manages Cloud Run now |
| `terraform/main.tf` | Terraform provider | Lithops manages Cloud Run now |

These can be removed once the Lithops-native deployment is confirmed stable for production use.

The `service_account/` directory (Terraform for SA, IAM, Artifact Registry) is still actively used.
